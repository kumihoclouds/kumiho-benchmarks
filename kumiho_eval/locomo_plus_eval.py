"""
LoCoMo-Plus Evaluation for Kumiho Cognitive Memory.

Evaluates Level-2 Cognitive Memory — implicit constraint recall across
long conversations with cue–trigger semantic disconnect.

Unlike LoCoMo (factual QA, categories 1-5), LoCoMo-Plus tests whether
the memory system can connect a later trigger query to an earlier cue
dialogue without explicit surface-level overlap.

Four constraint types: causal, state (+ goal, value in future releases)

Dataset: xjtuleeyf/Locomo-Plus (401 entries stitched into LoCoMo-10 conversations)
Paper: arXiv 2602.10715

Reference scores (cognitive category only):
  RAG methods:     23–29%
  Mem0/SeCom/A-Mem: 41–42%
  GPT/Gemini:      40–45%
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from requests.exceptions import ConnectionError as RequestsConnectionError
from tqdm import tqdm

from .common import (
    BenchmarkConfig,
    EvalResult,
    KumihoMemoryAdapter,
    compute_aggregate_metrics,
    generate_answer,
    save_results,
)

_RETRYABLE_ERRORS = (OSError, RequestsConnectionError, ConnectionError, TimeoutError)
MAX_ENTRY_RETRIES = 3
RETRY_BASE_DELAY = 10

logger = logging.getLogger("kumiho_eval.locomo_plus")

DATA_DIR = Path(__file__).resolve().parent.parent / "locomo" / "data"
LOCOMO_PLUS_DATA = DATA_DIR / "locomo_plus.json"
LOCOMO_BASE_DATA = DATA_DIR / "locomo10.json"

# ---------------------------------------------------------------------------
# Checkpoint / resume (same pattern as LongMemEval / MAB)
# ---------------------------------------------------------------------------


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "_checkpoint.jsonl"


def _load_checkpoint(output_dir: Path) -> tuple[list[EvalResult], set[str]]:
    ckpt = _checkpoint_path(output_dir)
    if not ckpt.exists():
        return [], set()

    results: list[EvalResult] = []
    completed: set[str] = set()
    for line in ckpt.read_text(encoding="utf-8").strip().splitlines():
        try:
            data = json.loads(line)
            r = EvalResult(
                question_id=data["question_id"],
                question=data["question"],
                question_type=data["question_type"],
                ground_truth=data["ground_truth"],
                prediction=data["prediction"],
                recalled_context=data.get("recalled_context", ""),
                f1_score=data.get("f1_score", 0.0),
                judge_score=data.get("judge_score", False),
                exact_match=data.get("exact_match", False),
                latency_ingest_ms=data.get("latency_ingest_ms", 0.0),
                latency_recall_ms=data.get("latency_recall_ms", 0.0),
                latency_answer_ms=data.get("latency_answer_ms", 0.0),
                metadata=data.get("metadata", {}),
            )
            results.append(r)
            completed.add(data["question_id"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Skipping corrupt checkpoint line: %s", e)
    logger.info("Loaded checkpoint: %d completed entries", len(completed))
    return results, completed


def _save_checkpoint_line(output_dir: Path, result: EvalResult) -> None:
    ckpt = _checkpoint_path(output_dir)
    data = {
        "question_id": result.question_id,
        "question": result.question,
        "question_type": result.question_type,
        "ground_truth": result.ground_truth,
        "prediction": result.prediction,
        "recalled_context": result.recalled_context[:500],
        "f1_score": result.f1_score,
        "judge_score": result.judge_score,
        "exact_match": result.exact_match,
        "latency_ingest_ms": result.latency_ingest_ms,
        "latency_recall_ms": result.latency_recall_ms,
        "latency_answer_ms": result.latency_answer_ms,
        "metadata": result.metadata,
    }
    with open(ckpt, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_locomo_plus(path: str | Path | None = None) -> list[dict]:
    """Load the LoCoMo-Plus cognitive memory dataset."""
    path = Path(path) if path else LOCOMO_PLUS_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo-Plus data not found at {path}. "
            "Download from: https://github.com/xjtuleeyf/Locomo-Plus"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_locomo_base(path: str | Path | None = None) -> list[dict]:
    """Load the LoCoMo-10 base conversations."""
    path = Path(path) if path else LOCOMO_BASE_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo base data not found at {path}. "
            "Run: git submodule update --init locomo"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Time gap parsing & cue-stitching (reimplemented from LoCoMo-Plus build_conv)
# ---------------------------------------------------------------------------

_WORD_NUMS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
    "eleven": 11, "twelve": 12, "several": 4,
}

_TIME_GAP_RE = re.compile(
    r"(?:about\s+|around\s+)?(\w+)\s+(week|month|year)s?\s+later",
    re.IGNORECASE,
)


def parse_time_gap(gap_str: str) -> int:
    """Convert 'two weeks later' → number of days."""
    m = _TIME_GAP_RE.search(gap_str)
    if not m:
        return 14  # default: 2 weeks
    num_word, unit = m.group(1).lower(), m.group(2).lower()
    num = _WORD_NUMS.get(num_word)
    if num is None:
        try:
            num = int(num_word)
        except ValueError:
            num = 2
    if unit == "week":
        return num * 7
    elif unit == "month":
        return num * 30
    elif unit == "year":
        return num * 365
    return num * 7


_DATE_PATTERNS = [
    # "1:56 pm on 8 May, 2023"
    re.compile(r"(\d{1,2}:\d{2}\s*[ap]m)\s+on\s+(\d{1,2}\s+\w+,?\s+\d{4})", re.I),
    # "8 May, 2023"
    re.compile(r"(\d{1,2}\s+\w+,?\s+\d{4})", re.I),
]


def _parse_session_date(dt_str: str) -> datetime | None:
    """Parse LoCoMo's date_time strings into datetime objects."""
    for pat in _DATE_PATTERNS:
        m = pat.search(dt_str)
        if m:
            groups = m.groups()
            date_part = groups[-1].replace(",", "")
            try:
                return datetime.strptime(date_part, "%d %B %Y")
            except ValueError:
                pass
    return None


def _extract_sessions(conversation: dict) -> list[dict]:
    """Extract ordered sessions with parsed dates from a LoCoMo conversation."""
    sessions = []
    idx = 1
    while True:
        key = f"session_{idx}"
        if key not in conversation:
            break
        dt_str = conversation.get(f"session_{idx}_date_time", "")
        sessions.append({
            "session_num": idx,
            "date_time": dt_str,
            "parsed_date": _parse_session_date(dt_str),
            "turns": conversation[key],
        })
        idx += 1
    return sessions


def _map_ab_speakers(text: str, speaker_a: str, speaker_b: str) -> str:
    """Replace A:/B: with actual speaker names in dialogue text."""
    lines = []
    for line in text.strip().split("\n"):
        line = line.strip()
        if line.startswith("A:"):
            lines.append(f"{speaker_a}: {line[2:].strip()}")
        elif line.startswith("B:"):
            lines.append(f"{speaker_b}: {line[2:].strip()}")
        else:
            lines.append(line)
    return "\n".join(lines)


def stitch_conversation(
    conversation: dict,
    cue_dialogue: str,
    trigger_query: str,
    time_gap_str: str,
) -> tuple[list[dict], str, str]:
    """
    Stitch a cue dialogue and trigger query into a LoCoMo conversation.

    Returns:
        (sessions, mapped_cue, mapped_trigger) — sessions list with cue/trigger
        inserted at appropriate temporal positions.
    """
    speaker_a = conversation.get("speaker_a", "A")
    speaker_b = conversation.get("speaker_b", "B")

    sessions = _extract_sessions(conversation)
    if not sessions:
        return [], cue_dialogue, trigger_query

    # Map A/B to actual speaker names
    mapped_cue = _map_ab_speakers(cue_dialogue, speaker_a, speaker_b)
    mapped_trigger = _map_ab_speakers(trigger_query, speaker_a, speaker_b)

    # Compute insertion points:
    # trigger goes 7 days after the last session
    # cue goes (trigger_date - time_gap_days) before the trigger
    time_gap_days = parse_time_gap(time_gap_str)
    last_date = None
    for s in reversed(sessions):
        if s["parsed_date"]:
            last_date = s["parsed_date"]
            break

    if last_date is None:
        # Fallback: put cue before last session, trigger after
        cue_insert_idx = max(0, len(sessions) - 2)
        trigger_insert_idx = len(sessions)
    else:
        trigger_date = last_date + timedelta(days=7)
        cue_date = trigger_date - timedelta(days=time_gap_days)

        # Find insertion index for cue (first session after cue_date)
        cue_insert_idx = len(sessions)
        for i, s in enumerate(sessions):
            if s["parsed_date"] and s["parsed_date"] >= cue_date:
                cue_insert_idx = i
                break

        trigger_insert_idx = len(sessions) + 1  # after all + cue

    # Build cue turns
    cue_turns = []
    for line in mapped_cue.split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, text = line.split(":", 1)
            cue_turns.append({"speaker": speaker.strip(), "text": text.strip()})
        else:
            cue_turns.append({"speaker": speaker_a, "text": line})

    cue_dt = ""
    if last_date and cue_insert_idx < len(sessions):
        # Use the date of the session we're inserting before
        cue_dt = sessions[cue_insert_idx]["date_time"]
    elif last_date:
        cue_target = last_date + timedelta(days=7) - timedelta(days=time_gap_days)
        cue_dt = cue_target.strftime("%-I:%M %p on %-d %B, %Y") if hasattr(cue_target, "strftime") else ""
        # Windows strftime doesn't support %-
        try:
            cue_dt = cue_target.strftime("%-I:%M %p on %-d %B, %Y")
        except ValueError:
            cue_dt = cue_target.strftime("%I:%M %p on %d %B, %Y").lstrip("0")

    cue_session = {
        "session_num": -1,  # synthetic
        "date_time": cue_dt,
        "parsed_date": None,
        "turns": cue_turns,
        "is_cue": True,
    }

    # Build trigger turn
    trigger_turns = []
    for line in mapped_trigger.split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, text = line.split(":", 1)
            trigger_turns.append({"speaker": speaker.strip(), "text": text.strip()})
        else:
            trigger_turns.append({"speaker": speaker_a, "text": line})

    trigger_dt = ""
    if last_date:
        trigger_target = last_date + timedelta(days=7)
        try:
            trigger_dt = trigger_target.strftime("%-I:%M %p on %-d %B, %Y")
        except ValueError:
            trigger_dt = trigger_target.strftime("%I:%M %p on %d %B, %Y").lstrip("0")

    trigger_session = {
        "session_num": -2,
        "date_time": trigger_dt,
        "parsed_date": None,
        "turns": trigger_turns,
        "is_trigger": True,
    }

    # Insert cue and trigger
    result = list(sessions)
    result.insert(cue_insert_idx, cue_session)
    result.append(trigger_session)

    return result, mapped_cue, mapped_trigger


# ---------------------------------------------------------------------------
# Session ingestion helpers
# ---------------------------------------------------------------------------


def _session_to_messages(session: dict, speaker_a: str) -> list[dict[str, str]]:
    """Convert a session to user/assistant messages for ingestion."""
    dt_str = session.get("date_time", "")
    messages = []
    for turn in session["turns"]:
        role = "user" if turn["speaker"] == speaker_a else "assistant"
        content = f"[{dt_str}] {turn['speaker']}: {turn['text']}"
        messages.append({"role": role, "content": content})
    return messages


# ---------------------------------------------------------------------------
# Cognitive judge
# ---------------------------------------------------------------------------

_COGNITIVE_JUDGE_SYSTEM = "You are a Memory Awareness Judge."

_COGNITIVE_JUDGE_TEMPLATE = """Your task: Judge whether the Model Prediction considers or is linked to the Evidence. If there is a clear connection, the answer is correct (score 1); if not, it is wrong (no score).

Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint). Give 1 point.
- "wrong": The prediction does not show such a link to the evidence. No point.

Memory/Evidence:
{evidence}

Model Prediction:
{prediction}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<Does the prediction relate to the evidence?>"}}
"""


async def cognitive_judge(
    evidence: str,
    prediction: str,
    *,
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
) -> tuple[bool, str]:
    """
    Judge whether a response demonstrates awareness of the cue evidence.

    Matches LoCoMo-Plus paper methodology (Table 7 / prompt.py):
    - Only evidence + prediction are shown to the judge (no trigger)
    - Default judge model: gpt-4o-mini (paper default)

    Returns (is_correct, reason).
    """
    import os

    import backoff
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    prompt = _COGNITIVE_JUDGE_TEMPLATE.format(
        evidence=evidence, prediction=prediction,
    )

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    async def _call():
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _COGNITIVE_JUDGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return resp.choices[0].message.content.strip()

    raw = await _call()

    # Parse JSON response
    try:
        parsed = json.loads(raw)
        label = parsed.get("label", "wrong").lower()
        reason = parsed.get("reason", "")
    except json.JSONDecodeError:
        # Fallback: check for keywords
        lower = raw.lower()
        if '"correct"' in lower or "'correct'" in lower:
            label, reason = "correct", raw
        else:
            label, reason = "wrong", raw

    return label == "correct", reason


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


async def evaluate_locomo_plus(
    config: BenchmarkConfig,
    data_path: str | Path | None = None,
    base_path: str | Path | None = None,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Run the LoCoMo-Plus cognitive memory evaluation.

    For each cognitive entry:
      1. Pick base conversation (cyclic index)
      2. Stitch cue + trigger at appropriate temporal positions
      3. Ingest all sessions → consolidate
      4. Use trigger as recall query → generate response
      5. Judge: does response reflect the cue evidence?

    Returns dict with results, metrics, and per-type breakdown.
    """
    plus_data = load_locomo_plus(data_path)
    base_data = load_locomo_base(base_path)

    if config.max_samples:
        plus_data = plus_data[:config.max_samples]

    adapter = KumihoMemoryAdapter(config)
    output_dir = Path(config.output_dir) / "locomo_plus"
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        all_results, completed_ids = _load_checkpoint(output_dir)
    else:
        all_results, completed_ids = [], set()

    logger.info("LoCoMo-Plus: %d cognitive entries, %d base conversations", len(plus_data), len(base_data))

    # Session-level semaphore (limits concurrent ingestion RPCs to kumiho-server)
    session_sem = asyncio.Semaphore(config.concurrency)
    # Entry-level semaphore (limits concurrent entries being processed)
    entry_sem = asyncio.Semaphore(max(1, config.entry_concurrency))
    # Lock for thread-safe checkpoint writes and result list appends
    checkpoint_lock = asyncio.Lock()
    pbar = tqdm(total=len(plus_data) - len(completed_ids), desc="LoCoMo-Plus")

    async def _process_entry(ei: int, entry: dict) -> None:
        """Process a single cognitive entry end-to-end."""
        entry_id = f"cog-{ei}"

        relation_type = entry.get("relation_type", "unknown")
        cue_dialogue = entry.get("cue_dialogue", "")
        trigger_query_raw = entry.get("trigger_query", "")
        time_gap = entry.get("time_gap", "two weeks later")

        if not cue_dialogue or not trigger_query_raw:
            pbar.update(1)
            return

        # Cyclic assignment to base conversation
        base_conv = base_data[ei % len(base_data)]
        conversation = base_conv["conversation"]
        speaker_a = conversation.get("speaker_a", "A")

        # Stitch cue + trigger into conversation
        stitched_sessions, mapped_cue, mapped_trigger = stitch_conversation(
            conversation, cue_dialogue, trigger_query_raw, time_gap,
        )

        if not stitched_sessions:
            pbar.update(1)
            return

        # Retry on transient errors
        last_error: Exception | None = None
        for attempt in range(1, MAX_ENTRY_RETRIES + 1):
            try:
                user_id = f"locomo-plus-{entry_id}"
                # Space is derived automatically from user_id during
                # consolidation (context/user_id → personal/locomo-plus-cog-N).
                # We still construct it here for recall scoping.
                user_space = f"{config.project_name}/personal/{user_id}"

                # --- Phase 1: Ingest all sessions (parallel) ---
                t_ingest = time.perf_counter()

                # Exclude trigger session from ingestion — it's the query
                ingest_sessions = [
                    s for s in stitched_sessions
                    if not s.get("is_trigger")
                ]

                async def _ingest_one(session: dict) -> str | None:
                    async with session_sem:
                        messages = _session_to_messages(session, speaker_a)
                        if not messages:
                            return None
                        result = await adapter.ingest_session(
                            user_id=user_id,
                            session_messages=messages,
                            context="personal",
                        )
                        sid = result.get("session_id")
                        if sid:
                            try:
                                cons = await adapter.consolidate(
                                    sid,
                                    user_id=user_id,
                                    context="personal",
                                )
                                # Post-consolidation: discover & create edges
                                if config.graph_augmented and cons.get("success"):
                                    store_res = cons.get("store_result", {})
                                    rev_kref = store_res.get("revision_kref", "")
                                    summary = cons.get("summary", "")
                                    if rev_kref and summary:
                                        try:
                                            await adapter.discover_and_link_edges(
                                                rev_kref, summary,
                                            )
                                        except Exception as e:
                                            logger.debug("Edge discovery failed: %s", e)
                            except Exception as e:
                                logger.warning("Consolidation failed: %s", e)
                        return sid

                ingest_results = await asyncio.gather(
                    *[_ingest_one(s) for s in ingest_sessions],
                    return_exceptions=True,
                )
                for r in ingest_results:
                    if isinstance(r, _RETRYABLE_ERRORS):
                        raise r

                ingest_ms = (time.perf_counter() - t_ingest) * 1000

                # --- Phase 2: Recall using trigger query ---
                trigger_text = " ".join(
                    t["text"] for t in stitched_sessions[-1].get("turns", [])
                )

                t_recall = time.perf_counter()
                if config.graph_augmented:
                    memories = await adapter.recall_with_graph_augmentation(
                        trigger_text,
                        limit=config.recall_limit,
                        space_paths=[user_space],
                    )
                else:
                    memories = await adapter.recall(
                        trigger_text,
                        limit=config.recall_limit,
                        space_paths=[user_space],
                    )
                recall_ms = (time.perf_counter() - t_recall) * 1000

                recalled_context = adapter.build_recalled_context(memories)

                # --- Phase 3: Generate response ---
                # Match LoCoMo-Plus INSTRUCTION_COGNITIVE (utils.py)
                system = (
                    "This is a memory-aware dialogue setting. "
                    "You are continuing or reflecting on a prior conversation. "
                    "Show that you are aware of the relevant memory or context "
                    "from your past interactions when responding."
                )

                t_answer = time.perf_counter()
                prediction = await generate_answer(
                    trigger_text,
                    recalled_context,
                    system_prompt=system,
                    model=config.answer_model,
                    api_key=config.openai_api_key,
                    max_tokens=256,
                )
                answer_ms = (time.perf_counter() - t_answer) * 1000

                # --- Phase 4: Cognitive judge ---
                # Paper judge only sees evidence + prediction (no trigger)
                judge_ok, judge_reason = await cognitive_judge(
                    evidence=mapped_cue,
                    prediction=prediction,
                    model=config.judge_model,
                    api_key=config.openai_api_key,
                )

                result = EvalResult(
                    question_id=entry_id,
                    question=trigger_text,
                    question_type=f"cognitive/{relation_type}",
                    ground_truth=mapped_cue,
                    prediction=prediction,
                    recalled_context=recalled_context,
                    f1_score=1.0 if judge_ok else 0.0,
                    judge_score=judge_ok,
                    exact_match=False,
                    latency_ingest_ms=ingest_ms,
                    latency_recall_ms=recall_ms,
                    latency_answer_ms=answer_ms,
                    metadata={
                        "relation_type": relation_type,
                        "time_gap": time_gap,
                        "time_gap_days": parse_time_gap(time_gap),
                        "base_conv_idx": ei % len(base_data),
                        "memories_recalled": len(memories),
                        "judge_reason": judge_reason,
                    },
                )
                async with checkpoint_lock:
                    all_results.append(result)
                    _save_checkpoint_line(output_dir, result)
                pbar.update(1)

                last_error = None
                break

            except _RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < MAX_ENTRY_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    logger.warning(
                        "Network error on %s (attempt %d/%d), retrying in %ds: %s",
                        entry_id, attempt, MAX_ENTRY_RETRIES, delay, e,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Failed entry %s after %d attempts: %s",
                        entry_id, MAX_ENTRY_RETRIES, e,
                    )

        if last_error is not None:
            logger.error("Skipping entry %s due to persistent errors", entry_id)
            pbar.update(1)

    try:
        # Build tasks for all pending entries
        async def _sem_entry(ei: int, entry: dict) -> None:
            async with entry_sem:
                await _process_entry(ei, entry)

        pending = [
            (ei, entry) for ei, entry in enumerate(plus_data)
            if f"cog-{ei}" not in completed_ids
        ]
        await asyncio.gather(
            *[_sem_entry(ei, entry) for ei, entry in pending],
            return_exceptions=True,
        )
    finally:
        pbar.close()
        await adapter.cleanup()

    # Save all results
    save_results(all_results, output_dir / "all_results.json")
    metrics = compute_aggregate_metrics(all_results)

    # Per-relation_type breakdown
    type_metrics: dict[str, Any] = {}
    for rtype in ["causal", "state"]:
        type_results = [
            r for r in all_results
            if r.metadata.get("relation_type") == rtype
        ]
        if type_results:
            type_metrics[rtype] = {
                "count": len(type_results),
                "judge_accuracy": float(np.mean([r.judge_score for r in type_results])),
            }

    # Per time-gap bucket
    gap_metrics: dict[str, Any] = {}
    for r in all_results:
        days = r.metadata.get("time_gap_days", 0)
        if days <= 14:
            bucket = "<=2wk"
        elif days <= 30:
            bucket = "2wk-1mo"
        elif days <= 90:
            bucket = "1-3mo"
        elif days <= 180:
            bucket = "3-6mo"
        else:
            bucket = ">6mo"
        gap_metrics.setdefault(bucket, []).append(r)

    gap_scores = {}
    for bucket, results in gap_metrics.items():
        gap_scores[bucket] = {
            "count": len(results),
            "judge_accuracy": float(np.mean([r.judge_score for r in results])),
        }

    metrics["locomo_plus"] = {
        "total_entries": len(all_results),
        "overall_judge_accuracy": float(np.mean([r.judge_score for r in all_results])) if all_results else 0,
        "by_relation_type": type_metrics,
        "by_time_gap": gap_scores,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    # Print results — cognitive eval uses Judge Accuracy only (no F1/EM)
    lp = metrics["locomo_plus"]
    print(f"\n{'=' * 70}")
    print(f"  LoCoMo-Plus (Cognitive Memory) — Kumiho Evaluation")
    print(f"{'=' * 70}")
    print(f"  Total entries:           {lp['total_entries']}")
    print(f"  Judge Accuracy:          {lp['overall_judge_accuracy']:.4f}")
    print(f"  Avg Recall Latency:      {metrics.get('avg_latency_recall_ms', 0):.1f} ms")
    print(f"  Avg Answer Latency:      {metrics.get('avg_latency_answer_ms', 0):.1f} ms")

    print(f"\n  By Relation Type:")
    print(f"  {'Type':<20} {'Count':>6} {'Judge Acc':>10}")
    print(f"  {'-' * 38}")
    for rtype in ["causal", "state"]:
        if rtype in type_metrics:
            vals = type_metrics[rtype]
            print(f"  {rtype:<20} {vals['count']:>6} {vals['judge_accuracy']:>10.4f}")

    print(f"\n  By Time Gap:")
    print(f"  {'Gap':<20} {'Count':>6} {'Judge Acc':>10}")
    print(f"  {'-' * 38}")
    for bucket in ["<=2wk", "2wk-1mo", "1-3mo", "3-6mo", ">6mo"]:
        if bucket in gap_scores:
            v = gap_scores[bucket]
            print(f"  {bucket:<20} {v['count']:>6} {v['judge_accuracy']:>10.4f}")

    print(f"{'=' * 70}\n")

    return {"results": all_results, "metrics": metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LoCoMo-Plus cognitive memory benchmark on Kumiho")
    parser.add_argument("--data", type=str, default=None, help="Path to locomo_plus.json")
    parser.add_argument("--base-data", type=str, default=None, help="Path to locomo10.json")
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit entries")
    parser.add_argument("--answer-model", type=str, default="gpt-4o")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    parser.add_argument("--recall-limit", type=int, default=10)
    parser.add_argument("--recall-mode", type=str, default="full",
                        choices=["full", "summarized"],
                        help="Recall mode: full (artifact content) or summarized (title+summary)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max parallel session ingestions (shared across entries)")
    parser.add_argument("--entry-concurrency", type=int, default=1,
                        help="Max entries processed in parallel (pipeline parallelism)")
    parser.add_argument("--project", type=str, default="benchmark-locomo-plus")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh instead of resuming from checkpoint")
    parser.add_argument("--graph-augmented", action="store_true",
                        help="Enable graph-augmented recall (follow edges from recalled memories)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = BenchmarkConfig(
        project_name=args.project,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        output_dir=args.output,
        max_samples=args.max_samples,
        recall_limit=args.recall_limit,
        recall_mode=args.recall_mode,
        concurrency=args.concurrency,
        entry_concurrency=args.entry_concurrency,
        graph_augmented=args.graph_augmented,
    )

    asyncio.run(evaluate_locomo_plus(
        config,
        data_path=args.data,
        base_path=args.base_data,
        resume=not args.no_resume,
    ))


if __name__ == "__main__":
    main()
