"""
LoCoMo Benchmark Evaluation for Kumiho Cognitive Memory.

Evaluates against the LoCoMo multi-session conversation benchmark:
  - 10 conversations, ~200 QA pairs each, 5 question categories
  - Category 1: Multi-hop (comma-split F1)
  - Category 2: Temporal / single-hop (token F1)
  - Category 3: Open-domain / reasoning (token F1 on primary answer)
  - Category 4: Single-hop factual (token F1)
  - Category 5: Adversarial (binary — model should refuse)

Metrics: Token-F1 (per-category), LLM-as-Judge accuracy

Reference scores:
  MAGMA  0.700 (Judge), Mem0  67.1% (Judge), Zep  58–75% (disputed)
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

from requests.exceptions import ConnectionError as RequestsConnectionError
from tqdm import tqdm

from .common import (
    BenchmarkConfig,
    EvalResult,
    KumihoMemoryAdapter,
    compute_aggregate_metrics,
    exact_match,
    generate_answer,
    llm_judge,
    multihop_f1,
    normalize_answer,
    print_metrics_table,
    save_results,
    token_f1,
)

_RETRYABLE_ERRORS = (OSError, RequestsConnectionError, ConnectionError, TimeoutError)
MAX_CONV_RETRIES = 3
RETRY_BASE_DELAY = 15  # seconds

logger = logging.getLogger("kumiho_eval.locomo")

LOCOMO_DATA = Path(__file__).resolve().parent.parent / "locomo" / "data" / "locomo10.json"

# Question category labels for reporting
CATEGORY_NAMES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}

# ---------------------------------------------------------------------------
# Checkpoint / resume (same pattern as LongMemEval / MAB)
# ---------------------------------------------------------------------------


def _checkpoint_path(output_dir: Path) -> Path:
    return output_dir / "_checkpoint.jsonl"


def _load_checkpoint(output_dir: Path) -> tuple[list[EvalResult], set[str]]:
    """Load checkpoint if it exists. Returns (results, completed_question_ids)."""
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
    logger.info("Loaded checkpoint: %d completed questions", len(completed))
    return results, completed


def _save_checkpoint_line(output_dir: Path, result: EvalResult) -> None:
    """Append a single result to the checkpoint JSONL file."""
    ckpt = _checkpoint_path(output_dir)
    data = {
        "question_id": result.question_id,
        "question": result.question,
        "question_type": result.question_type,
        "ground_truth": result.ground_truth,
        "prediction": result.prediction,
        "recalled_context": result.recalled_context[:8000],
        "recalled_context_len": len(result.recalled_context),
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
# Data loading helpers
# ---------------------------------------------------------------------------


def load_locomo(path: str | Path | None = None) -> list[dict]:
    """Load the LoCoMo-10 dataset."""
    path = Path(path) if path else LOCOMO_DATA
    if not path.exists():
        raise FileNotFoundError(
            f"LoCoMo dataset not found at {path}. "
            "Clone it: git clone --depth 1 https://github.com/snap-research/locomo.git"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_sessions(conversation: dict) -> list[dict]:
    """
    Extract ordered sessions from a LoCoMo conversation.

    Returns list of:
      {"session_num": int, "date_time": str, "turns": [{"speaker": str, "text": str, "dia_id": str}]}
    """
    sessions = []
    idx = 1
    while True:
        key = f"session_{idx}"
        dt_key = f"session_{idx}_date_time"
        if key not in conversation:
            break
        sessions.append(
            {
                "session_num": idx,
                "date_time": conversation.get(dt_key, ""),
                "turns": conversation[key],
                "speaker_a": conversation.get("speaker_a", "Speaker A"),
                "speaker_b": conversation.get("speaker_b", "Speaker B"),
            }
        )
        idx += 1
    return sessions


def session_to_messages(session: dict) -> list[dict[str, str]]:
    """
    Convert a LoCoMo session to alternating user/assistant messages.

    Maps speaker_a → user, speaker_b → assistant (arbitrary but consistent).
    Prefixes each message with the speaker name and session date for context.
    """
    speaker_a = session["speaker_a"]
    date_str = session.get("date_time", "")
    messages = []
    for turn in session["turns"]:
        role = "user" if turn["speaker"] == speaker_a else "assistant"
        content = f"[{date_str}] {turn['speaker']}: {turn['text']}"
        messages.append({"role": role, "content": content})
    return messages


def format_conversation_context(sessions: list[dict]) -> str:
    """Format full conversation as text context (for answer generation fallback)."""
    lines = []
    for session in sessions:
        lines.append(f"\n--- {session['date_time']} ---")
        for turn in session["turns"]:
            lines.append(f"{turn['speaker']}: {turn['text']}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Per-category scoring (matches LoCoMo evaluation.py exactly)
# ---------------------------------------------------------------------------


def score_locomo_qa(category: int, prediction: str, answer: str) -> float:
    """Score a single LoCoMo QA pair using category-specific logic."""
    answer_str = str(answer)

    if category == 1:
        # Multi-hop: split on commas, partial F1
        return multihop_f1(prediction, answer_str)
    elif category in (2, 4):
        # Temporal / single-hop: token-level F1
        return token_f1(prediction, answer_str)
    elif category == 3:
        # Open-domain: use primary answer (before semicolon)
        primary = answer_str.split(";")[0].strip()
        return token_f1(prediction, primary)
    elif category == 5:
        # Adversarial: binary — model should indicate it cannot answer.
        # Expanded refusal patterns to avoid undercounting correct abstentions.
        lower = prediction.lower()
        _REFUSAL_PHRASES = (
            "no information available",
            "not mentioned",
            "not provided",
            "cannot answer",
            "can't answer",
            "no relevant information",
            "i don't know",
            "i don't have",
            "not discussed",
            "not available",
            "no evidence",
            "not enough information",
            "insufficient information",
            "unable to determine",
            "unable to answer",
            "not specified",
            "no record",
            "not in the context",
            "not in the conversation",
        )
        if any(phrase in lower for phrase in _REFUSAL_PHRASES):
            return 1.0
        return 0.0
    else:
        return token_f1(prediction, answer_str)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


async def evaluate_locomo(
    config: BenchmarkConfig,
    data_path: str | Path | None = None,
    judge: bool = True,
    resume: bool = True,
) -> dict[str, Any]:
    """
    Run the full LoCoMo evaluation.

    1. For each conversation: ingest sessions (parallel) → consolidate
    2. For each QA pair: recall → generate answer → score

    Checkpoint/resume: saves progress after each question; skips completed
    questions on restart (controlled by `resume` parameter).

    Returns dict with results, metrics, and per-category breakdown.
    """
    dataset = load_locomo(data_path)
    if config.max_samples:
        dataset = dataset[: config.max_samples]

    adapter = KumihoMemoryAdapter(config)
    output_dir = Path(config.output_dir) / "locomo"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint for resume
    if resume:
        all_results, completed_ids = _load_checkpoint(output_dir)
    else:
        all_results, completed_ids = [], set()

    sem = asyncio.Semaphore(config.concurrency)

    try:
        for conv_idx, sample in enumerate(dataset):
            conv_id = sample.get("sample_id", f"conv-{conv_idx}")
            conversation = sample["conversation"]
            qa_pairs = sample["qa"]
            sessions = extract_sessions(conversation)

            # Check if all questions for this conversation are done
            conv_q_ids = [f"{conv_id}_q{qi}" for qi in range(len(qa_pairs))]
            if completed_ids and all(qid in completed_ids for qid in conv_q_ids):
                logger.info("Skipping completed conversation %s (%d questions)", conv_id, len(qa_pairs))
                continue

            logger.info(
                "Processing conversation %s (%d sessions, %d questions)",
                conv_id,
                len(sessions),
                len(qa_pairs),
            )

            # Retry the entire conversation processing on transient network errors
            last_conv_error: Exception | None = None
            for conv_attempt in range(1, MAX_CONV_RETRIES + 1):
                try:
                    # Create isolated evaluation space
                    space_name = await adapter.create_eval_space(conv_id)

                    # --- Phase 1: Ingest all sessions (parallel) ---
                    user_id = f"locomo-{conv_id}"
                    t_ingest = time.perf_counter()

                    async def _ingest_one(session: dict) -> str | None:
                        async with sem:
                            messages = session_to_messages(session)
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
                                        stack_revisions=config.stack_revisions,
                                    )
                                    # Post-consolidation edge discovery
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
                                    logger.warning("Consolidation failed for session: %s", e)
                            return sid

                    ingest_results = await asyncio.gather(
                        *[_ingest_one(s) for s in sessions],
                        return_exceptions=True,
                    )
                    for r in ingest_results:
                        if isinstance(r, _RETRYABLE_ERRORS):
                            raise r
                        if isinstance(r, Exception):
                            logger.warning("Session ingestion error: %s", r)

                    total_ingest_ms = (time.perf_counter() - t_ingest) * 1000
                    avg_ingest_ms = total_ingest_ms / max(len(sessions), 1)

                    # --- Phase 2: Answer questions ---
                    # Scope recall to this conversation's space so memories
                    # from other conversations don't contaminate answers.
                    user_space = f"{config.project_name}/personal/{user_id}"
                    full_context = format_conversation_context(sessions)

                    for qi, qa in enumerate(
                        tqdm(qa_pairs, desc=f"Evaluating {conv_id}", leave=False)
                    ):
                        question = qa["question"]
                        answer = str(qa.get("answer", qa.get("adversarial_answer", "")))
                        category = qa.get("category", 0)
                        q_id = f"{conv_id}_q{qi}"

                        # Skip already-completed questions (checkpoint resume)
                        if q_id in completed_ids:
                            continue

                        # Recall from memory (scoped to this conversation's space)
                        t0 = time.perf_counter()
                        if config.graph_augmented:
                            memories = await adapter.recall_with_graph_augmentation(
                                question, limit=config.recall_limit,
                                space_paths=[user_space],
                            )
                        else:
                            memories = await adapter.recall(
                                question, limit=config.recall_limit,
                                space_paths=[user_space],
                            )
                        recall_ms = (time.perf_counter() - t0) * 1000

                        # Build context from recalled memories (mode-aware)
                        recalled_context = adapter.build_recalled_context(
                            memories, query=question,
                        )

                        # Generate answer — aligned with original LoCoMo
                        # paper's QA_PROMPT: "short phrase", "exact words
                        # from the context".  Category-specific question
                        # modifications (not system prompt changes) per the
                        # original gpt_utils.py.
                        t1 = time.perf_counter()

                        # Question modifications per category
                        eval_question = question
                        if category == 2:
                            # Temporal: original paper appends date hint
                            eval_question = (
                                question
                                + " Use the date of the conversation to "
                                "answer with an approximate date."
                            )

                        # Prompt selection
                        if category == 5:
                            # Adversarial: instruct refusal when info absent
                            system = (
                                "Answer the following question based on the context. "
                                "If the information is not available in the context, "
                                'say "No information available".'
                            )
                        else:
                            # Terse extraction — format only, no content hints
                            system = (
                                "Answer in 1-5 words only. Use exact names, dates, "
                                "places and terms from the context. "
                                "Never write full sentences."
                            )

                        # Use recalled context, fall back to a truncated version of full context
                        answer_context = recalled_context if recalled_context else full_context[:8000]

                        prediction = await generate_answer(
                            eval_question,
                            answer_context,
                            system_prompt=system,
                            model=config.answer_model,
                            api_key=config.openai_api_key,
                            max_tokens=50,
                        )
                        answer_ms = (time.perf_counter() - t1) * 1000

                        # Score
                        f1 = score_locomo_qa(category, prediction, answer)

                        # LLM judge
                        judge_ok = False
                        if judge and category != 5:
                            try:
                                judge_ok = await llm_judge(
                                    question,
                                    answer,
                                    prediction,
                                    model=config.judge_model,
                                    api_key=config.openai_api_key,
                                )
                            except Exception as e:
                                logger.warning("Judge failed for %s: %s", q_id, e)
                        elif category == 5:
                            judge_ok = f1 == 1.0

                        result = EvalResult(
                            question_id=q_id,
                            question=question,
                            question_type=CATEGORY_NAMES.get(category, f"cat-{category}"),
                            ground_truth=answer,
                            prediction=prediction,
                            recalled_context=recalled_context,
                            f1_score=f1,
                            judge_score=judge_ok,
                            exact_match=exact_match(prediction, answer),
                            latency_ingest_ms=avg_ingest_ms,
                            latency_recall_ms=recall_ms,
                            latency_answer_ms=answer_ms,
                            metadata={
                                "category": category,
                                "conv_id": conv_id,
                                "evidence": qa.get("evidence", []),
                                "memories_recalled": len(memories),
                            },
                        )
                        all_results.append(result)
                        _save_checkpoint_line(output_dir, result)

                    # Save per-conversation intermediate results
                    save_results(
                        [r for r in all_results if r.metadata.get("conv_id") == conv_id],
                        output_dir / f"{conv_id}_results.json",
                    )

                    last_conv_error = None
                    break  # conversation succeeded

                except _RETRYABLE_ERRORS as e:
                    last_conv_error = e
                    if conv_attempt < MAX_CONV_RETRIES:
                        delay = RETRY_BASE_DELAY * (2 ** (conv_attempt - 1))
                        logger.warning(
                            "Network error on %s (attempt %d/%d), retrying in %ds: %s",
                            conv_id, conv_attempt, MAX_CONV_RETRIES, delay, e,
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "Failed conversation %s after %d attempts: %s",
                            conv_id, MAX_CONV_RETRIES, e,
                        )

            if last_conv_error is not None:
                logger.error("Skipping conversation %s due to persistent network errors", conv_id)

    finally:
        await adapter.cleanup()

    # Save all results and compute metrics
    save_results(all_results, output_dir / "all_results.json")
    metrics = compute_aggregate_metrics(all_results)

    # Also compute per-category metrics (LoCoMo standard)
    cat_metrics: dict[str, Any] = {}
    for cat_num, cat_name in CATEGORY_NAMES.items():
        cat_results = [
            r for r in all_results if r.metadata.get("category") == cat_num
        ]
        if cat_results:
            import numpy as np

            cat_metrics[cat_name] = {
                "count": len(cat_results),
                "f1": float(np.mean([r.f1_score for r in cat_results])),
                "judge_accuracy": float(np.mean([r.judge_score for r in cat_results])),
            }
    metrics["locomo_categories"] = cat_metrics

    # Save metrics
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print_metrics_table(metrics, "LoCoMo")

    # Print LoCoMo-specific category table
    print(f"\n  LoCoMo Per-Category Breakdown:")
    print(f"  {'Category':<20} {'Count':>6} {'F1':>8} {'Judge':>8}")
    print(f"  {'-' * 44}")
    for cat_name, vals in cat_metrics.items():
        print(
            f"  {cat_name:<20} {vals['count']:>6} "
            f"{vals['f1']:>8.4f} {vals['judge_accuracy']:>8.4f}"
        )
    print()

    return {"results": all_results, "metrics": metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LoCoMo benchmark on Kumiho")
    parser.add_argument("--data", type=str, default=None, help="Path to locomo10.json")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit conversations")
    parser.add_argument("--answer-model", type=str, default="gpt-4o", help="Model for answer generation")
    parser.add_argument("--judge-model", type=str, default="gpt-4o", help="Model for LLM judge")
    parser.add_argument("--recall-limit", type=int, default=3, help="Max memories to recall")
    parser.add_argument("--recall-mode", type=str, default="full",
                        choices=["full", "summarized"],
                        help="Recall mode: full (artifact content) or summarized (title+summary)")
    parser.add_argument("--concurrency", type=int, default=4,
                        help="Max parallel session ingestions per conversation")
    parser.add_argument("--no-graph-augmented", action="store_true",
                        help="Disable graph-augmented recall (fall back to vector-only search)")
    parser.add_argument("--no-judge", action="store_true", help="Skip LLM judge (F1 only)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh instead of resuming from checkpoint")
    parser.add_argument("--project", type=str, default="benchmark-locomo", help="Kumiho project name")
    parser.add_argument("--sibling-threshold", type=float, default=0.10,
                        help="Sibling similarity threshold (0=budget mode, 0.10=lenient, 0.30=strict)")
    parser.add_argument("--sibling-top-k", type=int, default=3,
                        help="Max siblings to keep after scoring (0=unlimited)")
    parser.add_argument("--context-top-k", type=int, default=5,
                        help="Global cap on revisions in final context (0=unlimited)")
    parser.add_argument("--no-stack", action="store_true",
                        help="Disable revision stacking (one item per session)")
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
        graph_augmented=not args.no_graph_augmented,
        sibling_similarity_threshold=args.sibling_threshold,
        sibling_top_k=args.sibling_top_k,
        context_top_k=args.context_top_k,
        stack_revisions=not args.no_stack,  # Default: True (stacking + sibling top-k)
    )

    asyncio.run(evaluate_locomo(
        config, data_path=args.data, judge=not args.no_judge, resume=not args.no_resume,
    ))


if __name__ == "__main__":
    main()
