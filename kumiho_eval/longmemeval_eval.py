"""
LongMemEval Benchmark Evaluation for Kumiho Cognitive Memory.

Evaluates against the LongMemEval long-term memory benchmark (ICLR 2025):
  - 500 questions across 5 core memory abilities
  - Information Extraction, Multi-Session Reasoning, Temporal Reasoning,
    Knowledge Updates, Abstention
  - Uses GPT-4o judge with task-specific prompts (matching official eval)

Dataset: HuggingFace xiaowu0162/longmemeval-cleaned

Reference scores:
  Zep 71.2% (gpt-4o), MAGMA 61.2%, Full-context baseline 60.2%
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import backoff
import numpy as np
from tqdm import tqdm

from .common import (
    BenchmarkConfig,
    EvalResult,
    KumihoMemoryAdapter,
    compute_aggregate_metrics,
    generate_answer,
    print_metrics_table,
    save_results,
    token_f1,
)

logger = logging.getLogger("kumiho_eval.longmemeval")

LONGMEMEVAL_DATA_DIR = Path(__file__).resolve().parent.parent / "LongMemEval" / "data"

# Mapping from question_type to human-readable label
QUESTION_TYPE_LABELS = {
    "single-session-user": "Info Extraction (user)",
    "single-session-assistant": "Info Extraction (assistant)",
    "single-session-preference": "Preference Recall",
    "multi-session": "Multi-Session Reasoning",
    "temporal-reasoning": "Temporal Reasoning",
    "temporal_reasoning_explicit": "Temporal (explicit)",
    "temporal_reasoning_implicit": "Temporal (implicit)",
    "knowledge-update": "Knowledge Update",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_longmemeval(
    variant: str = "s",
    data_dir: str | Path | None = None,
) -> list[dict]:
    """
    Load LongMemEval dataset.

    Args:
        variant: "s" (small, ~115K tokens), "m" (medium, ~500 sessions),
                 or "oracle" (evidence-only sessions)
        data_dir: Override data directory

    Returns list of question entries.
    """
    data_dir = Path(data_dir) if data_dir else LONGMEMEVAL_DATA_DIR

    # Try local files first
    local_path = data_dir / f"longmemeval_{variant}.json"
    if local_path.exists():
        with open(local_path, encoding="utf-8") as f:
            return json.load(f)

    # Fall back to HuggingFace download
    logger.info("Local data not found, downloading from HuggingFace...")
    try:
        from datasets import load_dataset

        ds = load_dataset("xiaowu0162/longmemeval-cleaned")
        # The dataset has train split with all entries
        entries = list(ds["train"])
        # Filter by variant if needed (all variants are in the same dataset)
        # Save locally for future runs
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        return entries
    except ImportError:
        raise ImportError(
            "Install `datasets` package: pip install datasets\n"
            "Or manually download from https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned"
        )


def extract_haystack_sessions(entry: dict) -> list[dict]:
    """
    Extract haystack sessions from a LongMemEval entry.

    Each session is a list of {"role": "user"|"assistant", "content": "..."} turns.
    """
    sessions = entry.get("haystack_sessions", [])
    dates = entry.get("haystack_dates", [])
    session_ids = entry.get("haystack_session_ids", [])

    result = []
    for i, session in enumerate(sessions):
        date = dates[i] if i < len(dates) else ""
        sid = session_ids[i] if i < len(session_ids) else f"session_{i}"
        result.append(
            {
                "session_id": sid,
                "date": date,
                "turns": session,
            }
        )
    return result


# ---------------------------------------------------------------------------
# LLM Judge (matches official LongMemEval evaluate_qa.py prompts)
# ---------------------------------------------------------------------------

_JUDGE_PROMPTS = {
    "default": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no."
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {prediction}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "temporal-reasoning": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response is equivalent to the correct answer or contains all the intermediate "
        "steps to get the correct answer, you should also answer yes. If the response only "
        "contains a subset of the information required by the answer, answer no. "
        "In addition, do not penalize off-by-one errors for the number of days. "
        "If the question asks for the number of days/weeks/months, etc., and the model makes "
        "off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's "
        "response is still correct."
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {prediction}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "knowledge-update": (
        "I will give you a question, a correct answer, and a response from a model. "
        "Please answer yes if the response contains the correct answer. Otherwise, answer no. "
        "If the response contains some previous information along with an updated answer, "
        "the response should be considered as correct as long as the updated answer is the "
        "required answer."
        "\n\nQuestion: {question}\n\nCorrect Answer: {answer}\n\nModel Response: {prediction}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "single-session-preference": (
        "I will give you a question, a rubric for desired personalized response, and a "
        "response from a model. Please answer yes if the response satisfies the desired "
        "response. Otherwise, answer no. The model does not need to reflect all the points "
        "in the rubric. The response is correct as long as it recalls and utilizes the user's "
        "personal information correctly."
        "\n\nQuestion: {question}\n\nRubric: {answer}\n\nModel Response: {prediction}"
        "\n\nIs the model response correct? Answer yes or no only."
    ),
    "abstention": (
        "I will give you an unanswerable question, an explanation, and a response from a "
        "model. Please answer yes if the model correctly identifies the question as "
        "unanswerable. The model could say that the information is incomplete, or some "
        "other information is given but the asked information is not."
        "\n\nQuestion: {question}\n\nExplanation: {answer}\n\nModel Response: {prediction}"
        "\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only."
    ),
}


def _get_judge_template(question_type: str, question_id: str) -> str:
    """Get the appropriate judge prompt template for a question type."""
    is_abstention = "_abs" in question_id

    if is_abstention:
        return _JUDGE_PROMPTS["abstention"]
    elif question_type in _JUDGE_PROMPTS:
        return _JUDGE_PROMPTS[question_type]
    elif "temporal" in question_type:
        return _JUDGE_PROMPTS["temporal-reasoning"]
    else:
        return _JUDGE_PROMPTS["default"]


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def longmemeval_judge(
    question: str,
    answer: str,
    prediction: str,
    question_type: str,
    question_id: str,
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> bool:
    """Run LongMemEval's official GPT-4o judge evaluation."""
    import os

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

    template = _get_judge_template(question_type, question_id)
    prompt = template.format(question=question, answer=answer, prediction=prediction)

    resp = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=10,
        temperature=0.0,
    )
    verdict = resp.choices[0].message.content.strip().lower()
    return "yes" in verdict


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


async def evaluate_longmemeval(
    config: BenchmarkConfig,
    variant: str = "s",
    data_dir: str | Path | None = None,
) -> dict[str, Any]:
    """
    Run the full LongMemEval evaluation.

    1. For each question: ingest haystack sessions → consolidate
    2. Recall → generate answer → judge

    Returns dict with results, metrics, and per-type breakdown.
    """
    dataset = load_longmemeval(variant=variant, data_dir=data_dir)
    if config.max_samples:
        dataset = dataset[: config.max_samples]

    adapter = KumihoMemoryAdapter(config)
    all_results: list[EvalResult] = []
    output_dir = Path(config.output_dir) / "longmemeval"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Also produce JSONL for official evaluate_qa.py compatibility
    hyp_file = output_dir / "hypotheses.jsonl"

    try:
        for qi, entry in enumerate(tqdm(dataset, desc="LongMemEval")):
            q_id = entry.get("question_id", f"q{qi}")
            question = entry["question"]
            answer = entry["answer"]
            q_type = entry.get("question_type", "unknown")
            q_date = entry.get("question_date", "")

            # Extract sessions
            sessions = extract_haystack_sessions(entry)

            if not sessions:
                logger.warning("No haystack sessions for %s, skipping", q_id)
                continue

            # Create evaluation space for this question
            space_name = await adapter.create_eval_space(f"lme-{q_id}")
            user_id = f"longmemeval-{q_id}"

            # --- Phase 1: Ingest sessions ---
            t_ingest_start = time.perf_counter()
            session_id_list = []

            for session in sessions:
                # Prefix turns with date for temporal context
                date_prefix = f"[{session['date']}] " if session["date"] else ""
                messages = []
                for turn in session["turns"]:
                    content = date_prefix + turn.get("content", "")
                    messages.append({"role": turn.get("role", "user"), "content": content})

                result = await adapter.ingest_session(
                    user_id=user_id,
                    session_messages=messages,
                    context="personal",
                )
                sid = result.get("session_id")
                if sid:
                    session_id_list.append(sid)
                    try:
                        await adapter.consolidate(sid)
                    except Exception as e:
                        logger.warning("Consolidation failed: %s", e)

            ingest_ms = (time.perf_counter() - t_ingest_start) * 1000

            # --- Phase 2: Recall & answer ---
            t_recall = time.perf_counter()
            memories = await adapter.recall(question, limit=config.recall_limit)
            recall_ms = (time.perf_counter() - t_recall) * 1000

            # Build context from recalled memories
            recalled_texts = []
            for mem in memories:
                summary = mem.get("summary", "")
                title = mem.get("title", "")
                if summary:
                    recalled_texts.append(f"{title}: {summary}" if title else summary)
            recalled_context = "\n\n".join(recalled_texts) if recalled_texts else ""

            # Determine system prompt based on question type
            is_abstention = "_abs" in q_id
            if is_abstention:
                system = (
                    "You are answering questions about a user's conversation history. "
                    "If the specific information asked about was never discussed, "
                    'clearly state that the information is not available or "I don\'t know".'
                )
            elif "temporal" in q_type:
                system = (
                    "You are answering questions about a user's conversation history. "
                    "Pay careful attention to dates and temporal ordering. "
                    f"The current date is {q_date}. "
                    "Answer with specific dates or time spans when asked."
                )
            elif q_type == "knowledge-update":
                system = (
                    "You are answering questions about a user's conversation history. "
                    "The user's information may have changed over time. "
                    "Always answer with the most recent/updated information."
                )
            else:
                system = (
                    "You are answering questions about a user's conversation history. "
                    "Answer concisely based on the available context."
                )

            t_answer = time.perf_counter()
            prediction = await generate_answer(
                question,
                recalled_context,
                system_prompt=system,
                model=config.answer_model,
                api_key=config.openai_api_key,
                max_tokens=200,
            )
            answer_ms = (time.perf_counter() - t_answer) * 1000

            # Judge
            try:
                judge_ok = await longmemeval_judge(
                    question,
                    answer,
                    prediction,
                    q_type,
                    q_id,
                    model=config.judge_model,
                    api_key=config.openai_api_key,
                )
            except Exception as e:
                logger.warning("Judge failed for %s: %s", q_id, e)
                judge_ok = False

            result = EvalResult(
                question_id=q_id,
                question=question,
                question_type=q_type,
                ground_truth=answer,
                prediction=prediction,
                recalled_context=recalled_context,
                f1_score=token_f1(prediction, answer),
                judge_score=judge_ok,
                latency_ingest_ms=ingest_ms,
                latency_recall_ms=recall_ms,
                latency_answer_ms=answer_ms,
                metadata={
                    "question_date": q_date,
                    "is_abstention": is_abstention,
                    "memories_recalled": len(memories),
                    "sessions_ingested": len(sessions),
                },
            )
            all_results.append(result)

            # Write JSONL line for official evaluator compatibility
            with open(hyp_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {"question_id": q_id, "hypothesis": prediction},
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    finally:
        await adapter.cleanup()

    # Save results and compute metrics
    save_results(all_results, output_dir / "all_results.json")
    metrics = compute_aggregate_metrics(all_results)

    # Compute per-ability metrics (LongMemEval standard)
    ability_metrics: dict[str, Any] = {}
    for result in all_results:
        qt = result.question_type
        if qt not in ability_metrics:
            ability_metrics[qt] = {"correct": 0, "total": 0}
        ability_metrics[qt]["total"] += 1
        if result.judge_score:
            ability_metrics[qt]["correct"] += 1

    for qt in ability_metrics:
        t = ability_metrics[qt]["total"]
        c = ability_metrics[qt]["correct"]
        ability_metrics[qt]["accuracy"] = c / t if t > 0 else 0.0

    # Abstention separate
    abs_results = [r for r in all_results if r.metadata.get("is_abstention")]
    non_abs_results = [r for r in all_results if not r.metadata.get("is_abstention")]

    metrics["longmemeval"] = {
        "overall_accuracy": float(np.mean([r.judge_score for r in all_results])),
        "non_abstention_accuracy": float(np.mean([r.judge_score for r in non_abs_results]))
        if non_abs_results
        else 0.0,
        "abstention_accuracy": float(np.mean([r.judge_score for r in abs_results]))
        if abs_results
        else 0.0,
        "task_averaged_accuracy": float(
            np.mean([v["accuracy"] for v in ability_metrics.values()])
        ),
        "per_ability": ability_metrics,
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print_metrics_table(metrics, "LongMemEval")

    # Print LongMemEval-specific ability table
    lme = metrics["longmemeval"]
    print(f"  LongMemEval Ability Breakdown:")
    print(f"  Overall Accuracy:         {lme['overall_accuracy']:.4f}")
    print(f"  Task-Averaged Accuracy:   {lme['task_averaged_accuracy']:.4f}")
    print(f"  Abstention Accuracy:      {lme['abstention_accuracy']:.4f}")
    print(f"\n  {'Ability':<35} {'Count':>6} {'Accuracy':>10}")
    print(f"  {'-' * 53}")
    for qt, vals in sorted(ability_metrics.items()):
        label = QUESTION_TYPE_LABELS.get(qt, qt)
        print(f"  {label:<35} {vals['total']:>6} {vals['accuracy']:>10.4f}")
    print()

    logger.info("Hypotheses JSONL saved to %s (compatible with official evaluate_qa.py)", hyp_file)

    return {"results": all_results, "metrics": metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run LongMemEval benchmark on Kumiho")
    parser.add_argument("--variant", type=str, default="s", choices=["s", "m", "oracle"],
                        help="Dataset variant: s (small), m (medium), oracle")
    parser.add_argument("--data-dir", type=str, default=None, help="Data directory override")
    parser.add_argument("--output", type=str, default="./results", help="Output directory")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit questions")
    parser.add_argument("--answer-model", type=str, default="gpt-4o")
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--recall-limit", type=int, default=10)
    parser.add_argument("--project", type=str, default="benchmark-longmemeval")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = BenchmarkConfig(
        project_name=args.project,
        answer_model=args.answer_model,
        judge_model=args.judge_model,
        output_dir=args.output,
        max_samples=args.max_samples,
        recall_limit=args.recall_limit,
    )

    asyncio.run(evaluate_longmemeval(config, variant=args.variant, data_dir=args.data_dir))


if __name__ == "__main__":
    main()
