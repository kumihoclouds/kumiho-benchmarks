"""
MemoryAgentBench Evaluation for Kumiho Cognitive Memory.

Evaluates against the MemoryAgentBench four-competency framework (ICLR 2026):
  - Accurate Retrieval (AR): precise information from long histories
  - Test-Time Learning (TTL): learning new rules/procedures during dialogue
  - Long-Range Understanding (LRU): comprehension across extended contexts
  - Conflict Resolution (CR): handling contradictory information

Dataset: HuggingFace ai-hyz/MemoryAgentBench (146 examples, contexts up to 3.17MB)

Reference scores (memory agents):
  Mem0: AR 29.4, TTL 27.0, LRU 0.8, CR-SH 18.0, CR-MH 2.0
  MemGPT/Letta: AR 31.7, TTL 55.3, LRU 2.5, CR-SH 28.0, CR-MH 3.0
  Cognee: AR 27.1, TTL 31.9, LRU 2.3, CR-SH 28.0, CR-MH 3.0
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import string
import time
from collections import Counter
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
    exact_match,
    generate_answer,
    llm_judge,
    normalize_answer,
    print_metrics_table,
    save_results,
    token_f1,
)

_RETRYABLE_ERRORS = (OSError, RequestsConnectionError, ConnectionError, TimeoutError)
MAX_SAMPLE_RETRIES = 3
RETRY_BASE_DELAY = 10  # seconds

logger = logging.getLogger("kumiho_eval.memoryagentbench")

# ---------------------------------------------------------------------------
# Dataset-specific metrics (matching MemoryAgentBench eval_other_utils.py)
# ---------------------------------------------------------------------------


def _rouge_scores(prediction: str, ground_truth: str) -> dict[str, float]:
    """Compute ROUGE-1, ROUGE-2, ROUGE-L scores."""
    try:
        from rouge_score import rouge_scorer

        scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
        scores = scorer.score(ground_truth, prediction)
        return {
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rouge1_recall": scores["rouge1"].recall,
            "rouge2_f1": scores["rouge2"].fmeasure,
            "rouge2_recall": scores["rouge2"].recall,
            "rougeL_f1": scores["rougeL"].fmeasure,
            "rougeL_recall": scores["rougeL"].recall,
        }
    except ImportError:
        return {}


def substring_em(prediction: str, answer: str) -> bool:
    """Substring exact match — answer found within prediction (normalised)."""
    return normalize_answer(answer) in normalize_answer(prediction)


def eventqa_recall(prediction: str, answers: list[str]) -> float:
    """EventQA metric: binary recall — all answer elements found in prediction."""
    pred_norm = normalize_answer(prediction)
    for ans in answers:
        if normalize_answer(ans) not in pred_norm:
            return 0.0
    return 1.0


def conflict_resolution_accuracy(prediction: str, answer: str) -> float:
    """Score for FactConsolidation tasks — exact substring match."""
    return 1.0 if substring_em(prediction, answer) else 0.0


def compute_mab_metrics(
    prediction: str,
    answer: str | list[str],
    source: str,
) -> dict[str, float]:
    """
    Compute dataset-specific metrics for a MemoryAgentBench question.

    Args:
        prediction: Model output
        answer: Ground truth (string or list)
        source: Dataset source identifier (e.g., "eventqa_65536", "longmemeval_s")

    Returns metrics dict with scores.
    """
    metrics: dict[str, float] = {}

    pred_str = prediction
    if isinstance(answer, list):
        ans_str = " ".join(str(a) for a in answer)
        ans_list = [str(a) for a in answer]
    else:
        ans_str = str(answer)
        ans_list = [ans_str]

    # Universal metrics
    metrics["f1"] = token_f1(pred_str, ans_str)
    metrics["exact_match"] = 1.0 if exact_match(pred_str, ans_str) else 0.0
    metrics["substring_em"] = 1.0 if substring_em(pred_str, ans_str) else 0.0

    # ROUGE
    rouge = _rouge_scores(pred_str, ans_str)
    metrics.update(rouge)

    # Dataset-specific
    if "eventqa" in source:
        metrics["eventqa_recall"] = eventqa_recall(pred_str, ans_list)
        metrics["primary_metric"] = metrics["eventqa_recall"]
    elif "factconsolidation" in source:
        metrics["conflict_accuracy"] = conflict_resolution_accuracy(pred_str, ans_str)
        metrics["primary_metric"] = metrics["conflict_accuracy"]
    elif "ruler" in source:
        # RULER uses token-F1 (drqa_f1)
        metrics["primary_metric"] = metrics["f1"]
    elif "longmemeval" in source:
        # LongMemEval within MAB uses judge-based accuracy (handled separately)
        metrics["primary_metric"] = metrics["f1"]
    elif "infbench" in source:
        metrics["primary_metric"] = metrics.get("rougeL_f1", metrics["f1"])
    else:
        metrics["primary_metric"] = metrics["f1"]

    return metrics


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

SPLIT_MAP = {
    "AR": "Accurate_Retrieval",
    "TTL": "Test_Time_Learning",
    "LRU": "Long_Range_Understanding",
    "CR": "Conflict_Resolution",
}


def load_memoryagentbench(
    split: str = "AR",
    max_samples: int | None = None,
) -> list[dict]:
    """
    Load MemoryAgentBench dataset from HuggingFace.

    Args:
        split: "AR", "TTL", "LRU", or "CR"
        max_samples: Limit number of samples
    """
    hf_split = SPLIT_MAP.get(split, split)

    try:
        from datasets import load_dataset

        ds = load_dataset("ai-hyz/MemoryAgentBench", split=hf_split)
        entries = list(ds)
        if max_samples:
            entries = entries[:max_samples]
        return entries
    except ImportError:
        raise ImportError(
            "Install `datasets` package: pip install datasets\n"
            "Dataset: https://huggingface.co/datasets/ai-hyz/MemoryAgentBench"
        )


def chunk_context(context: str, chunk_size: int = 4096) -> list[str]:
    """Split context into chunks for memory ingestion."""
    words = context.split()
    chunks = []
    current: list[str] = []
    current_len = 0

    for word in words:
        current.append(word)
        current_len += len(word) + 1
        if current_len >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_len = 0

    if current:
        chunks.append(" ".join(current))

    return chunks


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


async def evaluate_memoryagentbench(
    config: BenchmarkConfig,
    splits: list[str] | None = None,
    chunk_size: int = 4096,
) -> dict[str, Any]:
    """
    Run the full MemoryAgentBench evaluation.

    For each sample:
      1. Chunk the context → ingest chunks into memory → consolidate
      2. For each question: recall → generate → score

    Returns dict with results, metrics, and per-split/per-competency breakdown.
    """
    if splits is None:
        splits = ["AR", "TTL", "LRU", "CR"]

    adapter = KumihoMemoryAdapter(config)
    all_results: list[EvalResult] = []
    output_dir = Path(config.output_dir) / "memoryagentbench"
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        for split in splits:
            logger.info("Loading MemoryAgentBench split: %s", split)
            dataset = load_memoryagentbench(split=split, max_samples=config.max_samples)
            logger.info("Loaded %d samples for split %s", len(dataset), split)

            for si, sample in enumerate(tqdm(dataset, desc=f"MAB-{split}")):
                context = sample.get("context", "")
                questions = sample.get("questions", [])
                answers = sample.get("answers", [])
                metadata = sample.get("metadata", {})
                source = metadata.get("source", split)
                q_types = metadata.get("question_types", [])
                q_ids = metadata.get("question_ids", [])

                if not context or not questions:
                    continue

                sample_id = f"{split}-{si}"

                # Retry per-sample on transient network errors
                last_sample_error: Exception | None = None
                for attempt in range(1, MAX_SAMPLE_RETRIES + 1):
                    try:
                        # Create eval space
                        await adapter.create_eval_space(f"mab-{sample_id}")
                        user_id = f"mab-{sample_id}"

                        # --- Phase 1: Ingest context chunks ---
                        chunks = chunk_context(context, chunk_size=chunk_size)
                        t_ingest = time.perf_counter()
                        session_ids = []

                        for ci, chunk_text in enumerate(chunks):
                            messages = [{"role": "user", "content": chunk_text}]
                            result = await adapter.ingest_session(
                                user_id=user_id,
                                session_messages=messages,
                                context="work",
                            )
                            sid = result.get("session_id")
                            if sid:
                                session_ids.append(sid)

                        # Consolidate all sessions
                        for sid in session_ids:
                            try:
                                await adapter.consolidate(sid)
                            except Exception as e:
                                logger.warning("Consolidation failed: %s", e)

                        ingest_ms = (time.perf_counter() - t_ingest) * 1000

                        # --- Phase 2: Query ---
                        for qi in range(len(questions)):
                            question = questions[qi]
                            answer = answers[qi] if qi < len(answers) else ""
                            q_type = q_types[qi] if qi < len(q_types) else split
                            q_id = q_ids[qi] if qi < len(q_ids) else f"{sample_id}_q{qi}"

                            # Recall
                            t_recall = time.perf_counter()
                            memories = await adapter.recall(question, limit=config.recall_limit)
                            recall_ms = (time.perf_counter() - t_recall) * 1000

                            recalled_context = adapter.build_recalled_context(memories)

                            # Determine system prompt based on competency
                            if split == "CR":
                                system = (
                                    "You are answering questions where the context may contain "
                                    "conflicting information. Always use the MOST RECENT or "
                                    "LATEST information. If facts have been updated or corrected, "
                                    "use the corrected version."
                                )
                            elif split == "TTL":
                                system = (
                                    "You are answering questions where the context teaches you "
                                    "new rules, labels, or procedures. Apply what you learned "
                                    "from the context to answer the question."
                                )
                            else:
                                system = (
                                    "You are answering questions based on information from a "
                                    "long context. Answer precisely with exact information."
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

                            mab_metrics = compute_mab_metrics(prediction, answer, source)

                            # Also run LLM judge for LongMemEval-sourced questions
                            judge_ok = False
                            if "longmemeval" in source:
                                try:
                                    judge_ok = await llm_judge(
                                        question,
                                        str(answer),
                                        prediction,
                                        model=config.judge_model,
                                        api_key=config.openai_api_key,
                                    )
                                except Exception:
                                    pass

                            result = EvalResult(
                                question_id=str(q_id),
                                question=question,
                                question_type=f"{split}/{q_type}",
                                ground_truth=str(answer),
                                prediction=prediction,
                                recalled_context=recalled_context,
                                f1_score=mab_metrics.get("primary_metric", 0.0),
                                judge_score=judge_ok,
                                exact_match=mab_metrics.get("exact_match", 0.0) == 1.0,
                                latency_ingest_ms=ingest_ms / max(len(questions), 1),
                                latency_recall_ms=recall_ms,
                                latency_answer_ms=answer_ms,
                                metadata={
                                    "split": split,
                                    "source": source,
                                    "sample_index": si,
                                    "chunks_ingested": len(chunks),
                                    "memories_recalled": len(memories),
                                    **{
                                        k: v
                                        for k, v in mab_metrics.items()
                                        if k != "primary_metric"
                                    },
                                },
                            )
                            all_results.append(result)

                        last_sample_error = None
                        break  # sample succeeded

                    except _RETRYABLE_ERRORS as e:
                        last_sample_error = e
                        if attempt < MAX_SAMPLE_RETRIES:
                            delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                            logger.warning(
                                "Network error on %s (attempt %d/%d), retrying in %ds: %s",
                                sample_id, attempt, MAX_SAMPLE_RETRIES, delay, e,
                            )
                            await asyncio.sleep(delay)
                        else:
                            logger.error(
                                "Failed sample %s after %d attempts: %s",
                                sample_id, MAX_SAMPLE_RETRIES, e,
                            )

                if last_sample_error is not None:
                    logger.error("Skipping sample %s due to persistent network errors", sample_id)

            # Save per-split results
            split_results = [r for r in all_results if r.metadata.get("split") == split]
            save_results(split_results, output_dir / f"{split}_results.json")

    finally:
        await adapter.cleanup()

    # Save all and compute metrics
    save_results(all_results, output_dir / "all_results.json")
    metrics = compute_aggregate_metrics(all_results)

    # Per-competency metrics (MemoryAgentBench standard)
    competency_metrics: dict[str, Any] = {}
    for split in splits:
        split_results = [r for r in all_results if r.metadata.get("split") == split]
        if split_results:
            # Group by source dataset within the split
            by_source: dict[str, list[EvalResult]] = {}
            for r in split_results:
                src = r.metadata.get("source", split)
                by_source.setdefault(src, []).append(r)

            source_scores = {}
            for src, src_results in by_source.items():
                source_scores[src] = {
                    "count": len(src_results),
                    "primary_metric_avg": float(
                        np.mean([r.f1_score for r in src_results])
                    ),
                    "f1_avg": float(
                        np.mean(
                            [r.metadata.get("f1", r.f1_score) for r in src_results]
                        )
                    ),
                    "substring_em_avg": float(
                        np.mean(
                            [r.metadata.get("substring_em", 0) for r in src_results]
                        )
                    ),
                }

            competency_metrics[split] = {
                "count": len(split_results),
                "avg_primary_metric": float(
                    np.mean([r.f1_score for r in split_results])
                ),
                "by_source": source_scores,
            }

    metrics["memoryagentbench"] = {
        "per_competency": competency_metrics,
        "total_questions": len(all_results),
    }

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print_metrics_table(metrics, "MemoryAgentBench")

    # Print competency breakdown
    print(f"  MemoryAgentBench Competency Breakdown:")
    print(f"  {'Competency':<12} {'Count':>6} {'Primary Metric':>16}")
    print(f"  {'-' * 36}")
    for split_name, vals in competency_metrics.items():
        print(
            f"  {split_name:<12} {vals['count']:>6} "
            f"{vals['avg_primary_metric']:>16.4f}"
        )
        for src, src_vals in vals.get("by_source", {}).items():
            print(
                f"    {src:<24} {src_vals['count']:>6} "
                f"{src_vals['primary_metric_avg']:>10.4f}"
            )
    print()

    return {"results": all_results, "metrics": metrics}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run MemoryAgentBench on Kumiho")
    parser.add_argument("--splits", type=str, default="AR,TTL,LRU,CR",
                        help="Comma-separated splits to evaluate")
    parser.add_argument("--output", type=str, default="./results")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit samples per split")
    parser.add_argument("--chunk-size", type=int, default=4096,
                        help="Context chunk size for memory ingestion")
    parser.add_argument("--answer-model", type=str, default="gpt-4o")
    parser.add_argument("--judge-model", type=str, default="gpt-4o")
    parser.add_argument("--recall-limit", type=int, default=10)
    parser.add_argument("--project", type=str, default="benchmark-mab")
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

    splits = [s.strip() for s in args.splits.split(",")]
    asyncio.run(evaluate_memoryagentbench(config, splits=splits, chunk_size=args.chunk_size))


if __name__ == "__main__":
    main()
