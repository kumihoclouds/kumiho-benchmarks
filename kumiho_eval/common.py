"""
Shared utilities for Kumiho benchmark evaluation.

Provides:
- KumihoMemoryAdapter: wraps the kumiho-python SDK for benchmark harnesses
- LLM judge utilities (GPT-4o based scoring)
- Token-level F1 / BLEU / normalization functions matching benchmark conventions
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import string
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import backoff
import numpy as np

logger = logging.getLogger("kumiho_eval")

# ---------------------------------------------------------------------------
# Load .env.local if present (keeps secrets out of shell env)
# ---------------------------------------------------------------------------

_ENV_LOCAL = Path(__file__).resolve().parent / ".env.local"
if _ENV_LOCAL.exists():
    with open(_ENV_LOCAL) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
    logger.debug("Loaded env from %s", _ENV_LOCAL)

# ---------------------------------------------------------------------------
# Text normalisation (mirrors LoCoMo / LongMemEval conventions)
# ---------------------------------------------------------------------------

_ARTICLES_RE = re.compile(r"\b(a|an|the|and)\b", re.IGNORECASE)


def normalize_answer(text: str) -> str:
    """Lowercase, strip articles/punctuation/whitespace — matches LoCoMo eval."""
    text = text.replace(",", "")
    text = _ARTICLES_RE.sub(" ", text)
    text = " ".join(text.split())
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return text.lower().strip()


def token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between normalised strings (Porter-stemmed)."""
    try:
        from nltk.stem import PorterStemmer

        ps = PorterStemmer()
    except ImportError:
        ps = None

    pred_tokens = normalize_answer(prediction).split()
    gt_tokens = normalize_answer(ground_truth).split()

    if ps:
        pred_tokens = [ps.stem(w) for w in pred_tokens]
        gt_tokens = [ps.stem(w) for w in gt_tokens]

    if not pred_tokens or not gt_tokens:
        return 0.0

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def multihop_f1(prediction: str, ground_truth: str) -> float:
    """F1 for multi-hop: split on commas, best-match per ground-truth item."""
    preds = [p.strip() for p in prediction.split(",")]
    gts = [g.strip() for g in ground_truth.split(",")]
    if not gts:
        return 0.0
    scores = []
    for gt in gts:
        best = max((token_f1(p, gt) for p in preds), default=0.0)
        scores.append(best)
    return float(np.mean(scores))


def substring_exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(ground_truth) in normalize_answer(prediction)


def exact_match(prediction: str, ground_truth: str) -> bool:
    return normalize_answer(prediction) == normalize_answer(ground_truth)


# ---------------------------------------------------------------------------
# LLM Judge (GPT-4o)
# ---------------------------------------------------------------------------

_JUDGE_SYSTEM = (
    "You are an impartial judge evaluating whether a model's response "
    "correctly answers a question given the ground truth. Respond with ONLY "
    '"correct" or "incorrect". Be lenient with phrasing differences but '
    "strict on factual accuracy."
)

_JUDGE_TEMPLATE = """Question: {question}
Ground truth answer: {answer}
Model's response: {prediction}

Does the model's response correctly answer the question? Consider:
- Factual equivalence (different phrasing is OK)
- Completeness (all key facts present)
- For temporal questions, allow off-by-one for day/week/month counts

Answer "correct" or "incorrect":"""


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def llm_judge(
    question: str,
    answer: str,
    prediction: str,
    *,
    model: str = "gpt-4o",
    api_key: str | None = None,
) -> bool:
    """Use GPT-4o to judge whether prediction matches ground truth."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    prompt = _JUDGE_TEMPLATE.format(
        question=question, answer=answer, prediction=prediction
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": _JUDGE_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        max_tokens=10,
        temperature=0.0,
    )
    verdict = resp.choices[0].message.content.strip().lower()
    return "correct" in verdict


# ---------------------------------------------------------------------------
# Answer generation via LLM
# ---------------------------------------------------------------------------


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def generate_answer(
    question: str,
    context: str,
    *,
    system_prompt: str = "",
    model: str = "gpt-4o",
    api_key: str | None = None,
    max_tokens: int = 256,
) -> str:
    """Generate an answer to a question given retrieved context."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    user_msg = (
        f"Based on the following memory context, answer the question.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\n"
        f"Answer concisely with exact information from the context."
    )
    resp = await client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt or "You are a helpful assistant."},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=max_tokens,
        temperature=0.0,
    )
    return resp.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Kumiho Memory Adapter
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    project_name: str = "benchmark-eval"
    judge_model: str = "gpt-4o"
    answer_model: str = "gpt-4o"
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-4o-mini"
    kumiho_endpoint: str | None = None
    kumiho_token: str | None = None
    redis_url: str | None = None
    output_dir: str = "./results"
    max_samples: int | None = None
    consolidation_threshold: int = 20
    recall_limit: int = 10
    recall_mode: str = "full"  # "full" = artifact content, "summarized" = title+summary only
    concurrency: int = 4


@dataclass
class EvalResult:
    """Single question evaluation result."""

    question_id: str
    question: str
    question_type: str
    ground_truth: str
    prediction: str
    recalled_context: str = ""
    f1_score: float = 0.0
    judge_score: bool = False
    exact_match: bool = False
    latency_ingest_ms: float = 0.0
    latency_recall_ms: float = 0.0
    latency_answer_ms: float = 0.0
    metadata: dict = field(default_factory=dict)


class KumihoMemoryAdapter:
    """
    Wraps the kumiho-python SDK for benchmark evaluation.

    Lifecycle per conversation:
      1. create_eval_space()  — project + space for the conversation
      2. ingest_session()     — feed session messages through memory manager
      3. consolidate()        — consolidate session to long-term memory
      4. recall()             — query long-term memory
      5. cleanup()            — remove evaluation data
    """

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._manager: Any = None
        self._kumiho_client: Any = None
        self._initialised = False

    async def initialise(self) -> None:
        """Lazily initialise the Kumiho client and memory manager."""
        if self._initialised:
            return

        import kumiho
        from kumiho_memory import (
            RedisMemoryBuffer,
            UniversalMemoryManager,
            MemorySummarizer,
            PIIRedactor,
        )

        # Connect SDK
        endpoint = self.config.kumiho_endpoint or os.environ.get("KUMIHO_ENDPOINT")
        token = self.config.kumiho_token or os.environ.get("KUMIHO_AUTH_TOKEN")

        if endpoint and token:
            self._kumiho_client = kumiho.connect(endpoint=endpoint, token=token)
        else:
            self._kumiho_client = kumiho.connect()

        # Build memory manager components
        redis_buf = RedisMemoryBuffer(
            redis_url=self.config.redis_url or os.environ.get("KUMIHO_UPSTASH_REDIS_URL"),
        )

        summarizer = MemorySummarizer(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
            api_key=(
                self.config.openai_api_key
                or self.config.anthropic_api_key
                or os.environ.get("KUMIHO_LLM_API_KEY")
            ),
        )

        pii_redactor = PIIRedactor()

        # Build store/retrieve callables from the SDK
        async def _store(**kwargs: Any) -> dict:
            """Store memory via SDK — maps to kumiho_memory_store MCP tool."""
            return kumiho.memory_store(**kwargs) if hasattr(kumiho, "memory_store") else {}

        async def _retrieve(**kwargs: Any) -> list:
            """Retrieve memory via SDK — maps to kumiho_memory_retrieve MCP tool."""
            return kumiho.memory_retrieve(**kwargs) if hasattr(kumiho, "memory_retrieve") else []

        self._manager = UniversalMemoryManager(
            project=self.config.project_name,
            consolidation_threshold=self.config.consolidation_threshold,
            redis_buffer=redis_buf,
            summarizer=summarizer,
            pii_redactor=pii_redactor,
        )

        self._initialised = True

    async def create_eval_space(self, conv_id: str) -> str:
        """Create an isolated project space for a conversation evaluation."""
        await self.initialise()
        import kumiho

        space_name = f"eval-{conv_id}"
        try:
            kumiho.create_space(
                project_name=self.config.project_name,
                space_name=space_name,
            )
        except Exception:
            pass  # Space may already exist
        return space_name

    async def ingest_session(
        self,
        *,
        user_id: str,
        session_messages: list[dict[str, str]],
        context: str = "personal",
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Ingest a single session's messages into working memory.

        Args:
            user_id: Stable user identifier for the conversation
            session_messages: List of {"role": "user"|"assistant", "content": "..."}
            context: Memory context (personal, work, etc.)
            session_id: Optional explicit session ID

        Returns:
            Dict with session_id, message_count, timing info
        """
        await self.initialise()

        t0 = time.perf_counter()
        result = {"session_id": session_id, "message_count": 0}

        for msg in session_messages:
            role = msg.get("role", msg.get("speaker", "user"))
            content = msg.get("content", msg.get("text", ""))

            if role in ("user", "human"):
                resp = await self._manager.ingest_message(
                    user_id=user_id,
                    message=content,
                    role="user",
                    context=context,
                    session_id=session_id,
                )
                session_id = resp.get("session_id", session_id)
                result["message_count"] = resp.get("message_count", 0)
            else:
                if session_id:
                    await self._manager.add_assistant_response(
                        session_id=session_id,
                        response=content,
                    )
                    result["message_count"] += 1

        result["session_id"] = session_id
        result["ingest_ms"] = (time.perf_counter() - t0) * 1000
        return result

    async def consolidate(self, session_id: str) -> dict[str, Any]:
        """Consolidate a session into long-term graph memory."""
        await self.initialise()
        t0 = time.perf_counter()
        result = await self._manager.consolidate_session(session_id=session_id)
        result["consolidate_ms"] = (time.perf_counter() - t0) * 1000
        return result

    async def recall(
        self,
        query: str,
        *,
        limit: int | None = None,
        space_paths: list[str] | None = None,
        memory_types: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Recall memories relevant to a query."""
        await self.initialise()
        return await self._manager.recall_memories(
            query,
            limit=limit or self.config.recall_limit,
            space_paths=space_paths,
            memory_types=memory_types,
        )

    async def cleanup(self) -> None:
        """Close connections."""
        if self._manager:
            await self._manager.close()

    def build_recalled_context(self, memories: list[dict[str, Any]]) -> str:
        """
        Build text context from recalled memories based on recall_mode.

        - "full": includes artifact content (raw conversation text) — lossless
        - "summarized": only title + summary — lossy, comparable to Mem0/Graphiti
        """
        texts = []
        for mem in memories:
            title = mem.get("title", "")
            summary = mem.get("summary", "")
            content = mem.get("content", "")

            if self.config.recall_mode == "full" and content:
                texts.append(content[:4000])
            elif summary:
                texts.append(f"{title}: {summary}" if title else summary)

        return "\n\n".join(texts) if texts else ""


# ---------------------------------------------------------------------------
# Results I/O
# ---------------------------------------------------------------------------


def save_results(results: list[EvalResult], path: str | Path) -> None:
    """Save evaluation results to JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = []
    for r in results:
        data.append(
            {
                "question_id": r.question_id,
                "question": r.question,
                "question_type": r.question_type,
                "ground_truth": r.ground_truth,
                "prediction": r.prediction,
                "recalled_context": r.recalled_context[:500],  # truncate for readability
                "f1_score": r.f1_score,
                "judge_score": r.judge_score,
                "exact_match": r.exact_match,
                "latency_ingest_ms": r.latency_ingest_ms,
                "latency_recall_ms": r.latency_recall_ms,
                "latency_answer_ms": r.latency_answer_ms,
                "metadata": r.metadata,
            }
        )

    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Saved %d results to %s", len(data), path)


def compute_aggregate_metrics(results: list[EvalResult]) -> dict[str, Any]:
    """Compute aggregate metrics across all results."""
    if not results:
        return {}

    by_type: dict[str, list[EvalResult]] = {}
    for r in results:
        by_type.setdefault(r.question_type, []).append(r)

    metrics: dict[str, Any] = {
        "total_questions": len(results),
        "overall_f1": float(np.mean([r.f1_score for r in results])),
        "overall_judge_accuracy": float(np.mean([r.judge_score for r in results])),
        "overall_exact_match": float(np.mean([r.exact_match for r in results])),
        "avg_latency_recall_ms": float(np.mean([r.latency_recall_ms for r in results])),
        "avg_latency_answer_ms": float(np.mean([r.latency_answer_ms for r in results])),
        "by_type": {},
    }

    for qtype, type_results in sorted(by_type.items()):
        metrics["by_type"][qtype] = {
            "count": len(type_results),
            "f1": float(np.mean([r.f1_score for r in type_results])),
            "judge_accuracy": float(np.mean([r.judge_score for r in type_results])),
            "exact_match": float(np.mean([r.exact_match for r in type_results])),
        }

    return metrics


def print_metrics_table(metrics: dict[str, Any], benchmark_name: str) -> None:
    """Print a formatted metrics summary table."""
    print(f"\n{'=' * 70}")
    print(f"  {benchmark_name} — Kumiho Cognitive Memory Evaluation")
    print(f"{'=' * 70}")
    print(f"  Total questions: {metrics.get('total_questions', 0)}")
    print(f"  Overall F1:              {metrics.get('overall_f1', 0):.4f}")
    print(f"  Overall Judge Accuracy:  {metrics.get('overall_judge_accuracy', 0):.4f}")
    print(f"  Overall Exact Match:     {metrics.get('overall_exact_match', 0):.4f}")
    print(f"  Avg Recall Latency:      {metrics.get('avg_latency_recall_ms', 0):.1f} ms")
    print(f"  Avg Answer Latency:      {metrics.get('avg_latency_answer_ms', 0):.1f} ms")

    by_type = metrics.get("by_type", {})
    if by_type:
        print(f"\n  {'Category':<30} {'Count':>6} {'F1':>8} {'Judge':>8} {'EM':>8}")
        print(f"  {'-' * 62}")
        for qtype, vals in by_type.items():
            print(
                f"  {qtype:<30} {vals['count']:>6} "
                f"{vals['f1']:>8.4f} {vals['judge_accuracy']:>8.4f} "
                f"{vals['exact_match']:>8.4f}"
            )

    print(f"{'=' * 70}\n")
