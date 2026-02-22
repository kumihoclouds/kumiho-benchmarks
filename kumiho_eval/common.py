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
import subprocess
import threading
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import backoff
import numpy as np
from requests.exceptions import ConnectionError as RequestsConnectionError

# OpenAI SDK exceptions (uses httpx internally)
try:
    from openai import APIError as OpenAIAPIError
    from openai import APIConnectionError as OpenAIConnectionError
    from openai import APITimeoutError as OpenAITimeoutError
    from openai import RateLimitError as OpenAIRateLimitError
    from openai import InternalServerError as OpenAIInternalServerError
    _OPENAI_ERRORS: tuple = (
        OpenAIAPIError, OpenAIConnectionError, OpenAITimeoutError,
        OpenAIRateLimitError, OpenAIInternalServerError,
    )
except ImportError:
    _OPENAI_ERRORS = ()

# Network error types that warrant retry in adapter methods
_NETWORK_ERRORS = (
    OSError, RequestsConnectionError, ConnectionError, TimeoutError,
    *_OPENAI_ERRORS,
)
_MAX_ADAPTER_RETRIES = 5
_ADAPTER_RETRY_BASE = 5  # seconds
_CALL_TIMEOUT = 120  # seconds — per SDK call timeout

logger = logging.getLogger("kumiho_eval")

# ---------------------------------------------------------------------------
# Load .env.local if present (keeps secrets out of shell env)
# ---------------------------------------------------------------------------

_ENV_LOCAL = Path(__file__).resolve().parent / ".env.local"
if _ENV_LOCAL.exists():
    with open(_ENV_LOCAL, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())
    logger.debug("Loaded env from %s", _ENV_LOCAL)


# ---------------------------------------------------------------------------
# Token usage tracking
# ---------------------------------------------------------------------------


class TokenTracker:
    """Thread-safe token usage tracker aggregated by phase.

    Records prompt_tokens, completion_tokens, and total_tokens from OpenAI
    API responses.  Aggregates by phase (e.g. "judge", "answer",
    "retrieval_judge") so cost claims can be verified from the run manifest.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._phases: dict[str, dict[str, int]] = {}

    def record(self, phase: str, response: Any) -> dict[str, int]:
        """Extract and record token usage from an OpenAI chat response."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        tokens = {
            "prompt_tokens": getattr(usage, "prompt_tokens", 0) or 0,
            "completion_tokens": getattr(usage, "completion_tokens", 0) or 0,
            "total_tokens": getattr(usage, "total_tokens", 0) or 0,
        }

        with self._lock:
            if phase not in self._phases:
                self._phases[phase] = {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "calls": 0,
                }
            self._phases[phase]["prompt_tokens"] += tokens["prompt_tokens"]
            self._phases[phase]["completion_tokens"] += tokens["completion_tokens"]
            self._phases[phase]["total_tokens"] += tokens["total_tokens"]
            self._phases[phase]["calls"] += 1

        return tokens

    def summary(self) -> dict[str, Any]:
        """Return per-phase and total token usage."""
        with self._lock:
            total = {
                "prompt_tokens": 0, "completion_tokens": 0,
                "total_tokens": 0, "calls": 0,
            }
            for phase_data in self._phases.values():
                for key in total:
                    total[key] += phase_data[key]
            return {
                "by_phase": {k: dict(v) for k, v in self._phases.items()},
                "total": total,
            }

    def reset(self) -> None:
        """Reset all counters (call before each benchmark run)."""
        with self._lock:
            self._phases.clear()


token_tracker = TokenTracker()


# ---------------------------------------------------------------------------
# Prompt template registry (for manifest hash generation)
# ---------------------------------------------------------------------------

_PROMPT_TEMPLATE_REGISTRY: dict[str, str] = {}


def register_prompt_template(name: str, template: str) -> None:
    """Register a prompt template for manifest hash generation."""
    _PROMPT_TEMPLATE_REGISTRY[name] = template


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

register_prompt_template("llm_judge", _JUDGE_TEMPLATE)


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
    token_tracker.record("judge", resp)
    raw_verdict = resp.choices[0].message.content.strip()
    # Extract first token and match strictly — "incorrect" must NOT match "correct"
    verdict = raw_verdict.lower().split()[0] if raw_verdict.strip() else ""
    return verdict == "correct"


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
    temperature: float = 0.0,
) -> str:
    """Generate an answer to a question given retrieved context."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    user_msg = f"Context:\n{context}\n\n{question}"
    for _answer_attempt in range(3):
        resp = await client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or "You are a helpful assistant."},
                {"role": "user", "content": user_msg},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        token_tracker.record("answer", resp)
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
        logger.warning("Empty response from %s (attempt %d/3), retrying", model, _answer_attempt + 1)
    return text  # return empty after 3 tries rather than loop forever


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
    start_at: int = 0  # Skip entries before this index (e.g. 201 for cog-201)
    consolidation_threshold: int = 20
    recall_limit: int = 5
    recall_mode: str = "full"  # "full" = artifact content, "summarized" = title+summary only
    concurrency: int = 4
    entry_concurrency: int = 1  # How many entries to process in parallel (pipeline parallelism)
    graph_augmented: bool = True  # Graph-native: edge traversal + multi-query recall (Kumiho default)
    sibling_similarity_threshold: float = 0.30  # Min cosine similarity for siblings (0=off)
    sibling_top_k: int = 0  # Max siblings to keep after scoring (0=unlimited, use threshold only)
    context_top_k: int = 0  # Global cap on revisions in final context (0=unlimited)
    stack_revisions: bool = True  # True = stack similar sessions; False = one item per session


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

        # Connect SDK — pass whatever auth we have; connect() handles discovery
        endpoint = (
            self.config.kumiho_endpoint
            or os.environ.get("KUMIHO_ENDPOINT")
            or os.environ.get("KUMIHO_SERVER_ENDPOINT")
        )
        token = self.config.kumiho_token or os.environ.get("KUMIHO_AUTH_TOKEN")

        connect_kwargs: dict[str, Any] = {}
        if endpoint:
            connect_kwargs["endpoint"] = endpoint
            # When an explicit endpoint is given, skip discovery so we don't
            # override it with a cloud server_url.
            connect_kwargs["use_discovery"] = False
        if token:
            connect_kwargs["token"] = token
        self._kumiho_client = kumiho.connect(**connect_kwargs)
        # Set as global default so kumiho.memory_store(), kumiho.get_revision(),
        # etc. all use this client instead of bootstrapping a new one via discovery.
        kumiho.configure_default_client(self._kumiho_client)

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

        # Import store/retrieve from the MCP server module, then patch
        # _ensure_configured to a no-op so it doesn't re-discover and
        # override our local client on every call.
        import kumiho.mcp_server as _mcp_mod
        _mcp_mod._ensure_configured = lambda: True
        _mcp_store = _mcp_mod.tool_memory_store
        _mcp_retrieve = _mcp_mod.tool_memory_retrieve

        async def _store(**kwargs: Any) -> dict:
            """Store memory via SDK — wraps kumiho.mcp_server.tool_memory_store."""
            return _mcp_store(**kwargs)

        async def _retrieve(**kwargs: Any) -> list:
            """Retrieve memory via SDK — wraps kumiho.mcp_server.tool_memory_retrieve."""
            return _mcp_retrieve(**kwargs)

        self._manager = UniversalMemoryManager(
            project=self.config.project_name,
            consolidation_threshold=self.config.consolidation_threshold,
            redis_buffer=redis_buf,
            summarizer=summarizer,
            pii_redactor=pii_redactor,
            memory_store=_store,
            memory_retrieve=_retrieve,
            recall_mode=self.config.recall_mode,
            sibling_similarity_threshold=self.config.sibling_similarity_threshold,
            sibling_top_k=self.config.sibling_top_k,
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

        Includes retry with exponential backoff for transient network errors.

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
                resp = await self._retry_network(
                    self._manager.ingest_message,
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
                    await self._retry_network(
                        self._manager.add_assistant_response,
                        session_id=session_id,
                        response=content,
                    )
                    result["message_count"] += 1

        result["session_id"] = session_id
        result["ingest_ms"] = (time.perf_counter() - t0) * 1000
        return result

    async def _retry_network(self, coro_func, *args, timeout: float = _CALL_TIMEOUT, **kwargs) -> Any:
        """
        Call an async function with retry on transient network errors.

        Each attempt is wrapped in asyncio.wait_for with *timeout* seconds
        to prevent indefinite hangs from unresponsive servers.
        Uses exponential backoff: 5s, 10s, 20s, 40s, 80s.
        """
        last_err: Exception | None = None
        for attempt in range(1, _MAX_ADAPTER_RETRIES + 1):
            try:
                return await asyncio.wait_for(
                    coro_func(*args, **kwargs), timeout=timeout,
                )
            except _NETWORK_ERRORS as e:
                last_err = e
                if attempt < _MAX_ADAPTER_RETRIES:
                    delay = _ADAPTER_RETRY_BASE * (2 ** (attempt - 1))
                    logger.warning(
                        "Network error (attempt %d/%d), retrying in %ds: %s",
                        attempt, _MAX_ADAPTER_RETRIES, delay, e,
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Network error persists after %d attempts: %s",
                        _MAX_ADAPTER_RETRIES, e,
                    )
                    raise

        # Should not reach here, but just in case
        raise last_err  # type: ignore[misc]

    async def consolidate(
        self,
        session_id: str,
        *,
        space_path: str | None = None,
        user_id: str | None = None,
        context: str | None = None,
        stack_revisions: bool | None = None,
    ) -> dict[str, Any]:
        """Consolidate a session into long-term graph memory.

        When *user_id* and *context* are provided, the memory is stored
        into a user-scoped space (``{context}/{user_id}``).  An explicit
        *space_path* overrides everything.

        Set *stack_revisions* to ``False`` to create a new item per session
        instead of stacking onto an existing similar item.
        """
        await self.initialise()
        t0 = time.perf_counter()
        kwargs: dict[str, Any] = {"session_id": session_id}
        if space_path:
            kwargs["space_path"] = space_path
        if user_id:
            kwargs["user_id"] = user_id
        if context:
            kwargs["context"] = context
        if stack_revisions is not None:
            kwargs["stack_revisions"] = stack_revisions
        result = await self._retry_network(
            self._manager.consolidate_session, **kwargs,
        )
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
        return await self._retry_network(
            self._manager.recall_memories,
            query,
            limit=limit or self.config.recall_limit,
            space_paths=space_paths,
            memory_types=memory_types,
        )

    async def _reformulate_query(self, query: str) -> list[str]:
        """
        Generate alternative search queries that capture different semantic
        angles of a trigger message.  Bridges the cue-trigger semantic
        disconnect by reformulating around underlying emotion, causal event,
        and related concepts.

        Returns 2-3 reformulated queries (does NOT include the original).
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.config.openai_api_key
            or os.environ.get("OPENAI_API_KEY"),
        )
        system = (
            "You generate alternative memory search queries. "
            "Given a conversational message, produce 2-3 short search queries "
            "that capture different semantic angles of what this person might "
            "be referring to from their past. Focus on:\n"
            "- The underlying emotion or concern\n"
            "- A possible causal event that led to this behavior\n"
            "- Related situations or consequences\n"
            "Return ONLY the queries, one per line, no numbering or bullets."
        )
        try:
            resp = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": query},
                ],
                max_tokens=100,
                temperature=0.3,
            )
            token_tracker.record("recall_reformulation", resp)
            raw = resp.choices[0].message.content.strip()
            queries = [
                line.strip().lstrip("0123456789.-) ")
                for line in raw.splitlines()
                if line.strip()
            ]
            logger.info(
                "Multi-query reformulation: %d queries from trigger",
                len(queries),
            )
            return queries[:3]
        except Exception as e:
            logger.warning("Query reformulation failed: %s", e)
            return []

    @staticmethod
    def _collect_top_revisions(
        memories: list[dict[str, Any]],
        limit: int,
    ) -> list[dict[str, Any]]:
        """Flatten sibling revisions, return top-*limit* by score.

        When siblings exist, the primary memory is skipped (it's the
        item-level shell whose recall score is on a different scale).
        Each returned dict has at least ``title``, ``summary``, ``kref``,
        and ``_score`` keys.
        """
        candidates: list[dict[str, Any]] = []
        for mem in memories:
            siblings = mem.get("sibling_revisions", [])
            if siblings:
                for sib in siblings:
                    candidates.append({
                        "kref": sib.get("kref", ""),
                        "title": sib.get("title", ""),
                        "summary": sib.get("summary", ""),
                        "_score": sib.get("_score", 0.0),
                    })
            else:
                candidates.append({
                    "kref": mem.get("kref", ""),
                    "title": mem.get("title", ""),
                    "summary": mem.get("summary", ""),
                    "_score": mem.get("score", 0.0),
                })
        candidates.sort(key=lambda x: x.get("_score", 0.0), reverse=True)
        return candidates[:limit]

    async def recall_with_graph_augmentation(
        self,
        query: str,
        *,
        limit: int | None = None,
        max_total: int | None = None,
        max_hops: int = 1,
        edge_types: list[str] | None = None,
        space_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        Recall with graph augmentation:
          multi-query reformulation → parallel recall → edge traversal → merge.

        After standard vector recall, follows edges from each recalled memory
        to discover connected memories that vector similarity alone would miss.
        This is critical for implicit-constraint tasks (e.g. LoCoMo-Plus) where
        the trigger query has intentionally low semantic overlap with the cue.

        Falls back to multi-hop semantic recall if the graph API is unavailable.
        """
        await self.initialise()
        import kumiho

        base_limit = limit or self.config.recall_limit

        # --- Multi-query recall: reformulate trigger into multiple angles ---
        alt_queries = await self._reformulate_query(query)
        all_queries = [query] + alt_queries

        # Run all queries in parallel
        recall_tasks = [
            self.recall(q, limit=base_limit, space_paths=space_paths)
            for q in all_queries
        ]
        recall_results = await asyncio.gather(*recall_tasks, return_exceptions=True)

        # Merge and deduplicate by kref, keeping highest score
        best_by_kref: dict[str, dict[str, Any]] = {}
        for result in recall_results:
            if isinstance(result, Exception):
                logger.debug("Recall query failed: %s", result)
                continue
            for mem in result:
                kref = mem.get("kref", "")
                if not kref:
                    continue
                existing = best_by_kref.get(kref)
                if existing is None or mem.get("score", 0) > existing.get("score", 0):
                    best_by_kref[kref] = mem

        # Sort by score descending, take top base_limit * 2
        memories = sorted(
            best_by_kref.values(),
            key=lambda m: m.get("score", 0),
            reverse=True,
        )[: base_limit * 2]

        if len(all_queries) > 1:
            logger.info(
                "Multi-query recall: %d queries → %d unique memories (from %d total)",
                len(all_queries),
                len(memories),
                sum(
                    len(r) for r in recall_results
                    if not isinstance(r, Exception)
                ),
            )

        if not memories:
            return memories

        seen_krefs: set[str] = set()
        for m in memories:
            kref = m.get("kref", "")
            if kref:
                seen_krefs.add(kref)
            # Also mark sibling krefs as seen so edge traversal
            # doesn't re-discover revisions we already have.
            for sib in m.get("sibling_revisions", []):
                sib_kref = sib.get("kref", "")
                if sib_kref:
                    seen_krefs.add(sib_kref)

        augmented = list(memories)
        edge_filter = set(edge_types or [
            "DERIVED_FROM", "DEPENDS_ON", "REFERENCED",
            "CONTAINS", "CREATED_FROM", "SUPERSEDES",
        ])

        # --- Collect top-K scored revisions for edge traversal ---
        # Instead of traversing from primary recalled items (which with
        # stacking are always the same 1-2 items), flatten sibling
        # revisions, rank by _score, and traverse from the top-K.
        # Primary memory is skipped when siblings exist (score scale
        # mismatch: recall ~3.0 vs sibling cosine 0-1).
        revision_candidates: list[tuple[str, float]] = []
        for mem in memories:
            siblings = mem.get("sibling_revisions", [])
            if siblings:
                for sib in siblings:
                    sib_kref = sib.get("kref", "")
                    if sib_kref:
                        revision_candidates.append((sib_kref, sib.get("_score", 0.0)))
            else:
                kref = mem.get("kref", "")
                if kref:
                    revision_candidates.append((kref, mem.get("score", 0.0)))

        # Sort by score descending, pick top-K for traversal
        revision_candidates.sort(key=lambda x: x[1], reverse=True)
        traverse_limit = self.config.context_top_k or 5
        traverse_krefs = [
            kref for kref, _ in revision_candidates[:traverse_limit]
        ]

        if traverse_krefs:
            logger.info(
                "Graph augmentation: traversing edges from %d top-scored revisions (top score=%.3f)",
                len(traverse_krefs),
                revision_candidates[0][1] if revision_candidates else 0,
            )

        # --- Strategy A: Edge traversal via kumiho SDK ---
        # kumiho.get_revision / rev.get_edges are *synchronous* gRPC calls
        # that can hang indefinitely on Windows.  Neither asyncio.wait_for
        # nor asyncio.wait reliably timeout these calls because the Windows
        # ProactorEventLoop doesn't process timer callbacks while to_thread
        # futures are pending.  We use a bare daemon thread + threading.Event
        # with an OS-level timeout (WaitForSingleObject) which is guaranteed
        # to return regardless of asyncio state.
        _GRAPH_TRAVERSAL_TIMEOUT = 30  # seconds

        graph_found = 0
        graph_augmented_results: list[dict] = []

        def _sync_graph_traverse() -> int:
            """Run all sync gRPC calls in a plain thread.

            Traverses edges from the top-K scored revisions (not the
            primary recalled items) so that graph augmentation discovers
            connections relevant to the specific question.
            """
            found = 0
            for kref_str in traverse_krefs:
                try:
                    rev = kumiho.get_revision(kref_str)
                    edges = rev.get_edges(direction=kumiho.BOTH)
                    for edge in edges:
                        if edge.edge_type not in edge_filter:
                            continue
                        connected_uri = (
                            edge.target_kref.uri
                            if edge.source_kref.uri == kref_str
                            else edge.source_kref.uri
                        )
                        if not connected_uri or connected_uri in seen_krefs:
                            continue
                        seen_krefs.add(connected_uri)
                        try:
                            connected_rev = kumiho.get_revision(connected_uri)
                            graph_augmented_results.append({
                                "kref": connected_uri,
                                "title": connected_rev.metadata.get("title", ""),
                                "summary": connected_rev.metadata.get("summary", ""),
                                "content": connected_rev.metadata.get("content", ""),
                                "score": 0.0,
                                "graph_augmented": True,
                                "edge_type": edge.edge_type,
                                "from_kref": kref_str,
                            })
                            found += 1
                        except Exception as e:
                            logger.debug("Failed to fetch connected revision %s: %s", connected_uri, e)
                except Exception as e:
                    logger.debug("Failed to get edges for %s: %s", kref_str, e)
            return found

        # Run in a daemon thread with OS-level timeout via threading.Event
        _done_event = threading.Event()
        _traverse_result: list[int] = []  # mutable container for thread result

        def _worker():
            try:
                _traverse_result.append(_sync_graph_traverse())
            except Exception as e:
                logger.debug("Graph traversal thread error: %s", e)
            finally:
                _done_event.set()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        # Poll the event from the event loop — does NOT consume thread pool
        # threads (unlike asyncio.to_thread which can starve when many
        # concurrent entries hold pool threads in edge-creation waits).
        _deadline = time.monotonic() + _GRAPH_TRAVERSAL_TIMEOUT
        while not _done_event.is_set():
            if time.monotonic() >= _deadline:
                break
            await asyncio.sleep(0.5)
        completed = _done_event.is_set()

        if completed and _traverse_result:
            graph_found = _traverse_result[0]
            augmented.extend(graph_augmented_results)
        else:
            logger.warning(
                "\033[1;33mGraph traversal timed out after %ds — falling back to semantic recall\033[0m",
                _GRAPH_TRAVERSAL_TIMEOUT,
            )

        # --- Strategy B: Multi-hop semantic recall (fallback if no edges found) ---
        # Use top-K scored revisions (not primary items) so the semantic
        # fallback is also question-specific when items are stacked.
        if graph_found == 0 and max_hops >= 1:
            logger.debug("No graph edges found, falling back to multi-hop semantic recall")
            # Gather title/summary from the top-K scored revisions
            secondary_terms = []
            _top_revisions_for_fallback = self._collect_top_revisions(memories, traverse_limit)
            for rev_info in _top_revisions_for_fallback:
                title = rev_info.get("title", "")
                summary = rev_info.get("summary", "")
                if title:
                    secondary_terms.append(title)
                elif summary:
                    secondary_terms.append(summary[:100])

            if secondary_terms:
                augmented_query = " ".join(secondary_terms)
                hop_memories = await self.recall(augmented_query, limit=base_limit)
                for mem in hop_memories:
                    kref = mem.get("kref", "")
                    if kref and kref not in seen_krefs:
                        seen_krefs.add(kref)
                        mem["graph_augmented"] = True
                        mem["hop"] = 1
                        augmented.append(mem)

        if len(augmented) > len(memories):
            logger.info(
                "Graph augmentation: %d base + %d augmented = %d total memories",
                len(memories), len(augmented) - len(memories), len(augmented),
            )

        # Cap total — graph augmentation adds targeted connections, not flood
        cap = max_total or (base_limit + 5)
        if len(augmented) > cap:
            logger.info("Capping augmented memories from %d to %d", len(augmented), cap)
            augmented = augmented[:cap]

        return augmented

    # ------------------------------------------------------------------
    # Post-consolidation edge discovery (Option 3: LLM-driven linking)
    # ------------------------------------------------------------------

    async def discover_and_link_edges(
        self,
        revision_kref: str,
        summary: str,
        *,
        max_queries: int = 5,
        max_edges: int = 3,
        min_score: float = 0.3,
        edge_type: str = "REFERENCED",
        space_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """
        After consolidation, use an LLM to discover and create edges
        between the new revision and existing related memories.

        This bridges the cue–trigger semantic disconnect: the LLM generates
        "implication queries" — future scenarios where this memory would be
        relevant — and links to existing memories that match.

        When *space_paths* is not provided, the space is automatically
        derived from *revision_kref* so edge discovery only searches
        within the same user/space scope as the source memory.

        Returns list of created edges with metadata.
        """
        await self.initialise()
        import kumiho

        if not revision_kref or not summary:
            return []

        # Auto-derive space scope from the revision kref when not provided.
        # e.g. kref://project/personal/user-1/item.kind?r=1
        #      → space_paths = ["project/personal/user-1"]
        if space_paths is None:
            try:
                # Strip "kref://" prefix, then split off the item (last segment
                # before '?') to get the space path.
                path_part = revision_kref.split("?")[0]
                if path_part.startswith("kref://"):
                    path_part = path_part[len("kref://"):]
                # path_part = "project/space/.../item_name.kind"
                # The item is the last segment; everything before it is the
                # space path.
                segments = path_part.strip("/").split("/")
                if len(segments) >= 3:
                    # project + at least one space + item
                    space_paths = ["/".join(segments[:-1])]
            except Exception:
                pass  # Fall back to un-scoped search

        # Step 1: Ask LLM for implication queries
        queries = await self._generate_implication_queries(summary, max_queries=max_queries)
        if not queries:
            return []

        logger.debug(
            "Edge discovery for %s: generated %d implication queries (space_paths=%s)",
            revision_kref, len(queries), space_paths,
        )

        # Step 2: Search existing memories with each query (parallel)
        candidates: dict[str, dict[str, Any]] = {}  # kref → {memory, best_score, query}

        async def _search_one(q: str) -> list[tuple[str, dict]]:
            try:
                mems = await self.recall(q, limit=3, space_paths=space_paths)
                return [(q, m) for m in mems]
            except Exception as e:
                logger.debug("Edge discovery recall failed for query %r: %s", q, e)
                return []

        search_results = await asyncio.gather(*[_search_one(q) for q in queries])
        for hits in search_results:
            for q, mem in hits:
                kref = mem.get("kref", "")
                score = mem.get("score", 0.0)
                if not kref or kref == revision_kref:
                    continue
                if score < min_score:
                    continue
                if kref not in candidates or score > candidates[kref]["score"]:
                    candidates[kref] = {
                        "memory": mem,
                        "score": score,
                        "query": q,
                    }

        if not candidates:
            logger.debug("Edge discovery: no candidates above threshold %.2f", min_score)
            return []

        # Step 3: Create edges to top-N candidates
        sorted_candidates = sorted(
            candidates.values(), key=lambda c: c["score"], reverse=True,
        )[:max_edges]

        # Get the source revision, create edges — all sync gRPC calls.
        # Same daemon-thread + threading.Event pattern as Strategy A in
        # recall_with_graph_augmentation.  asyncio.wait / asyncio.wait_for
        # do NOT reliably timeout asyncio.to_thread on Windows.
        _EDGE_CREATION_TIMEOUT = 60  # seconds

        _edge_results: list[dict[str, Any]] = []

        def _sync_create_edges() -> list[dict[str, Any]]:
            """Run all sync gRPC edge-creation calls in a plain thread."""
            source_rev = None
            for attempt in range(1, 4):
                try:
                    source_rev = kumiho.get_revision(revision_kref)
                    break
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" in str(e) and attempt < 3:
                        time.sleep(0.05 * attempt)
                    else:
                        logger.warning("Failed to get source revision %s: %s", revision_kref, e)
                        return []

            edges_out: list[dict[str, Any]] = []
            for cand in sorted_candidates:
                target_kref = cand["memory"].get("kref", "")
                for attempt in range(1, 4):
                    try:
                        target_rev = kumiho.get_revision(target_kref)
                        source_rev.create_edge(
                            target_rev,
                            edge_type,
                            metadata={
                                "reason": f"LLM implication: {cand['query'][:100]}",
                                "score": str(round(cand["score"], 3)),
                            },
                        )
                        edges_out.append({
                            "source": revision_kref,
                            "target": target_kref,
                            "edge_type": edge_type,
                            "query": cand["query"],
                            "score": cand["score"],
                        })
                        logger.debug(
                            "Created edge %s → %s (type=%s, query=%r, score=%.3f)",
                            revision_kref, target_kref, edge_type,
                            cand["query"][:60], cand["score"],
                        )
                        break  # success
                    except Exception as e:
                        err_str = str(e)
                        if "RESOURCE_EXHAUSTED" in err_str and attempt < 3:
                            wait_ms = 50 * attempt
                            logger.debug(
                                "Rate limited on edge %s → %s (attempt %d), retrying in %dms",
                                revision_kref, target_kref, attempt, wait_ms,
                            )
                            time.sleep(wait_ms / 1000)
                        else:
                            logger.warning("Failed to create edge %s → %s: %s", revision_kref, target_kref, e)
                            break
            return edges_out

        _edge_done = threading.Event()

        def _edge_worker():
            try:
                _edge_results.extend(_sync_create_edges())
            except Exception as e:
                logger.debug("Edge creation thread error: %s", e)
            finally:
                _edge_done.set()

        t = threading.Thread(target=_edge_worker, daemon=True)
        t.start()

        # Poll — same pattern as graph traversal, avoids thread pool starvation
        _edge_deadline = time.monotonic() + _EDGE_CREATION_TIMEOUT
        while not _edge_done.is_set():
            if time.monotonic() >= _edge_deadline:
                break
            await asyncio.sleep(0.5)
        completed = _edge_done.is_set()

        if completed:
            created_edges = list(_edge_results)
        else:
            logger.warning(
                "\033[1;33mEdge creation timed out after %ds for %s\033[0m",
                _EDGE_CREATION_TIMEOUT, revision_kref,
            )
            created_edges = []

        return created_edges

    @staticmethod
    async def _generate_implication_queries(
        summary: str,
        *,
        max_queries: int = 5,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
    ) -> list[str]:
        """
        Generate search queries for scenarios where this memory would be relevant.

        The LLM thinks beyond literal content to identify implicit constraints,
        life themes, and future situations this memory affects.
        """
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))

        prompt = f"""Given this conversation memory, generate {max_queries} search queries that would help find this memory in the FUTURE when someone is in a related situation.

Think BEYOND the literal content. Consider:
- What implicit constraints or decisions were established?
- What life situations or problems would this memory be relevant to?
- What future scenarios might this memory affect?
- What emotional states or challenges connect to this topic?

Memory:
{summary[:2000]}

Return ONLY a JSON array of {max_queries} short search queries (each 3-8 words). No explanation.
Example: ["feeling overwhelmed with commitments", "declining social invitations", "work-life balance stress"]"""

        raw = ""
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.7,
            )
            token_tracker.record("implication_queries", resp)
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*\n?", "", raw)
                raw = re.sub(r"\n?```\s*$", "", raw)
                raw = raw.strip()
            # Parse JSON array
            queries = json.loads(raw)
            if isinstance(queries, list):
                return [str(q).strip() for q in queries if q][:max_queries]
        except Exception as e:
            logger.warning("Failed to generate implication queries: %s (raw=%r)", e, raw[:200])

        return []

    async def cleanup(self) -> None:
        """Close connections."""
        if self._manager:
            await self._manager.close()

    # -----------------------------------------------------------------
    # Embedding-based sibling relevance scoring
    # -----------------------------------------------------------------

    @staticmethod
    def _embed_texts(
        texts: list[str],
        api_key: str | None,
        model: str = "text-embedding-3-small",
    ) -> np.ndarray:
        """Batch-embed texts via OpenAI and return an (N, dim) numpy array."""
        from openai import OpenAI

        client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
        resp = client.embeddings.create(input=texts, model=model)
        return np.array([item.embedding for item in resp.data])

    @staticmethod
    def _cosine_similarities(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity between a single query vector and a matrix of rows."""
        norms = np.linalg.norm(matrix, axis=1) * np.linalg.norm(query_vec)
        norms = np.where(norms == 0, 1e-9, norms)
        return matrix @ query_vec / norms

    def _filter_siblings_by_embedding(
        self,
        siblings: list[dict[str, Any]],
        query: str,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Keep only siblings whose embedding similarity to *query* exceeds *threshold*."""
        if not siblings or not query or threshold <= 0:
            return siblings

        sib_texts = []
        for sib in siblings:
            t = sib.get("title", "")
            s = sib.get("summary", "")
            sib_texts.append(f"{t}: {s}" if t else s)

        try:
            all_texts = [query] + sib_texts
            embeddings = self._embed_texts(all_texts, self.config.openai_api_key)
            query_vec = embeddings[0]
            sib_matrix = embeddings[1:]
            scores = self._cosine_similarities(query_vec, sib_matrix)

            kept = []
            for i, sib in enumerate(siblings):
                if scores[i] >= threshold:
                    kept.append(sib)
            logger.debug(
                "Sibling embedding filter: %d/%d kept (threshold=%.2f, scores=%s)",
                len(kept), len(siblings), threshold,
                [f"{s:.3f}" for s in scores],
            )
            return kept
        except Exception as e:
            logger.warning("Sibling embedding filter failed, keeping all: %s", e)
            return siblings

    # -----------------------------------------------------------------
    # Context builder
    # -----------------------------------------------------------------

    def build_recalled_context(
        self,
        memories: list[dict[str, Any]],
        query: str = "",
    ) -> str:
        """
        Build text context from recalled memories based on recall_mode.

        - "full": includes artifact content (raw conversation text) — lossless
        - "summarized": only title + summary — lossy, comparable to Mem0/Graphiti

        **Revision-centric assembly**: flattens all revisions (primary +
        siblings) from every recalled memory, ranks them globally by
        ``_score``, and takes the top ``context_top_k``.  Item-level
        metadata is skipped — only revision-level content appears.
        """
        full_mode = self.config.recall_mode == "full"

        # --- Collect all revisions across all memories into one flat list ---
        all_revisions: list[dict[str, Any]] = []

        for mem in memories:
            siblings = mem.get("sibling_revisions", [])

            if siblings:
                # With stacking, the primary memory is the *item-level*
                # shell (latest revision's generic title/summary).  Its
                # recall score (~3.0+) is on a different scale than sibling
                # _scores (cosine 0-1), so including it would always make
                # it dominate the top-K.  Skip it — the siblings ARE the
                # individually scored revisions we want.
                for sib in siblings:
                    all_revisions.append({
                        "title": sib.get("title", ""),
                        "summary": sib.get("summary", ""),
                        "content": sib.get("content", ""),
                        "_score": sib.get("_score", 0.0),
                    })
            else:
                # No siblings (non-stacked item or single revision) —
                # use the primary memory directly.
                all_revisions.append({
                    "title": mem.get("title", ""),
                    "summary": mem.get("summary", ""),
                    "content": mem.get("content", ""),
                    "_score": mem.get("score", 0.0),
                })

        # --- Global ranking by score (best revisions first) ---
        has_scores = any(r.get("_score", 0) > 0 for r in all_revisions)
        if has_scores:
            all_revisions.sort(key=lambda r: r.get("_score", 0.0), reverse=True)

        # --- Apply global top-K cap ---
        top_k = self.config.context_top_k
        if top_k > 0 and len(all_revisions) > top_k:
            all_revisions = all_revisions[:top_k]

        # --- Build text from surviving revisions ---
        texts = []
        for rev in all_revisions:
            title = rev.get("title", "")
            summary = rev.get("summary", "")
            content = rev.get("content", "")

            if full_mode and content:
                texts.append(content[:8000])
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


# ---------------------------------------------------------------------------
# Run manifest (reproducibility)
# ---------------------------------------------------------------------------


def _get_git_sha(repo_path: Path) -> str:
    """Get the HEAD commit SHA of a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path),
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _get_submodule_shas(repo_root: Path) -> dict[str, str]:
    """Get commit SHAs for dataset submodules."""
    shas = {}
    for submod in ["locomo", "LongMemEval", "MemoryAgentBench"]:
        submod_path = repo_root / submod
        if submod_path.is_dir():
            shas[submod] = _get_git_sha(submod_path)
    return shas


def generate_run_manifest(
    config: BenchmarkConfig,
    benchmarks: list[str],
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Generate a reproducibility manifest for a benchmark run.

    Captures: harness git commit, dataset SHAs, model names, prompt template
    hashes, config flags, and timestamps.  Written alongside metrics so any
    reviewer can verify the exact evaluation environment.
    """
    if repo_root is None:
        repo_root = Path(__file__).resolve().parent.parent

    # Collect all registered prompt template hashes
    template_hashes = {}
    for name, template in sorted(_PROMPT_TEMPLATE_REGISTRY.items()):
        template_hashes[name] = hashlib.sha256(
            template.encode("utf-8"),
        ).hexdigest()[:16]

    return {
        "harness_git_sha": _get_git_sha(repo_root),
        "dataset_shas": _get_submodule_shas(repo_root),
        "benchmarks": benchmarks,
        "config": {
            "answer_model": config.answer_model,
            "judge_model": config.judge_model,
            "llm_model": config.llm_model,
            "llm_provider": config.llm_provider,
            "recall_limit": config.recall_limit,
            "recall_mode": config.recall_mode,
            "graph_augmented": config.graph_augmented,
            "sibling_similarity_threshold": config.sibling_similarity_threshold,
            "consolidation_threshold": config.consolidation_threshold,
            "max_samples": config.max_samples,
            "start_at": config.start_at,
            "concurrency": config.concurrency,
            "entry_concurrency": config.entry_concurrency,
        },
        "prompt_template_hashes": template_hashes,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
    }
