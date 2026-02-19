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
from requests.exceptions import ConnectionError as RequestsConnectionError

# Network error types that warrant retry in adapter methods
_NETWORK_ERRORS = (OSError, RequestsConnectionError, ConnectionError, TimeoutError)
_MAX_ADAPTER_RETRIES = 5
_ADAPTER_RETRY_BASE = 5  # seconds

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
    entry_concurrency: int = 1  # How many entries to process in parallel (pipeline parallelism)
    graph_augmented: bool = False  # Follow edges from recalled memories to discover related ones


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

    async def _retry_network(self, coro_func, *args, **kwargs) -> Any:
        """
        Call an async function with retry on transient network errors.

        Uses exponential backoff: 5s, 10s, 20s, 40s, 80s.
        """
        last_err: Exception | None = None
        for attempt in range(1, _MAX_ADAPTER_RETRIES + 1):
            try:
                return await coro_func(*args, **kwargs)
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
    ) -> dict[str, Any]:
        """Consolidate a session into long-term graph memory.

        When *user_id* and *context* are provided, the memory is stored
        into a user-scoped space (``{context}/{user_id}``).  An explicit
        *space_path* overrides everything.
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

        augmented = list(memories)
        edge_filter = set(edge_types or [
            "DERIVED_FROM", "DEPENDS_ON", "REFERENCED",
            "CONTAINS", "CREATED_FROM", "SUPERSEDES",
        ])

        # --- Strategy A: Edge traversal via kumiho SDK ---
        graph_found = 0
        for mem in memories[:5]:  # Top-5 to avoid explosion
            kref_str = mem.get("kref", "")
            if not kref_str:
                continue
            try:
                rev = kumiho.get_revision(kref_str)
                edges = rev.get_edges(direction=kumiho.BOTH)
                for edge in edges:
                    if edge.edge_type not in edge_filter:
                        continue
                    # Pick the other end of the edge
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
                        augmented.append({
                            "kref": connected_uri,
                            "title": connected_rev.metadata.get("title", ""),
                            "summary": connected_rev.metadata.get("summary", ""),
                            "content": connected_rev.metadata.get("content", ""),
                            "score": 0.0,
                            "graph_augmented": True,
                            "edge_type": edge.edge_type,
                            "from_kref": kref_str,
                        })
                        graph_found += 1
                    except Exception as e:
                        logger.debug("Failed to fetch connected revision %s: %s", connected_uri, e)
            except Exception as e:
                logger.debug("Failed to get edges for %s: %s", kref_str, e)

        # --- Strategy B: Multi-hop semantic recall (fallback if no edges found) ---
        if graph_found == 0 and max_hops >= 1:
            logger.info("No graph edges found, falling back to multi-hop semantic recall")
            secondary_terms = []
            for mem in memories[:5]:
                title = mem.get("title", "")
                summary = mem.get("summary", "")
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

        # Cap total to prevent context noise from drowning signal
        cap = max_total or (base_limit * 3)
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

        logger.info(
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
            logger.info("Edge discovery: no candidates above threshold %.2f", min_score)
            return []

        # Step 3: Create edges to top-N candidates
        sorted_candidates = sorted(
            candidates.values(), key=lambda c: c["score"], reverse=True,
        )[:max_edges]

        # Get the source revision object once (with retry for rate limits)
        source_rev = None
        for attempt in range(1, 4):
            try:
                source_rev = kumiho.get_revision(revision_kref)
                break
            except Exception as e:
                if "RESOURCE_EXHAUSTED" in str(e) and attempt < 3:
                    await asyncio.sleep(0.05 * attempt)
                else:
                    logger.warning("Failed to get source revision %s: %s", revision_kref, e)
                    return []

        created_edges: list[dict[str, Any]] = []
        for cand in sorted_candidates:
            target_kref = cand["memory"].get("kref", "")
            # Retry loop for rate-limited edge creation (server marks
            # CreateEdge as "expensive" — 10 req/s, burst 20).
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
                    created_edges.append({
                        "source": revision_kref,
                        "target": target_kref,
                        "edge_type": edge_type,
                        "query": cand["query"],
                        "score": cand["score"],
                    })
                    logger.info(
                        "Created edge %s → %s (type=%s, query=%r, score=%.3f)",
                        revision_kref, target_kref, edge_type,
                        cand["query"][:60], cand["score"],
                    )
                    break  # success
                except Exception as e:
                    err_str = str(e)
                    if "RESOURCE_EXHAUSTED" in err_str and attempt < 3:
                        wait_ms = 50 * attempt  # 50ms, 100ms
                        logger.debug(
                            "Rate limited on edge %s → %s (attempt %d), retrying in %dms",
                            revision_kref, target_kref, attempt, wait_ms,
                        )
                        await asyncio.sleep(wait_ms / 1000)
                    else:
                        logger.warning("Failed to create edge %s → %s: %s", revision_kref, target_kref, e)
                        break

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

    def build_recalled_context(self, memories: list[dict[str, Any]]) -> str:
        """
        Build text context from recalled memories based on recall_mode.

        - "full": includes artifact content (raw conversation text) — lossless
        - "summarized": only title + summary — lossy, comparable to Mem0/Graphiti

        For stacked items with sibling revisions, the siblings' summaries
        are appended so the LLM sees the full conversation progression
        without each revision consuming a separate recall slot.
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

            # Unfold sibling revisions (conversation progression)
            for sib in mem.get("sibling_revisions", []):
                sib_title = sib.get("title", "")
                sib_summary = sib.get("summary", "")
                if sib_summary:
                    texts.append(
                        f"{sib_title}: {sib_summary}" if sib_title else sib_summary
                    )

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
