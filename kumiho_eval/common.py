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

    def record_usage(self, phase: str, tokens: dict[str, Any]) -> dict[str, int]:
        """Record pre-extracted token counts (dict form).

        Bridge for the SDK's ``GraphAugmentationConfig.on_llm_usage`` hook,
        which reports ``{model, prompt_tokens, completion_tokens,
        total_tokens}`` dicts for LLM calls made inside kumiho-memory
        (e.g. ``recall_reformulation``, ``implication_queries``).
        """
        counts = {
            "prompt_tokens": int(tokens.get("prompt_tokens") or 0),
            "completion_tokens": int(tokens.get("completion_tokens") or 0),
            "total_tokens": int(tokens.get("total_tokens") or 0),
        }
        with self._lock:
            if phase not in self._phases:
                self._phases[phase] = {
                    "prompt_tokens": 0, "completion_tokens": 0,
                    "total_tokens": 0, "calls": 0,
                }
            for key, value in counts.items():
                self._phases[phase][key] += value
            self._phases[phase]["calls"] += 1
        return counts

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

# Runtime (no gold-label) detector for date/duration questions. Used to decide
# whether to surface atomic facts in the recalled context: facts sharpen
# factual recall but their competing dates distract a temporal answer, so we
# withhold them ONLY when the question itself asks about time. This infers the
# query type from the question text — exactly what a real deployment can do —
# instead of reading the dataset's ground-truth category label.
_TEMPORAL_QUERY_RE = re.compile(
    r"\b(when|what year|which year|what date|what day|what month|what time|"
    r"how long|how many (?:years?|months?|weeks?|days?|hours?)|how old|"
    r"since when|for how long|how much time|at what point)\b",
    re.IGNORECASE,
)


def is_temporal_query(question: str) -> bool:
    """True if the question asks for a date/duration (query-text heuristic)."""
    return bool(_TEMPORAL_QUERY_RE.search(question or ""))


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
    '"correct" or "incorrect". Be generous: if the response conveys the same '
    "core fact as the ground truth, count it correct even if it is phrased "
    "differently, less complete, or formatted differently. Only mark it "
    "incorrect if it contradicts the ground truth or fails to convey the "
    "specific fact being asked for."
)

# Aligned with the field protocol used to score Mem0 (arXiv 2504.19413,
# Appendix A): the judge is given only the question, gold answer, and
# prediction, and is told to be generous — "as long as it touches on the
# same topic as the gold answer, it should be counted as CORRECT" — with
# format-insensitive matching for dates. We keep this lenient-on-phrasing /
# strict-on-facts framing: a contradicting or missing fact is still wrong.
_JUDGE_TEMPLATE = """Question: {question}
Ground truth answer: {answer}
Model's response: {prediction}

Does the model's response correctly answer the question? Be generous with
grading:
- As long as the response conveys the same core fact as the ground truth,
  count it CORRECT — even if the phrasing, formatting, or level of detail
  differs.
- For dates, treat different formats of the same date as CORRECT
  (e.g. "7 May 2023" == "2023-05-07" == "May 7th, 2023"). Also allow
  off-by-one for day/week/month counts.
- Only mark it INCORRECT if it contradicts the ground truth or omits the
  specific fact being asked for entirely.

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


# Pinned default for answer generation. A bare alias ("gpt-4o") is a moving
# target: on 2026-07-10 a frozen-context re-answer of the record run's own
# stored contexts scored 0.486 where the same contexts had scored 0.543 two
# days earlier — the provider changed the model behind the name, silently.
# Scores produced through an alias are irreproducible; pin, and log what the
# API actually served (see ``answer_model_registry``).
DEFAULT_ANSWER_MODEL = "gpt-4o-2024-08-06"

#: Aggregate of what the API ACTUALLY served this process:
#: {(resolved_model, system_fingerprint): call_count}. Harnesses dump this
#: into metrics.json so a silent model swap is visible in the artifact.
answer_model_registry: dict[tuple[str, str | None], int] = {}


def warn_if_alias(model: str, *, role: str = "answer") -> None:
    """Warn when *model* has no date suffix — an alias the provider can move."""
    import re

    if not re.search(r"\d{4}-\d{2}-\d{2}$", model):
        logger.warning(
            "%s model %r is a bare alias — the provider can silently repoint "
            "it (measured 2026-07-10: -0.056 F1 on frozen contexts). Pin a "
            "dated snapshot for reproducible scores.", role, model,
        )


@backoff.on_exception(backoff.expo, Exception, max_tries=3)
async def generate_answer(
    question: str,
    context: str,
    *,
    system_prompt: str = "",
    model: str = DEFAULT_ANSWER_MODEL,
    api_key: str | None = None,
    max_tokens: int = 256,
    user_instruction: str = "Answer concisely with exact information from the context.",
    temperature: float = 0.0,
    meta_out: dict | None = None,
) -> str:
    """Generate an answer to a question given retrieved context.

    When *meta_out* is given, the resolved model id (``resp.model``) and
    ``system_fingerprint`` of the successful call are written into it —
    callers persist these per row so a silent provider-side model change
    is detectable after the fact.

    ``user_instruction`` is the trailing instruction in the user turn. It
    defaults to context-grounded extraction (correct for single/multi-hop/
    temporal), but callers can relax it for categories like open-domain that
    legitimately require combining the context with world/commonsense knowledge.
    """
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    user_msg = f"Context:\n{context}\n\n{question}\n\n{user_instruction}"
    text = ""
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
        resolved = getattr(resp, "model", model) or model
        fingerprint = getattr(resp, "system_fingerprint", None)
        answer_model_registry[(resolved, fingerprint)] = (
            answer_model_registry.get((resolved, fingerprint), 0) + 1
        )
        if meta_out is not None:
            meta_out["answer_model"] = resolved
            meta_out["system_fingerprint"] = fingerprint
        text = (resp.choices[0].message.content or "").strip()
        if text:
            return text
        logger.warning("Empty response from %s (attempt %d/3), retrying", model, _answer_attempt + 1)
    return text  # return empty after 3 tries rather than loop forever


# ---------------------------------------------------------------------------
# Relation decomposition (opt-in ingestion stage)
# ---------------------------------------------------------------------------
#
# The product's consolidation summarizer schema deliberately OMITS relations
# (measured: relations in the summary schema regressed based-on base recall).
# In production, entity->entity relation edges are written by an in-loop agent
# that calls ``kumiho_memory.ontology.decompose_and_link_agent`` AFTER
# consolidation. This harness stage simulates that agent so a LoCoMo
# relation_traversal pair run (read flag OFF vs ON) actually has edges to
# traverse. One extra LLM call (the run's configured summarizer model) extracts
# a lean decomposition from the CONSOLIDATED SUMMARY — never the raw transcript.

_DECOMPOSE_SYSTEM = (
    "You extract a knowledge-graph decomposition from a consolidated memory "
    "summary. Return STRICT JSON only — no prose, no code fences."
)

_DECOMPOSE_TEMPLATE = """From the memory summary below, extract the salient entities, facts, and relations.

Rules:
- entities: the concrete people, places, organizations, products, or things the summary is about. Each: {{"name": str, "type": str, "aliases": [str]}}. Use a short lowercase type (e.g. "person", "place", "organization", "activity", "object"). aliases may be empty.
- facts: standalone factual statements the summary asserts. Each: {{"statement": str, "about": [str]}} where "about" lists entity names the fact concerns.
- relations: directed links BETWEEN two entities. Each: {{"subject": str, "predicate": str, "object": str}}. subject and object MUST be names from the entities list. predicate is a short verb phrase (e.g. "works at", "lives in", "owns", "married to", "part of").

Extract at most 10 of each. Only include a relation when both endpoints are entities you listed. If a category has nothing, use an empty list.

Summary:
{summary}

Return JSON exactly of the form:
{{"entities": [...], "facts": [...], "relations": [...]}}
"""

register_prompt_template("decompose_relations", _DECOMPOSE_TEMPLATE)


def _parse_decomposition(raw: str) -> dict[str, Any]:
    """Parse an LLM decomposition response into a validated dict.

    Tolerant by design: strips accidental markdown fences and returns ``{}``
    on ANY malformation (bad JSON, wrong shape, no entities) rather than
    raising, so a single bad response never fails the run. Only dict-shaped
    entries survive; a decomposition with no entities is dropped since
    relations have no anchors to link onto.
    """
    if not raw:
        return {}
    text = raw.strip()
    # Tolerate accidental markdown fences around the JSON body.
    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    entities = [e for e in (parsed.get("entities") or []) if isinstance(e, dict)]
    facts = [f for f in (parsed.get("facts") or []) if isinstance(f, dict)]
    relations = [r for r in (parsed.get("relations") or []) if isinstance(r, dict)]
    if not entities:
        return {}
    return {"entities": entities, "facts": facts, "relations": relations}


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
    recall_candidate_multiplier: float = 1.0  # SDK retrieve-wide-then-trim: over-fetch ceil(limit * multiplier) candidates, rerank, trim back to limit (1.0 = off). Uniform across all queries -- never keyed on benchmark categories.
    answer_only: bool = False  # Skip ingest; answer against an existing project's corpus. Enables same-corpus A/B across SDK builds (one ingest, N answer runs) -- removes consolidation nondeterminism from cross-build comparisons.
    recall_mode: str = "full"  # "full" = artifact content, "summarized" = title+summary only
    concurrency: int = 4
    entry_concurrency: int = 1  # How many entries to process in parallel (pipeline parallelism)
    graph_augmented: bool = True  # Graph-native: edge traversal + multi-query recall (Kumiho default)
    sibling_similarity_threshold: float = 0.30  # Min cosine similarity for siblings (0=off)
    sibling_top_k: int = 0  # Max siblings to keep after scoring (0=unlimited, use threshold only)
    context_top_k: int = 0  # Global cap on revisions in final context (0=unlimited)
    stack_revisions: bool = True  # True = stack similar sessions; False = one item per session
    two_pass_rerank: bool = False  # Re-rank siblings with focused embeddings (title+summary only)
    sibling_score_fields: list[str] | None = None  # Server-side focused scoring fields (e.g. ["title", "summary"])
    decompose_relations: bool = False  # Opt-in: after each consolidation, one extra LLM call (llm_model) extracts a lean decomposition from the summary and writes entity->entity relation edges via kumiho_memory.ontology.decompose_and_link_agent. OFF by default. Must be ON in BOTH arms of a relation_traversal pair run (writes are shared; only the read flag KUMIHO_MEMORY_RELATION_TRAVERSAL differs). Requires kumiho-memory>=0.18.0.


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
        self._embed_adapter: Any = None
        self._embed_adapter_failed = False

    async def initialise(self) -> None:
        """Lazily initialise the Kumiho client and memory manager."""
        if self._initialised:
            return

        # KUMIHO_LLM_MODEL drives consolidation, reformulation, AND the LLM
        # half of sibling ranking — recall quality itself. An unpinned alias
        # here is the same reproducibility hole as the answer model.
        env_model = os.environ.get("KUMIHO_LLM_MODEL", "")
        if env_model:
            warn_if_alias(env_model, role="KUMIHO_LLM_MODEL (consolidation/sibling-ranking)")

        import kumiho
        from kumiho_memory import (
            RedisMemoryBuffer,
            UniversalMemoryManager,
            MemorySummarizer,
            PIIRedactor,
        )
        from kumiho_memory.graph_augmentation import GraphAugmentationConfig

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

        # Sibling ranking: give the manager an embedding adapter so stacked-
        # revision ranking uses cosine similarity (discriminative for direct
        # factual recall) instead of the LLM-only reranker fallback, which
        # hard-filters to 1-3 picks and drops the fact-bearing revision on
        # direct single-hop questions. No key → None → SDK falls back as before.
        embedding_adapter = None
        _emb_key = self.config.openai_api_key or os.environ.get("OPENAI_API_KEY")
        if _emb_key:
            from kumiho_memory import OpenAICompatEmbeddingAdapter
            embedding_adapter = OpenAICompatEmbeddingAdapter.create(
                api_key=_emb_key,
                base_url=os.environ.get("OPENAI_EMBEDDINGS_BASE_URL"),
                model="text-embedding-3-small",
            )

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

        # Graph-augmented recall is SDK business logic (multi-query
        # reformulation via the summarizer's provider-agnostic adapter, edge
        # traversal, sibling-seeded seeding, semantic fallback).  The harness
        # only supplies configuration + token accounting.
        graph_cfg = GraphAugmentationConfig(
            sibling_seeded_traversal=True,  # seed traversal from scored revisions
            on_llm_usage=lambda phase, info: token_tracker.record_usage(phase, info),
        )

        self._manager = UniversalMemoryManager(
            project=self.config.project_name,
            consolidation_threshold=self.config.consolidation_threshold,
            redis_buffer=redis_buf,
            summarizer=summarizer,
            pii_redactor=pii_redactor,
            memory_store=_store,
            memory_retrieve=_retrieve,
            recall_mode=self.config.recall_mode,
            embedding_adapter=embedding_adapter,
            sibling_similarity_threshold=self.config.sibling_similarity_threshold,
            sibling_top_k=self.config.sibling_top_k,
            sibling_score_fields=self.config.sibling_score_fields,
            recall_candidate_multiplier=self.config.recall_candidate_multiplier,
            graph_augmentation=graph_cfg,
            # NOTE: embedding_adapter is deliberately NOT passed to the
            # manager — with sibling_similarity_threshold > 0 it would switch
            # sibling selection to pure-embedding mode, bypassing the LLM
            # sibling reranker (+ its deterministic fallback).  The two-pass
            # rerank gets its adapter directly (see rerank_memories below).
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

    async def recall_with_graph_augmentation(
        self,
        query: str,
        *,
        limit: int | None = None,
        space_paths: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Graph-augmented recall — delegates to kumiho-memory.

        The full cognitive pipeline lives in the SDK (multi-query
        reformulation via the manager's provider-agnostic LLM adapter,
        parallel recall with kref-dedup merge, edge traversal seeded from
        top-scored sibling revisions, multi-hop semantic fallback, evidence
        weighting, the post-recall rerank stack, and sibling enrichment).
        The harness holds no retrieval logic of its own — it supplies
        configuration and reads results.  LLM token usage inside the SDK is
        reported back through ``GraphAugmentationConfig.on_llm_usage`` (see
        ``initialise()``).
        """
        await self.initialise()
        return await self._retry_network(
            self._manager.recall_memories,
            query,
            limit=limit or self.config.recall_limit,
            space_paths=space_paths,
            graph_augmented=True,
        )

    # ------------------------------------------------------------------
    # Post-consolidation edge discovery (Option 3: LLM-driven linking)
    # ------------------------------------------------------------------

    async def discover_and_link_edges(
        self,
        revision_kref: str,
        summary: str,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Post-consolidation edge discovery — delegates to kumiho-memory.

        Implication-query generation, candidate search + thresholding, and
        edge creation (with space auto-derivation from the revision kref) are
        SDK logic: ``UniversalMemoryManager.discover_edges_post_consolidation``
        → ``GraphAugmentedRecall.discover_edges``.
        """
        await self.initialise()
        return await self._manager.discover_edges_post_consolidation(
            revision_kref, summary, **kwargs,
        )

    # ------------------------------------------------------------------
    # Opt-in relation decomposition (simulates the in-loop decompose agent)
    # ------------------------------------------------------------------

    async def decompose_and_link_relations(
        self,
        revision_kref: str,
        summary: str,
    ) -> dict[str, Any]:
        """Extract a lean decomposition from *summary* and materialize
        entity->entity relation edges via the SDK's agent-driven writer.

        The production consolidation summary omits relations by design; in a
        live loop an agent calls ``decompose_and_link_agent`` after
        consolidation to write the relation edges. This mirrors that so a
        ``relation_traversal`` pair run has edges to traverse. One LLM call
        using the run's configured summarizer model (``config.llm_model``),
        the same structured-JSON convention as the harness's other extraction
        calls, over the CONSOLIDATED SUMMARY (not the raw transcript).

        Best-effort: any failure (LLM error, malformed JSON, SDK write) is
        logged and swallowed — a single conversation never fails the run.
        Token usage is recorded under the ``decompose_relations`` phase.
        Returns the SDK write stats (``{entities, facts, relations, edges}``)
        or ``{}``. Requires kumiho-memory>=0.18.0.
        """
        await self.initialise()
        if not revision_kref or not summary or not summary.strip():
            return {}

        from openai import AsyncOpenAI

        client = AsyncOpenAI(
            api_key=self.config.openai_api_key or os.environ.get("OPENAI_API_KEY"),
        )
        prompt = _DECOMPOSE_TEMPLATE.format(summary=summary[:6000])

        @backoff.on_exception(backoff.expo, Exception, max_tries=3)
        async def _call() -> str:
            resp = await client.chat.completions.create(
                model=self.config.llm_model,
                messages=[
                    {"role": "system", "content": _DECOMPOSE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=800,
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            token_tracker.record("decompose_relations", resp)
            return (resp.choices[0].message.content or "").strip()

        try:
            raw = await _call()
        except Exception as e:  # LLM call failed after retries — skip, keep going
            logger.warning(
                "Relation decomposition LLM call failed (%s): %s", revision_kref, e,
            )
            return {}

        decomposition = _parse_decomposition(raw)
        if not decomposition:
            logger.debug(
                "Relation decomposition produced nothing usable for %s", revision_kref,
            )
            return {}

        try:
            from kumiho_memory.ontology import decompose_and_link_agent

            stats = await decompose_and_link_agent(
                revision_kref, decomposition, project_name=self.config.project_name,
            )
            logger.debug("Relation decomposition linked %s: %s", revision_kref, stats)
            return stats or {}
        except Exception as e:  # SDK write failed — best-effort, keep the run going
            logger.warning(
                "decompose_and_link_agent failed (%s): %s", revision_kref, e,
            )
            return {}

    async def cleanup(self) -> None:
        """Close connections."""
        if self._manager:
            await self._manager.close()

    # -----------------------------------------------------------------
    # Two-pass re-ranking
    # -----------------------------------------------------------------

    def _get_embedding_adapter(self) -> Any:
        """Lazily build the embedding adapter for two-pass rerank.

        Returns ``None`` when no API key / openai package is available —
        ``two_pass_rerank`` no-ops safely on a ``None`` adapter.

        Deliberately NOT passed to ``UniversalMemoryManager``: with
        ``sibling_similarity_threshold > 0`` the manager would switch sibling
        selection to pure-embedding mode, bypassing the LLM sibling reranker
        and its deterministic fallback.
        """
        if self._embed_adapter is None and not self._embed_adapter_failed:
            from kumiho_memory import OpenAICompatEmbeddingAdapter

            try:
                self._embed_adapter = OpenAICompatEmbeddingAdapter.create(
                    api_key=self.config.openai_api_key
                    or os.environ.get("OPENAI_API_KEY"),
                )
            except Exception as e:
                logger.warning(
                    "Embedding adapter unavailable for two-pass rerank: %s", e,
                )
                self._embed_adapter_failed = True
        return self._embed_adapter

    def rerank_memories(
        self,
        memories: list[dict[str, Any]],
        query: str,
    ) -> list[dict[str, Any]]:
        """Two-pass focused rerank — delegates to ``kumiho_memory.two_pass_rerank``.

        Re-scores primaries AND siblings with focused title+summary
        embeddings on one cosine scale; safe no-op without an embedding
        adapter.
        """
        from kumiho_memory import two_pass_rerank

        return two_pass_rerank(query, memories, self._get_embedding_adapter())

    # -----------------------------------------------------------------
    # Context builder
    # -----------------------------------------------------------------

    def build_recalled_context(
        self,
        memories: list[dict[str, Any]],
        query: str = "",
        *,
        top_k: int | None = None,
    ) -> str:
        """Context assembly — delegates to ``kumiho_memory.compose_context``.

        Revision-centric assembly (siblings subsume the primary, global score
        ranking, top-k cap, full/summarized modes) is SDK logic.  *top_k*
        overrides ``config.context_top_k`` for this call.
        """
        from kumiho_memory import compose_context

        return compose_context(
            memories,
            query,
            mode=self.config.recall_mode,
            top_k=top_k if top_k is not None else self.config.context_top_k,
        )


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

    # SDK provenance: benchmarks measure kumiho-memory, so every manifest
    # must prove exactly which build ran (a stale site-packages install once
    # went unnoticed because nothing recorded this).
    try:
        import kumiho_memory

        sdk_provenance = {
            "kumiho_memory_version": getattr(kumiho_memory, "__version__", "?"),
            "kumiho_memory_path": str(getattr(kumiho_memory, "__file__", "?")),
        }
    except Exception:
        sdk_provenance = {
            "kumiho_memory_version": None, "kumiho_memory_path": None,
        }

    return {
        "harness_git_sha": _get_git_sha(repo_root),
        "dataset_shas": _get_submodule_shas(repo_root),
        **sdk_provenance,
        "benchmarks": benchmarks,
        "config": {
            "answer_model": config.answer_model,
            "judge_model": config.judge_model,
            "llm_model": config.llm_model,
            "llm_provider": config.llm_provider,
            "recall_limit": config.recall_limit,
            "recall_candidate_multiplier": config.recall_candidate_multiplier,
            "recall_mode": config.recall_mode,
            "graph_augmented": config.graph_augmented,
            "decompose_relations": config.decompose_relations,
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
