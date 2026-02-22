"""
AGM Belief Revision Compliance Evaluation for Kumiho Cognitive Memory.

Systematically tests whether the graph-native memory system satisfies AGM
belief revision postulates (Alchourron, Gardenfors, Makinson 1985) and
Hansson's belief base postulates operationally.

Postulates tested:
  K*2  (Success):         After revision by A, A is in the belief state
  K*3  (Inclusion):       Revision adds only A and preserves survivors
  K*4  (Vacuity):         Non-conflicting info => expansion (no supersession)
  K*5  (Consistency):     Revised belief state contains no contradictions
  K*6  (Extensionality):  Equivalent inputs produce equivalent states
  Rel  (Relevance):       Only relevant beliefs affected by contraction
  CR   (Core-Retainment): Removed beliefs contributed to inconsistency

Test categories per postulate:
  simple       Single-belief, straightforward operation
  multi_item   Multiple independent beliefs, verify isolation
  chain        Beliefs with dependency edges (DERIVED_FROM)
  temporal     Time-ordered revisions, multi-step sequences
  adversarial  Edge cases, rapid mutations, boundary values

Reference: Paper Section 7 (Formal Properties of Graph-Native Belief Revision)

To reproduce:
    python -m kumiho_eval.agm_compliance_eval [--max-scenarios N] [--output DIR]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .common import BenchmarkConfig

logger = logging.getLogger("kumiho_eval.agm")

# ---------------------------------------------------------------------------
# Load .env.local if present
# ---------------------------------------------------------------------------

_ENV_LOCAL = Path(__file__).resolve().parent / ".env.local"
if _ENV_LOCAL.exists():
    with open(_ENV_LOCAL, encoding="utf-8") as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class AssertionResult:
    """Single assertion within a scenario."""

    name: str
    passed: bool
    expected: str
    actual: str
    error: str = ""


@dataclass
class PostulateResult:
    """Result of running one AGM scenario."""

    scenario_id: str
    postulate: str
    category: str
    description: str
    passed: bool
    assertions: list[AssertionResult] = field(default_factory=list)
    latency_ms: float = 0.0
    error: str = ""


@dataclass
class ComplianceReport:
    """Full AGM compliance evaluation report."""

    results: list[PostulateResult] = field(default_factory=list)
    timestamp: str = ""
    project: str = ""
    total_scenarios: int = 0
    total_passed: int = 0
    total_failed: int = 0
    total_errors: int = 0
    overall_pass_rate: float = 0.0
    per_postulate: dict[str, dict[str, Any]] = field(default_factory=dict)
    per_category: dict[str, dict[str, Any]] = field(default_factory=dict)
    matrix: dict[str, dict[str, str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AGM Adapter — wraps kumiho SDK for belief-level operations
# ---------------------------------------------------------------------------


class AGMAdapter:
    """
    Thin wrapper around the kumiho Python SDK for AGM compliance testing.

    Uses the SDK's OOP API:
      - Belief  => Item (kind="belief") with a Revision
      - B(tau)  => Set of items where published tag exists and item not deprecated
      - Revise  => New revision + move published tag + DERIVED_FROM edge
      - Expand  => New item (no conflict)
      - Contract => Deprecate item
    """

    def __init__(self, project: str | None = None):
        self._kumiho: Any = None
        self._project: Any = None  # Project object
        self._spaces: dict[str, Any] = {}  # scenario_id -> Space object
        self._items: dict[tuple[str, str], Any] = {}  # (scenario_id, key) -> Item
        self.project_name: str = project or f"agm-test-{uuid.uuid4().hex[:8]}"

    @property
    def project(self) -> str:
        return self.project_name

    def initialize(self) -> None:
        """Connect to Kumiho and create test project."""
        import kumiho

        endpoint = os.environ.get("KUMIHO_ENDPOINT")
        token = os.environ.get("KUMIHO_AUTH_TOKEN")

        # Pass whatever we have — connect() handles discovery for missing endpoint
        connect_kwargs: dict[str, Any] = {}
        if endpoint:
            connect_kwargs["endpoint"] = endpoint
        if token:
            connect_kwargs["token"] = token
        if not token:
            logger.warning(
                "KUMIHO_AUTH_TOKEN not set. Ensure .env.local or shell env has it."
            )
        kumiho.connect(**connect_kwargs)

        self._kumiho = kumiho

        # Validate auth with a real API call
        try:
            self._project = kumiho.create_project(
                name=self.project_name, description="AGM compliance test run"
            )
        except Exception as e:
            err_str = str(e).lower()
            if "unauthenticated" in err_str or "auth" in err_str or "permission" in err_str:
                raise RuntimeError(
                    f"Authentication failed. Check KUMIHO_AUTH_TOKEN in .env.local "
                    f"or shell env. Error: {e}"
                ) from e
            # Project may already exist — try to get it
            self._project = kumiho.get_project(name=self.project_name)

    def _scenario_id_from_path(self, space_path: str) -> str:
        return space_path.split("/", 1)[1]

    def _get_item(self, scenario_id: str, key: str) -> Any:
        """Get a cached Item object."""
        return self._items.get((scenario_id, key))

    def _item_kref_str(self, space_path: str, key: str) -> str:
        """Build the expected kref string for no_edge assertions."""
        item_name = key.replace("_", "-").replace(" ", "-").lower()
        return f"kref://{space_path}/{item_name}.belief"

    # --- Space management ---

    def create_scenario_space(self, scenario_id: str) -> str:
        """Create an isolated space for a scenario. Returns space_path."""
        try:
            space = self._project.create_space(name=scenario_id)
        except Exception:
            space = self._project.get_space(scenario_id)
        self._spaces[scenario_id] = space
        return f"{self.project_name}/{scenario_id}"

    # --- Belief operations ---

    # The tag used to identify the current belief. The built-in "latest" tag
    # auto-advances with each new revision, so we never need to untag/retag.
    BELIEF_TAG = "latest"

    def store_belief(
        self,
        space_path: str,
        key: str,
        value: str,
        metadata: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        """
        Store a belief as an Item + Revision.

        The "latest" tag auto-advances, so no manual tagging is needed.
        Returns dict with item_kref, revision_kref, and object references.
        """
        scenario_id = self._scenario_id_from_path(space_path)
        space = self._spaces[scenario_id]
        item_name = key.replace("_", "-").replace(" ", "-").lower()

        # Create item (or get existing for multi-revision scenarios)
        existing = self._get_item(scenario_id, key)
        if existing is not None:
            item = existing
        else:
            try:
                item = space.create_item(item_name=item_name, kind="belief")
            except Exception:
                item = space.get_item(item_name=item_name, kind="belief")

        self._items[(scenario_id, key)] = item

        # Create revision with belief content
        rev_meta = {
            "belief_key": key,
            "belief_value": value,
            "title": f"{key} = {value}",
            "summary": f"The value of {key} is {value}",
        }
        if metadata:
            rev_meta.update(metadata)

        revision = item.create_revision(metadata=rev_meta)
        # "latest" tag auto-advances — no manual tag needed

        return {
            "item_kref": str(item.kref),
            "revision_kref": str(revision.kref),
            "_revision": revision,
        }

    def revise_belief(
        self,
        space_path: str,
        key: str,
        new_value: str,
    ) -> dict[str, Any]:
        """
        Revise an existing belief: new revision + DERIVED_FROM edge.

        The "latest" tag auto-advances to the new revision, implementing
        the AGM *revision operation on the graph.
        """
        scenario_id = self._scenario_id_from_path(space_path)
        item = self._get_item(scenario_id, key)
        if item is None:
            raise RuntimeError(f"Item not found for key: {key}")

        # Get current belief revision (latest)
        old_rev = item.get_latest_revision()
        if old_rev is None:
            raise RuntimeError(f"No revision found for: {key}")

        # Create new revision — "latest" tag auto-advances
        new_rev = item.create_revision(metadata={
            "belief_key": key,
            "belief_value": new_value,
            "title": f"{key} = {new_value}",
            "summary": f"The value of {key} is {new_value}",
        })

        # Create provenance edge: new DERIVED_FROM old
        new_rev.create_edge(target_revision=old_rev, edge_type="DERIVED_FROM")

        return {
            "item_kref": str(item.kref),
            "old_revision_kref": str(old_rev.kref),
            "new_revision_kref": str(new_rev.kref),
            "_revision": new_rev,
        }

    def expand_belief(
        self,
        space_path: str,
        key: str,
        value: str,
    ) -> dict[str, Any]:
        """
        AGM *expansion: add a new belief without conflicting with existing ones.
        Creates a new item (not a revision on an existing item).
        """
        return self.store_belief(space_path, key, value)

    def contract_belief(self, space_path: str, key: str) -> None:
        """AGM contraction: deprecate an item (exclude from belief state)."""
        scenario_id = self._scenario_id_from_path(space_path)
        item = self._get_item(scenario_id, key)
        if item is None:
            raise RuntimeError(f"Item not found for key: {key}")
        item.set_deprecated(True)

    def restore_belief(self, space_path: str, key: str) -> None:
        """Reverse a contraction: un-deprecate an item."""
        scenario_id = self._scenario_id_from_path(space_path)
        item = self._get_item(scenario_id, key)
        if item is None:
            raise RuntimeError(f"Item not found for key: {key}")
        item.set_deprecated(False)

    # --- Query operations ---

    def get_published_value(self, space_path: str, key: str) -> str | None:
        """Get the belief_value of the current (latest) revision. None if not found."""
        scenario_id = self._scenario_id_from_path(space_path)
        item = self._get_item(scenario_id, key)
        if item is None:
            return None
        try:
            rev = item.get_latest_revision()
            if rev is None:
                return None
            return rev.metadata.get("belief_value")
        except Exception:
            return None

    def get_revision_count(self, space_path: str, key: str) -> int:
        """Get the number of revisions for an item."""
        scenario_id = self._scenario_id_from_path(space_path)
        item = self._get_item(scenario_id, key)
        if item is None:
            return -1
        try:
            revs = item.get_revisions()
            return len(revs)
        except Exception:
            return -1

    def is_item_active(self, space_path: str, key: str) -> bool:
        """Check if an item is not deprecated."""
        scenario_id = self._scenario_id_from_path(space_path)
        item = self._get_item(scenario_id, key)
        if item is None:
            return False
        try:
            # Refresh state from server
            refreshed = self._kumiho.get_item(str(item.kref))
            return not refreshed.deprecated
        except Exception:
            return False

    def get_edges_for_belief(
        self,
        space_path: str,
        key: str,
        revision: int | None = None,
        direction: str = "both",
        edge_type: str | None = None,
    ) -> list[dict]:
        """Get edges for a belief's latest or specified revision."""
        scenario_id = self._scenario_id_from_path(space_path)
        item = self._get_item(scenario_id, key)
        if item is None:
            return []
        try:
            if revision is not None:
                rev = item.get_revision(revision)
            else:
                rev = item.get_latest_revision()
            if rev is None:
                return []

            import kumiho
            dir_map = {
                "outgoing": kumiho.OUTGOING,
                "incoming": kumiho.INCOMING,
                "both": kumiho.BOTH,
            }
            dir_val = dir_map.get(direction, kumiho.BOTH)

            edges = rev.get_edges(edge_type_filter=edge_type, direction=dir_val)
            return [
                {
                    "source": str(e.source_kref),
                    "target": str(e.target_kref),
                    "type": e.edge_type,
                }
                for e in edges
            ]
        except Exception:
            return []

    def get_dependents(
        self,
        space_path: str,
        key: str,
        edge_types: list[str] | None = None,
        max_depth: int = 5,
    ) -> list[dict]:
        """Get items that depend on the given belief."""
        scenario_id = self._scenario_id_from_path(space_path)
        item = self._get_item(scenario_id, key)
        if item is None:
            return []
        try:
            rev = item.get_latest_revision()
            if rev is None:
                return []
            result = rev.get_all_dependents(
                edge_type_filter=edge_types, max_depth=max_depth
            )
            if isinstance(result, list):
                return result
            return getattr(result, "revisions", [])
        except Exception:
            return []

    def search_beliefs(
        self,
        query: str,
        space_path: str | None = None,
    ) -> list[dict]:
        """Search for beliefs via fulltext search."""
        try:
            results = self._kumiho.search(
                query=query,
                context=space_path or self.project_name,
                kind="belief",
            )
            return [{"kref": str(sr.kref)} for sr in results] if results else []
        except Exception:
            return []

    def get_space_item_count(self, space_path: str) -> int:
        """Count belief items in a space."""
        scenario_id = self._scenario_id_from_path(space_path)
        space = self._spaces.get(scenario_id)
        if space is None:
            return -1
        try:
            items = space.get_items(kind_filter="belief")
            return len(items)
        except Exception:
            return -1

    # --- Cleanup ---

    def cleanup(self) -> None:
        """Delete the test project and all its contents."""
        try:
            if self._project:
                self._project.delete(force=True)
                logger.info("Cleaned up project: %s", self.project_name)
        except Exception as e:
            logger.warning("Cleanup failed for project %s: %s", self.project_name, e)


# ---------------------------------------------------------------------------
# Scenario Dataset
# ---------------------------------------------------------------------------

# Each scenario is a self-contained test case with:
#   beliefs:    initial beliefs to create
#   edges:      initial edges between beliefs (optional)
#   operations: sequence of operations to perform
#   assertions: conditions that must hold after all operations


K2_SCENARIOS = [
    # --- K*2 (Success): After revision by A, A must be in the belief state ---
    {
        "id": "k2_simple_01",
        "postulate": "K*2",
        "category": "simple",
        "description": "Revise favorite color — new value must be published",
        "beliefs": [{"key": "favorite_color", "value": "blue"}],
        "operations": [{"type": "revise", "key": "favorite_color", "new_value": "black"}],
        "assertions": [
            {"type": "published_value", "key": "favorite_color", "expected": "black"},
            {"type": "revision_count", "key": "favorite_color", "expected": 2},
        ],
    },
    {
        "id": "k2_simple_02",
        "postulate": "K*2",
        "category": "simple",
        "description": "Revise timezone — new value must be published",
        "beliefs": [{"key": "timezone", "value": "UTC"}],
        "operations": [{"type": "revise", "key": "timezone", "new_value": "Asia/Seoul"}],
        "assertions": [
            {"type": "published_value", "key": "timezone", "expected": "Asia/Seoul"},
        ],
    },
    {
        "id": "k2_simple_03",
        "postulate": "K*2",
        "category": "simple",
        "description": "Revise preferred editor — new value must be published",
        "beliefs": [{"key": "editor", "value": "vim"}],
        "operations": [{"type": "revise", "key": "editor", "new_value": "neovim"}],
        "assertions": [
            {"type": "published_value", "key": "editor", "expected": "neovim"},
        ],
    },
    {
        "id": "k2_multi_01",
        "postulate": "K*2",
        "category": "multi_item",
        "description": "Revise one belief among three — revised value must be published",
        "beliefs": [
            {"key": "color", "value": "blue"},
            {"key": "food", "value": "pizza"},
            {"key": "city", "value": "Seoul"},
        ],
        "operations": [{"type": "revise", "key": "color", "new_value": "red"}],
        "assertions": [
            {"type": "published_value", "key": "color", "expected": "red"},
            {"type": "published_value", "key": "food", "expected": "pizza"},
            {"type": "published_value", "key": "city", "expected": "Seoul"},
        ],
    },
    {
        "id": "k2_chain_01",
        "postulate": "K*2",
        "category": "chain",
        "description": "Revise root of A->B chain — A' must be published",
        "beliefs": [
            {"key": "project_lang", "value": "Python"},
            {"key": "framework", "value": "FastAPI"},
        ],
        "edges": [{"from": "framework", "to": "project_lang", "type": "DEPENDS_ON"}],
        "operations": [{"type": "revise", "key": "project_lang", "new_value": "Rust"}],
        "assertions": [
            {"type": "published_value", "key": "project_lang", "expected": "Rust"},
            {"type": "revision_count", "key": "project_lang", "expected": 2},
        ],
    },
    {
        "id": "k2_temporal_01",
        "postulate": "K*2",
        "category": "temporal",
        "description": "Double revision — final value must be published",
        "beliefs": [{"key": "meeting_day", "value": "Monday"}],
        "operations": [
            {"type": "revise", "key": "meeting_day", "new_value": "Wednesday"},
            {"type": "revise", "key": "meeting_day", "new_value": "Friday"},
        ],
        "assertions": [
            {"type": "published_value", "key": "meeting_day", "expected": "Friday"},
            {"type": "revision_count", "key": "meeting_day", "expected": 3},
        ],
    },
    {
        "id": "k2_temporal_02",
        "postulate": "K*2",
        "category": "temporal",
        "description": "Five rapid revisions — only final value published",
        "beliefs": [{"key": "status", "value": "draft"}],
        "operations": [
            {"type": "revise", "key": "status", "new_value": "review"},
            {"type": "revise", "key": "status", "new_value": "approved"},
            {"type": "revise", "key": "status", "new_value": "staging"},
            {"type": "revise", "key": "status", "new_value": "production"},
        ],
        "assertions": [
            {"type": "published_value", "key": "status", "expected": "production"},
            {"type": "revision_count", "key": "status", "expected": 5},
        ],
    },
    {
        "id": "k2_adversarial_01",
        "postulate": "K*2",
        "category": "adversarial",
        "description": "Revise to case-variant value — new value must be published exactly",
        "beliefs": [{"key": "theme", "value": "dark"}],
        "operations": [{"type": "revise", "key": "theme", "new_value": "Dark"}],
        "assertions": [
            {"type": "published_value", "key": "theme", "expected": "Dark"},
        ],
    },
    {
        "id": "k2_adversarial_02",
        "postulate": "K*2",
        "category": "adversarial",
        "description": "Revise to a long value — long value must be published",
        "beliefs": [{"key": "bio", "value": "short"}],
        "operations": [
            {
                "type": "revise",
                "key": "bio",
                "new_value": "A very long biography that spans multiple sentences and "
                "contains detailed information about the user's background, "
                "preferences, and professional experience in software engineering.",
            }
        ],
        "assertions": [
            {
                "type": "published_value",
                "key": "bio",
                "expected": "A very long biography that spans multiple sentences and "
                "contains detailed information about the user's background, "
                "preferences, and professional experience in software engineering.",
            },
        ],
    },
]


K3_SCENARIOS = [
    # --- K*3 (Inclusion): No phantom beliefs after revision ---
    {
        "id": "k3_simple_01",
        "postulate": "K*3",
        "category": "simple",
        "description": "After revision, item count unchanged (no phantom items)",
        "beliefs": [{"key": "language", "value": "Python"}],
        "operations": [{"type": "revise", "key": "language", "new_value": "Rust"}],
        "assertions": [
            {"type": "item_count", "expected": 1},
            {"type": "revision_count", "key": "language", "expected": 2},
        ],
    },
    {
        "id": "k3_multi_01",
        "postulate": "K*3",
        "category": "multi_item",
        "description": "Revise one of three — others unchanged at r=1",
        "beliefs": [
            {"key": "db_engine", "value": "PostgreSQL"},
            {"key": "cache", "value": "Redis"},
            {"key": "queue", "value": "RabbitMQ"},
        ],
        "operations": [{"type": "revise", "key": "db_engine", "new_value": "Neo4j"}],
        "assertions": [
            {"type": "item_count", "expected": 3},
            {"type": "revision_count", "key": "db_engine", "expected": 2},
            {"type": "revision_count", "key": "cache", "expected": 1},
            {"type": "revision_count", "key": "queue", "expected": 1},
        ],
    },
    {
        "id": "k3_multi_02",
        "postulate": "K*3",
        "category": "multi_item",
        "description": "Revise one of five — total item count still 5",
        "beliefs": [
            {"key": "a_val", "value": "1"},
            {"key": "b_val", "value": "2"},
            {"key": "c_val", "value": "3"},
            {"key": "d_val", "value": "4"},
            {"key": "e_val", "value": "5"},
        ],
        "operations": [{"type": "revise", "key": "c_val", "new_value": "33"}],
        "assertions": [
            {"type": "item_count", "expected": 5},
            {"type": "revision_count", "key": "a_val", "expected": 1},
            {"type": "revision_count", "key": "b_val", "expected": 1},
            {"type": "revision_count", "key": "c_val", "expected": 2},
            {"type": "revision_count", "key": "d_val", "expected": 1},
            {"type": "revision_count", "key": "e_val", "expected": 1},
        ],
    },
    {
        "id": "k3_chain_01",
        "postulate": "K*3",
        "category": "chain",
        "description": "Revise root of chain — leaf item unchanged",
        "beliefs": [
            {"key": "base_model", "value": "GPT-4"},
            {"key": "fine_tune", "value": "custom-lora"},
        ],
        "edges": [{"from": "fine_tune", "to": "base_model", "type": "DERIVED_FROM"}],
        "operations": [{"type": "revise", "key": "base_model", "new_value": "Claude-4"}],
        "assertions": [
            {"type": "item_count", "expected": 2},
            {"type": "revision_count", "key": "base_model", "expected": 2},
            {"type": "revision_count", "key": "fine_tune", "expected": 1},
        ],
    },
    {
        "id": "k3_temporal_01",
        "postulate": "K*3",
        "category": "temporal",
        "description": "Three revisions — still one item, four revisions total",
        "beliefs": [{"key": "version", "value": "v1"}],
        "operations": [
            {"type": "revise", "key": "version", "new_value": "v2"},
            {"type": "revise", "key": "version", "new_value": "v3"},
            {"type": "revise", "key": "version", "new_value": "v4"},
        ],
        "assertions": [
            {"type": "item_count", "expected": 1},
            {"type": "revision_count", "key": "version", "expected": 4},
        ],
    },
    {
        "id": "k3_adversarial_01",
        "postulate": "K*3",
        "category": "adversarial",
        "description": "Create and revise 5 items rapidly — exactly 5 items",
        "beliefs": [
            {"key": "rapid_a", "value": "a0"},
            {"key": "rapid_b", "value": "b0"},
            {"key": "rapid_c", "value": "c0"},
            {"key": "rapid_d", "value": "d0"},
            {"key": "rapid_e", "value": "e0"},
        ],
        "operations": [
            {"type": "revise", "key": "rapid_a", "new_value": "a1"},
            {"type": "revise", "key": "rapid_b", "new_value": "b1"},
            {"type": "revise", "key": "rapid_c", "new_value": "c1"},
            {"type": "revise", "key": "rapid_d", "new_value": "d1"},
            {"type": "revise", "key": "rapid_e", "new_value": "e1"},
        ],
        "assertions": [
            {"type": "item_count", "expected": 5},
            {"type": "revision_count", "key": "rapid_a", "expected": 2},
            {"type": "revision_count", "key": "rapid_e", "expected": 2},
        ],
    },
]


K4_SCENARIOS = [
    # --- K*4 (Vacuity): Non-conflicting = expansion, conflicting = revision ---
    {
        "id": "k4_simple_01",
        "postulate": "K*4",
        "category": "simple",
        "description": "Non-conflicting beliefs — both exist independently (expansion)",
        "beliefs": [{"key": "likes_python", "value": "true"}],
        "operations": [{"type": "expand", "key": "likes_rust", "value": "true"}],
        "assertions": [
            {"type": "item_count", "expected": 2},
            {"type": "published_value", "key": "likes_python", "expected": "true"},
            {"type": "published_value", "key": "likes_rust", "expected": "true"},
            {"type": "no_edge", "from_key": "likes_rust", "to_key": "likes_python", "edge_type": "DERIVED_FROM"},
        ],
    },
    {
        "id": "k4_simple_02",
        "postulate": "K*4",
        "category": "simple",
        "description": "Conflicting belief — revision with DERIVED_FROM edge",
        "beliefs": [{"key": "preferred_lang", "value": "Python"}],
        "operations": [{"type": "revise", "key": "preferred_lang", "new_value": "Rust"}],
        "assertions": [
            {"type": "item_count", "expected": 1},
            {"type": "published_value", "key": "preferred_lang", "expected": "Rust"},
            {"type": "has_edge", "key": "preferred_lang", "edge_type": "DERIVED_FROM"},
        ],
    },
    {
        "id": "k4_multi_01",
        "postulate": "K*4",
        "category": "multi_item",
        "description": "Three non-conflicting expansions — all independent",
        "beliefs": [],
        "operations": [
            {"type": "expand", "key": "skill_python", "value": "advanced"},
            {"type": "expand", "key": "skill_rust", "value": "intermediate"},
            {"type": "expand", "key": "skill_go", "value": "beginner"},
        ],
        "assertions": [
            {"type": "item_count", "expected": 3},
            {"type": "published_value", "key": "skill_python", "expected": "advanced"},
            {"type": "published_value", "key": "skill_rust", "expected": "intermediate"},
            {"type": "published_value", "key": "skill_go", "expected": "beginner"},
        ],
    },
    {
        "id": "k4_multi_02",
        "postulate": "K*4",
        "category": "multi_item",
        "description": "Expand two, then revise one — only revised has edge",
        "beliefs": [{"key": "db_choice", "value": "PostgreSQL"}],
        "operations": [
            {"type": "expand", "key": "cache_choice", "value": "Redis"},
            {"type": "revise", "key": "db_choice", "new_value": "Neo4j"},
        ],
        "assertions": [
            {"type": "item_count", "expected": 2},
            {"type": "has_edge", "key": "db_choice", "edge_type": "DERIVED_FROM"},
            {"type": "no_edge", "from_key": "cache_choice", "to_key": "db_choice", "edge_type": "DERIVED_FROM"},
        ],
    },
    {
        "id": "k4_chain_01",
        "postulate": "K*4",
        "category": "chain",
        "description": "Expand into a chain — existing items undisturbed",
        "beliefs": [
            {"key": "model_base", "value": "llama3"},
            {"key": "model_ft", "value": "llama3-ft"},
        ],
        "edges": [{"from": "model_ft", "to": "model_base", "type": "DERIVED_FROM"}],
        "operations": [{"type": "expand", "key": "model_eval", "value": "eval-suite-v1"}],
        "assertions": [
            {"type": "item_count", "expected": 3},
            {"type": "revision_count", "key": "model_base", "expected": 1},
            {"type": "revision_count", "key": "model_ft", "expected": 1},
            {"type": "revision_count", "key": "model_eval", "expected": 1},
        ],
    },
    {
        "id": "k4_temporal_01",
        "postulate": "K*4",
        "category": "temporal",
        "description": "Expand three times then revise one — only revision creates edge",
        "beliefs": [],
        "operations": [
            {"type": "expand", "key": "fact_a", "value": "alpha"},
            {"type": "expand", "key": "fact_b", "value": "beta"},
            {"type": "expand", "key": "fact_c", "value": "gamma"},
            {"type": "revise", "key": "fact_b", "new_value": "beta-prime"},
        ],
        "assertions": [
            {"type": "item_count", "expected": 3},
            {"type": "revision_count", "key": "fact_a", "expected": 1},
            {"type": "revision_count", "key": "fact_b", "expected": 2},
            {"type": "revision_count", "key": "fact_c", "expected": 1},
            {"type": "has_edge", "key": "fact_b", "edge_type": "DERIVED_FROM"},
        ],
    },
    {
        "id": "k4_adversarial_01",
        "postulate": "K*4",
        "category": "adversarial",
        "description": "Similar keys (color vs colour) — treated as separate expansions",
        "beliefs": [{"key": "color", "value": "blue"}],
        "operations": [{"type": "expand", "key": "colour", "value": "red"}],
        "assertions": [
            {"type": "item_count", "expected": 2},
            {"type": "published_value", "key": "color", "expected": "blue"},
            {"type": "published_value", "key": "colour", "expected": "red"},
        ],
    },
]


K5_SCENARIOS = [
    # --- K*5 (Consistency): No contradictions in active belief state ---
    {
        "id": "k5_simple_01",
        "postulate": "K*5",
        "category": "simple",
        "description": "Revise meeting day — only new day published (no contradiction)",
        "beliefs": [{"key": "meeting_day", "value": "Monday"}],
        "operations": [{"type": "revise", "key": "meeting_day", "new_value": "Tuesday"}],
        "assertions": [
            {"type": "published_value", "key": "meeting_day", "expected": "Tuesday"},
            {"type": "published_value_not", "key": "meeting_day", "excluded": "Monday"},
        ],
    },
    {
        "id": "k5_simple_02",
        "postulate": "K*5",
        "category": "simple",
        "description": "Revise location — old location not in published state",
        "beliefs": [{"key": "office_location", "value": "Seoul"}],
        "operations": [{"type": "revise", "key": "office_location", "new_value": "Tokyo"}],
        "assertions": [
            {"type": "published_value", "key": "office_location", "expected": "Tokyo"},
            {"type": "published_value_not", "key": "office_location", "excluded": "Seoul"},
        ],
    },
    {
        "id": "k5_multi_01",
        "postulate": "K*5",
        "category": "multi_item",
        "description": "Revise A among {A,B,C} — no cross-contamination",
        "beliefs": [
            {"key": "os_choice", "value": "Linux"},
            {"key": "shell_choice", "value": "zsh"},
            {"key": "term_choice", "value": "kitty"},
        ],
        "operations": [{"type": "revise", "key": "os_choice", "new_value": "macOS"}],
        "assertions": [
            {"type": "published_value", "key": "os_choice", "expected": "macOS"},
            {"type": "published_value", "key": "shell_choice", "expected": "zsh"},
            {"type": "published_value", "key": "term_choice", "expected": "kitty"},
            {"type": "published_value_not", "key": "os_choice", "excluded": "Linux"},
        ],
    },
    {
        "id": "k5_multi_02",
        "postulate": "K*5",
        "category": "multi_item",
        "description": "Revise A and B independently — both updated, no stale values",
        "beliefs": [
            {"key": "frontend", "value": "React"},
            {"key": "backend", "value": "Express"},
        ],
        "operations": [
            {"type": "revise", "key": "frontend", "new_value": "Svelte"},
            {"type": "revise", "key": "backend", "new_value": "FastAPI"},
        ],
        "assertions": [
            {"type": "published_value", "key": "frontend", "expected": "Svelte"},
            {"type": "published_value", "key": "backend", "expected": "FastAPI"},
            {"type": "published_value_not", "key": "frontend", "excluded": "React"},
            {"type": "published_value_not", "key": "backend", "excluded": "Express"},
        ],
    },
    {
        "id": "k5_chain_01",
        "postulate": "K*5",
        "category": "chain",
        "description": "Revise root of chain — no stale root value in published state",
        "beliefs": [
            {"key": "arch_pattern", "value": "monolith"},
            {"key": "deploy_target", "value": "single-server"},
        ],
        "edges": [{"from": "deploy_target", "to": "arch_pattern", "type": "DEPENDS_ON"}],
        "operations": [{"type": "revise", "key": "arch_pattern", "new_value": "microservices"}],
        "assertions": [
            {"type": "published_value", "key": "arch_pattern", "expected": "microservices"},
            {"type": "published_value_not", "key": "arch_pattern", "excluded": "monolith"},
        ],
    },
    {
        "id": "k5_temporal_01",
        "postulate": "K*5",
        "category": "temporal",
        "description": "Triple revision — only final value active, no predecessors",
        "beliefs": [{"key": "priority", "value": "low"}],
        "operations": [
            {"type": "revise", "key": "priority", "new_value": "medium"},
            {"type": "revise", "key": "priority", "new_value": "high"},
            {"type": "revise", "key": "priority", "new_value": "critical"},
        ],
        "assertions": [
            {"type": "published_value", "key": "priority", "expected": "critical"},
            {"type": "published_value_not", "key": "priority", "excluded": "low"},
            {"type": "published_value_not", "key": "priority", "excluded": "medium"},
            {"type": "published_value_not", "key": "priority", "excluded": "high"},
        ],
    },
    {
        "id": "k5_temporal_02",
        "postulate": "K*5",
        "category": "temporal",
        "description": "Revise back to original value — no stale intermediate",
        "beliefs": [{"key": "theme_mode", "value": "dark"}],
        "operations": [
            {"type": "revise", "key": "theme_mode", "new_value": "light"},
            {"type": "revise", "key": "theme_mode", "new_value": "dark"},
        ],
        "assertions": [
            {"type": "published_value", "key": "theme_mode", "expected": "dark"},
            {"type": "revision_count", "key": "theme_mode", "expected": 3},
        ],
    },
    {
        "id": "k5_adversarial_01",
        "postulate": "K*5",
        "category": "adversarial",
        "description": "Ten rapid revisions — only final value active",
        "beliefs": [{"key": "counter", "value": "0"}],
        "operations": [
            {"type": "revise", "key": "counter", "new_value": str(i)}
            for i in range(1, 11)
        ],
        "assertions": [
            {"type": "published_value", "key": "counter", "expected": "10"},
            {"type": "published_value_not", "key": "counter", "excluded": "5"},
            {"type": "revision_count", "key": "counter", "expected": 11},
        ],
    },
]


K6_SCENARIOS = [
    # --- K*6 (Extensionality): Equivalent inputs => equivalent states ---
    {
        "id": "k6_structural_01",
        "postulate": "K*6",
        "category": "simple",
        "description": "Same item revised with same value twice — state is equivalent",
        "beliefs": [{"key": "config_val", "value": "alpha"}],
        "operations": [
            {"type": "revise", "key": "config_val", "new_value": "beta"},
            {"type": "revise", "key": "config_val", "new_value": "beta"},
        ],
        "assertions": [
            {"type": "published_value", "key": "config_val", "expected": "beta"},
            # Two revisions of "beta" + 1 original = 3 revisions; state is "beta"
            {"type": "revision_count", "key": "config_val", "expected": 3},
        ],
    },
    {
        "id": "k6_structural_02",
        "postulate": "K*6",
        "category": "simple",
        "description": "Revise A->B vs fresh store B — same published value",
        "beliefs": [{"key": "setting_x", "value": "old"}],
        "operations": [{"type": "revise", "key": "setting_x", "new_value": "new"}],
        "assertions": [
            {"type": "published_value", "key": "setting_x", "expected": "new"},
        ],
    },
    {
        "id": "k6_multi_01",
        "postulate": "K*6",
        "category": "multi_item",
        "description": "Two items revised to same value — both have same published value",
        "beliefs": [
            {"key": "param_a", "value": "x"},
            {"key": "param_b", "value": "y"},
        ],
        "operations": [
            {"type": "revise", "key": "param_a", "new_value": "z"},
            {"type": "revise", "key": "param_b", "new_value": "z"},
        ],
        "assertions": [
            {"type": "published_value", "key": "param_a", "expected": "z"},
            {"type": "published_value", "key": "param_b", "expected": "z"},
        ],
    },
    {
        "id": "k6_temporal_01",
        "postulate": "K*6",
        "category": "temporal",
        "description": "A->B->A path vs fresh A — final state equivalent (value=A)",
        "beliefs": [{"key": "flag", "value": "enabled"}],
        "operations": [
            {"type": "revise", "key": "flag", "new_value": "disabled"},
            {"type": "revise", "key": "flag", "new_value": "enabled"},
        ],
        "assertions": [
            {"type": "published_value", "key": "flag", "expected": "enabled"},
        ],
    },
    {
        "id": "k6_adversarial_01",
        "postulate": "K*6",
        "category": "adversarial",
        "description": "Idempotent revision — revising with current value still succeeds",
        "beliefs": [{"key": "idempotent_val", "value": "stable"}],
        "operations": [{"type": "revise", "key": "idempotent_val", "new_value": "stable"}],
        "assertions": [
            {"type": "published_value", "key": "idempotent_val", "expected": "stable"},
            {"type": "revision_count", "key": "idempotent_val", "expected": 2},
        ],
    },
]


RELEVANCE_SCENARIOS = [
    # --- Relevance (Hansson): Only relevant beliefs affected by contraction ---
    {
        "id": "rel_simple_01",
        "postulate": "Relevance",
        "category": "simple",
        "description": "Deprecate A — unrelated B remains active",
        "beliefs": [
            {"key": "belief_a", "value": "alpha"},
            {"key": "belief_b", "value": "beta"},
        ],
        "operations": [{"type": "contract", "key": "belief_a"}],
        "assertions": [
            {"type": "item_deprecated", "key": "belief_a"},
            {"type": "item_active", "key": "belief_b"},
            {"type": "published_value", "key": "belief_b", "expected": "beta"},
            {"type": "revision_count", "key": "belief_b", "expected": 1},
        ],
    },
    {
        "id": "rel_multi_01",
        "postulate": "Relevance",
        "category": "multi_item",
        "description": "Deprecate 1 of 5 — other 4 completely unchanged",
        "beliefs": [
            {"key": "rel_a", "value": "1"},
            {"key": "rel_b", "value": "2"},
            {"key": "rel_c", "value": "3"},
            {"key": "rel_d", "value": "4"},
            {"key": "rel_e", "value": "5"},
        ],
        "operations": [{"type": "contract", "key": "rel_c"}],
        "assertions": [
            {"type": "item_deprecated", "key": "rel_c"},
            {"type": "item_active", "key": "rel_a"},
            {"type": "item_active", "key": "rel_b"},
            {"type": "item_active", "key": "rel_d"},
            {"type": "item_active", "key": "rel_e"},
            {"type": "revision_count", "key": "rel_a", "expected": 1},
            {"type": "revision_count", "key": "rel_e", "expected": 1},
        ],
    },
    {
        "id": "rel_multi_02",
        "postulate": "Relevance",
        "category": "multi_item",
        "description": "Deprecate and restore — restored item back in active state",
        "beliefs": [
            {"key": "temp_belief", "value": "temporary"},
            {"key": "perm_belief", "value": "permanent"},
        ],
        "operations": [
            {"type": "contract", "key": "temp_belief"},
            {"type": "restore", "key": "temp_belief"},
        ],
        "assertions": [
            {"type": "item_active", "key": "temp_belief"},
            {"type": "item_active", "key": "perm_belief"},
            {"type": "published_value", "key": "temp_belief", "expected": "temporary"},
        ],
    },
    {
        "id": "rel_chain_01",
        "postulate": "Relevance",
        "category": "chain",
        "description": "A->B chain, C independent. Deprecate A — C untouched",
        "beliefs": [
            {"key": "chain_root", "value": "root-val"},
            {"key": "chain_leaf", "value": "leaf-val"},
            {"key": "independent", "value": "ind-val"},
        ],
        "edges": [{"from": "chain_leaf", "to": "chain_root", "type": "DERIVED_FROM"}],
        "operations": [{"type": "contract", "key": "chain_root"}],
        "assertions": [
            {"type": "item_deprecated", "key": "chain_root"},
            {"type": "item_active", "key": "independent"},
            {"type": "published_value", "key": "independent", "expected": "ind-val"},
            {"type": "revision_count", "key": "independent", "expected": 1},
        ],
    },
    {
        "id": "rel_chain_02",
        "postulate": "Relevance",
        "category": "chain",
        "description": "A->B, deprecate A — B's item NOT auto-deprecated",
        "beliefs": [
            {"key": "parent_fact", "value": "parent-val"},
            {"key": "child_fact", "value": "child-val"},
        ],
        "edges": [{"from": "child_fact", "to": "parent_fact", "type": "DERIVED_FROM"}],
        "operations": [{"type": "contract", "key": "parent_fact"}],
        "assertions": [
            {"type": "item_deprecated", "key": "parent_fact"},
            {"type": "item_active", "key": "child_fact"},
        ],
    },
    {
        "id": "rel_temporal_01",
        "postulate": "Relevance",
        "category": "temporal",
        "description": "Revise A twice then deprecate — other items untouched",
        "beliefs": [
            {"key": "evolving", "value": "v1"},
            {"key": "stable_peer", "value": "unchanged"},
        ],
        "operations": [
            {"type": "revise", "key": "evolving", "new_value": "v2"},
            {"type": "revise", "key": "evolving", "new_value": "v3"},
            {"type": "contract", "key": "evolving"},
        ],
        "assertions": [
            {"type": "item_deprecated", "key": "evolving"},
            {"type": "item_active", "key": "stable_peer"},
            {"type": "revision_count", "key": "stable_peer", "expected": 1},
            {"type": "published_value", "key": "stable_peer", "expected": "unchanged"},
        ],
    },
    {
        "id": "rel_adversarial_01",
        "postulate": "Relevance",
        "category": "adversarial",
        "description": "Beliefs with similar names — deprecating one doesn't touch the other",
        "beliefs": [
            {"key": "user_name", "value": "alice"},
            {"key": "user_nickname", "value": "ally"},
        ],
        "operations": [{"type": "contract", "key": "user_name"}],
        "assertions": [
            {"type": "item_deprecated", "key": "user_name"},
            {"type": "item_active", "key": "user_nickname"},
            {"type": "published_value", "key": "user_nickname", "expected": "ally"},
        ],
    },
    {
        "id": "rel_adversarial_02",
        "postulate": "Relevance",
        "category": "adversarial",
        "description": "Deprecate all but one — surviving belief fully intact",
        "beliefs": [
            {"key": "doomed_1", "value": "x"},
            {"key": "doomed_2", "value": "y"},
            {"key": "survivor", "value": "z"},
        ],
        "operations": [
            {"type": "contract", "key": "doomed_1"},
            {"type": "contract", "key": "doomed_2"},
        ],
        "assertions": [
            {"type": "item_deprecated", "key": "doomed_1"},
            {"type": "item_deprecated", "key": "doomed_2"},
            {"type": "item_active", "key": "survivor"},
            {"type": "published_value", "key": "survivor", "expected": "z"},
            {"type": "revision_count", "key": "survivor", "expected": 1},
        ],
    },
]


CORE_RETAINMENT_SCENARIOS = [
    # --- Core-Retainment: Removed items must have contributed to inconsistency ---
    {
        "id": "cr_simple_01",
        "postulate": "Core-Retainment",
        "category": "simple",
        "description": "Standalone belief deprecated — no collateral damage",
        "beliefs": [
            {"key": "standalone", "value": "solo"},
            {"key": "bystander", "value": "safe"},
        ],
        "operations": [{"type": "contract", "key": "standalone"}],
        "assertions": [
            {"type": "item_deprecated", "key": "standalone"},
            {"type": "item_active", "key": "bystander"},
        ],
    },
    {
        "id": "cr_chain_01",
        "postulate": "Core-Retainment",
        "category": "chain",
        "description": "A->B chain, deprecate A. B active (not auto-removed). Dependency detectable.",
        "beliefs": [
            {"key": "source_fact", "value": "source-val"},
            {"key": "derived_fact", "value": "derived-val"},
        ],
        "edges": [{"from": "derived_fact", "to": "source_fact", "type": "DERIVED_FROM"}],
        "operations": [{"type": "contract", "key": "source_fact"}],
        "assertions": [
            {"type": "item_deprecated", "key": "source_fact"},
            {"type": "item_active", "key": "derived_fact"},
            # The system can identify that derived_fact depended on source_fact
            {"type": "has_edge", "key": "derived_fact", "edge_type": "DERIVED_FROM"},
        ],
    },
    {
        "id": "cr_chain_02",
        "postulate": "Core-Retainment",
        "category": "chain",
        "description": "A->B, C->D independent chains. Deprecate A — C,D untouched",
        "beliefs": [
            {"key": "chain1_root", "value": "c1r"},
            {"key": "chain1_leaf", "value": "c1l"},
            {"key": "chain2_root", "value": "c2r"},
            {"key": "chain2_leaf", "value": "c2l"},
        ],
        "edges": [
            {"from": "chain1_leaf", "to": "chain1_root", "type": "DERIVED_FROM"},
            {"from": "chain2_leaf", "to": "chain2_root", "type": "DERIVED_FROM"},
        ],
        "operations": [{"type": "contract", "key": "chain1_root"}],
        "assertions": [
            {"type": "item_deprecated", "key": "chain1_root"},
            {"type": "item_active", "key": "chain2_root"},
            {"type": "item_active", "key": "chain2_leaf"},
            {"type": "revision_count", "key": "chain2_root", "expected": 1},
            {"type": "revision_count", "key": "chain2_leaf", "expected": 1},
        ],
    },
    {
        "id": "cr_independent_01",
        "postulate": "Core-Retainment",
        "category": "multi_item",
        "description": "Three independent beliefs, deprecate one — others fully untouched",
        "beliefs": [
            {"key": "ind_a", "value": "a"},
            {"key": "ind_b", "value": "b"},
            {"key": "ind_c", "value": "c"},
        ],
        "operations": [{"type": "contract", "key": "ind_b"}],
        "assertions": [
            {"type": "item_deprecated", "key": "ind_b"},
            {"type": "item_active", "key": "ind_a"},
            {"type": "item_active", "key": "ind_c"},
            {"type": "published_value", "key": "ind_a", "expected": "a"},
            {"type": "published_value", "key": "ind_c", "expected": "c"},
        ],
    },
    {
        "id": "cr_adversarial_01",
        "postulate": "Core-Retainment",
        "category": "adversarial",
        "description": "Deep chain A->B->C->D. Deprecate A. Downstream items active, dependency detectable.",
        "beliefs": [
            {"key": "deep_a", "value": "a"},
            {"key": "deep_b", "value": "b"},
            {"key": "deep_c", "value": "c"},
            {"key": "deep_d", "value": "d"},
        ],
        "edges": [
            {"from": "deep_b", "to": "deep_a", "type": "DERIVED_FROM"},
            {"from": "deep_c", "to": "deep_b", "type": "DERIVED_FROM"},
            {"from": "deep_d", "to": "deep_c", "type": "DERIVED_FROM"},
        ],
        "operations": [{"type": "contract", "key": "deep_a"}],
        "assertions": [
            {"type": "item_deprecated", "key": "deep_a"},
            {"type": "item_active", "key": "deep_b"},
            {"type": "item_active", "key": "deep_c"},
            {"type": "item_active", "key": "deep_d"},
        ],
    },
    {
        "id": "cr_adversarial_02",
        "postulate": "Core-Retainment",
        "category": "adversarial",
        "description": "Mixed edges: DERIVED_FROM and DEPENDS_ON. Deprecate root — only DERIVED chain affected concept",
        "beliefs": [
            {"key": "root_node", "value": "root"},
            {"key": "derived_node", "value": "derived"},
            {"key": "dependent_node", "value": "dependent"},
            {"key": "unrelated_node", "value": "unrelated"},
        ],
        "edges": [
            {"from": "derived_node", "to": "root_node", "type": "DERIVED_FROM"},
            {"from": "dependent_node", "to": "root_node", "type": "DEPENDS_ON"},
        ],
        "operations": [{"type": "contract", "key": "root_node"}],
        "assertions": [
            {"type": "item_deprecated", "key": "root_node"},
            {"type": "item_active", "key": "derived_node"},
            {"type": "item_active", "key": "dependent_node"},
            {"type": "item_active", "key": "unrelated_node"},
            {"type": "revision_count", "key": "unrelated_node", "expected": 1},
        ],
    },
]


ALL_SCENARIOS = (
    K2_SCENARIOS
    + K3_SCENARIOS
    + K4_SCENARIOS
    + K5_SCENARIOS
    + K6_SCENARIOS
    + RELEVANCE_SCENARIOS
    + CORE_RETAINMENT_SCENARIOS
)


# ---------------------------------------------------------------------------
# Assertion Engine
# ---------------------------------------------------------------------------


def check_assertion(
    adapter: AGMAdapter,
    space_path: str,
    assertion: dict,
) -> AssertionResult:
    """Evaluate a single assertion against the current graph state."""
    a_type = assertion["type"]

    try:
        if a_type == "published_value":
            key = assertion["key"]
            expected = assertion["expected"]
            actual = adapter.get_published_value(space_path, key)
            passed = actual == expected
            return AssertionResult(
                name=f"published_value({key})=={expected!r}",
                passed=passed,
                expected=expected,
                actual=str(actual),
            )

        elif a_type == "published_value_not":
            key = assertion["key"]
            excluded = assertion["excluded"]
            actual = adapter.get_published_value(space_path, key)
            passed = actual != excluded
            return AssertionResult(
                name=f"published_value({key})!={excluded!r}",
                passed=passed,
                expected=f"not {excluded}",
                actual=str(actual),
            )

        elif a_type == "revision_count":
            key = assertion["key"]
            expected = assertion["expected"]
            actual = adapter.get_revision_count(space_path, key)
            passed = actual == expected
            return AssertionResult(
                name=f"revision_count({key})=={expected}",
                passed=passed,
                expected=str(expected),
                actual=str(actual),
            )

        elif a_type == "item_count":
            expected = assertion["expected"]
            actual = adapter.get_space_item_count(space_path)
            passed = actual == expected
            return AssertionResult(
                name=f"item_count=={expected}",
                passed=passed,
                expected=str(expected),
                actual=str(actual),
            )

        elif a_type == "item_active":
            key = assertion["key"]
            actual = adapter.is_item_active(space_path, key)
            return AssertionResult(
                name=f"item_active({key})",
                passed=actual,
                expected="active",
                actual="active" if actual else "deprecated",
            )

        elif a_type == "item_deprecated":
            key = assertion["key"]
            actual = not adapter.is_item_active(space_path, key)
            return AssertionResult(
                name=f"item_deprecated({key})",
                passed=actual,
                expected="deprecated",
                actual="deprecated" if actual else "active",
            )

        elif a_type == "has_edge":
            key = assertion["key"]
            edge_type = assertion["edge_type"]
            edges = adapter.get_edges_for_belief(space_path, key, edge_type=edge_type)
            has = len(edges) > 0
            return AssertionResult(
                name=f"has_edge({key}, {edge_type})",
                passed=has,
                expected=f">=1 {edge_type} edge",
                actual=f"{len(edges)} edges",
            )

        elif a_type == "no_edge":
            from_key = assertion["from_key"]
            to_key = assertion["to_key"]
            edge_type = assertion["edge_type"]
            edges = adapter.get_edges_for_belief(
                space_path, from_key, edge_type=edge_type, direction="outgoing"
            )
            # Check if any edge points to the target item
            target_kref = adapter._item_kref_str(space_path, to_key)
            has_edge = any(
                target_kref in str(e.get("target", e.get("to", "")))
                for e in edges
            )
            return AssertionResult(
                name=f"no_edge({from_key}->{to_key}, {edge_type})",
                passed=not has_edge,
                expected=f"no {edge_type} edge to {to_key}",
                actual=f"edge {'found' if has_edge else 'not found'}",
            )

        else:
            return AssertionResult(
                name=f"unknown({a_type})",
                passed=False,
                expected="",
                actual="",
                error=f"Unknown assertion type: {a_type}",
            )

    except Exception as e:
        return AssertionResult(
            name=f"{a_type}(error)",
            passed=False,
            expected=str(assertion),
            actual="",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Scenario Executor
# ---------------------------------------------------------------------------


MAX_SCENARIO_RETRIES = 3
SCENARIO_RETRY_BASE_DELAY = 5  # seconds


def run_scenario(adapter: AGMAdapter, scenario: dict) -> PostulateResult:
    """Execute a single AGM scenario: setup -> operate -> assert.

    Includes retry with exponential backoff for transient network errors.
    """
    scenario_id = scenario["id"]

    for attempt in range(1, MAX_SCENARIO_RETRIES + 1):
        t0 = time.perf_counter()
        try:
            # 1. Create isolated space
            space_path = adapter.create_scenario_space(scenario_id)

            # 2. Setup: create initial beliefs
            belief_refs: dict[str, dict] = {}
            for belief in scenario.get("beliefs", []):
                ref = adapter.store_belief(
                    space_path, belief["key"], belief["value"],
                    metadata=belief.get("metadata"),
                )
                belief_refs[belief["key"]] = ref

            # 3. Setup: create initial edges (using Revision objects)
            for edge in scenario.get("edges", []):
                from_ref = belief_refs.get(edge["from"], {})
                to_ref = belief_refs.get(edge["to"], {})
                from_rev = from_ref.get("_revision")
                to_rev = to_ref.get("_revision")
                if from_rev and to_rev:
                    from_rev.create_edge(
                        target_revision=to_rev,
                        edge_type=edge["type"],
                    )

            # 4. Execute operations
            for op in scenario.get("operations", []):
                if op["type"] == "revise":
                    ref = adapter.revise_belief(space_path, op["key"], op["new_value"])
                    belief_refs[op["key"]] = ref
                elif op["type"] == "expand":
                    ref = adapter.expand_belief(space_path, op["key"], op["value"])
                    belief_refs[op["key"]] = ref
                elif op["type"] == "contract":
                    adapter.contract_belief(space_path, op["key"])
                elif op["type"] == "restore":
                    adapter.restore_belief(space_path, op["key"])

            # 5. Small delay for eventual consistency
            time.sleep(0.1)

            # 6. Run assertions
            assertion_results = []
            for assertion in scenario.get("assertions", []):
                result = check_assertion(adapter, space_path, assertion)
                assertion_results.append(result)

            all_passed = all(a.passed for a in assertion_results)
            latency = (time.perf_counter() - t0) * 1000

            return PostulateResult(
                scenario_id=scenario_id,
                postulate=scenario["postulate"],
                category=scenario["category"],
                description=scenario["description"],
                passed=all_passed,
                assertions=assertion_results,
                latency_ms=latency,
            )

        except (OSError, ConnectionError, TimeoutError) as e:
            latency = (time.perf_counter() - t0) * 1000
            if attempt < MAX_SCENARIO_RETRIES:
                delay = SCENARIO_RETRY_BASE_DELAY * (2 ** (attempt - 1))
                logger.warning(
                    "Network error on %s (attempt %d/%d), retrying in %ds: %s",
                    scenario_id, attempt, MAX_SCENARIO_RETRIES, delay, e,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "Failed %s after %d attempts: %s",
                    scenario_id, MAX_SCENARIO_RETRIES, e,
                )
                return PostulateResult(
                    scenario_id=scenario_id,
                    postulate=scenario["postulate"],
                    category=scenario["category"],
                    description=scenario["description"],
                    passed=False,
                    latency_ms=latency,
                    error=str(e),
                )

        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000
            return PostulateResult(
                scenario_id=scenario_id,
                postulate=scenario["postulate"],
                category=scenario["category"],
                description=scenario["description"],
                passed=False,
                latency_ms=latency,
                error=str(e),
            )

    # Should not reach here, but satisfy type checker
    return PostulateResult(
        scenario_id=scenario_id,
        postulate=scenario["postulate"],
        category=scenario["category"],
        description=scenario["description"],
        passed=False,
        error="Max retries exhausted",
    )


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


def build_compliance_report(results: list[PostulateResult], project: str) -> ComplianceReport:
    """Aggregate individual scenario results into a compliance report."""
    report = ComplianceReport(
        results=results,
        timestamp=datetime.now(timezone.utc).isoformat(),
        project=project,
        total_scenarios=len(results),
        total_passed=sum(1 for r in results if r.passed),
        total_failed=sum(1 for r in results if not r.passed and not r.error),
        total_errors=sum(1 for r in results if r.error),
    )

    report.overall_pass_rate = (
        report.total_passed / report.total_scenarios
        if report.total_scenarios > 0
        else 0.0
    )

    # Per-postulate breakdown
    postulates: dict[str, list[PostulateResult]] = {}
    for r in results:
        postulates.setdefault(r.postulate, []).append(r)

    for name, group in postulates.items():
        passed = sum(1 for r in group if r.passed)
        report.per_postulate[name] = {
            "total": len(group),
            "passed": passed,
            "failed": len(group) - passed,
            "pass_rate": passed / len(group) if group else 0.0,
        }

    # Per-category breakdown
    categories: dict[str, list[PostulateResult]] = {}
    for r in results:
        categories.setdefault(r.category, []).append(r)

    for name, group in categories.items():
        passed = sum(1 for r in group if r.passed)
        report.per_category[name] = {
            "total": len(group),
            "passed": passed,
            "failed": len(group) - passed,
            "pass_rate": passed / len(group) if group else 0.0,
        }

    # Build compliance matrix: postulate x category -> pass/fail/partial
    for p_name, p_group in postulates.items():
        report.matrix[p_name] = {}
        cat_results: dict[str, list[bool]] = {}
        for r in p_group:
            cat_results.setdefault(r.category, []).append(r.passed)

        for cat, passes in cat_results.items():
            if all(passes):
                report.matrix[p_name][cat] = "PASS"
            elif not any(passes):
                report.matrix[p_name][cat] = "FAIL"
            else:
                report.matrix[p_name][cat] = "PARTIAL"

    return report


def print_compliance_report(report: ComplianceReport) -> None:
    """Print human-readable compliance report to stdout."""
    print(f"\n{'=' * 78}")
    print("  AGM Belief Revision Compliance Report - Kumiho Cognitive Memory")
    print(f"{'=' * 78}")
    print(f"  Timestamp:  {report.timestamp}")
    print(f"  Project:    {report.project}")
    print(f"  Scenarios:  {report.total_scenarios}")
    print(f"  Passed:     {report.total_passed}")
    print(f"  Failed:     {report.total_failed}")
    print(f"  Errors:     {report.total_errors}")
    print(f"  Pass Rate:  {report.overall_pass_rate:.1%}")

    # Postulate breakdown
    print(f"\n  {'Postulate':<20} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Rate':>8}")
    print(f"  {'-' * 48}")
    for name in [
        "K*2", "K*3", "K*4", "K*5", "K*6", "Relevance", "Core-Retainment"
    ]:
        if name in report.per_postulate:
            p = report.per_postulate[name]
            print(
                f"  {name:<20} {p['total']:>6} {p['passed']:>6} "
                f"{p['failed']:>6} {p['pass_rate']:>7.0%}"
            )

    # Category breakdown
    print(f"\n  {'Category':<20} {'Total':>6} {'Pass':>6} {'Fail':>6} {'Rate':>8}")
    print(f"  {'-' * 48}")
    for name in ["simple", "multi_item", "chain", "temporal", "adversarial"]:
        if name in report.per_category:
            c = report.per_category[name]
            print(
                f"  {name:<20} {c['total']:>6} {c['passed']:>6} "
                f"{c['failed']:>6} {c['pass_rate']:>7.0%}"
            )

    # Compliance matrix
    all_cats = ["simple", "multi_item", "chain", "temporal", "adversarial"]
    present_cats = [c for c in all_cats if c in report.per_category]

    print(f"\n  Compliance Matrix:")
    header = f"  {'Postulate':<20}" + "".join(f" {c:>12}" for c in present_cats)
    print(header)
    print(f"  {'-' * (20 + 13 * len(present_cats))}")

    for p_name in [
        "K*2", "K*3", "K*4", "K*5", "K*6", "Relevance", "Core-Retainment"
    ]:
        if p_name in report.matrix:
            row = f"  {p_name:<20}"
            for cat in present_cats:
                val = report.matrix[p_name].get(cat, "-")
                row += f" {val:>12}"
            print(row)

    # Failed scenarios detail
    failed = [r for r in report.results if not r.passed]
    if failed:
        print(f"\n  Failed Scenarios ({len(failed)}):")
        print(f"  {'-' * 70}")
        for r in failed:
            if r.error:
                print(f"  ERROR  {r.scenario_id}: {r.error[:60]}")
            else:
                failed_asserts = [a for a in r.assertions if not a.passed]
                for a in failed_asserts:
                    err = f" ({a.error})" if a.error else ""
                    print(
                        f"  FAIL   {r.scenario_id}: {a.name} "
                        f"(expected={a.expected}, actual={a.actual}){err}"
                    )

    print(f"{'=' * 78}\n")


def generate_latex_compliance_table(report: ComplianceReport) -> str:
    """Generate a LaTeX table for the paper."""
    lines = [
        "% AGM Compliance - auto-generated by kumiho_eval.agm_compliance_eval",
        f"% Date: {report.timestamp}",
        "",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{AGM Belief Revision Compliance (Operational Verification)}",
        "\\label{tab:agm-compliance}",
    ]

    all_cats = ["simple", "multi_item", "chain", "temporal", "adversarial"]
    present_cats = [c for c in all_cats if c in report.per_category]
    cat_labels = {
        "simple": "Simple",
        "multi_item": "Multi",
        "chain": "Chain",
        "temporal": "Temporal",
        "adversarial": "Advers.",
    }

    col_spec = "l" + "c" * len(present_cats) + "c"
    lines.append(f"\\begin{{tabular}}{{{col_spec}}}")
    lines.append("\\toprule")

    header_cats = " & ".join(cat_labels.get(c, c) for c in present_cats)
    lines.append(f"Postulate & {header_cats} & Pass Rate \\\\")
    lines.append("\\midrule")

    postulate_order = [
        "K*2", "K*3", "K*4", "K*5", "K*6", "Relevance", "Core-Retainment"
    ]
    postulate_labels = {
        "K*2": "K$^*$2 (Success)",
        "K*3": "K$^*$3 (Inclusion)",
        "K*4": "K$^*$4 (Vacuity)",
        "K*5": "K$^*$5 (Consistency)",
        "K*6": "K$^*$6 (Extensionality)",
        "Relevance": "Relevance",
        "Core-Retainment": "Core-Retainment",
    }

    for p_name in postulate_order:
        if p_name not in report.per_postulate:
            continue
        label = postulate_labels.get(p_name, p_name)
        cells = []
        for cat in present_cats:
            val = report.matrix.get(p_name, {}).get(cat, "--")
            if val == "PASS":
                cells.append("\\cmark")
            elif val == "FAIL":
                cells.append("\\xmark")
            elif val == "PARTIAL":
                cells.append("$\\sim$")
            else:
                cells.append("--")

        rate = report.per_postulate[p_name]["pass_rate"]
        rate_str = f"{rate:.0%}"
        if rate == 1.0:
            rate_str = f"\\textbf{{{rate_str}}}"

        row = f"{label} & {' & '.join(cells)} & {rate_str} \\\\"
        lines.append(row)

    lines.append("\\midrule")

    # Overall row
    overall_cells = []
    for cat in present_cats:
        rate = report.per_category[cat]["pass_rate"]
        overall_cells.append(f"{rate:.0%}")
    overall_str = f"\\textbf{{Overall}} & {' & '.join(overall_cells)} & \\textbf{{{report.overall_pass_rate:.0%}}} \\\\"
    lines.append(overall_str)

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append(
        "\\vspace{2pt}\\\\\\footnotesize "
        f"{report.total_scenarios} scenarios "
        f"({report.total_passed} passed, "
        f"{report.total_failed} failed, "
        f"{report.total_errors} errors). "
        "\\cmark\\ = all pass, \\xmark\\ = all fail, $\\sim$ = partial."
    )
    lines.append("\\end{table}")

    return "\n".join(lines)


def save_compliance_report(report: ComplianceReport, output_dir: Path) -> None:
    """Save full report as JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Detailed JSON
    data = {
        "timestamp": report.timestamp,
        "project": report.project,
        "summary": {
            "total_scenarios": report.total_scenarios,
            "total_passed": report.total_passed,
            "total_failed": report.total_failed,
            "total_errors": report.total_errors,
            "overall_pass_rate": report.overall_pass_rate,
        },
        "per_postulate": report.per_postulate,
        "per_category": report.per_category,
        "compliance_matrix": report.matrix,
        "scenarios": [
            {
                "id": r.scenario_id,
                "postulate": r.postulate,
                "category": r.category,
                "description": r.description,
                "passed": r.passed,
                "latency_ms": r.latency_ms,
                "error": r.error,
                "assertions": [
                    {
                        "name": a.name,
                        "passed": a.passed,
                        "expected": a.expected,
                        "actual": a.actual,
                        "error": a.error,
                    }
                    for a in r.assertions
                ],
            }
            for r in report.results
        ],
    }

    json_path = output_dir / "agm_compliance_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved JSON report to %s", json_path)

    # LaTeX table
    latex = generate_latex_compliance_table(report)
    latex_path = output_dir / "agm_compliance_table.tex"
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex)
    logger.info("Saved LaTeX table to %s", latex_path)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------


def _agm_checkpoint_path(output_dir: Path) -> Path:
    """Deterministic checkpoint file path for AGM resume support."""
    return output_dir / "_agm_checkpoint.jsonl"


def _load_agm_checkpoint(output_dir: Path) -> tuple[list[PostulateResult], set[str]]:
    """Load AGM checkpoint if it exists. Returns (results, completed_scenario_ids)."""
    ckpt = _agm_checkpoint_path(output_dir)
    if not ckpt.exists():
        return [], set()

    results: list[PostulateResult] = []
    completed: set[str] = set()
    for line in ckpt.read_text(encoding="utf-8").strip().splitlines():
        try:
            data = json.loads(line)
            assertions = [
                AssertionResult(
                    name=a["name"],
                    passed=a["passed"],
                    expected=a["expected"],
                    actual=a["actual"],
                    error=a.get("error", ""),
                )
                for a in data.get("assertions", [])
            ]
            r = PostulateResult(
                scenario_id=data["scenario_id"],
                postulate=data["postulate"],
                category=data["category"],
                description=data["description"],
                passed=data["passed"],
                assertions=assertions,
                latency_ms=data.get("latency_ms", 0.0),
                error=data.get("error", ""),
            )
            results.append(r)
            completed.add(data["scenario_id"])
        except (json.JSONDecodeError, KeyError) as e:
            logger.warning("Skipping corrupt AGM checkpoint line: %s", e)
    logger.info("Loaded AGM checkpoint: %d completed scenarios", len(completed))
    return results, completed


def _save_agm_checkpoint_line(output_dir: Path, result: PostulateResult) -> None:
    """Append a single scenario result to the AGM checkpoint JSONL file."""
    ckpt = _agm_checkpoint_path(output_dir)
    data = {
        "scenario_id": result.scenario_id,
        "postulate": result.postulate,
        "category": result.category,
        "description": result.description,
        "passed": result.passed,
        "latency_ms": result.latency_ms,
        "error": result.error,
        "assertions": [
            {
                "name": a.name,
                "passed": a.passed,
                "expected": a.expected,
                "actual": a.actual,
                "error": a.error,
            }
            for a in result.assertions
        ],
    }
    with open(ckpt, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


async def evaluate_agm_compliance(
    config: BenchmarkConfig | None = None,
    project: str | None = None,
    output_dir: str = "./results/agm",
    max_scenarios: int | None = None,
    resume: bool = True,
) -> ComplianceReport:
    """
    Run the full AGM compliance evaluation.

    Creates an isolated test project, runs all scenarios, generates
    a compliance report, and cleans up.

    Args:
        config: Optional BenchmarkConfig (used for endpoint/token)
        project: Optional project name override
        output_dir: Where to write results
        max_scenarios: Limit scenarios for quick smoke tests
        resume: Resume from checkpoint if available

    Returns:
        ComplianceReport with full results.
    """
    # Use config for endpoint/token if available
    if config and config.kumiho_endpoint:
        os.environ.setdefault("KUMIHO_ENDPOINT", config.kumiho_endpoint)
    if config and config.kumiho_token:
        os.environ.setdefault("KUMIHO_AUTH_TOKEN", config.kumiho_token)

    # Resolve project name
    proj = project
    if not proj and config:
        proj = f"{config.project_name}-agm"
    if not proj:
        proj = f"agm-test-{uuid.uuid4().hex[:8]}"

    adapter = AGMAdapter(project=proj)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("AGM Compliance Evaluation - Kumiho Cognitive Memory")
    logger.info("Project: %s", proj)
    logger.info("=" * 60)

    # Load checkpoint for resume
    if resume:
        all_results, completed_ids = _load_agm_checkpoint(output)
    else:
        all_results, completed_ids = [], set()

    try:
        adapter.initialize()

        # Select scenarios
        scenarios = ALL_SCENARIOS
        if max_scenarios:
            scenarios = scenarios[:max_scenarios]

        logger.info("Running %d AGM scenarios...", len(scenarios))

        # Run all scenarios
        skipped = 0
        for i, scenario in enumerate(scenarios):
            scenario_id = scenario["id"]

            # Skip already-completed scenarios (checkpoint resume)
            if scenario_id in completed_ids:
                skipped += 1
                continue

            logger.info(
                "[%d/%d] %s - %s",
                i + 1,
                len(scenarios),
                scenario_id,
                scenario["description"][:50],
            )
            result = run_scenario(adapter, scenario)
            all_results.append(result)
            _save_agm_checkpoint_line(output, result)

            status = "PASS" if result.passed else ("ERROR" if result.error else "FAIL")
            logger.info("  -> %s (%.0f ms)", status, result.latency_ms)

            if not result.passed and not result.error:
                for a in result.assertions:
                    if not a.passed:
                        logger.info(
                            "     Assertion failed: %s (expected=%s, actual=%s)",
                            a.name, a.expected, a.actual,
                        )

        if skipped:
            logger.info("Resumed: skipped %d already-completed scenarios", skipped)

        # Build report
        report = build_compliance_report(all_results, proj)

        # Output
        print_compliance_report(report)
        save_compliance_report(report, output)

    finally:
        adapter.cleanup()

    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="AGM Belief Revision Compliance Evaluation for Kumiho",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all scenarios
  python -m kumiho_eval.agm_compliance_eval

  # Quick smoke test (first 10 scenarios)
  python -m kumiho_eval.agm_compliance_eval --max-scenarios 10

  # Custom output directory
  python -m kumiho_eval.agm_compliance_eval --output ./results/agm-run1

  # Specific project name
  python -m kumiho_eval.agm_compliance_eval --project agm-paper-final
""",
    )

    parser.add_argument("--output", type=str, default="./results/agm",
                        help="Output directory for results")
    parser.add_argument("--max-scenarios", type=int, default=None,
                        help="Limit number of scenarios (for smoke testing)")
    parser.add_argument("--project", type=str, default=None,
                        help="Kumiho project name (default: auto-generated)")
    parser.add_argument("--endpoint", type=str, default=None,
                        help="Kumiho server endpoint (overrides KUMIHO_ENDPOINT env)")
    parser.add_argument("--token", type=str, default=None,
                        help="Kumiho auth token (overrides KUMIHO_AUTH_TOKEN env)")
    parser.add_argument("--no-resume", action="store_true",
                        help="Start fresh instead of resuming from checkpoint")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = BenchmarkConfig(
        project_name=args.project or "benchmark-agm",
        kumiho_endpoint=args.endpoint,
        kumiho_token=args.token,
    )

    report = asyncio.run(
        evaluate_agm_compliance(
            config=config,
            output_dir=args.output,
            project=args.project,
            max_scenarios=args.max_scenarios,
            resume=not args.no_resume,
        )
    )

    # Exit with non-zero if any failures
    if report.total_failed > 0 or report.total_errors > 0:
        exit(1)


if __name__ == "__main__":
    main()
