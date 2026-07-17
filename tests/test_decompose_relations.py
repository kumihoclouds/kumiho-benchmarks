"""Unit tests for the opt-in relation-decomposition ingestion stage.

Hermetic: no network, no paid LLM calls, no real kumiho-memory. The OpenAI
client and the ``kumiho_memory.ontology.decompose_and_link_agent`` seam are
faked (the same fake-adapter / fake-kumiho-seam approach relation_ab.py uses
against a live backend, inverted here for a pure unit test).

Runnable two ways:
    pytest tests/test_decompose_relations.py
    python  tests/test_decompose_relations.py

Async cases run via ``asyncio.run`` inside sync test functions, so no
pytest-asyncio plugin is required.
"""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path

# Make ``import kumiho_eval`` work regardless of the invocation cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kumiho_eval.common import (  # noqa: E402
    BenchmarkConfig,
    KumihoMemoryAdapter,
    _parse_decomposition,
    generate_run_manifest,
    token_tracker,
)

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeUsage:
    prompt_tokens = 120
    completion_tokens = 40
    total_tokens = 160


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage()
        self.model = "fake-summarizer"
        self.system_fingerprint = "fp_fake"


# What the fake LLM returns, and a log of the create() kwargs it was called with.
_FAKE_CONTENT = {"value": ""}
_CREATE_CALLS: list[dict] = []


class _FakeCompletions:
    async def create(self, **kwargs):
        _CREATE_CALLS.append(kwargs)
        return _FakeResponse(_FAKE_CONTENT["value"])


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeAsyncOpenAI:
    def __init__(self, **kwargs) -> None:
        self.chat = _FakeChat()


# Log of decompose_and_link_agent invocations.
_LINK_CALLS: list[dict] = []


async def _fake_decompose_and_link_agent(
    conversation_kref, decomposition, *, project_name, **_kw
):
    _LINK_CALLS.append(
        {
            "kref": conversation_kref,
            "decomposition": decomposition,
            "project_name": project_name,
        }
    )
    rels = decomposition.get("relations", [])
    return {
        "entities": len(decomposition.get("entities", [])),
        "facts": len(decomposition.get("facts", [])),
        "relations": len(rels),
        "edges": len(rels),
    }


class _Patched:
    """Context manager that installs the fake openai client + fake kumiho seam
    and restores the originals afterwards (restore, never pop — a bare pop
    would shadow a real kumiho_memory install for the rest of the process)."""

    def __enter__(self):
        import openai

        self._openai = openai
        self._orig_async_openai = getattr(openai, "AsyncOpenAI", None)
        openai.AsyncOpenAI = _FakeAsyncOpenAI

        self._had_ontology = "kumiho_memory.ontology" in sys.modules
        self._orig_ontology = sys.modules.get("kumiho_memory.ontology")
        fake = types.ModuleType("kumiho_memory.ontology")
        fake.decompose_and_link_agent = _fake_decompose_and_link_agent
        sys.modules["kumiho_memory.ontology"] = fake

        _CREATE_CALLS.clear()
        _LINK_CALLS.clear()
        return self

    def __exit__(self, *exc):
        self._openai.AsyncOpenAI = self._orig_async_openai
        if self._had_ontology:
            sys.modules["kumiho_memory.ontology"] = self._orig_ontology
        else:
            sys.modules.pop("kumiho_memory.ontology", None)
        return False


def _make_adapter() -> KumihoMemoryAdapter:
    adapter = KumihoMemoryAdapter(BenchmarkConfig(project_name="test-proj"))
    adapter._initialised = True  # skip real SDK/network initialise()
    return adapter


# ---------------------------------------------------------------------------
# 1. Stage is OFF by default
# ---------------------------------------------------------------------------


def test_decompose_relations_off_by_default():
    assert BenchmarkConfig().decompose_relations is False


def test_manifest_records_the_flag():
    manifest = generate_run_manifest(
        BenchmarkConfig(decompose_relations=True), ["locomo"]
    )
    assert manifest["config"]["decompose_relations"] is True
    off = generate_run_manifest(BenchmarkConfig(), ["locomo"])
    assert off["config"]["decompose_relations"] is False


# ---------------------------------------------------------------------------
# 2. Valid decomposition JSON -> decompose_and_link_agent call shape
# ---------------------------------------------------------------------------

_VALID_JSON = (
    '{"entities": [{"name": "Caroline", "type": "person", "aliases": []},'
    ' {"name": "Seattle", "type": "place", "aliases": []}],'
    ' "facts": [{"statement": "Caroline lives in Seattle", "about": ["Caroline"]}],'
    ' "relations": [{"subject": "Caroline", "predicate": "lives in",'
    ' "object": "Seattle"}]}'
)


def test_valid_decomposition_calls_link_with_right_shape():
    token_tracker.reset()
    adapter = _make_adapter()
    with _Patched():
        _FAKE_CONTENT["value"] = _VALID_JSON
        kref = "kref://test-proj/personal/rev.conversation?r=1"
        stats = asyncio.run(
            adapter.decompose_and_link_relations(kref, "Caroline lives in Seattle.")
        )

    # The SDK writer was invoked exactly once with the consolidation kref,
    # the run's project name, and a well-formed decomposition.
    assert len(_LINK_CALLS) == 1
    call = _LINK_CALLS[0]
    assert call["kref"] == kref
    assert call["project_name"] == "test-proj"
    decomp = call["decomposition"]
    assert [e["name"] for e in decomp["entities"]] == ["Caroline", "Seattle"]
    assert decomp["relations"][0] == {
        "subject": "Caroline", "predicate": "lives in", "object": "Seattle",
    }
    assert stats["relations"] == 1 and stats["edges"] == 1

    # Exactly one LLM call, made with the configured summarizer model, and its
    # cost was recorded under the decompose_relations phase.
    assert len(_CREATE_CALLS) == 1
    assert _CREATE_CALLS[0]["model"] == BenchmarkConfig().llm_model
    phases = token_tracker.summary()["by_phase"]
    assert phases["decompose_relations"]["calls"] == 1
    assert phases["decompose_relations"]["total_tokens"] == 160


# ---------------------------------------------------------------------------
# 3. Malformed LLM output is tolerated (skip, don't raise, don't write)
# ---------------------------------------------------------------------------


def test_malformed_output_is_skipped_not_raised():
    adapter = _make_adapter()
    with _Patched():
        _FAKE_CONTENT["value"] = "here is your answer: not json at all {{{"
        stats = asyncio.run(
            adapter.decompose_and_link_relations(
                "kref://test-proj/personal/rev.conversation?r=1", "some summary"
            )
        )
    # No raise; empty result; the SDK writer was never called.
    assert stats == {}
    assert _LINK_CALLS == []


def test_empty_summary_short_circuits():
    adapter = _make_adapter()
    with _Patched():
        _FAKE_CONTENT["value"] = _VALID_JSON
        stats = asyncio.run(
            adapter.decompose_and_link_relations("kref://x?r=1", "   ")
        )
    # Blank summary: no LLM call, no write.
    assert stats == {}
    assert _CREATE_CALLS == []
    assert _LINK_CALLS == []


# ---------------------------------------------------------------------------
# 4. _parse_decomposition tolerance unit-level
# ---------------------------------------------------------------------------


def test_parse_decomposition_variants():
    # Well-formed.
    d = _parse_decomposition(_VALID_JSON)
    assert len(d["entities"]) == 2 and len(d["relations"]) == 1
    # Markdown-fenced JSON is unwrapped.
    fenced = "```json\n" + _VALID_JSON + "\n```"
    assert _parse_decomposition(fenced)["entities"]
    # Garbage -> {}.
    assert _parse_decomposition("nope") == {}
    assert _parse_decomposition("") == {}
    # Valid JSON but wrong shape -> {}.
    assert _parse_decomposition("[1, 2, 3]") == {}
    # No entities -> dropped (relations would have no anchors).
    assert _parse_decomposition('{"entities": [], "relations": []}') == {}
    # Non-dict entries are filtered out.
    d2 = _parse_decomposition(
        '{"entities": [{"name": "A"}, "junk"], "facts": null, "relations": []}'
    )
    assert d2 == {"entities": [{"name": "A"}], "facts": [], "relations": []}


# ---------------------------------------------------------------------------
# Direct runner (no pytest required)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {t.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
