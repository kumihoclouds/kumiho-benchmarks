"""Unit tests for the held-out validation gate (issue #109).

Covers:
  - kumiho_eval.heldout: deterministic subset selection, manifest I/O,
    resolution (materialized / unmaterialized / drift / missing), filtering,
    apply_heldout, and the freeze/show CLI.
  - kumiho_eval.pair_gate: metric extraction, verdict logic, report shape,
    rendering, and the CLI runner exit codes.

Hermetic: no network, no paid LLM calls, no real dataset. The LongMemEval
loader is faked where the freeze CLI needs it. Runnable two ways:
    pytest tests/test_heldout_gate.py
    python  tests/test_heldout_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Make ``import kumiho_eval`` work regardless of the invocation cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from kumiho_eval import heldout, pair_gate  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _entries(n: int) -> list[dict]:
    return [{"question_id": f"lme_q{i:03d}", "answer": str(i)} for i in range(n)]


def _write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


# ===========================================================================
# heldout.select_heldout_ids
# ===========================================================================


def test_select_known_vector_locks_the_rule():
    """Regression lock: pins the sha256-lex-v1 output for a fixed input.

    If the algorithm ever silently changes (e.g. sha256 -> md5, or the
    slice/sort order), this hardcoded vector breaks.
    """
    got = heldout.select_heldout_ids(_entries(10), size=4)
    assert got == ["lme_q000", "lme_q001", "lme_q007", "lme_q009"]


def test_select_is_permutation_invariant():
    import random

    entries = _entries(30)
    baseline = heldout.select_heldout_ids(entries, size=7)
    shuffled = entries[:]
    random.Random(99).shuffle(shuffled)
    assert heldout.select_heldout_ids(shuffled, size=7) == baseline


def test_select_dedups_and_skips_invalid_ids():
    entries = _entries(10)
    base = heldout.select_heldout_ids(entries, size=4)
    noisy = (
        entries
        + entries  # duplicates
        + [{"answer": "no id"}, {"question_id": ""}, {"question_id": None}, {"question_id": 5}]
    )
    assert heldout.select_heldout_ids(noisy, size=4) == base


def test_select_size_larger_than_pool_returns_all_sorted():
    entries = _entries(5)
    got = heldout.select_heldout_ids(entries, size=999)
    assert got == sorted(e["question_id"] for e in entries)


def test_select_size_zero_returns_empty():
    assert heldout.select_heldout_ids(_entries(5), size=0) == []


def test_select_negative_size_raises():
    with pytest.raises(ValueError):
        heldout.select_heldout_ids(_entries(5), size=-1)


def test_select_output_is_sorted():
    got = heldout.select_heldout_ids(_entries(50), size=20)
    assert got == sorted(got)
    assert len(got) == 20


# ===========================================================================
# heldout manifest I/O
# ===========================================================================


def test_load_manifest_default_is_committed_and_wellformed():
    m = heldout.load_manifest()
    assert m["benchmark"] == "longmemeval"
    assert m["variant"] == "s"
    assert m["size"] == 100
    assert m["rule"] == "sha256-lex-v1"
    assert m["never_tune_on_this"] is True
    # committed manifest ships unmaterialized (CI has no vendored data)
    assert heldout.is_materialized(m) is False


def test_load_manifest_missing_raises(tmp_path):
    with pytest.raises(heldout.HeldoutError):
        heldout.load_manifest(tmp_path / "nope.json")


def test_load_manifest_bad_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(heldout.HeldoutError):
        heldout.load_manifest(p)


def test_load_manifest_non_object_raises(tmp_path):
    p = _write_json(tmp_path / "list.json", ["a", "b"])
    with pytest.raises(heldout.HeldoutError):
        heldout.load_manifest(p)


def test_freeze_manifest_shape():
    entries = _entries(200)
    m = heldout.freeze_manifest(entries, size=100, variant="s")
    assert m["materialized"] is True
    assert m["count"] == 100
    assert m["size"] == 100
    assert m["variant"] == "s"
    assert m["rule"] == "sha256-lex-v1"
    assert m["question_ids"] == sorted(m["question_ids"])
    assert m["question_ids"] == heldout.select_heldout_ids(entries, size=100)


def test_write_then_load_round_trip(tmp_path):
    entries = _entries(120)
    m = heldout.freeze_manifest(entries, size=30)
    out = heldout.write_manifest(m, tmp_path / "sub" / "hm.json")
    assert out.exists()
    reloaded = heldout.load_manifest(out)
    assert reloaded == m
    # resolving against the same dataset returns the frozen ids
    assert heldout.resolve_heldout_ids(entries, reloaded) == m["question_ids"]


# ===========================================================================
# heldout.is_materialized
# ===========================================================================


def test_is_materialized_variants():
    assert heldout.is_materialized({"materialized": True, "question_ids": ["a"]}) is True
    assert heldout.is_materialized({"materialized": False, "question_ids": ["a"]}) is False
    assert heldout.is_materialized({"materialized": True, "question_ids": []}) is False
    assert heldout.is_materialized({}) is False


# ===========================================================================
# heldout.resolve_heldout_ids
# ===========================================================================


def test_resolve_unmaterialized_computes_from_rule():
    entries = _entries(40)
    assert heldout.resolve_heldout_ids(entries, {}) == heldout.select_heldout_ids(entries)


def test_resolve_none_manifest_treated_as_unmaterialized():
    entries = _entries(40)
    assert heldout.resolve_heldout_ids(entries, None) == heldout.select_heldout_ids(entries)


def test_resolve_unmaterialized_empty_entries_raises():
    with pytest.raises(heldout.HeldoutError):
        heldout.resolve_heldout_ids([], {})


def test_resolve_unmaterialized_non_int_size_falls_back():
    entries = _entries(40)
    got = heldout.resolve_heldout_ids(entries, {"size": "oops"})
    assert got == heldout.select_heldout_ids(entries, size=heldout.HELDOUT_SIZE)


def test_resolve_materialized_returns_frozen_no_drift():
    entries = _entries(20)
    sel = heldout.select_heldout_ids(entries, size=5)
    manifest = {"materialized": True, "size": 5, "question_ids": sel}
    assert heldout.resolve_heldout_ids(entries, manifest) == sel


def test_resolve_materialized_skips_drift_check_when_size_mismatch():
    entries = _entries(20)
    sel = heldout.select_heldout_ids(entries, size=5)
    # size (100) != count (5) -> drift branch skipped, frozen returned as-is
    manifest = {"materialized": True, "size": 100, "question_ids": sel}
    assert heldout.resolve_heldout_ids(entries, manifest) == sel


def test_resolve_materialized_empty_entries_returns_frozen_unverified():
    sel = ["z_id_1", "z_id_2"]
    manifest = {"materialized": True, "size": 2, "question_ids": sel}
    assert heldout.resolve_heldout_ids([], manifest) == sorted(sel)


def test_resolve_materialized_missing_id_strict_raises():
    entries = _entries(20)
    sel = heldout.select_heldout_ids(entries, size=5)
    manifest = {"materialized": True, "size": 6, "question_ids": sel + ["ghost_id"]}
    with pytest.raises(heldout.HeldoutError):
        heldout.resolve_heldout_ids(entries, manifest, strict=True)


def test_resolve_materialized_missing_id_non_strict_drops():
    entries = _entries(20)
    sel = heldout.select_heldout_ids(entries, size=5)
    manifest = {"materialized": True, "size": 6, "question_ids": sel + ["ghost_id"]}
    got = heldout.resolve_heldout_ids(entries, manifest, strict=False)
    assert "ghost_id" not in got
    assert set(got) == set(sel)


def test_resolve_materialized_drift_strict_raises():
    entries = _entries(20)
    all_ids = sorted(e["question_id"] for e in entries)
    sel = heldout.select_heldout_ids(entries, size=5)
    # A present-but-wrong set of the same size -> drift.
    wrong = [q for q in all_ids if q not in sel][:5]
    assert wrong != sel and len(wrong) == 5
    manifest = {"materialized": True, "size": 5, "question_ids": wrong}
    with pytest.raises(heldout.HeldoutError):
        heldout.resolve_heldout_ids(entries, manifest, strict=True)


def test_resolve_materialized_drift_non_strict_returns_frozen():
    entries = _entries(20)
    all_ids = sorted(e["question_id"] for e in entries)
    sel = heldout.select_heldout_ids(entries, size=5)
    wrong = [q for q in all_ids if q not in sel][:5]
    manifest = {"materialized": True, "size": 5, "question_ids": wrong}
    got = heldout.resolve_heldout_ids(entries, manifest, strict=False)
    assert got == sorted(wrong)


def test_resolve_materialized_bad_ids_type_raises():
    manifest = {"materialized": True, "question_ids": [1, 2, 3]}
    with pytest.raises(heldout.HeldoutError):
        heldout.resolve_heldout_ids(_entries(10), manifest)


def test_resolve_materialized_non_list_ids_raises():
    # a truthy non-list question_ids trips is_materialized then fails validation
    manifest = {"materialized": True, "question_ids": "lme_q000"}
    with pytest.raises(heldout.HeldoutError):
        heldout.resolve_heldout_ids(_entries(10), manifest)


# ===========================================================================
# heldout.filter_entries
# ===========================================================================


def test_filter_entries_orders_by_ids():
    entries = _entries(10)
    ids = ["lme_q003", "lme_q000", "lme_q007"]
    got = heldout.filter_entries(entries, ids)
    assert [e["question_id"] for e in got] == ids


def test_filter_entries_missing_strict_raises():
    with pytest.raises(heldout.HeldoutError):
        heldout.filter_entries(_entries(5), ["lme_q000", "absent"], strict=True)


def test_filter_entries_missing_non_strict_drops():
    got = heldout.filter_entries(_entries(5), ["lme_q000", "absent"], strict=False)
    assert [e["question_id"] for e in got] == ["lme_q000"]


def test_filter_entries_dedups_first_wins_and_skips_bad():
    dup = _entries(3) + [{"question_id": "lme_q000", "answer": "SECOND"}, {"answer": "no id"}]
    got = heldout.filter_entries(dup, ["lme_q000"])
    assert len(got) == 1
    assert got[0]["answer"] == "0"  # first occurrence wins


# ===========================================================================
# heldout.apply_heldout
# ===========================================================================


def test_apply_heldout_default_manifest_computes(monkeypatch):
    # committed default manifest exists but is unmaterialized -> compute
    entries = _entries(40)
    filtered, info = heldout.apply_heldout(entries)
    assert info["materialized"] is False
    assert info["rule"] == "sha256-lex-v1"
    assert info["selected_count"] == len(filtered)
    assert [e["question_id"] for e in filtered] == info["question_ids"]


def test_apply_heldout_default_absent_falls_back_to_rule(monkeypatch, tmp_path):
    monkeypatch.setattr(heldout, "DEFAULT_MANIFEST_PATH", tmp_path / "gone.json")
    entries = _entries(40)
    filtered, info = heldout.apply_heldout(entries)
    assert info["materialized"] is False
    assert [e["question_id"] for e in filtered] == heldout.select_heldout_ids(entries)


def test_apply_heldout_explicit_missing_manifest_raises(tmp_path):
    with pytest.raises(heldout.HeldoutError):
        heldout.apply_heldout(_entries(10), manifest_path=tmp_path / "nope.json")


def test_apply_heldout_materialized_manifest(tmp_path):
    entries = _entries(60)
    manifest = heldout.freeze_manifest(entries, size=10)
    mpath = heldout.write_manifest(manifest, tmp_path / "hm.json")
    filtered, info = heldout.apply_heldout(entries, manifest_path=mpath)
    assert info["materialized"] is True
    assert info["selected_count"] == 10
    assert [e["question_id"] for e in filtered] == manifest["question_ids"]


# ===========================================================================
# heldout CLI (freeze / show)
# ===========================================================================


def test_cli_show_default(capsys):
    rc = heldout.main(["show"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sha256-lex-v1" in out
    assert "materialized=False" in out


def test_cli_show_explicit_manifest(tmp_path, capsys):
    entries = _entries(40)
    manifest = heldout.freeze_manifest(entries, size=5)
    mpath = heldout.write_manifest(manifest, tmp_path / "hm.json")
    rc = heldout.main(["show", "--manifest", str(mpath)])
    assert rc == 0
    assert "materialized=True" in capsys.readouterr().out


def test_cli_freeze_writes_manifest(tmp_path, monkeypatch, capsys):
    entries = _entries(150)

    from kumiho_eval import longmemeval_eval

    def _fake_loader(variant="s", data_dir=None):
        assert variant == "s"
        return entries

    monkeypatch.setattr(longmemeval_eval, "load_longmemeval", _fake_loader)

    out = tmp_path / "frozen.json"
    rc = heldout.main(["freeze", "--size", "20", "--out", str(out)])
    assert rc == 0
    assert "Froze 20 held-out ids" in capsys.readouterr().out

    written = heldout.load_manifest(out)
    assert heldout.is_materialized(written) is True
    assert written["count"] == 20
    assert written["question_ids"] == heldout.select_heldout_ids(entries, size=20)


# ===========================================================================
# pair_gate.load_metrics
# ===========================================================================


def test_pg_load_metrics_ok(tmp_path):
    p = _write_json(tmp_path / "m.json", {"overall_f1": 0.5})
    assert pair_gate.load_metrics(p) == {"overall_f1": 0.5}


def test_pg_load_metrics_missing_raises(tmp_path):
    with pytest.raises(pair_gate.GateInputError):
        pair_gate.load_metrics(tmp_path / "nope.json")


def test_pg_load_metrics_bad_json_raises(tmp_path):
    p = tmp_path / "bad.json"
    p.write_text("{oops", encoding="utf-8")
    with pytest.raises(pair_gate.GateInputError):
        pair_gate.load_metrics(p)


def test_pg_load_metrics_non_object_raises(tmp_path):
    p = _write_json(tmp_path / "arr.json", [1, 2, 3])
    with pytest.raises(pair_gate.GateInputError):
        pair_gate.load_metrics(p)


# ===========================================================================
# pair_gate metric extraction
# ===========================================================================


def test_extract_locomo_4cat_prefers_report():
    m = {"locomo_field_report": {"token_f1": {"overall_4cat": 0.408}}}
    assert pair_gate.extract_locomo_f1(m, "4cat") == pytest.approx(0.408)


def test_extract_locomo_4cat_means_categories_when_no_report():
    m = {
        "locomo_categories": {
            "single-hop": {"f1": 0.4},
            "multi-hop": {"f1": 0.3},
            "temporal": {"f1": 0.5},
            "open-domain": {"f1": 0.2},
            "adversarial": {"f1": 0.9},  # excluded
        }
    }
    assert pair_gate.extract_locomo_f1(m, "4cat") == pytest.approx((0.4 + 0.3 + 0.5 + 0.2) / 4)


def test_extract_locomo_4cat_falls_back_to_5cat():
    m = {"overall_f1": 0.53}  # no 4cat, no categories -> 5cat via overall_f1
    assert pair_gate.extract_locomo_f1(m, "4cat") == pytest.approx(0.53)


def test_extract_locomo_4cat_zero_is_not_treated_as_missing():
    # a legitimate 4cat F1 of 0.0 must be returned, not fall through to 5cat
    m = {"locomo_field_report": {"token_f1": {"overall_4cat": 0.0, "overall_5cat": 0.9}}}
    assert pair_gate.extract_locomo_f1(m, "4cat") == 0.0


def test_extract_locomo_5cat_and_overall():
    m = {"locomo_field_report": {"token_f1": {"overall_5cat": 0.565}}, "overall_f1": 0.5}
    assert pair_gate.extract_locomo_f1(m, "5cat") == pytest.approx(0.565)
    assert pair_gate.extract_locomo_f1(m, "overall") == pytest.approx(0.5)


def test_extract_locomo_missing_returns_none():
    assert pair_gate.extract_locomo_f1({}, "5cat") is None
    assert pair_gate.extract_locomo_f1({}, "overall") is None
    assert pair_gate.extract_locomo_f1({}, "4cat") is None


def test_extract_locomo_unknown_selector_raises():
    with pytest.raises(ValueError):
        pair_gate.extract_locomo_f1({}, "bogus")


def test_extract_heldout_f1_and_accuracy():
    assert pair_gate.extract_heldout_f1({"overall_f1": 0.42}, "f1") == pytest.approx(0.42)
    m = {"longmemeval": {"overall_accuracy": 0.71}}
    assert pair_gate.extract_heldout_f1(m, "accuracy") == pytest.approx(0.71)


def test_extract_heldout_accuracy_falls_back_to_judge():
    m = {"overall_judge_accuracy": 0.6}
    assert pair_gate.extract_heldout_f1(m, "accuracy") == pytest.approx(0.6)


def test_extract_heldout_missing_returns_none():
    assert pair_gate.extract_heldout_f1({}, "f1") is None
    assert pair_gate.extract_heldout_f1({}, "accuracy") is None


def test_extract_heldout_unknown_selector_raises():
    with pytest.raises(ValueError):
        pair_gate.extract_heldout_f1({}, "bogus")


# ===========================================================================
# pair_gate.evaluate_pair_gate
# ===========================================================================


def test_gate_pass_locomo_up_heldout_flat():
    v = pair_gate.evaluate_pair_gate(0.01, 0.0)
    assert v["passed"] is True
    assert v["reasons"] == []
    assert v["locomo_improved"] is True


def test_gate_pass_heldout_within_tolerance():
    v = pair_gate.evaluate_pair_gate(0.02, -0.009)
    assert v["passed"] is True


def test_gate_boundary_heldout_exactly_tolerance_is_ok():
    v = pair_gate.evaluate_pair_gate(0.0, -0.01)
    assert v["heldout_regressed"] is False
    assert v["passed"] is True


def test_gate_fail_heldout_regression():
    v = pair_gate.evaluate_pair_gate(0.05, -0.02)
    assert v["passed"] is False
    assert v["heldout_regressed"] is True
    assert any("held-out" in r for r in v["reasons"])


def test_gate_fail_locomo_regression():
    v = pair_gate.evaluate_pair_gate(-0.001, 0.0)
    assert v["passed"] is False
    assert v["locomo_meets_min"] is False
    assert any("LoCoMo" in r for r in v["reasons"])


def test_gate_positive_min_delta_requires_real_gain():
    # +0.003 gain does not clear a +0.005 required margin
    v = pair_gate.evaluate_pair_gate(0.003, 0.0, locomo_min_delta=0.005)
    assert v["passed"] is False
    assert v["locomo_improved"] is True  # improved, but not enough
    assert v["locomo_meets_min"] is False


def test_gate_pass_flat_locomo_notes_non_improvement():
    # min_delta 0.0 lets an exactly-flat LoCoMo pass, but a note is recorded
    v = pair_gate.evaluate_pair_gate(0.0, 0.0)
    assert v["passed"] is True
    assert v["locomo_improved"] is False
    assert any("did not strictly improve" in r for r in v["reasons"])


def test_gate_tolerance_is_magnitude_not_sign():
    # a positive tolerance arg and a negative one behave identically
    a = pair_gate.evaluate_pair_gate(0.01, -0.02, heldout_tolerance=0.01)
    b = pair_gate.evaluate_pair_gate(0.01, -0.02, heldout_tolerance=-0.01)
    assert a["heldout_regressed"] == b["heldout_regressed"] is True


# ===========================================================================
# pair_gate.build_report + render
# ===========================================================================


def _locomo(cand: float) -> dict:
    return {"locomo_field_report": {"token_f1": {"overall_4cat": cand}}}


def _heldout(cand: float) -> dict:
    return {"overall_f1": cand}


def test_build_report_shape_and_deltas():
    rep = pair_gate.build_report(_locomo(0.42), _heldout(0.41), _locomo(0.40), _heldout(0.415))
    assert rep["gate"] == "pair-gate/heldout"
    assert rep["locomo"]["delta"] == pytest.approx(0.02)
    assert rep["heldout"]["delta"] == pytest.approx(-0.005)
    assert rep["heldout"]["metric"] == "token_f1"
    assert rep["thresholds"]["heldout_tolerance"] == pytest.approx(0.01)
    assert rep["verdict"]["passed"] is True
    assert "generated_at" in rep


def test_build_report_accuracy_metric_labels():
    rep = pair_gate.build_report(
        _locomo(0.42),
        {"longmemeval": {"overall_accuracy": 0.70}},
        _locomo(0.41),
        {"longmemeval": {"overall_accuracy": 0.71}},
        heldout_metric="accuracy",
    )
    assert rep["heldout"]["metric"] == "judge_accuracy"
    assert rep["verdict"]["heldout_regressed"] is False  # -0.01 within tolerance


def test_build_report_missing_metric_raises():
    with pytest.raises(pair_gate.GateInputError):
        pair_gate.build_report(_locomo(0.42), {}, _locomo(0.40), _heldout(0.41))


def test_render_table_contains_verdict_and_numbers():
    rep = pair_gate.build_report(_locomo(0.42), _heldout(0.41), _locomo(0.40), _heldout(0.42))
    table = pair_gate.render_table(rep)
    assert "VERDICT: PASS" in table
    assert "held-out" in table
    assert "LoCoMo" in table


def test_render_table_fail_shows_reasons():
    rep = pair_gate.build_report(_locomo(0.42), _heldout(0.30), _locomo(0.40), _heldout(0.42))
    table = pair_gate.render_table(rep)
    assert "VERDICT: FAIL" in table
    assert "regressed" in table


def test_fmt_helpers_handle_none():
    assert pair_gate._fmt(None) == "n/a"
    assert pair_gate._fmt(0.5) == "0.5000"
    assert pair_gate._fmt_delta(None) == "n/a"
    assert pair_gate._fmt_delta(0.01) == "+0.0100"
    assert pair_gate._fmt_delta(-0.01) == "-0.0100"


# ===========================================================================
# pair_gate.run (CLI) + baseline resolution
# ===========================================================================


def _cli_files(tmp_path, lc, hc, lb, hb):
    return [
        "--locomo-candidate", str(_write_json(tmp_path / "lc.json", lc)),
        "--heldout-candidate", str(_write_json(tmp_path / "hc.json", hc)),
        "--locomo-baseline", str(_write_json(tmp_path / "lb.json", lb)),
        "--heldout-baseline", str(_write_json(tmp_path / "hb.json", hb)),
    ]


def test_run_pass_returns_zero_and_writes_report(tmp_path, capsys):
    argv = _cli_files(tmp_path, _locomo(0.42), _heldout(0.42), _locomo(0.40), _heldout(0.42))
    out = tmp_path / "report.json"
    argv += ["--out", str(out)]
    rc = pair_gate.run(argv)
    assert rc == 0
    assert "VERDICT: PASS" in capsys.readouterr().out
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["verdict"]["passed"] is True


def test_run_fail_returns_one(tmp_path):
    argv = _cli_files(tmp_path, _locomo(0.42), _heldout(0.30), _locomo(0.40), _heldout(0.42))
    assert pair_gate.run(argv) == 1


def test_run_bad_input_returns_two(tmp_path):
    argv = [
        "--locomo-candidate", str(tmp_path / "missing.json"),
        "--heldout-candidate", str(_write_json(tmp_path / "hc.json", _heldout(0.4))),
        "--heldout-baseline", str(_write_json(tmp_path / "hb.json", _heldout(0.4))),
    ]
    assert pair_gate.run(argv) == 2


def test_run_feb_baseline_default(tmp_path):
    # default --locomo-baseline is 'feb'; 4cat of feb ~ 0.410
    argv = [
        "--locomo-candidate", str(_write_json(tmp_path / "lc.json", _locomo(0.45))),
        "--heldout-candidate", str(_write_json(tmp_path / "hc.json", _heldout(0.42))),
        "--heldout-baseline", str(_write_json(tmp_path / "hb.json", _heldout(0.42))),
    ]
    assert pair_gate.run(argv) == 0


def test_resolve_locomo_baseline_feb():
    base = pair_gate._resolve_locomo_baseline("feb")
    assert base["label"] == "feb-2026-cosine"
    assert pair_gate.extract_locomo_f1(base, "4cat") == pytest.approx(0.41, abs=1e-3)


def test_run_missing_metric_returns_two(tmp_path):
    # heldout candidate has no readable metric -> GateInputError -> exit 2
    argv = _cli_files(tmp_path, _locomo(0.42), {}, _locomo(0.40), _heldout(0.42))
    assert pair_gate.run(argv) == 2


# ===========================================================================
# pair_gate cp949 console safety (adversarial review finding 1)
# ===========================================================================


def test_render_table_is_console_encodable_no_cp949_crash():
    """Regression lock for the cp949 UnicodeEncodeError.

    render_table's header previously used an em-dash (U+2014), which a Korean
    Windows console (cp949) cannot encode: `print(render_table(...))` aborted
    with UnicodeEncodeError before the `--out` report was written. The rendered
    table must be pure ASCII (a strict subset of cp949), so both encodings are
    safe. Pre-fix, the `.encode("ascii")` line raises.
    """
    for rep in (
        pair_gate.build_report(_locomo(0.42), _heldout(0.41), _locomo(0.40), _heldout(0.42)),
        pair_gate.build_report(_locomo(0.30), _heldout(0.20), _locomo(0.40), _heldout(0.42)),
    ):
        table = pair_gate.render_table(rep)
        table.encode("ascii")  # raises pre-fix (em-dash U+2014)
        table.encode("cp949")  # the exact codec from the traced crash


def test_run_writes_report_and_prints_without_encode_error(tmp_path, capsys):
    """The gate's stdout table must not carry a character that would abort the
    run on a cp949 console (which would also skip the --out JSON write)."""
    argv = _cli_files(tmp_path, _locomo(0.42), _heldout(0.42), _locomo(0.40), _heldout(0.42))
    out = tmp_path / "report.json"
    argv += ["--out", str(out)]
    rc = pair_gate.run(argv)
    assert rc == 0
    printed = capsys.readouterr().out
    printed.encode("cp949")  # would raise pre-fix
    assert out.exists()


# ===========================================================================
# pair_gate null-nested-metric robustness (adversarial review findings 6 & 8)
# ===========================================================================


def test_extract_locomo_null_nested_returns_none_not_attributeerror():
    # An explicit JSON null at a nested field must map to the clean "missing"
    # path (None), never raise AttributeError from a chained `.get`.
    assert pair_gate.extract_locomo_f1({"locomo_field_report": None}, "4cat") is None
    assert pair_gate.extract_locomo_f1({"locomo_field_report": None}, "5cat") is None
    assert (
        pair_gate.extract_locomo_f1({"locomo_field_report": {"token_f1": None}}, "5cat")
        is None
    )
    assert pair_gate.extract_locomo_f1({"locomo_categories": None}, "4cat") is None
    assert (
        pair_gate.extract_locomo_f1({"locomo_categories": {"multi-hop": None}}, "4cat")
        is None
    )


def test_extract_heldout_null_longmemeval_returns_none():
    assert pair_gate.extract_heldout_f1({"longmemeval": None}, "accuracy") is None


def test_run_null_nested_metric_returns_two_not_traceback(tmp_path):
    # End-to-end: a metrics.json with a null nested dict must exit 2 (documented
    # bad-input), not die with an uncaught AttributeError / default exit 1.
    argv = _cli_files(
        tmp_path,
        {"locomo_field_report": None},  # null nested -> no readable LoCoMo metric
        _heldout(0.42),
        _locomo(0.40),
        _heldout(0.42),
    )
    assert pair_gate.run(argv) == 2


# ===========================================================================
# longmemeval_eval.evaluate_longmemeval held-out wiring (finding 5 + 7/9)
#
# The heldout=True integration branch (apply_heldout call, longmemeval_heldout
# output subdir, metrics['heldout'] population) had zero coverage. These drive
# it fully offline: load_longmemeval, KumihoMemoryAdapter, and the per-question
# processor are faked, so no dataset, no SDK, and no LLM calls are involved.
# ===========================================================================


def _lme_entries(n: int) -> list[dict]:
    return [
        {
            "question_id": f"lme_q{i:03d}",
            "question": f"q{i}?",
            "answer": str(i),
            "question_type": "single-session-user",
        }
        for i in range(n)
    ]


def _run_offline_eval(tmp_path, monkeypatch, *, heldout: bool, max_samples=None, n: int = 12):
    """Run evaluate_longmemeval with every network/LLM/SDK seam faked out."""
    import asyncio
    import contextlib
    import io

    from kumiho_eval import common, longmemeval_eval

    entries = _lme_entries(n)

    monkeypatch.setattr(
        longmemeval_eval, "load_longmemeval", lambda variant="s", data_dir=None: entries
    )

    class _FakeAdapter:
        def __init__(self, config):
            self.config = config

        async def cleanup(self):
            return None

    monkeypatch.setattr(longmemeval_eval, "KumihoMemoryAdapter", _FakeAdapter)

    async def _fake_process(adapter, entry, qi, config):
        return common.EvalResult(
            question_id=entry["question_id"],
            question=entry["question"],
            question_type=entry.get("question_type", "unknown"),
            ground_truth=entry["answer"],
            prediction=entry["answer"],
            f1_score=0.5,
            judge_score=True,
            metadata={"is_abstention": False},
        )

    monkeypatch.setattr(longmemeval_eval, "_process_single_question", _fake_process)

    config = common.BenchmarkConfig(output_dir=str(tmp_path), max_samples=max_samples)

    # redirect_stdout guards against print_metrics_table's own em-dash aborting
    # the test on a cp949 console (that print lives in common.py, out of scope).
    with contextlib.redirect_stdout(io.StringIO()):
        asyncio.run(
            longmemeval_eval.evaluate_longmemeval(config, heldout=heldout, resume=False)
        )

    subdir = "longmemeval_heldout" if heldout else "longmemeval"
    out_dir = tmp_path / subdir
    metrics = json.loads((out_dir / "metrics.json").read_text(encoding="utf-8"))
    return out_dir, metrics


def test_evaluate_longmemeval_heldout_wiring(tmp_path, monkeypatch):
    out_dir, metrics = _run_offline_eval(tmp_path, monkeypatch, heldout=True, n=12)
    # writes under the held-out subdir, never clobbering a full-run subdir
    assert out_dir.name == "longmemeval_heldout"
    assert not (tmp_path / "longmemeval" / "metrics.json").exists()
    # self-describing held-out block is populated
    assert "heldout" in metrics
    hd = metrics["heldout"]
    assert hd["benchmark"] == "longmemeval"
    assert hd["rule"] == "sha256-lex-v1"
    assert hd["materialized"] is False
    # 12 entries < size 100 -> the whole set is the slice
    assert hd["selected_count"] == 12
    assert len(hd["question_ids"]) == 12
    assert "truncated_by_max_samples" not in hd
    # the pair gate reads overall_f1 from exactly this file
    assert metrics["overall_f1"] == pytest.approx(0.5)


def test_evaluate_longmemeval_full_run_has_no_heldout_block(tmp_path, monkeypatch):
    out_dir, metrics = _run_offline_eval(tmp_path, monkeypatch, heldout=False, n=6)
    assert out_dir.name == "longmemeval"
    assert "heldout" not in metrics


def test_evaluate_longmemeval_heldout_max_samples_records_truncation(tmp_path, monkeypatch):
    # findings 7/9: --max-samples truncates the slice after selection; the
    # recorded selected_count / question_ids must reflect what actually ran.
    _, metrics = _run_offline_eval(tmp_path, monkeypatch, heldout=True, max_samples=5, n=12)
    hd = metrics["heldout"]
    assert hd["selected_count"] == 5
    assert len(hd["question_ids"]) == 5
    assert hd["truncated_by_max_samples"] is True


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
