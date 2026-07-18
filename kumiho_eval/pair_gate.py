#!/usr/bin/env python3
"""pair_gate.py — Held-out validation gate for LoCoMo-tuned constants (#109).

Produces a SIDE-BY-SIDE report of LoCoMo metrics and held-out (LongMemEval)
metrics for a candidate run versus a baseline, and renders a PASS/FAIL verdict.

The gate this issue adds: a tuned-constant change (evidence delta, recency,
MMR lambda, half-life, ...) must improve LoCoMo *without regressing the
held-out slice beyond a small tolerance*. LoCoMo alone is not enough — the
constants are tuned on it, so LoCoMo going up proves nothing about
generalisation. The held-out slice (see ``kumiho_eval.heldout``) is never
tuned on, so a held-out regression means the LoCoMo gain was overfitting.

This gate is offline: it consumes already-produced ``metrics.json`` files
(exactly like ``scripts/release_gate.py``). Produce the held-out metrics with
``python -m kumiho_eval.longmemeval_eval --heldout`` and the LoCoMo metrics
with ``python -m kumiho_eval.locomo_eval``.

Verdict (defaults):
  - held-out F1 must not drop by more than ``--heldout-tolerance`` (0.01), AND
  - LoCoMo F1 must gain at least ``--locomo-min-delta`` (0.0 = must not regress;
    set a positive margin for a tuned-constant PR, e.g. 0.005).

Exit codes:
  0  gate passed
  1  gate failed (regression)
  2  bad input (missing/malformed JSON, unreadable metric)

Usage:
    python -m kumiho_eval.pair_gate \
        --locomo-candidate results/locomo/metrics.json \
        --heldout-candidate results/longmemeval_heldout/metrics.json \
        --locomo-baseline old/locomo/metrics.json \
        --heldout-baseline old/longmemeval_heldout/metrics.json \
        --out results/pair_gate.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Reuse the documented Feb 2026 cosine reference for LoCoMo baselines.
try:  # pragma: no cover - trivial import shim
    from scripts.release_gate import FEB_2026_BASELINE
except Exception:  # pragma: no cover
    FEB_2026_BASELINE = {
        "label": "feb-2026-cosine",
        "total_conversations": 10,
        "locomo_field_report": {"token_f1": {"overall_5cat": 0.565}},
        "locomo_categories": {
            "single-hop": {"f1": 0.462},
            "multi-hop": {"f1": 0.355},
            "temporal": {"f1": 0.533},
            "open-domain": {"f1": 0.290},
            "adversarial": {"f1": 0.975},
        },
    }

CATEGORY_NAMES = ["multi-hop", "temporal", "open-domain", "single-hop", "adversarial"]
ADVERSARIAL_CATEGORY = "adversarial"

DEFAULT_HELDOUT_TOLERANCE = 0.01
DEFAULT_LOCOMO_MIN_DELTA = 0.0

# Boundary slack for float subtraction. A delta of e.g. 0.42 - 0.41 lands at
# -0.010000000000000009, which would spuriously trip an exact -0.01 tolerance;
# this epsilon keeps the boundary inclusive against that rounding.
_FLOAT_EPS = 1e-9


class GateInputError(ValueError):
    """Raised for missing/malformed inputs (maps to exit code 2)."""


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def load_metrics(path: str | Path) -> dict[str, Any]:
    """Load a metrics.json, raising GateInputError on any problem."""
    p = Path(path)
    if not p.exists():
        raise GateInputError(f"metrics file not found: {path}")
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise GateInputError(f"{path} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise GateInputError(f"{path} must be a JSON object")
    return data


def _four_cat_f1(metrics: dict[str, Any]) -> float | None:
    report = metrics.get("locomo_field_report", {}).get("token_f1", {})
    if report.get("overall_4cat") is not None:
        return float(report["overall_4cat"])
    vals = [
        metrics.get("locomo_categories", {}).get(cat, {}).get("f1")
        for cat in CATEGORY_NAMES
        if cat != ADVERSARIAL_CATEGORY
    ]
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _five_cat_f1(metrics: dict[str, Any]) -> float | None:
    report = metrics.get("locomo_field_report", {}).get("token_f1", {})
    if report.get("overall_5cat") is not None:
        return float(report["overall_5cat"])
    v = metrics.get("overall_f1")
    return float(v) if v is not None else None


def extract_locomo_f1(metrics: dict[str, Any], which: str = "4cat") -> float | None:
    """LoCoMo F1 for the gate.

    ``which``:
      - ``4cat``: adversarial-stripped aggregate (field-comparable; default),
        falling back to 5cat then ``overall_f1``.
      - ``5cat``: full aggregate including adversarial.
      - ``overall``: the raw ``overall_f1`` field.
    """
    if which == "overall":
        v = metrics.get("overall_f1")
        return float(v) if v is not None else None
    if which == "5cat":
        return _five_cat_f1(metrics)
    if which == "4cat":
        # explicit None check: a legitimate 4cat F1 of 0.0 must not fall through
        v = _four_cat_f1(metrics)
        return v if v is not None else _five_cat_f1(metrics)
    raise ValueError(f"unknown locomo metric selector: {which!r}")


def extract_heldout_f1(metrics: dict[str, Any], which: str = "f1") -> float | None:
    """Held-out (LongMemEval) primary metric.

    ``which``:
      - ``f1``: token ``overall_f1`` (default; the tolerance is stated in F1).
      - ``accuracy``: LLM-judge ``longmemeval.overall_accuracy`` (falls back to
        top-level ``overall_judge_accuracy``).
    """
    if which == "f1":
        v = metrics.get("overall_f1")
        return float(v) if v is not None else None
    if which == "accuracy":
        lme = metrics.get("longmemeval", {})
        v = lme.get("overall_accuracy")
        if v is None:
            v = metrics.get("overall_judge_accuracy")
        return float(v) if v is not None else None
    raise ValueError(f"unknown held-out metric selector: {which!r}")


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------


def evaluate_pair_gate(
    locomo_delta: float,
    heldout_delta: float,
    *,
    heldout_tolerance: float = DEFAULT_HELDOUT_TOLERANCE,
    locomo_min_delta: float = DEFAULT_LOCOMO_MIN_DELTA,
) -> dict[str, Any]:
    """Pure gate decision from the two deltas.

    ``heldout_tolerance`` is a positive magnitude: held-out may drop by at most
    this much. A delta exactly equal to ``-heldout_tolerance`` is NOT a
    regression (boundary is inclusive). ``locomo_min_delta`` is the minimum
    LoCoMo gain required (0.0 = must not regress).
    """
    tol = abs(heldout_tolerance)
    heldout_regressed = heldout_delta < -tol - _FLOAT_EPS
    locomo_meets_min = locomo_delta >= locomo_min_delta - _FLOAT_EPS
    locomo_improved = locomo_delta > _FLOAT_EPS
    passed = locomo_meets_min and not heldout_regressed

    reasons: list[str] = []
    if not locomo_meets_min:
        reasons.append(
            f"LoCoMo delta {locomo_delta:+.4f} below required minimum "
            f"{locomo_min_delta:+.4f}"
        )
    if heldout_regressed:
        reasons.append(
            f"held-out F1 regressed {heldout_delta:+.4f} "
            f"(beyond -{tol:.4f} tolerance)"
        )
    if passed and not locomo_improved:
        reasons.append(
            f"note: LoCoMo did not strictly improve ({locomo_delta:+.4f}); "
            "passed on non-regression only"
        )

    return {
        "passed": passed,
        "locomo_delta": locomo_delta,
        "heldout_delta": heldout_delta,
        "heldout_tolerance": tol,
        "locomo_min_delta": locomo_min_delta,
        "locomo_meets_min": locomo_meets_min,
        "locomo_improved": locomo_improved,
        "heldout_regressed": heldout_regressed,
        "reasons": reasons,
    }


def build_report(
    locomo_candidate: dict[str, Any],
    heldout_candidate: dict[str, Any],
    locomo_baseline: dict[str, Any],
    heldout_baseline: dict[str, Any],
    *,
    locomo_metric: str = "4cat",
    heldout_metric: str = "f1",
    heldout_tolerance: float = DEFAULT_HELDOUT_TOLERANCE,
    locomo_min_delta: float = DEFAULT_LOCOMO_MIN_DELTA,
) -> dict[str, Any]:
    """Assemble the side-by-side report + verdict.

    Raises GateInputError if a required metric is absent from any input.
    """
    lc = extract_locomo_f1(locomo_candidate, locomo_metric)
    lb = extract_locomo_f1(locomo_baseline, locomo_metric)
    hc = extract_heldout_f1(heldout_candidate, heldout_metric)
    hb = extract_heldout_f1(heldout_baseline, heldout_metric)

    missing = [
        name
        for name, val in (
            ("locomo-candidate", lc),
            ("locomo-baseline", lb),
            ("heldout-candidate", hc),
            ("heldout-baseline", hb),
        )
        if val is None
    ]
    if missing:
        raise GateInputError(
            f"could not read required metric from: {', '.join(missing)} "
            f"(locomo_metric={locomo_metric}, heldout_metric={heldout_metric})"
        )

    locomo_delta = lc - lb
    heldout_delta = hc - hb
    verdict = evaluate_pair_gate(
        locomo_delta,
        heldout_delta,
        heldout_tolerance=heldout_tolerance,
        locomo_min_delta=locomo_min_delta,
    )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gate": "pair-gate/heldout",
        "locomo": {
            "metric": f"token_f1_{locomo_metric}",
            "candidate": lc,
            "baseline": lb,
            "delta": locomo_delta,
        },
        "heldout": {
            "benchmark": "longmemeval",
            "metric": ("token_f1" if heldout_metric == "f1" else "judge_accuracy"),
            "candidate": hc,
            "baseline": hb,
            "delta": heldout_delta,
            "tolerance": verdict["heldout_tolerance"],
        },
        "thresholds": {
            "heldout_tolerance": verdict["heldout_tolerance"],
            "locomo_min_delta": locomo_min_delta,
        },
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _fmt(v: float | None) -> str:
    return f"{v:.4f}" if v is not None else "n/a"


def _fmt_delta(v: float | None) -> str:
    if v is None:
        return "n/a"
    return f"{v:+.4f}"


def render_table(report: dict[str, Any]) -> str:
    """Render the side-by-side report as a fixed-width text table."""
    lc, hd = report["locomo"], report["heldout"]
    verdict = report["verdict"]
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("  Pair Gate — LoCoMo (tuned) vs held-out (LongMemEval, never tuned)")
    lines.append("=" * 72)
    lines.append(f"  {'Track':<22} {'candidate':>10} {'baseline':>10} {'delta':>10}")
    lines.append(f"  {'-' * 54}")
    lines.append(
        f"  {'LoCoMo ' + lc['metric']:<22} {_fmt(lc['candidate']):>10} "
        f"{_fmt(lc['baseline']):>10} {_fmt_delta(lc['delta']):>10}"
    )
    lines.append(
        f"  {'held-out ' + hd['metric']:<22} {_fmt(hd['candidate']):>10} "
        f"{_fmt(hd['baseline']):>10} {_fmt_delta(hd['delta']):>10}"
    )
    lines.append(f"  {'-' * 54}")
    lines.append(
        f"  held-out tolerance: -{report['thresholds']['heldout_tolerance']:.4f} F1"
        f"   |   LoCoMo min gain: {report['thresholds']['locomo_min_delta']:+.4f} F1"
    )
    lines.append("=" * 72)
    status = "PASS" if verdict["passed"] else "FAIL"
    lines.append(f"  VERDICT: {status}")
    for reason in verdict["reasons"]:
        lines.append(f"    - {reason}")
    lines.append("=" * 72)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _resolve_locomo_baseline(spec: str) -> dict[str, Any]:
    if spec.strip().lower() == "feb":
        return FEB_2026_BASELINE
    return load_metrics(spec)


def run(argv: list[str] | None = None) -> int:
    """Argparse entry point returning an exit code (testable without sys.exit)."""
    parser = argparse.ArgumentParser(
        description="Held-out validation gate: LoCoMo + held-out side by side.",
    )
    parser.add_argument("--locomo-candidate", required=True, help="Candidate LoCoMo metrics.json")
    parser.add_argument("--heldout-candidate", required=True, help="Candidate held-out metrics.json")
    parser.add_argument(
        "--locomo-baseline", default="feb",
        help="Baseline LoCoMo metrics.json, or 'feb' for the documented Feb 2026 "
        "cosine reference (default: feb)",
    )
    parser.add_argument("--heldout-baseline", required=True, help="Baseline held-out metrics.json")
    parser.add_argument("--locomo-metric", default="4cat", choices=["4cat", "5cat", "overall"])
    parser.add_argument("--heldout-metric", default="f1", choices=["f1", "accuracy"])
    parser.add_argument(
        "--heldout-tolerance", type=float, default=DEFAULT_HELDOUT_TOLERANCE,
        help="Max allowed held-out F1 drop before failing (default: 0.01)",
    )
    parser.add_argument(
        "--locomo-min-delta", type=float, default=DEFAULT_LOCOMO_MIN_DELTA,
        help="Minimum LoCoMo F1 gain required (default: 0.0 = must not regress; "
        "set a positive margin for a tuned-constant PR)",
    )
    parser.add_argument("--out", default=None, help="Write the report JSON to this path")
    args = parser.parse_args(argv)

    try:
        report = build_report(
            load_metrics(args.locomo_candidate),
            load_metrics(args.heldout_candidate),
            _resolve_locomo_baseline(args.locomo_baseline),
            load_metrics(args.heldout_baseline),
            locomo_metric=args.locomo_metric,
            heldout_metric=args.heldout_metric,
            heldout_tolerance=args.heldout_tolerance,
            locomo_min_delta=args.locomo_min_delta,
        )
    except GateInputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    print(render_table(report))

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(f"\nReport written to {out}")

    return 0 if report["verdict"]["passed"] else 1


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - thin wrapper
    return run(argv)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
