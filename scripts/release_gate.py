#!/usr/bin/env python3
"""
release_gate.py — Release gate for LoCoMo benchmark runs.

Compares a candidate run's metrics.json (as persisted by
kumiho_eval.locomo_eval — see locomo_field_report / locomo_categories) against
a baseline, and fails the build on regression. Two baseline modes:

  --baseline feb            hardcoded, documented Feb 2026 cosine-run reference
  --baseline path/to.json   another run's metrics.json

Prints a per-category F1 + LLM-judge (J) table, candidate vs baseline, with
deltas, and always shows the adversarial-stripped (4-cat) aggregate next to
the full (5-cat) one -- 4-cat is the field-comparable number (see
locomo_eval.py ADVERSARIAL_CATEGORY / P0L2).

Exit codes:
  0  pass
  1  regression beyond threshold (gate failure)
  2  bad input (partial run without --allow-partial, missing/malformed JSON)

Usage:
    python scripts/release_gate.py results/locomo/metrics.json
    python scripts/release_gate.py results/locomo/metrics.json --baseline feb
    python scripts/release_gate.py results/locomo/metrics.json --baseline old_metrics.json --threshold 0.03
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Must match kumiho_eval.locomo_eval.CATEGORY_NAMES / ADVERSARIAL_CATEGORY.
CATEGORY_NAMES = ["multi-hop", "temporal", "open-domain", "single-hop", "adversarial"]
ADVERSARIAL_CATEGORY = "adversarial"

MIN_CONVERSATIONS = 10  # Below this, a run is a partial/noise slice (see Task 4 spec).

# ---------------------------------------------------------------------------
# Hardcoded, documented reference: Feb 2026 cosine-similarity recall run.
# No fabricated numbers -- these are the exact figures given in the release
# gate spec (P1L9): overall F1 0.565, single 0.462, multi 0.355, temporal
# 0.533, open 0.290, adversarial 0.975. No LLM-judge numbers were recorded
# for that run, so `judge_accuracy` is intentionally absent below --
# judge deltas against this baseline print as "n/a" and are never gated on.
# ---------------------------------------------------------------------------
FEB_2026_BASELINE: dict[str, Any] = {
    "label": "feb-2026-cosine",
    "total_conversations": 10,
    "locomo_categories": {
        "single-hop": {"f1": 0.462},
        "multi-hop": {"f1": 0.355},
        "temporal": {"f1": 0.533},
        "open-domain": {"f1": 0.290},
        "adversarial": {"f1": 0.975},
    },
    "locomo_field_report": {
        "token_f1": {
            "overall_5cat": 0.565,
        },
    },
}


def _load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(2)
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: {path} is not valid JSON: {e}", file=sys.stderr)
        sys.exit(2)


def _resolve_baseline(spec: str) -> dict[str, Any]:
    if spec.strip().lower() == "feb":
        return FEB_2026_BASELINE
    return _load_json(spec)


def _count_conversations(metrics: dict[str, Any]) -> int | None:
    """Best-effort conversation count. None means "unknown" (skip the check)."""
    return metrics.get("total_conversations")


def _cat_value(metrics: dict[str, Any], cat: str, metric: str) -> float | None:
    """`metric` is 'f1' or 'judge_accuracy'."""
    return metrics.get("locomo_categories", {}).get(cat, {}).get(metric)


def _four_cat_f1(metrics: dict[str, Any]) -> float | None:
    """Adversarial-stripped (4-cat) F1 aggregate.

    Prefers the precise, question-count-weighted value persisted under
    locomo_field_report.token_f1.overall_4cat (Task 1). Falls back to an
    unweighted mean across the four non-adversarial categories when only
    per-category values are available (e.g. the hardcoded 'feb' reference,
    or a metrics.json predating the Task-1 reporting changes).
    """
    report = metrics.get("locomo_field_report", {}).get("token_f1", {})
    if report.get("overall_4cat") is not None:
        return report["overall_4cat"]

    vals = [
        v for cat in CATEGORY_NAMES if cat != ADVERSARIAL_CATEGORY
        for v in [_cat_value(metrics, cat, "f1")] if v is not None
    ]
    return float(sum(vals) / len(vals)) if vals else None


def _five_cat_f1(metrics: dict[str, Any]) -> float | None:
    report = metrics.get("locomo_field_report", {}).get("token_f1", {})
    if report.get("overall_5cat") is not None:
        return report["overall_5cat"]
    return metrics.get("overall_f1")


def _four_cat_judge(metrics: dict[str, Any]) -> float | None:
    report = metrics.get("locomo_field_report", {}).get("llm_judge", {})
    return report.get("headline_4cat_excl_adversarial")


def _five_cat_judge(metrics: dict[str, Any]) -> float | None:
    report = metrics.get("locomo_field_report", {}).get("llm_judge", {})
    if report.get("overall_5cat") is not None:
        return report["overall_5cat"]
    return metrics.get("overall_judge_accuracy")


def _fmt(value: float | None) -> str:
    return f"{value:.4f}" if value is not None else "n/a"


def _fmt_delta(cand: float | None, base: float | None) -> str:
    if cand is None or base is None:
        return "n/a"
    delta = cand - base
    sign = "+" if delta >= 0 else ""
    return f"{sign}{delta:.4f}"


def _check_partial(metrics: dict[str, Any], label: str, allow_partial: bool) -> None:
    count = _count_conversations(metrics)
    if count is not None and count < MIN_CONVERSATIONS and not allow_partial:
        print(
            f"ERROR: {label} run has only {count} conversation(s) "
            f"(< {MIN_CONVERSATIONS}). Single/partial-conversation slices are "
            "documented as noise -- pass --allow-partial to override.",
            file=sys.stderr,
        )
        sys.exit(2)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Release gate: compare a LoCoMo metrics.json against a baseline "
        "and fail on F1 regression.",
    )
    parser.add_argument("candidate", help="Path to the candidate run's metrics.json")
    parser.add_argument(
        "--baseline", default="feb",
        help="Path to a baseline metrics.json, or 'feb' for the hardcoded Feb 2026 "
        "cosine-run reference numbers (default: feb)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.02,
        help="Max allowed per-category F1 regression before failing (default: 0.02)",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help=f"Allow candidate/baseline runs with fewer than {MIN_CONVERSATIONS} "
        "conversations (single-conversation slices are documented as noise)",
    )
    args = parser.parse_args()

    candidate = _load_json(args.candidate)
    baseline = _resolve_baseline(args.baseline)

    _check_partial(candidate, "candidate", args.allow_partial)
    _check_partial(baseline, "baseline", args.allow_partial)

    cand_label = Path(args.candidate).name
    base_label = baseline.get("label", args.baseline)

    print(f"\n{'=' * 78}")
    print(f"  LoCoMo Release Gate — {cand_label}  vs  baseline={base_label}")
    print(f"{'=' * 78}")
    print(f"  {'Category':<16} {'F1 cand':>9} {'F1 base':>9} {'ΔF1':>9} "
          f"{'J cand':>9} {'J base':>9} {'ΔJ':>9}")
    print(f"  {'-' * 74}")

    regressions: list[str] = []

    for cat in CATEGORY_NAMES:
        f1_c = _cat_value(candidate, cat, "f1")
        f1_b = _cat_value(baseline, cat, "f1")
        j_c = _cat_value(candidate, cat, "judge_accuracy")
        j_b = _cat_value(baseline, cat, "judge_accuracy")

        print(
            f"  {cat:<16} {_fmt(f1_c):>9} {_fmt(f1_b):>9} {_fmt_delta(f1_c, f1_b):>9} "
            f"{_fmt(j_c):>9} {_fmt(j_b):>9} {_fmt_delta(j_c, j_b):>9}"
        )

        if f1_c is not None and f1_b is not None and (f1_c - f1_b) < -args.threshold:
            regressions.append(
                f"category '{cat}' F1 regressed {f1_b:.4f} -> {f1_c:.4f} "
                f"(Δ{f1_c - f1_b:+.4f}, threshold -{args.threshold:.4f})"
            )

    print(f"  {'-' * 74}")

    five_f1_c, five_f1_b = _five_cat_f1(candidate), _five_cat_f1(baseline)
    four_f1_c, four_f1_b = _four_cat_f1(candidate), _four_cat_f1(baseline)
    five_j_c, five_j_b = _five_cat_judge(candidate), _five_cat_judge(baseline)
    four_j_c, four_j_b = _four_cat_judge(candidate), _four_cat_judge(baseline)

    print(
        f"  {'overall (5cat)':<16} {_fmt(five_f1_c):>9} {_fmt(five_f1_b):>9} "
        f"{_fmt_delta(five_f1_c, five_f1_b):>9} {_fmt(five_j_c):>9} {_fmt(five_j_b):>9} "
        f"{_fmt_delta(five_j_c, five_j_b):>9}"
    )
    print(
        f"  {'overall (4cat)':<16} {_fmt(four_f1_c):>9} {_fmt(four_f1_b):>9} "
        f"{_fmt_delta(four_f1_c, four_f1_b):>9} {_fmt(four_j_c):>9} {_fmt(four_j_b):>9} "
        f"{_fmt_delta(four_j_c, four_j_b):>9}"
    )
    print(f"{'=' * 78}")
    print(
        "  4cat = adversarial-stripped aggregate (field-comparable to Mem0/Zep). "
        "5cat includes adversarial."
    )

    if four_f1_c is not None and four_f1_b is not None and four_f1_c < four_f1_b:
        regressions.append(
            f"4-cat (adversarial-stripped) F1 aggregate regressed "
            f"{four_f1_b:.4f} -> {four_f1_c:.4f} (any regression fails this gate)"
        )

    print()
    if regressions:
        print("GATE FAILED:")
        for r in regressions:
            print(f"  - {r}")
        print()
        return 1

    print("GATE PASSED: no category F1 regression beyond threshold, "
          "4-cat aggregate did not regress.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
