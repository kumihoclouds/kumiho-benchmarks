"""Deterministic held-out subset for LoCoMo-tuned-constant validation.

The rerank / prior constants (evidence delta 0.15, recency 0.12, MMR lambda
0.72, half-life 45d, ...) are tuned against LoCoMo / LoCoMo-Plus. Nothing
proves that "LoCoMo score up" generalises to real-world recall. This module
defines a *held-out* slice of a different corpus (LongMemEval-S) that the pair
gate (`kumiho_eval.pair_gate`) guards against regression.

HARD RULE — this subset must NEVER be used for tuning. It is a validation-only
guard: a tuned-constant change must improve LoCoMo *without* regressing this
held-out slice. Fitting constants to these question IDs would defeat the entire
point of holding them out.

Sampling rule ("sha256-lex-v1")
-------------------------------
The subset is a content-addressed, seed-free, order-independent sample:

  1. Collect the unique ``question_id`` of every entry.
  2. Rank them by the hex ``sha256`` digest of the id (lexicographic).
  3. Take the first ``size`` (default 100) and return them sorted
     lexicographically for a stable, diff-friendly committed order.

Because LongMemEval-S is a fixed published dataset, this rule yields the *same*
100 ids every time — it needs no RNG seed and is invariant to input ordering.
The rule itself is therefore the authoritative subset definition. The committed
manifest (``heldout_sets/longmemeval_heldout.json``) caches that definition;
run ``python -m kumiho_eval.heldout freeze`` on a checkout that has the vendored
LongMemEval data to *materialise* the exact id list into the manifest, after
which it is treated as immutable and verified on load.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Subset specification (the authoritative definition)
# ---------------------------------------------------------------------------

#: Benchmark the held-out slice is drawn from.
HELDOUT_BENCHMARK = "longmemeval"

#: LongMemEval variant. The 500 question ids are shared across s/m/oracle, so
#: the selected id set is variant-independent; "s" is the run-side default.
HELDOUT_VARIANT = "s"

#: Number of question ids in the held-out slice (issue #109: 50-100).
HELDOUT_SIZE = 100

#: Identifier for the sampling rule, bumped if the algorithm ever changes.
HELDOUT_RULE = "sha256-lex-v1"

#: Default committed manifest path (next to this module, no name clash with it).
DEFAULT_MANIFEST_PATH = (
    Path(__file__).resolve().parent / "heldout_sets" / "longmemeval_heldout.json"
)


class HeldoutError(RuntimeError):
    """Raised when the held-out subset cannot be resolved consistently."""


# ---------------------------------------------------------------------------
# Sampling rule
# ---------------------------------------------------------------------------


def _question_ids(entries: list[dict[str, Any]]) -> list[str]:
    """Unique, non-empty ``question_id`` values across entries (sorted)."""
    seen: set[str] = set()
    for entry in entries:
        qid = entry.get("question_id")
        if isinstance(qid, str) and qid:
            seen.add(qid)
    return sorted(seen)


def select_heldout_ids(
    entries: list[dict[str, Any]],
    size: int = HELDOUT_SIZE,
) -> list[str]:
    """Deterministically select the held-out question ids from ``entries``.

    Implements the ``sha256-lex-v1`` rule (see module docstring). The result is
    invariant to the ordering of ``entries`` and to duplicate ids, and is
    returned sorted lexicographically. When fewer than ``size`` unique ids are
    present, all of them are returned.
    """
    if size < 0:
        raise ValueError(f"size must be non-negative, got {size}")

    ids = _question_ids(entries)
    ranked = sorted(ids, key=lambda qid: hashlib.sha256(qid.encode("utf-8")).hexdigest())
    return sorted(ranked[:size])


# ---------------------------------------------------------------------------
# Manifest I/O
# ---------------------------------------------------------------------------


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    """Load the held-out manifest JSON.

    Raises HeldoutError if the file is missing or not valid JSON.
    """
    p = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    if not p.exists():
        raise HeldoutError(f"held-out manifest not found: {p}")
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise HeldoutError(f"held-out manifest {p} is not valid JSON: {e}") from e
    if not isinstance(data, dict):
        raise HeldoutError(f"held-out manifest {p} must be a JSON object")
    return data


def freeze_manifest(
    entries: list[dict[str, Any]],
    *,
    size: int = HELDOUT_SIZE,
    variant: str = HELDOUT_VARIANT,
) -> dict[str, Any]:
    """Build a *materialised* manifest dict from a loaded dataset.

    The returned dict has ``materialized: true`` and a concrete, sorted
    ``question_ids`` list. Callers persist it with :func:`write_manifest`.
    """
    ids = select_heldout_ids(entries, size=size)
    return {
        "benchmark": HELDOUT_BENCHMARK,
        "variant": variant,
        "size": size,
        "rule": HELDOUT_RULE,
        "materialized": True,
        "count": len(ids),
        "never_tune_on_this": True,
        "description": (
            "Validation-only held-out slice of LongMemEval-"
            f"{variant}. Selected by the {HELDOUT_RULE} rule. NEVER tune "
            "constants against these question ids."
        ),
        "question_ids": ids,
    }


def write_manifest(manifest: dict[str, Any], path: str | Path | None = None) -> Path:
    """Write ``manifest`` as pretty JSON, creating parent dirs as needed."""
    p = Path(path) if path is not None else DEFAULT_MANIFEST_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return p


# ---------------------------------------------------------------------------
# Resolution / filtering
# ---------------------------------------------------------------------------


def _manifest_ids(manifest: dict[str, Any]) -> list[str]:
    ids = manifest.get("question_ids") or []
    if not isinstance(ids, list) or not all(isinstance(q, str) for q in ids):
        raise HeldoutError("manifest 'question_ids' must be a list of strings")
    return list(ids)


def is_materialized(manifest: dict[str, Any]) -> bool:
    """True if the manifest carries a concrete, frozen id list."""
    return bool(manifest.get("materialized")) and bool(manifest.get("question_ids"))


def resolve_heldout_ids(
    entries: list[dict[str, Any]],
    manifest: dict[str, Any] | None = None,
    *,
    strict: bool = True,
) -> list[str]:
    """Resolve the held-out ids to use for ``entries``.

    - Materialised manifest: the frozen ids are authoritative. When ``entries``
      is non-empty every frozen id must appear in it; missing ids raise (or, if
      ``strict`` is False, are dropped). The frozen list is also re-checked
      against the sampling rule when the frozen ``size`` matches this run's
      selection, so silent dataset drift is caught.
    - Unmaterialised manifest (or ``None``): the ids are computed from
      ``entries`` via the sampling rule. Requires a non-empty ``entries``.

    Returns ids sorted lexicographically.
    """
    manifest = manifest if manifest is not None else {}

    if is_materialized(manifest):
        frozen = sorted(_manifest_ids(manifest))
        if entries:
            present = set(_question_ids(entries))
            missing = [q for q in frozen if q not in present]
            if missing:
                if strict:
                    raise HeldoutError(
                        f"{len(missing)} frozen held-out id(s) absent from the "
                        f"dataset (e.g. {missing[0]!r}); the vendored corpus does "
                        "not match the frozen manifest"
                    )
                frozen = [q for q in frozen if q in present]
            # Drift guard: if the frozen size matches a fresh selection over
            # this dataset, they must agree.
            size = manifest.get("size", len(frozen))
            if isinstance(size, int) and size == len(_manifest_ids(manifest)):
                recomputed = select_heldout_ids(entries, size=size)
                if recomputed != sorted(_manifest_ids(manifest)) and strict:
                    raise HeldoutError(
                        "frozen held-out ids do not match the sampling rule over "
                        "the current dataset (manifest drift or dataset change)"
                    )
        return frozen

    if not entries:
        raise HeldoutError(
            "held-out manifest is not materialized and no dataset was provided "
            "to compute the subset; run `python -m kumiho_eval.heldout freeze` "
            "on a checkout with the vendored LongMemEval data"
        )
    size = manifest.get("size", HELDOUT_SIZE)
    if not isinstance(size, int):
        size = HELDOUT_SIZE
    return select_heldout_ids(entries, size=size)


def filter_entries(
    entries: list[dict[str, Any]],
    ids: list[str],
    *,
    strict: bool = True,
) -> list[dict[str, Any]]:
    """Return the entries whose ``question_id`` is in ``ids``.

    Output order follows ``ids`` (the deterministic held-out order), not the
    dataset order. When ``strict`` and an id has no matching entry, raise.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for entry in entries:
        qid = entry.get("question_id")
        if isinstance(qid, str) and qid and qid not in by_id:
            by_id[qid] = entry

    out: list[dict[str, Any]] = []
    missing: list[str] = []
    for qid in ids:
        entry = by_id.get(qid)
        if entry is None:
            missing.append(qid)
            continue
        out.append(entry)

    if missing and strict:
        raise HeldoutError(
            f"{len(missing)} held-out id(s) not present in the dataset "
            f"(e.g. {missing[0]!r})"
        )
    return out


def apply_heldout(
    entries: list[dict[str, Any]],
    *,
    manifest_path: str | Path | None = None,
    strict: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Restrict a loaded LongMemEval dataset to the held-out slice.

    Loads the manifest (default committed path when ``manifest_path`` is None,
    tolerating an absent default), resolves the ids, filters, and returns
    ``(filtered_entries, resolution_info)``. ``resolution_info`` records what
    was used so the run's metrics.json can be self-describing.
    """
    try:
        manifest = load_manifest(manifest_path)
    except HeldoutError:
        if manifest_path is not None:
            raise
        manifest = {}  # default manifest absent -> compute from rule

    ids = resolve_heldout_ids(entries, manifest, strict=strict)
    filtered = filter_entries(entries, ids, strict=strict)
    info = {
        "benchmark": HELDOUT_BENCHMARK,
        "variant": manifest.get("variant", HELDOUT_VARIANT),
        "rule": manifest.get("rule", HELDOUT_RULE),
        "size": manifest.get("size", HELDOUT_SIZE),
        "materialized": is_materialized(manifest),
        "selected_count": len(filtered),
        "question_ids": ids,
    }
    return filtered, info


# ---------------------------------------------------------------------------
# CLI: freeze the manifest from vendored data
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="Materialise the LongMemEval held-out manifest from the "
        "vendored dataset (validation-only; NEVER tune on this subset).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze", help="Write the frozen id list to the manifest")
    freeze.add_argument("--variant", default=HELDOUT_VARIANT, choices=["s", "m", "oracle"])
    freeze.add_argument("--size", type=int, default=HELDOUT_SIZE)
    freeze.add_argument("--data-dir", default=None, help="LongMemEval data dir override")
    freeze.add_argument("--out", default=None, help="Manifest output path override")

    show = sub.add_parser("show", help="Print the committed manifest summary")
    show.add_argument("--manifest", default=None, help="Manifest path override")

    args = parser.parse_args(argv)

    if args.command == "freeze":
        from .longmemeval_eval import load_longmemeval

        entries = load_longmemeval(variant=args.variant, data_dir=args.data_dir)
        manifest = freeze_manifest(entries, size=args.size, variant=args.variant)
        out = write_manifest(manifest, args.out)
        print(f"Froze {manifest['count']} held-out ids -> {out}")
        return 0

    if args.command == "show":
        manifest = load_manifest(args.manifest)
        print(
            f"benchmark={manifest.get('benchmark')} "
            f"variant={manifest.get('variant')} size={manifest.get('size')} "
            f"rule={manifest.get('rule')} "
            f"materialized={is_materialized(manifest)} "
            f"count={len(manifest.get('question_ids') or [])}"
        )
        return 0

    return 2  # pragma: no cover - argparse 'required=True' prevents this


if __name__ == "__main__":  # pragma: no cover
    import sys

    sys.exit(main())
