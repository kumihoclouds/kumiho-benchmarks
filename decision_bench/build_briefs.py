#!/usr/bin/env python3
"""Populate each corpus item's `memory_brief` from the LIVE Decision Memory.

For every item we call the real ``kumiho_memory.code_query.why()`` against the
``dogfood-code-memory`` project on the local CE (the exact call the product's
``kumiho_code_why`` MCP tool makes) and cache the rendered context brief plus
what it retrieved.  Run once; the benchmark run then holds this retrieved
context FIXED, so ``run.py`` measures the VALUE of the memory content, not
retrieval variance — and needs no CE.

Whether why() actually surfaces the expected decision for each item is itself
recorded (``retrieved_expected_commit``): if it does not, memory cannot help
that item and the benchmark says so honestly.

    # env: KUMIHO_MEMORY_CODE=1 (set here), no KUMIHO_AUTH_TOKEN, CE on :9190
    python build_briefs.py            # build briefs.json
    python build_briefs.py --rerank   # also engage the cross-encoder reranker
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = "dogfood-code-memory"

if os.environ.get("KUMIHO_AUTH_TOKEN"):
    sys.exit(
        "KUMIHO_AUTH_TOKEN is set — this would hit the cloud, not the local CE "
        "dogfood project. Unset it and retry."
    )


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rerank", action="store_true", help="engage the cross-encoder reranker")
    ap.add_argument("--limit", type=int, default=5)
    args = ap.parse_args()

    os.environ["KUMIHO_MEMORY_CODE"] = "1"
    from kumiho_memory.code_decisions import CodeMemoryConfig
    from kumiho_memory.code_query import why

    cfg = CodeMemoryConfig(repo="kumiho-SDKs")
    reranker = None
    if args.rerank:
        try:
            from kumiho_memory.recall_rerank import resolve_reranker_from_env

            os.environ.setdefault("KUMIHO_RERANK_CROSS_ENCODER", "1")
            reranker = resolve_reranker_from_env(env=os.environ)
        except Exception as exc:  # noqa: BLE001
            print(f"reranker unavailable ({exc}); continuing without it")

    with open(os.path.join(HERE, "corpus.json"), encoding="utf-8") as fh:
        corpus = json.load(fh)

    briefs: dict = {}
    for item in corpus["items"]:
        res = await why(
            question=item["why_question"], file=item["file"],
            project_name=PROJECT, config=cfg, limit=args.limit, reranker=reranker,
        )
        brief = res.get("context", "") or ""
        decisions = res.get("decisions", [])
        titles = [d.get("title", "") for d in decisions]
        shas = []
        for d in decisions:
            shas += [a.get("commit", "") for a in d.get("anchors", [])]
            shas += [c.get("sha", "") for c in d.get("commits", [])]
            shas.append(d.get("kref", ""))
        want = item["commit"][:7]
        retrieved = any(want in (s or "") for s in shas)
        briefs[item["id"]] = {
            "memory_brief": brief,
            "retrieved_titles": titles,
            "retrieved_expected_commit": retrieved,
            "chars": len(brief),
        }
        flag = "ok" if (brief and retrieved) else "!!"
        print(f"[{flag}] {item['id']:>22}  {len(brief):>4}ch  expect {want} retrieved={retrieved}  {titles[:3]}")

    with open(os.path.join(HERE, "briefs.json"), "w", encoding="utf-8") as fh:
        json.dump(briefs, fh, ensure_ascii=False, indent=2)

    empty = [k for k, v in briefs.items() if not v["memory_brief"]]
    missed = [k for k, v in briefs.items() if not v["retrieved_expected_commit"]]
    print(f"\nwrote briefs.json — {len(briefs)} items")
    if empty:
        print(f"  empty brief (why() returned nothing): {empty}")
    if missed:
        print(f"  expected commit NOT retrieved (memory can't help these): {missed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
