# relation_ab — relation-traversal A/B micro-benchmark

Deterministic, local-only A/B for kumiho-memory's flag-gated entity→entity
relation traversal (`GraphAugmentationConfig.relation_traversal`, env
`KUMIHO_MEMORY_RELATION_TRAVERSAL`), shipped in
[kumiho-SDKs PR #91](https://github.com/KumihoIO/kumiho-SDKs/pull/91).

No LLM, no cloud, no embedding service — an authored 22-conversation corpus is
materialized through the branch's real write path
(`decompose_and_link_agent`, so predicate folding is exercised end-to-end) and
queried through the production augmentation entrypoint
(`GraphAugmentedRecall.recall`) with the flag toggled as the only difference.

## Result (2026-07-17, local CE)

| metric | OFF | ON |
|---|--:|--:|
| relation-gold reach (gold only reachable via a relation edge) | 0/8 | **8/8** |
| direct-gold reach (control) | 3/3 | 3/3 |
| negative relation-path leak (control) | 0 | 0 |
| per-recall latency delta | — | within noise (±10 ms, sign flips) |

Predicate folding verified in the same run: `uses/utilizes/employs → USES`,
`depends on/relies on/requires → DEPENDS_ON`, `part of → PART_OF`, and the
unregistered `monitors → RELATES_TO` fallback.

Full report with configuration, methodology (seed-confidence gate), per-query
table, and caveats: [`relation_report.md`](relation_report.md).

## Run

Requires: a local Kumiho Community Edition on `127.0.0.1:9190`, and a
kumiho-memory checkout at or after PR #91.

```bash
cd relation_ab
env -u KUMIHO_AUTH_TOKEN \
  PYTHONPATH=/path/to/kumiho-SDKs/python/kumiho-memory \
  python relation_ab.py            # setup -> A/B -> report -> teardown
# flags: --keep, --index-wait N, --allow-token
```

The script refuses to run with `KUMIHO_AUTH_TOKEN` set (it would target the
cloud); it isolates everything in a `RelationEvalBench` project and tears it
down unless `--keep` is passed.

## What this does and does not show

It shows the new read path reaches exactly the memories that are only
connected through relation edges, within the default caps, at no measurable
latency cost — and that the OFF condition (today's default) cannot reach them.
It is an authored single-gold micro-benchmark: it does **not** measure
end-task QA quality, and the default-ON decision still goes through the
pair-measured LoCoMo F1 + LoCoMo-Plus gate
([kumiho-SDKs #86](https://github.com/KumihoIO/kumiho-SDKs/issues/86)).
Note for that gate: the standard LoCoMo ingest (LLM summarizer path) writes no
entity→entity relation edges — only agent-driven decompose does — so a
standard LoCoMo pair run cannot exercise this feature; the ingest path must
write relations first.
