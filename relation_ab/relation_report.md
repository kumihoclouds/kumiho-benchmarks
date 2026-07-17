# Relation-traversal A/B benchmark (kumiho-memory, PR #91)

Deterministic, local-only measurement of the flag-gated entity->entity relation traversal in `kumiho_memory.graph_augmentation` (`GraphAugmentationConfig.relation_traversal`).

## Configuration

| setting | value |
|---|---|
| backend | local Community Edition (KUMIHO_AUTH_TOKEN unset, no cloud) |
| LLM | none (adapter=None; `reformulate_queries=False`) |
| retrieval | `kumiho.search` deep (`include_revision_metadata=True`), scored by the local CE's built-in similarity; no external embedding service, no LLM |
| seed feed | confidence gate: keep search hits within 0.90x the top score (admits exactly the 1 intended seed per query — see methodology) |
| entrypoint | `GraphAugmentedRecall.recall` (the production path the memory manager calls) |
| condition A | `entity_recall=True, relation_traversal=False` |
| condition B | `entity_recall=True, relation_traversal=True` |
| caps | GraphAugmentationConfig defaults (edges/anchor=3, neighbours=4, results=4, hub_degree_max=12) |
| corpus | 22 conversations: 8 seed+gold pairs (=16) + 3 direct + 2 negative in the searchable `chat` space, plus 1 relation-source conversation in `relsrc` (never searched) |

### Methodology: seed feed

The manager feeds `recall` a list of search hits (revision-kref + score dicts); the traversal seeds from them.  The local CE ranks by dense similarity, so short structurally-similar authored memos share a ~0.5-0.8 similarity floor and never drop out of the raw top-k — even with globally unique codewords per conversation, a raw top-5 feed co-retrieves other pairs' seeds (and sometimes the gold itself).  In every query the memory the text is genuinely about is rank #1 by a wide margin, so a relative gate (>= 0.90x the top score) admits exactly that one true seed: verified for 13/13 queries.  This excludes the direct-co-retrieval CONFOUND (a gold surfacing as its own dense hit is a retrieval artifact, not a graph-path finding) without masking any graph-path reachability — a gold reached in OFF could only come from direct co-retrieval (excluded on purpose) or from a seed sharing the gold's entity (which the corpus never builds).

## A/B results

| metric | OFF (A) | ON (B) |
|---|--:|--:|
| relation-gold reach | 0/8 | 8/8 |
| direct-gold reach (control) | 3/3 | 3/3 |
| negative relation-path leak (control) | 0 | 0 |
| relation-path results surfaced (relation class) | 0 | 8 |

## Latency (per `recall` call, all queries)

| condition | mean | p95 |
|---|--:|--:|
| OFF | 663.9ms | 728.0ms |
| ON | 664.3ms | 679.6ms |

Mean latency delta (ON - OFF): **+0.4ms** (this run) — within run-to-run noise (the sign flips across runs), i.e. no measurable latency penalty at these caps.

Latency notes (read the delta, not the absolute): a warmup pass runs both conditions before timing, so the numbers are not confounded by cold-start (an un-warmed sequential OFF-then-ON pass times OFF cold and ON hot, which inverts the sign misleadingly). The absolute floor (~0.6-0.7s) is dominated by the traversal's daemon-thread timeout poll cadence (`asyncio.sleep(0.5)` between readiness checks), not the graph work — the handful of extra `get_edges` round-trips the ON path adds is far below that granularity, so the ON-OFF delta stays in the single-to-low-tens of milliseconds and its sign is not stable.

## Predicate folding (write path, via predicate_registry)

| pair | verbatim predicate | expected edge | observed edge | ok |
|---|---|---|---|:-:|
| r1_uses | `uses` | USES | USES | yes |
| r2_utilizes | `utilizes` | USES | USES | yes |
| r3_employs | `employs` | USES | USES | yes |
| r4_depends_on | `depends on` | DEPENDS_ON | DEPENDS_ON | yes |
| r5_relies_on | `relies on` | DEPENDS_ON | DEPENDS_ON | yes |
| r6_requires | `requires` | DEPENDS_ON | DEPENDS_ON | yes |
| r7_part_of | `part of` | PART_OF | PART_OF | yes |
| r8_relates_to | `monitors` | RELATES_TO | RELATES_TO | yes |

## Per-query detail

| query | class | predicate | OFF gold | ON gold | ON via | OFF rel-paths | ON rel-paths |
|---|---|---|:-:|:-:|---|--:|--:|
| r1_uses | relation | `uses` | miss | hit | USES | 0 | 1 |
| r2_utilizes | relation | `utilizes` | miss | hit | USES | 0 | 1 |
| r3_employs | relation | `employs` | miss | hit | USES | 0 | 1 |
| r4_depends_on | relation | `depends on` | miss | hit | DEPENDS_ON | 0 | 1 |
| r5_relies_on | relation | `relies on` | miss | hit | DEPENDS_ON | 0 | 1 |
| r6_requires | relation | `requires` | miss | hit | DEPENDS_ON | 0 | 1 |
| r7_part_of | relation | `part of` | miss | hit | PART_OF | 0 | 1 |
| r8_relates_to | relation | `monitors` | miss | hit | RELATES_TO | 0 | 1 |
| d1_registry | direct |  | hit | hit |  | 0 | 0 |
| d2_flags | direct |  | hit | hit |  | 0 | 0 |
| d3_postmortem | direct |  | hit | hit |  | 0 | 0 |
| n1_wiki | negative |  | - | - |  | 0 | 0 |
| n2_survey | negative |  | - | - |  | 0 | 0 |

## Interpretation

- **OFF reaches 0/8 relation golds** — as expected: the gold is only entity-B-anchored and shares no entity with the seed, so ABOUT-sibling bridging cannot reach it. This validates the golds do not leak through any other path.
- **ON reaches 8/8 relation golds** — the relation edge is the sole path that surfaces them.
- **Direct controls: OFF 3/3, ON 3/3** — base recall is intact in both conditions.
- **Negative controls: 0 / 0 relation-path leaks** — an entity with no relation edges surfaces no relation results in either condition.

## Caveats (honest scope)

- **Authored micro-benchmark, not LoCoMo.** The corpus is hand-written to isolate one mechanism; it is not a natural-distribution memory workload and says nothing about end-task QA quality.
- **Reachability, not answer quality.** The metric is whether the gold conversation's kref appears in `recall`'s augmented output — i.e. whether the new graph path *can reach* the connected memory. It does not score ranking, downstream context selection, or answer correctness.
- **Single gold per query.** Each relation query has exactly one designated gold; there is no graded relevance.
- **Seed-confidence gate, not raw top-k.** The local CE ranks by dense similarity with a ~0.5-0.8 floor across these short authored memos, so seeds are fed via a relative score gate (see methodology) rather than raw top-k. This isolates each query's one genuine seed; a production run over a natural corpus feeds top-k, where the tail is genuinely related rather than a structural-similarity floor.
- **Traversal entries are score-less.** Relation-path results ride the sibling reserve and never outrank or evict a scored direct hit; this benchmark measures their presence in `recall` output, upstream of the manager's final rerank/trim.
- **Both conditions run `entity_recall=True`** (the ontology default), so the OFF baseline already includes the 2-hop ABOUT walk; the A/B delta isolates the relation extension specifically.
- **Corpus size favors isolation over minimalism.** Each of the 8 relation queries gets a dedicated seed + gold conversation so its OFF-unreachability is provable independently; that is why the corpus is a couple dozen conversations rather than a single shared dozen.

