# Decision-Reintroduction Benchmark v2 — report

- agent model: `gpt-5`  judge: `gpt-4o-mini`  trials/arm: 3
- behavioral items: 6  (headline); general_knowledge / recall reported separately

## Headline — behavioral items (R = evidence-backed choice)

| item | A R-rate | B' (decoy) R | B (memory) R | **content effect B−B'** | verified cite |
|---|--:|--:|--:|--:|--:|
| `based_on_schema` | 0.00 | 0.00 | 1.00 | +1.00 | 0.67 |
| `per_subquery_ce` | 1.00 | 0.00 | 1.00 | +1.00 | 1.00 |
| `additive_partition` | 1.00 | 0.00 | 1.00 | +1.00 | 0.33 |
| `sibling_reserve_on_top` | 0.00 | 0.00 | 1.00 | +1.00 | — |
| `gate_about_edges` | 1.00 | 1.00 | 1.00 | +0.00 | — |
| `skip_hub_entities` | 1.00 | 1.00 | 1.00 | +0.00 | 0.67 |

- baseline (A) R-rate: **0.67**
- decoy (B′) R-rate: **0.33**   → authority effect (B′−A): -0.33
- memory (B) R-rate: **1.00**
- **content effect (B − B′), memory-attributable: +0.67**
- raw gain (B − A): +0.33
- model already knew (A R≥0.5): `per_subquery_ce`, `additive_partition`, `gate_about_edges`, `skip_hub_entities`
- keyword/judge disagreements: 3

## General-knowledge (a-priori, NOT headline)

| item | A R-rate | B (memory) R | gain B−A |
|---|--:|--:|--:|
| `rerank_event_loop` | 1.00 | 1.00 | +0.00 |
| `dedup_key` | 1.00 | 1.00 | +0.00 |
| `fire_forget_entity` | 0.00 | 1.00 | +1.00 |
| `hybrid_sibling_rerank` | 1.00 | 1.00 | +0.00 |

## Internal-state recall (NOT behavioral)

| item | A R-rate | B (memory) R | gain B−A |
|---|--:|--:|--:|
| `ontology_default_on` | 0.00 | 1.00 | +1.00 |
| `space_scoped_fact_leg` | 1.00 | 1.00 | +0.00 |

## How to read this
- **content effect (B − B′)** is the memory-attributable number: how much more the agent adopts the evidence-backed choice with the REAL brief vs a generic decoy brief. Raw gain (B − A) also credits 'any authoritative in-context text.'
- R-adoption (not W-avoidance) is the metric; both/neither don't count as wins.
- verified cite = the item's measured token appeared verbatim (deterministic).