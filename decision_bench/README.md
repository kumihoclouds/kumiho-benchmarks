# decision_bench — the downstream Decision Memory benchmark

Every other Kumiho benchmark measures **recall** (did we retrieve the right
memory?). This one measures **behavior**: *does an agent that has Decision Memory
make a different, better engineering decision than one that doesn't?* That is the
one question the business case turns on, and nothing else in this repo answers it.

> **Claim under test:** when an agent faces a fork where the *plausible default is
> empirically wrong*, Decision Memory steers it to the evidence-backed choice —
> and the steering comes from the *specific captured evidence*, not from having
> any authoritative text in context.

## Result (v2, 2026-07-12)

Two independent frontier models, agent-under-test = **Claude Opus 4.8** (keyless)
and **gpt-5** (API); judge = a different, smaller model in each case (Haiku /
gpt-4o-mini), blind to which option is correct.

**Behavioral items (the headline). R = adopts the evidence-backed choice.**

| item | Claude A / B′ / B | Δ (B−B′) | gpt-5 A / B′ / B | Δ (B−B′) |
|---|---|--:|---|--:|
| `based_on_schema` | 0.00 / 0.00 / **1.00** | **+1.00** | 0.00 / 0.00 / **1.00** | **+1.00** |
| `sibling_reserve_on_top` | 0.00 / 0.00 / **1.00** | **+1.00** | 0.00 / 0.00 / **1.00** | **+1.00** |
| `per_subquery_ce` | 1.00 / 0.50 / 1.00 | +0.50 | 1.00 / 0.00 / 1.00 | +1.00 |
| `additive_partition` | 0.75 / 0.50 / 1.00 | +0.50 | 1.00 / 0.00 / 1.00 | +1.00 |
| `gate_about_edges` | 1.00 / 1.00 / 1.00 | 0.00 | 1.00 / 1.00 / 1.00 | 0.00 |
| `skip_hub_entities` | 1.00 / 1.00 / 1.00 | 0.00 | 1.00 / 1.00 / 1.00 | 0.00 |
| **mean** | .63 / .50 / **1.00** | **+0.50** | .67 / .33 / **1.00** | **+0.67** |

Three findings:

1. **With the real `why()` brief, R-adoption is 1.00 on every behavioral item, on
   both models.** The memory reliably produces the evidence-backed decision.
2. **The two clean-discriminating items (`based_on_schema`,
   `sibling_reserve_on_top`) replicate exactly across models** — the model always
   picks the wrong default alone (A=0) and with a generic decoy brief (B′=0), and
   always picks right with the real memory (B=1.0). Model-independent, +1.00.
3. **The control arm is validated on both, harder on gpt-5.** The *authority
   effect* (B′−A) is **negative** (−0.13 Claude, −0.33 gpt-5): a generic
   authoritative brief does **not** help. On gpt-5 the decoy actually dragged
   `per_subquery_ce` and `additive_partition` from a correct 1.00 down to 0.00,
   while the real memory held 1.00 — so the gain is the **specific captured
   evidence**, not "any in-context authority." This is the property v1 could not
   isolate.

**Cross-model divergence** (reported outside the headline): `fire_forget_entity`
is unknown to both models a priori (A=0) and memory helps (Claude +0.5, gpt-5
+1.0); `ontology_default_on` is a latest-not-stale item — Opus guessed the current
default, but gpt-5 guessed the *old* default and memory corrected it (+1.0).

## What makes this not recall-in-disguise

Each behavioral item is a real decision mined from `kumiho-SDKs` history into the
`dogfood-code-memory` graph, chosen so the **correct answer is wrong a priori** —
it contradicts the plausible default and is only knowable from a Kumiho-specific
measurement or incident. E.g. *"add an optional `based_on` provenance field?"* —
obvious **yes**; measured **no** (an emitted field shifts the summarizer's output
and weakens base recall). So avoiding the wrong choice is attributable to the
memory content, not general competence — and the decoy control arm proves it.

## Design

Three arms per behavioral item, judged **blind** to arm:

| arm | prompt | isolates |
|---|---|---|
| **A** baseline | the design task only | does the model reintroduce the wrong choice on its own? |
| **B′** decoy control | task + a *generic, number-free* prior-decision brief | how much is "any authoritative in-context text"? |
| **B** memory | task + the live `why()` brief | how much is the *specific* captured evidence? |

memory-attributable effect = **B − B′** (not B − A).

- **The judge is blind:** it sees only the bare actions (rationale and numbers
  stripped), in a deterministically-shuffled order as Option 1/2, on a *different*
  model than the agent. R-adoption (not W-avoidance) is the metric; both/neither
  never count as a win.
- **Verified citation:** a citation counts only when the item's measured token
  (e.g. `+0.031`, `-0.078`) appears verbatim — not merely when the judge says
  "cited."
- **Buckets:** only `behavioral` items form the headline. `general_knowledge`
  (a-priori best practice) and `recall` (internal-state) are reported separately.

## Running it

```bash
# 1) build the memory briefs from the live dogfood graph (needs CE on :9190)
python build_briefs.py

# 2a) keyless Claude leg — via Claude Code's Workflow tool (no API key):
#     Workflow({ scriptPath: ".../keyless_bench.mjs" })  # loader reads the files
# 2b) cross-model leg (GPT / any OpenAI-compatible endpoint):
OPENAI_API_KEY=... python run.py --model gpt-5 --judge-model gpt-4o-mini --trials 3
```

Outputs `report_v2.md` + `raw_results_v2.json`. Preflight first
(`run.py --only based_on_schema --trials 1`).

## Honest limitations (stated, not hidden)

- **Small n:** 6 behavioral items; only ~2–4 truly *discriminate* per model (the
  rest, the model already knows — reported, not counted as wins). This is a signal,
  not proof.
- **Single, non-blind corpus author** wrote the decisions, the options, and the
  signals. Selection was post-hoc from 31 candidates.
- The benchmark can fail: if `B − B′ ≈ 0`, the specific memory didn't change
  behavior, and the report says so per item.
- Tier: this is a **behavioral-effect** test (does memory change the decision),
  not proof of net outcome value at scale, retention, or team-loop value.

## History

v1 was built, then **failed a 4-lens adversarial review** (a superseded ground
truth, a non-blind judge, a near-tautological headline, and a-priori-derivable
items). v2 is the rebuild: control arm, blind judge, honest buckets, verified
citations, and cross-model runs. The review and its fixes are the reason the v2
numbers are trustworthy.

## Visual report

`report_v2.html` — a standalone, self-contained visual of the cross-model result (A/B'/B dot-plots per decision, Opus 4.8 vs gpt-5). Open it directly in a browser.
