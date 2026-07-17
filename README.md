# kumiho-eval: Empirical Benchmark Suite for AI Cognitive Memory

Reproducible evaluation harness for **Graph-Native Cognitive Memory** systems.
Tests long-term conversational memory, belief revision compliance, and retrieval
quality against established benchmarks and formal postulates.

Built to evaluate [Kumiho](https://kumiho.io)'s dual-store architecture
(Redis working memory + Neo4j long-term graph) and generate paper-ready tables
for the AI Cognitive Memory paper.

## Latest Results

### LoCoMo-Plus (Level-2 Cognitive Memory)

**93.3% judge accuracy** on the full 401-entry LoCoMo-Plus benchmark —
the highest score we are aware of as of February 2026 under the LoCoMo-Plus
evaluation protocol (gpt-4o-mini cognitive judge, evidence-vs-prediction).
Outperforms Gemini-2.5-Pro (45.7%) by 47.6 points.
**Recall accuracy: 98.5%** — the architecture retrieves the correct memory
in all but 6 of 401 entries.

All baseline scores below are **reported** (not reproduced) from the
LoCoMo-Plus publication ([Li et al. 2026, arXiv 2602.10715](https://arxiv.org/abs/2602.10715),
Table 2). See [`baselines.json`](baselines.json) for exact provenance of
each number including table/figure references and evaluation protocol details.

| System | Model | LoCoMo-Plus Accuracy |
|--------|-------|---------------------|
| RAG (text-embedding-large) | text-embedding-large | 29.8% |
| Mem0 | Various | 41.4% |
| A-Mem | Various | 42.4% |
| SeCom | Various | 42.6% |
| GPT-4.1 | GPT-4.1 (full context) | 43.6% |
| Gemini-2.5-Pro | Gemini-2.5-Pro (1M ctx) | 45.7% |
| **Kumiho (GPT-4o-mini answer)** | **GPT-4o-mini** | **~88%** |
| **Kumiho (GPT-4o answer)** | **GPT-4o** | **93.3%** |

Total cost for the full 401-entry run: **~$14** using GPT-4o-mini for
consolidation, event extraction, prospective indexing, reformulation, and
judging. GPT-4o for answer generation only. Token-level cost breakdown
available in `manifest_*.json` after each run.

#### By Constraint Type

| Type | GPT-4o Accuracy | GPT-4o-mini Accuracy | Description |
|------|-----------------|----------------------|-------------|
| Causal | 96.0% | 96.0% | Cause-effect reasoning |
| State | 96.0% | 95.0% | State-change tracking |
| Value | 96.0% | ~89% | Value/belief inference |
| Goal | 85.0% | ~73% | Goal/intention inference |

#### Key Architectural Innovations

- **Prospective indexing** — generates future-facing implications at write time,
  bridging the cue-trigger semantic gap
- **Event extraction** — preserves causal chains that narrative summarization drops
- **Sibling relevance filtering** — embedding-based quality control over retrieved context
- **Model-decoupled architecture** — recall accuracy (98.5%) is invariant to answer
  model choice; end-to-end accuracy scales with model reasoning capacity

See [docs/AI_Cognitive_memory_LoCoMo_Plus_benchmark.md](../docs/AI_Cognitive_memory_LoCoMo_Plus_benchmark.md)
for the full analysis including failure taxonomy, model comparison, time-gap
breakdown, and paper integration notes.

### LoCoMo (Original QA Benchmark)

**0.531 overall F1** on all 1,986 questions across 10 conversations, on the
**shipped ontology stack** (kumiho-memory **0.10.1** + kumiho-server **v1.5.0**)
with the answer model **pinned to `gpt-4o-2024-08-06`** and every response's
`system_fingerprint` logged. Achieved in **summarized mode** (title + summary
only, no raw conversation artifacts), demonstrating that the graph's metadata
layer alone is sufficient to outperform systems with full-context retrieval.

> **⏱️ Read the numbers by era, not by absolute value.** The February 2026
> peak (**0.565**) and the 2026-07-08 v0.9.0 run (**0.564**) were measured with
> *that day's* `gpt-4o`, which is **not version-stable** — the same recalled
> contexts, re-answered on later `gpt-4o` snapshots, scored **−0.056** across
> every snapshot we tested (frozen-context experiment). Those peaks are
> therefore **historical, not reproducible** on current models, and are **not
> directly comparable** to a run made on a different date. The **0.531** headline
> is the number a reader can reproduce *today* on a pinned model.
>
> **The apples-to-apples "did the ontology help" test is same-corpus,
> same-era, same-pins:** ontology **ON = 0.531** vs an **OFF-era control
> (0.521)** measured the same day on the same model → **+0.010** from the
> write-time ontology + graph-augmented recall. And the retrieval layer's
> **multi-hop F1 (0.361) beats the February record (0.355)** *despite* the
> weaker answer model — recall improved even as the answer model regressed.
> The recall layer is what Kumiho controls; end-to-end F1 additionally scales
> with whatever answer model the application chooses.
>
> **LoCoMo-Plus crown (93.3%) is unaffected** — a separate external benchmark
> ([arXiv 2602.10715](https://arxiv.org/abs/2602.10715)); see above.

The official LoCoMo evaluation metric is **token-level F1 with Porter stemming**
([Maharana et al. 2024](https://arxiv.org/abs/2402.17753), `evaluation.py`).
Many competing systems report LLM-as-judge accuracy instead, which inflates
scores by 1.5–2× and is not directly comparable. The table below uses **F1
only**.

All baseline F1 scores are sourced from the Mem0 research paper
([Chhablani et al. 2025, arXiv 2504.19413](https://arxiv.org/abs/2504.19413))
and Memobase's published evaluation
([memodb-io/memobase](https://github.com/memodb-io/memobase/blob/main/docs/experiments/locomo-benchmark/README.md)).

| System | Single-Hop | Multi-Hop | Temporal | Open-Domain | Overall F1 | Source |
| ------ | ---------- | --------- | -------- | ----------- | ---------- | ------ |
| Zep | 0.357 | 0.194 | 0.420 | 0.496 | — | arXiv 2504.19413 |
| OpenAI Memory | — | — | — | — | ~0.343 | arXiv 2504.19413 |
| Mem0 | 0.387 | 0.286 | 0.489 | 0.477 | ~0.40 | arXiv 2504.19413 |
| Mem0-Graph | 0.381 | 0.243 | 0.516 | 0.493 | ~0.40 | arXiv 2504.19413 |
| Memobase | 0.463 | 0.229 | 0.642 | 0.516 | — | GitHub |
| Kumiho (Feb 2026, cosine) <sup>†</sup> | 0.462 | 0.355 | 0.533 | 0.290 | 0.565 | This work |
| Kumiho (v0.9.0, 2026-07-08) <sup>†</sup> | 0.449 | 0.393 | 0.530 | 0.313 | 0.564 | This work |
| **Kumiho** (0.10.1 ontology, 2026-07-11, pinned) | **0.424** | **0.361** | **0.457** | **0.248** | **0.531** | This work |

<sup>†</sup> *Measured on an unpinned, since-drifted `gpt-4o`; historical peaks,
not reproducible on current snapshots (see era note above). The bold row is the
reproducible-today figure on `gpt-4o-2024-08-06`.*

*Kumiho's overall includes the adversarial category (0.955 F1, n=446), which
most baselines do not report separately. Excluding adversarial, Kumiho's F1
across the four standard categories is **0.408**.*

#### Per-Category Breakdown

| Category | Count | F1 | same-era OFF control |
| -------- | ----: | --: | --: |
| Single-hop | 841 | 0.424 | 0.401 |
| Multi-hop | 282 | **0.361** | 0.353 |
| Temporal | 321 | 0.457 | 0.471 |
| Open-domain | 96 | 0.248 | 0.232 |
| Adversarial | 446 | 0.955 | 0.951 |
| **Overall** | **1,986** | **0.531** | **0.521** |

<sub>kumiho-memory 0.10.1 (ontology ON by default) + kumiho-server v1.5.0, full
10-conversation run, token-F1, answer model pinned to `gpt-4o-2024-08-06`
(`system_fingerprint` logged per response), summarized mode. "Same-era OFF
control" = `KUMIHO_MEMORY_ONTOLOGY=0` on the same corpus, same day, same pins
(the gate-v2 control) — the like-for-like measure of the ontology's effect.
Multi-hop (0.361) exceeds the February peak (0.355) despite the weaker pinned
answer model. February 2026 (0.565) and v0.9.0 (0.564) retained above as
drift-era peaks.</sub>

Run configuration: `--recall-mode summarized --recall-limit 3 --context-top-k 5 --recall-candidate-multiplier 3 --answer-model gpt-4o-2024-08-06 --no-judge --graph-augmented (default, ontology on)`

---

## Benchmarks

### Conversational Memory Benchmarks

| Benchmark | Focus | Metric | Source |
|-----------|-------|--------|--------|
| **LoCoMo** | Long conversation QA (10 conversations, ~2,000 QA pairs across 5 categories) | Token-F1 (official) | [Maharana et al. 2024](https://arxiv.org/abs/2402.17753) |
| **LoCoMo-Plus** | Level-2 cognitive memory (401 entries, 4 constraint types, cue-trigger semantic disconnect) | LLM Cognitive Judge Accuracy | [arXiv 2602.10715](https://arxiv.org/abs/2602.10715) |
| **LongMemEval** | 5 core memory abilities (500 questions, multi-session, temporal) | Accuracy across ability categories | [ICLR 2025](https://arxiv.org/abs/2410.10813) |
| **MemoryAgentBench** | Agent competency (action recall, TTL, LRU, single/multi-hop CR) | Per-competency accuracy | [MemoryAgentBench](https://github.com/MemoryAgentBench) |

### AGM Belief Revision Compliance

Tests whether the memory system satisfies the formal AGM postulates
(Alchourron, Gardenfors, Makinson 1985) and Hansson's belief base postulates
operationally on the graph:

| Postulate | What It Tests |
|-----------|---------------|
| K\*2 (Success) | After revision by A, A is in the belief state |
| K\*3 (Inclusion) | Revision adds only A and preserves survivors |
| K\*4 (Vacuity) | Non-conflicting info expands without supersession |
| K\*5 (Consistency) | Revised belief state contains no contradictions |
| K\*6 (Extensionality) | Equivalent inputs produce equivalent states |
| Relevance | Only relevant beliefs affected by contraction |
| Core-Retainment | Removed beliefs contributed to inconsistency |

49 scenarios across 5 categories per postulate: `simple`, `multi_item`,
`chain`, `temporal`, `adversarial`.

## Setup

### Prerequisites

- Python 3.11+
- A [Kumiho](https://kumiho.io) account
- OpenAI API key (for answer generation and LLM-as-Judge)

### Install

```bash
git clone --recurse-submodules https://github.com/kumihoclouds/kumiho-benchmarks.git
cd kumiho-benchmarks
pip install -r kumiho_eval/requirements.txt
```

The benchmark datasets are included as git submodules (`locomo/`,
`LongMemEval/`, `MemoryAgentBench/`). The `--recurse-submodules` flag fetches
them automatically. If you already cloned without it:

```bash
git submodule update --init --recursive
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."
export KUMIHO_AUTH_TOKEN="your-kumiho-api-token"  # from kumiho.io dashboard
```

Or create a `.env.local` file in `kumiho_eval/`.

### Run without an API key (ChatGPT / Codex OAuth)

If you have a ChatGPT subscription and are logged in with the
[`codex`](https://github.com/openai/codex) CLI (`codex login`), you can run the
whole harness — answer generation, LLM-as-Judge, query reformulation, **and**
the kumiho-memory summarizer — through your subscription instead of a paid API
key. No `OPENAI_API_KEY` required.

`kumiho_eval/codex_proxy.py` is a small local OpenAI-compatible server that
translates `/v1/chat/completions` and `/v1/responses` into the ChatGPT backend
Responses API (`https://chatgpt.com/backend-api/codex/responses`), reusing the
OAuth token the `codex` CLI stored in `~/.codex/auth.json` (auto-refreshed).
Because the `openai` SDK honours `OPENAI_BASE_URL`, pointing it at the proxy
routes every LLM call through Codex without changing any harness code.

```bash
# 0. sanity-check the OAuth token works
python -m kumiho_eval.codex_proxy --self-test

# One-command runner (starts the proxy in-process, then runs the benchmark)
python -m kumiho_eval.run_codex --locomo --max-samples 1 \
    --recall-mode summarized --recall-limit 3 --graph-augmented

# ...or run the proxy and benchmark separately:
python -m kumiho_eval.codex_proxy --port 8123 &
export OPENAI_BASE_URL=http://127.0.0.1:8123/v1
export OPENAI_API_KEY=codex-oauth      # any non-empty placeholder
python -m kumiho_eval.run_benchmarks --locomo --max-samples 1
```

| Env var | Default | Description |
|---------|---------|-------------|
| `CODEX_MODEL` | `gpt-5.5` | Codex model to use (e.g. `gpt-5.4-mini` for speed). `gpt-4o*` requests are auto-mapped to this. |
| `CODEX_REASONING_EFFORT` | `low` | `none`/`low`/`medium`/`high`/`xhigh` (`minimal` is auto-clamped to `low`). |
| `CODEX_HOME` | `~/.codex` | Location of the `codex` CLI `auth.json`. |

**Limits.** ChatGPT-subscription Codex enforces per-plan rate limits (Plus <
Pro); large runs will hit 429s — the harness already retries with exponential
backoff. There is no embeddings endpoint, so client-side embedding features are
off by default under `run_codex` (sibling similarity threshold 0, no two-pass
rerank); core recall still embeds server-side on the Kumiho tenant. Runs made
this way use a reasoning LLM over your subscription, **not** the published
`gpt-4o` harness — label results accordingly.

## Usage

### Unified Runner

```bash
# Run all Tier 1 benchmarks
python -m kumiho_eval.run_benchmarks --all

# Run individual benchmarks
python -m kumiho_eval.run_benchmarks --locomo
python -m kumiho_eval.run_benchmarks --locomo-plus
python -m kumiho_eval.run_benchmarks --longmemeval
python -m kumiho_eval.run_benchmarks --mab

# Quick smoke test (1 sample each)
python -m kumiho_eval.run_benchmarks --all --max-samples 1

# Custom models
python -m kumiho_eval.run_benchmarks --all --answer-model gpt-4o --judge-model gpt-4o

# Disable graph-augmented recall (on by default)
python -m kumiho_eval.run_benchmarks --locomo-plus --no-graph-augmented

# Compare full vs summarized recall modes
python -m kumiho_eval.run_benchmarks --all --dual-mode

# Run only AGM compliance (Tier 3)
python -m kumiho_eval.run_benchmarks --agm

# Run everything
python -m kumiho_eval.run_benchmarks --all --agm
```

### Unified Runner CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--all` | | Run all Tier 1 benchmarks |
| `--locomo` | | Run LoCoMo benchmark |
| `--locomo-plus` | | Run LoCoMo-Plus cognitive memory benchmark |
| `--longmemeval` | | Run LongMemEval benchmark |
| `--mab` | | Run MemoryAgentBench |
| `--agm` | | Run AGM compliance evaluation (Tier 3) |
| `--output` | `./results` | Output directory |
| `--max-samples` | all | Limit samples per benchmark |
| `--answer-model` | `gpt-4o` | Model for answer generation |
| `--judge-model` | `gpt-4o` | Model for LLM-as-Judge evaluation |
| `--recall-limit` | `10` | Max memories recalled per query |
| `--recall-mode` | `full` | `full` (artifact content) or `summarized` (title+summary) |
| `--dual-mode` | | Run both full and summarized, then compare |
| `--no-graph-augmented` | | Disable graph-augmented recall (on by default) |
| `--decompose-relations` | off | Opt-in relation- + belief-change decomposition write stage (LoCoMo only; see [Relation Decomposition](#relation-decomposition-opt-in)) |
| `--project` | `benchmark-eval` | Kumiho project name prefix |
| `-v` | | Verbose logging |

### Standalone Evaluations

Each benchmark can also be run directly with finer-grained control:

```bash
# LoCoMo-Plus (recommended: summarized mode, graph-augmented is on by default)
python -m kumiho_eval.locomo_plus_eval \
  --concurrency 16 \
  --entry-concurrency 4 \
  --recall-mode summarized \
  --project benchmark-locomo-plus

# LoCoMo (original)
python -m kumiho_eval.locomo_eval \
  --concurrency 4 \
  --recall-mode full \
  --project benchmark-locomo

# LongMemEval
python -m kumiho_eval.longmemeval_eval \
  --variant s \
  --concurrency 4 \
  --project benchmark-longmemeval

# MemoryAgentBench
python -m kumiho_eval.memoryagentbench_eval \
  --splits AR,TTL,LRU,CR \
  --project benchmark-mab

# AGM compliance
python -m kumiho_eval.agm_compliance_eval [--max-scenarios N] [--output DIR]
```

### Standalone CLI Options

#### locomo_plus_eval.py

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | auto | Path to locomo_plus.json |
| `--base-data` | auto | Path to locomo10.json |
| `--concurrency` | `4` | Max parallel session ingestions |
| `--entry-concurrency` | `1` | Max entries processed in parallel |
| `--no-graph-augmented` | | Disable graph-augmented recall (on by default) |
| `--recall-mode` | `full` | `full` or `summarized` |
| `--recall-limit` | `10` | Max memories recalled per query |
| `--answer-model` | `gpt-4o` | Model for answer generation |
| `--judge-model` | `gpt-4o-mini` | Model for cognitive judge |
| `--project` | `benchmark-locomo-plus` | Kumiho project name |
| `--max-samples` | all | Limit entries |
| `--no-resume` | | Start fresh (ignore checkpoint) |

#### locomo_eval.py

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | auto | Path to locomo10.json |
| `--concurrency` | `4` | Max parallel session ingestions per conversation |
| `--recall-mode` | `full` | `full` or `summarized` |
| `--recall-limit` | `10` | Max memories recalled |
| `--answer-model` | `gpt-4o` | Model for answer generation |
| `--judge-model` | `gpt-4o` | Model for LLM judge |
| `--no-judge` | | Skip LLM judge (F1 only) |
| `--project` | `benchmark-locomo` | Kumiho project name |
| `--max-samples` | all | Limit conversations |
| `--no-resume` | | Start fresh (ignore checkpoint) |
| `--decompose-relations` | off | Opt-in relation- + belief-change decomposition write stage (see [Relation Decomposition](#relation-decomposition-opt-in)); also set by `KUMIHO_EVAL_DECOMPOSE_RELATIONS=1`. Relation edges need kumiho-memory>=0.18.0; belief-change (SUPERSEDES/CONTRADICTS) edges need >=0.19.0 (older SDKs ignore them gracefully) |

#### longmemeval_eval.py

| Flag | Default | Description |
|------|---------|-------------|
| `--variant` | `s` | Dataset variant: `s` (small), `m` (medium), `oracle` |
| `--data-dir` | auto | Data directory override |
| `--concurrency` | `4` | Max parallel session ingestions |
| `--recall-mode` | `full` | `full` or `summarized` |
| `--recall-limit` | `10` | Max memories recalled |
| `--answer-model` | `gpt-4o` | Model for answer generation |
| `--judge-model` | `gpt-4o` | Model for LLM judge |
| `--project` | `benchmark-longmemeval` | Kumiho project name |
| `--max-samples` | all | Limit questions |
| `--no-resume` | | Start fresh (ignore checkpoint) |

#### memoryagentbench_eval.py

| Flag | Default | Description |
|------|---------|-------------|
| `--splits` | `AR,TTL,LRU,CR` | Comma-separated competency splits |
| `--chunk-size` | `16384` | Context chunk size (chars) for ingestion |
| `--recall-mode` | `full` | `full` or `summarized` |
| `--recall-limit` | `10` | Max memories recalled |
| `--answer-model` | `gpt-4o` | Model for answer generation |
| `--judge-model` | `gpt-4o` | Model for LLM judge |
| `--project` | `benchmark-mab` | Kumiho project name |
| `--max-samples` | all | Limit samples per split |
| `--no-resume` | | Start fresh (ignore checkpoint) |

## Output

Each run produces results in the output directory with checkpoint/resume support:

```
results/
  locomo/
    _checkpoint.jsonl                             # Resume checkpoint
    all_results.json                              # Per-question results
    metrics.json                                  # Aggregate metrics
  locomo_plus/
    _checkpoint.jsonl                             # Resume checkpoint
    all_results.json                              # Per-entry results
    metrics.json                                  # Per relation type + time gap breakdown
  longmemeval/
    _checkpoint.jsonl
    all_results.json
    metrics.json
    hypotheses.jsonl                              # Compatible with official evaluate_qa.py
  mab/
    _checkpoint.jsonl
    all_results.json
    metrics.json
    AR_results.json / TTL_results.json / ...      # Per-competency results
  agm/
    agm_report_TIMESTAMP.json                     # Full compliance report
    agm_compliance_matrix.txt                     # Postulate x category matrix
    agm_latex_table.tex                           # Paper-ready LaTeX table
  tier1_metrics_TIMESTAMP.json                    # All Tier 1 combined
  manifest_TIMESTAMP.json                         # Run manifest (git SHAs, config, prompt hashes)
  paper_tables_TIMESTAMP.tex                      # LaTeX comparison vs baselines
```

All evaluation scripts support checkpoint/resume by default. If a run is
interrupted, re-run the same command to pick up where it left off. Use
`--no-resume` to start fresh.

### Recall Modes

The suite supports two recall modes that test different memory architectures:

- **`full`** — Recalls complete artifact content from BYO-storage. Lossless retrieval, higher token cost.
- **`summarized`** — Recalls only title + summary from the cloud graph. Lossy but lightweight.

Use `--dual-mode` to run both and quantify the accuracy delta — this is a key
result for the paper's BYO-storage contribution.

### Relation Decomposition (opt-in)

`--decompose-relations` (on `locomo_eval` and `run_benchmarks`; or
`KUMIHO_EVAL_DECOMPOSE_RELATIONS=1`, default **OFF**) adds one extra LLM call
after each session consolidation that extracts a lean decomposition (up to
10 entities / 10 facts / 10 relations / 10 supersedes / 10 contradicts,
JSON-constrained) from the **consolidated summary** — never the raw
transcript — and writes entity→entity relation edges through
`kumiho_memory.ontology.decompose_and_link_agent`. The
stage builds its own OpenAI-compatible client (it honors `OPENAI_BASE_URL`)
with the run's configured summarizer model (`KUMIHO_LLM_MODEL`, default
`gpt-4o-mini`) and the summarizer's key-resolution chain
(`openai_api_key` → `anthropic_api_key` → `KUMIHO_LLM_API_KEY` →
`OPENAI_API_KEY`). Its tokens are accounted under the `decompose_relations`
phase; the flag is recorded in the run manifest, and standalone `locomo_eval`
runs are self-auditing — `metrics.json` carries `decompose_relations`,
`decompose_relations_token_usage`, and `decompose_relations_write_stats`
(edges actually written; a run that wrote zero relation edges — or zero
belief-change edges — logs a loud null-gate warning, one line each).

**Belief-change extraction (kumiho-memory ≥ 0.19.0).** The same call also asks
the model for two belief-update lists and passes them through the *same*
decomposition dict:

- `supersedes`: a NEW fact REPLACES a prior one (job / home / plan / status
  changed) — `{"statement": <current fact>, "replaces": <prior fact>}`.
- `contradicts`: a NEW fact CONFLICTS with an earlier one without cleanly
  replacing it (a correction or reversal) —
  `{"statement": <current fact>, "conflicts_with": <earlier fact>}`.

The model only sees THIS session's summary, so it is instructed to emit a
belief change only when the summary itself signals one ("no longer", "used
to", "now", "instead of", "changed to"); the `replaces`/`conflicts_with`
target is the implied prior statement phrased as a standalone fact. Referential
integrity, like relations: **both** the new statement and its target must also
appear in `facts` (the SDK resolves each to a fact in the same call and drops
any belief change it can't anchor). The SDK lands these as `SUPERSEDES` /
`CONTRADICTS` edges with `basis=agent`; the heuristic lexical-overlap
`SUPERSEDES` still runs as a fallback and yields to agent declarations. On an
older kumiho-memory (< 0.19.0) the extra keys are simply ignored server-side,
so the stage degrades gracefully — relation edges still land and the
belief-change counts stay 0. Belief-change edges are what make the final
`0.19.0` LoCoMo run actually exercise kumiho-memory's `CONTRADICTS` read path.

**Why it exists.** The product's consolidation summarizer schema deliberately
omits relations — measured, adding relation fields to the summary regressed
`based_on` base recall. In production the relation edges are written by an
in-loop agent that calls `decompose_and_link_agent` after consolidation, so this
stage *simulates that in-loop agent* rather than changing the summarizer.

**Pair-run usage (important).** The read-side flag under test is
`KUMIHO_MEMORY_RELATION_TRAVERSAL` (`GraphAugmentationConfig.relation_traversal`,
default OFF). For a valid `relation_traversal` OFF-vs-ON comparison, turn
`--decompose-relations` **ON in BOTH arms** — the relation edges are shared
write-side state; only the *read* flag should differ between the two arms.
Enabling decomposition in just the ON arm would confound the write and read
changes. This ON-in-both-arms rule is unchanged for belief-change edges — they
are shared write-side state too. Relation edges require **kumiho-memory >=
0.18.0**; belief-change (SUPERSEDES/CONTRADICTS) edges require **>= 0.19.0**.

## Architecture

```text
kumiho_eval/
├── run_benchmarks.py          # Unified CLI runner
├── common.py                  # KumihoMemoryAdapter, BenchmarkConfig, metrics
├── locomo_eval.py             # LoCoMo benchmark (Tier 1)
├── locomo_plus_eval.py        # LoCoMo-Plus cognitive memory benchmark
├── longmemeval_eval.py        # LongMemEval benchmark (Tier 1)
├── memoryagentbench_eval.py   # MemoryAgentBench benchmark (Tier 1)
├── agm_compliance_eval.py     # AGM belief revision compliance (Tier 3)
└── requirements.txt

locomo/                        # LoCoMo + LoCoMo-Plus dataset (submodule)
LongMemEval/                   # LongMemEval dataset (submodule)
MemoryAgentBench/              # MemoryAgentBench dataset (submodule)
```

The `KumihoMemoryAdapter` in `common.py` wraps the Kumiho SDK and provides
a standard interface for all benchmarks:

1. **`create_eval_space()`** — Isolated project + space per conversation
2. **`ingest_session()`** — Feed conversation history through the memory manager
3. **`consolidate()`** — Trigger summarization and long-term storage
4. **`recall()`** — Query long-term memory (full or summarized mode)
5. **`cleanup()`** — Remove evaluation data after the run

## Contributing

Contributions welcome. To add a new benchmark:

1. Create `kumiho_eval/<benchmark>_eval.py` with an async evaluation function
2. Use `KumihoMemoryAdapter` from `common.py` for memory operations
3. Return results as `list[EvalResult]` with standard metric fields
4. Wire into `run_benchmarks.py` with a new CLI flag
5. Add reference scores to `REFERENCE_SCORES` dict

## Citation

If you use this benchmark suite in your research, please cite:

```bibtex
@software{kumiho_eval_2026,
  title   = {kumiho-eval: Empirical Benchmark Suite for AI Cognitive Memory},
  author  = {Kumiho Inc.},
  year    = {2026},
  url     = {https://github.com/kumihoclouds/kumiho-benchmarks},
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
