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

**0.533 overall F1** on all 1,986 questions across 10 conversations —
the highest score we are aware of on the official LoCoMo token-level F1 metric
as of February 2026.

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
| **Kumiho** | **0.423** | **0.314** | **0.493** | **0.262** | **0.533** | This work |

*Kumiho's overall includes adversarial category (0.966 F1, n=446) which most
baselines do not report separately. Excluding adversarial, Kumiho's F1 across
the four standard categories is 0.407.*

#### Per-Category Breakdown

| Category | Count | F1 |
| -------- | ----: | --: |
| Single-hop | 841 | 0.423 |
| Multi-hop | 282 | 0.314 |
| Temporal | 321 | 0.493 |
| Open-domain | 96 | 0.262 |
| Adversarial | 446 | 0.966 |
| **Overall** | **1,986** | **0.533** |

Run configuration: `--recall-mode summarized --recall-limit 3 --context-top-k 7 --no-judge`

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

# Enable graph-augmented recall (edge traversal) for LoCoMo-Plus
python -m kumiho_eval.run_benchmarks --locomo-plus --graph-augmented

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
| `--graph-augmented` | | Enable graph-augmented recall (edge traversal) |
| `--project` | `benchmark-eval` | Kumiho project name prefix |
| `-v` | | Verbose logging |

### Standalone Evaluations

Each benchmark can also be run directly with finer-grained control:

```bash
# LoCoMo-Plus (recommended: summarized + graph-augmented for best results)
python -m kumiho_eval.locomo_plus_eval \
  --concurrency 16 \
  --entry-concurrency 4 \
  --graph-augmented \
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
| `--graph-augmented` | | Enable graph-augmented recall (edge traversal) |
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
