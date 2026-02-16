# kumiho-eval: Empirical Benchmark Suite for AI Cognitive Memory

Reproducible evaluation harness for **Graph-Native Cognitive Memory** systems.
Tests long-term conversational memory, belief revision compliance, and retrieval
quality against established benchmarks and formal postulates.

Built to evaluate [Kumiho Cloud](https://kumiho.cloud)'s dual-store architecture
(Redis working memory + Neo4j long-term graph) and generate paper-ready tables
for the AI Cognitive Memory paper.

## Benchmarks

### Tier 1 — Conversational Memory Benchmarks

| Benchmark | Focus | Metric | Source |
|-----------|-------|--------|--------|
| **LoCoMo** | Long conversation QA (10 conversations, ~200 QA pairs across 5 categories) | Token-F1, LLM-as-Judge Accuracy | [Maharana et al. 2024](https://arxiv.org/abs/2402.14562) |
| **LongMemEval** | 5 core memory abilities (500 questions, multi-session, temporal) | Accuracy across ability categories | [ICLR 2025](https://arxiv.org/abs/2410.10813) |
| **MemoryAgentBench** | Agent competency (action recall, TTL, LRU, single/multi-hop CR) | Per-competency accuracy | [MemoryAgentBench](https://github.com/MemoryAgentBench) |

### Tier 3 — AGM Belief Revision Compliance

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
- A running [Kumiho Cloud](https://kumiho.cloud) instance (or local dev server)
- OpenAI API key (for answer generation and LLM-as-Judge)

### Install

```bash
git clone --recurse-submodules https://github.com/kumihoclouds/kumiho-benchmarks.git
cd kumiho-benchmarks
pip install -r kumiho_eval/requirements.txt
```

The three Tier 1 benchmark datasets are included as git submodules (`locomo/`,
`LongMemEval/`, `MemoryAgentBench/`). The `--recurse-submodules` flag fetches
them automatically. If you already cloned without it:

```bash
git submodule update --init --recursive
```

### Environment Variables

```bash
# Required
export OPENAI_API_KEY="sk-..."
export KUMIHO_AUTH_TOKEN="your-kumiho-api-token"

# Optional (defaults shown)
export KUMIHO_ENDPOINT=""                # auto-discover if empty
export KUMIHO_UPSTASH_REDIS_URL=""       # Redis for working memory
export KUMIHO_LLM_API_KEY=""             # LLM for memory summarization
```

Or create a `.env.local` file in `kumiho_eval/`.

## Usage

### Unified Runner

```bash
# Run all Tier 1 benchmarks
python -m kumiho_eval.run_benchmarks --all

# Run individual benchmarks
python -m kumiho_eval.run_benchmarks --locomo
python -m kumiho_eval.run_benchmarks --longmemeval
python -m kumiho_eval.run_benchmarks --mab

# Quick smoke test (1 sample each)
python -m kumiho_eval.run_benchmarks --all --max-samples 1

# Custom models
python -m kumiho_eval.run_benchmarks --all --answer-model gpt-4o --judge-model gpt-4o

# Compare full vs summarized recall modes
python -m kumiho_eval.run_benchmarks --all --dual-mode

# Run only AGM compliance (Tier 3)
python -m kumiho_eval.run_benchmarks --agm

# Run everything
python -m kumiho_eval.run_benchmarks --all --agm
```

### CLI Options

| Flag | Default | Description |
|------|---------|-------------|
| `--all` | | Run all Tier 1 benchmarks |
| `--locomo` | | Run LoCoMo benchmark |
| `--longmemeval` | | Run LongMemEval benchmark |
| `--mab` | | Run MemoryAgentBench |
| `--agm` | | Run AGM compliance evaluation (Tier 3) |
| `--output` | `./results` | Output directory for metrics and reports |
| `--max-samples` | all | Limit samples per benchmark |
| `--answer-model` | `gpt-4o` | Model for answer generation |
| `--judge-model` | `gpt-4o` | Model for LLM-as-Judge evaluation |
| `--recall-limit` | `10` | Max memories recalled per query |
| `--recall-mode` | `full` | `full` (artifact content) or `summarized` (title+summary) |
| `--dual-mode` | | Run twice: full recall then summarized recall |
| `--project` | `benchmark-eval` | Kumiho project name prefix |
| `-v` | | Verbose logging |

### Standalone AGM Compliance

```bash
python -m kumiho_eval.agm_compliance_eval [--max-scenarios N] [--output DIR]
```

## Output

Each run produces timestamped results in the output directory:

```
results/
  locomo/
    locomo_results_20260216T120000.json       # Per-question results
    locomo_metrics_20260216T120000.json        # Aggregate metrics
  longmemeval/
    longmemeval_results_20260216T120000.json
    longmemeval_metrics_20260216T120000.json
  mab/
    mab_results_20260216T120000.json
    mab_metrics_20260216T120000.json
  agm/
    agm_report_20260216T120000.json           # Full compliance report
    agm_compliance_matrix.txt                 # Postulate × category matrix
    agm_latex_table.tex                       # Paper-ready LaTeX table
  combined_metrics.json                       # All benchmarks in one file
  comparison_table.tex                        # LaTeX comparison vs baselines
```

### Recall Modes

The suite supports two recall modes that test different memory architectures:

- **`full`** — Recalls complete artifact content from BYO-storage. Lossless retrieval, higher token cost.
- **`summarized`** — Recalls only title + summary from the cloud graph. Lossy but lightweight.

Use `--dual-mode` to run both and quantify the accuracy delta — this is a key
result for the paper's BYO-storage contribution.

## Architecture

```
kumiho_eval/
├── run_benchmarks.py          # Unified CLI runner
├── common.py                  # KumihoMemoryAdapter, BenchmarkConfig, metrics
├── locomo_eval.py             # LoCoMo benchmark (Tier 1)
├── longmemeval_eval.py        # LongMemEval benchmark (Tier 1)
├── memoryagentbench_eval.py   # MemoryAgentBench benchmark (Tier 1)
├── agm_compliance_eval.py     # AGM belief revision compliance (Tier 3)
└── requirements.txt

locomo/                        # LoCoMo dataset (submodule)
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

## Reference Scores

The runner includes hard-coded reference scores from published systems for
paper comparison tables:

| System | LoCoMo Judge Acc. | Source |
|--------|-------------------|--------|
| MAGMA | 70.0% | arXiv 2601.03236 |
| Mem0 | 58.3% | arXiv 2504.19413 |
| Zep | 50.5% | arXiv 2504.19413 |
| A-Mem | 67.3% | arXiv 2504.01080 |
| ReadAgent | 56.0% | arXiv 2402.09727 |
| **Kumiho (full)** | **95.2%** | This work |

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
  author  = {Park, Youngbin},
  year    = {2026},
  url     = {https://github.com/kumihoclouds/kumiho-benchmarks},
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.
