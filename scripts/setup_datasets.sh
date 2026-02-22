#!/usr/bin/env bash
# setup_datasets.sh — Fetch and pin benchmark dataset submodules.
#
# Run from the kumiho-benchmarks root:
#   bash scripts/setup_datasets.sh
#
# This ensures all three dataset submodules are checked out at the exact
# commits used for published results.  Pinned SHAs are recorded here so
# that any reviewer can reproduce the evaluation environment exactly.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ---------------------------------------------------------------------------
# Pinned dataset commits (update these when re-running on new data)
# ---------------------------------------------------------------------------
LOCOMO_SHA="main"            # snap-research/locomo — locomo10.json
LONGMEMEVAL_SHA="main"       # xiaowu0162/LongMemEval
MEMORYAGENTBENCH_SHA="main"  # HUST-AI-HYZ/MemoryAgentBench
# LoCoMo-Plus is fetched from HuggingFace (xjtuleeyf/Locomo-Plus), not a submodule.

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
checkout_submodule() {
    local path="$1"
    local sha="$2"
    local url="$3"

    echo "--- Setting up $path (target: $sha) ---"

    if [ ! -d "$REPO_ROOT/$path/.git" ] && [ ! -f "$REPO_ROOT/$path/.git" ]; then
        echo "  Initializing submodule $path..."
        cd "$REPO_ROOT"
        git submodule update --init "$path"
    fi

    cd "$REPO_ROOT/$path"
    git fetch origin
    git checkout "$sha"
    actual_sha=$(git rev-parse HEAD)
    echo "  $path pinned at $actual_sha"
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
echo "=== Kumiho Benchmark Dataset Setup ==="
echo "Repository: $REPO_ROOT"
echo ""

checkout_submodule "locomo"          "$LOCOMO_SHA"          "https://github.com/snap-research/locomo.git"
checkout_submodule "LongMemEval"     "$LONGMEMEVAL_SHA"     "https://github.com/xiaowu0162/LongMemEval.git"
checkout_submodule "MemoryAgentBench" "$MEMORYAGENTBENCH_SHA" "https://github.com/HUST-AI-HYZ/MemoryAgentBench.git"

# LoCoMo-Plus: download from HuggingFace if not cached locally
LOCOMO_PLUS_PATH="$REPO_ROOT/locomo/data/locomo_plus.json"
if [ ! -f "$LOCOMO_PLUS_PATH" ]; then
    echo ""
    echo "--- Fetching LoCoMo-Plus dataset from HuggingFace ---"
    python3 -c "
from huggingface_hub import hf_hub_download
import shutil
p = hf_hub_download(repo_id='xjtuleeyf/Locomo-Plus', filename='locomo_plus.json', repo_type='dataset')
shutil.copy(p, '$LOCOMO_PLUS_PATH')
print(f'  Saved to $LOCOMO_PLUS_PATH')
" || echo "  WARNING: Could not download LoCoMo-Plus. Install huggingface_hub or download manually."
else
    echo ""
    echo "--- LoCoMo-Plus already cached at $LOCOMO_PLUS_PATH ---"
fi

# Print final status
echo ""
echo "=== Dataset Commit Manifest ==="
for sub in locomo LongMemEval MemoryAgentBench; do
    if [ -d "$REPO_ROOT/$sub/.git" ] || [ -f "$REPO_ROOT/$sub/.git" ]; then
        sha=$(cd "$REPO_ROOT/$sub" && git rev-parse HEAD)
        echo "  $sub: $sha"
    fi
done
echo ""
echo "Done. All datasets ready for evaluation."
