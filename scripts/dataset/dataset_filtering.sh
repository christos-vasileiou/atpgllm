#!/bin/bash
#SBATCH --job-name=asap7-ds-filter
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=jobs/asap7-filter-%j.out
#SBATCH --error=jobs/asap7-filter-%j.err

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
mkdir -p "$REPO_ROOT/jobs"
cd "$REPO_ROOT"

if [ -f "/work/cxv200006/myenv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source /work/cxv200006/myenv/bin/activate
elif [ -f "/proj/trela/christos/myenv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source /proj/trela/christos/myenv/bin/activate
fi

python "$SCRIPT_DIR/dataset_filtering.py"
