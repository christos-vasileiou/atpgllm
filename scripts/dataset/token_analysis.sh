#!/bin/bash
#SBATCH --job-name=token_analysis
#SBATCH --output=jobs/token_analysis_%j.out
#SBATCH --error=jobs/token_analysis_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --partition=normal

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

python "$SCRIPT_DIR/dataset_token_analysis.py"
