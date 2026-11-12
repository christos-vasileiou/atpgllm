#!/bin/bash
#SBATCH --job-name=retest_seq_labels
#SBATCH --partition=dev
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --output=runs/bridges_retest_seq/labels_%j.out
# Re-test label table for v5 (combinational and full-scan circuits), CPU only.
#
#   cd /work/cxv200006/atpg_retest_seq/libatpgllm && mkdir -p runs/bridges_retest_seq
#   sbatch scripts/train/juno_retest_seq_labels.sbatch
#
# v5 holds the v4 circuits as data-*.parquet and the sequential ones as
# seq-*.parquet, per split, so one table covers both kinds with one split.
set -euo pipefail
_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$_REPO_ROOT"
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
export PYTHONPATH="$_REPO_ROOT" FAULT_SIM_BACKEND=fast
export DATA_PREPROCESSING_ROOT="${DATA_PREPROCESSING_ROOT:-/work/cxv200006/transformers_atpg/data_preprocessing_seq-v5}"
DATA="${DATA:-/work/cxv200006/transformers_atpg/data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v5}"
LABELS="${LABELS:-../data/freeset/retest_labels_v5}"
echo "[labels] $(hostname) start=$(date -Is) data=$DATA out=$LABELS"
head -1 PROVENANCE.txt
python -m atpgllm.graph.scripts.build_retest_labels --data "$DATA" --out "$LABELS" --splits ${SPLITS:-validation train}
ls -la "$LABELS"
echo "[labels] done $(date -Is)"
