#!/bin/bash
#SBATCH --job-name=bridges_retest_seq
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=16:00:00
#SBATCH --output=runs/bridges_retest_seq/slurm_%j.out
# Graph model for both kinds of circuit: the R3 recipe on the v5 label table
# (combinational and full-scan), started from the R3 weights, with state cells.
#
#   cd /work/cxv200006/atpg_retest_seq/libatpgllm && mkdir -p runs/bridges_retest_seq
#   sbatch --dependency=afterok:<labels job> scripts/train/juno_bridges_retest_seq.sbatch
#
# Steps, in order:
#   smoke     two tiny steps on both kinds, so a broken setup fails in minutes
#   baseline  the R3 weights on the evaluation set of the run, before any step
#             (learning rate 0), split by kind and by state bits
#   train     R3 recipe, --state-cells, --init R3
#   check     graph-only detection on the same validation problems, R3 and new
# The tree is an rsync of the bridges-retest worktree: PROVENANCE.txt names the
# commit. Loading the train labels needs tens of GB of host memory.
set -euo pipefail
_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$_REPO_ROOT"
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
export PYTHONPATH="$_REPO_ROOT" HF_HUB_OFFLINE=1 FAULT_SIM_BACKEND=fast
export DATA_PREPROCESSING_ROOT="${DATA_PREPROCESSING_ROOT:-/work/cxv200006/transformers_atpg/data_preprocessing_seq-v5}"
LABELS="${LABELS:-../data/freeset/retest_labels_v5}"
R3="${R3:-/work/cxv200006/atpg_retest/libatpgllm/runs/bridges_retest/r3/best.pt}"
OUT="${OUT:-runs/bridges_retest_seq}"
STEPS="${STEPS:-20000}"
MAX_EVAL="${MAX_EVAL:-4000}"
echo "[retest-seq] $(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
head -1 PROVENANCE.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p "$OUT"

SMOKE="$OUT/smoke_$SLURM_JOB_ID"
python -m atpgllm.graph.scripts.train_retest --labels "$LABELS" --train-split validation --eval-split validation \
  --max-train 64 --max-eval 64 --steps 2 --batch-size 8 --node-dim 64 --gin-layers 2 --q-hidden 64 --q-layers 2 \
  --eval-every 2 --state-cells --text-weight 1 --out "$SMOKE/r3"
python -m atpgllm.graph.scripts.retest_graph_only --checkpoint "$SMOKE/r3/best.pt" --labels "$LABELS" \
  --split validation --max-problems 32 --out "$SMOKE/r4"
echo "[retest-seq] smoke passed $(date -Is)"

# Same seed and --max-eval as the run below, so both score the same graphs.
python -m atpgllm.graph.scripts.train_retest --labels "$LABELS" --train-split validation --eval-split validation \
  --max-train 64 --max-eval "$MAX_EVAL" --steps 1 --lr 0 --eval-every 1 --state-cells --init "$R3" \
  --out "$OUT/baseline_r3"
echo "[retest-seq] baseline done $(date -Is)"

python -m atpgllm.graph.scripts.train_retest --labels "$LABELS" --steps "$STEPS" --batch-size 64 \
  --max-eval "$MAX_EVAL" --eval-every 1000 --text-weight 1 --state-cells --init "$R3" --out "$OUT/r3s"
echo "[retest-seq] train done $(date -Is)"

for pair in "r3:$R3" "r3s:$OUT/r3s/best.pt"; do
  python -m atpgllm.graph.scripts.retest_graph_only --checkpoint "${pair#*:}" --labels "$LABELS" \
    --split validation --max-problems 6000 --out "$OUT/r4_${pair%%:*}"
done
echo "[retest-seq] done $(date -Is)"
