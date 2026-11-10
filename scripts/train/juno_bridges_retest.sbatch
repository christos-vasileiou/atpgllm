#!/bin/bash
#SBATCH --job-name=bridges_retest
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/bridges_retest/slurm_%j.out
# BRIDGES re-test, graph side (R2 to R4) on one juno GPU.
#
#   cd /work/cxv200006/atpg_retest/libatpgllm && mkdir -p runs/bridges_retest
#   sbatch scripts/train/juno_bridges_retest.sbatch
#
# The tree is an rsync of the uncommitted bridges-retest worktree, so there is
# no git here: PROVENANCE.txt and uncommitted.diff record the source commit and
# changes. A two-step CPU-sized smoke runs first, so a broken setup fails in
# minutes instead of after the queue wait. Loading the train labels peaks near
# 9 GB of host memory (measured on tamale3).
set -euo pipefail
_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$_REPO_ROOT"
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
export PYTHONPATH="$_REPO_ROOT" HF_HUB_OFFLINE=1 FAULT_SIM_BACKEND=fast
echo "[retest] $(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
head -1 PROVENANCE.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

LABELS=../data/freeset/retest_labels_v4
SMOKE=runs/bridges_retest/smoke_$SLURM_JOB_ID
S="--labels $LABELS --train-split validation --eval-split test --max-train 64 --max-eval 32 --steps 2
   --batch-size 8 --node-dim 64 --gin-layers 2 --q-hidden 64 --q-layers 2 --eval-every 2"
python -m atpgllm.graph.scripts.train_retest $S --query-weight 0 --collapse-weight 0 --select sp_mae --out $SMOKE/r2
python -m atpgllm.graph.scripts.train_retest $S --text-weight 1 --init $SMOKE/r2/best.pt --out $SMOKE/r3
python -m atpgllm.graph.scripts.retest_graph_only --checkpoint $SMOKE/r3/best.pt --labels $LABELS \
  --split validation --max-problems 16 --out $SMOKE/r4
echo "[retest] smoke passed $(date -Is)"

# Slurm already restricts the job to its card; pass that index through unchanged.
GPU="${CUDA_VISIBLE_DEVICES:-0}" OUT=runs/bridges_retest scripts/train/run_bridges_retest.sh
echo "[retest] done $(date -Is)"
