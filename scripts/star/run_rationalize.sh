#!/bin/bash
#SBATCH --job-name=star_rationalize
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=jobs/star_%j.out
# Hinted rationalization on one GPU with a managed TetraMAX service.
#
#   sbatch scripts/star/run_rationalize.sh --adapter <sft checkpoint> --dataset <v4 dir> --out runs/star_pilot ...
#   ./scripts/star/run_rationalize.sh ...            (no Slurm)
#
# Arguments go to scripts/star/rationalize.py. PYTHON_VENV, HF_HOME and
# HF_HUB_OFFLINE are taken from the environment when set.
set -euo pipefail
_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$_REPO_ROOT"
mkdir -p jobs
export PYTHONPATH="$_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [ -n "${PYTHON_VENV:-}" ]; then
    # shellcheck source=/dev/null
    source "$PYTHON_VENV/bin/activate"
fi
export FAULT_SIM_BACKEND=tetramax TMAX_MANAGE_SERVICE=True
export TMAX_MAX_CONCURRENT="${TMAX_MAX_CONCURRENT:-16}" TMAX_TIMEOUT_S="${TMAX_TIMEOUT_S:-120}"
export TMAX_ACQUIRE_TIMEOUT_S="${TMAX_ACQUIRE_TIMEOUT_S:-120}"
# shellcheck source=../train/_tetramax_lifecycle.sh
source scripts/train/_tetramax_lifecycle.sh
trap stop_tetramax_service EXIT
start_tetramax_service
preflight_tetramax_simulation
echo "[star] $(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-all} commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain | grep -qv '^??' && echo yes || echo no)"
python scripts/star/rationalize.py "$@"
