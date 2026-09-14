#!/bin/bash
#SBATCH --job-name=sft_preflight
#SBATCH --output=jobs/sft_preflight_%j.out
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

set -euo pipefail
cd /work/cxv200006/transformers_atpg/atpgllm

source /work/cxv200006/myenv/bin/activate

# Per-job writable caches; tokenizer files are read from the shared saved run.
preflight_cache="${SLURM_TMPDIR:-/tmp}/atpg-sft-preflight-${SLURM_JOB_ID:-manual}"
export HF_DATASETS_CACHE="$preflight_cache/datasets"
export MPLCONFIGDIR="$preflight_cache/matplotlib"
export XDG_CACHE_HOME="$preflight_cache/cache"
export MPLBACKEND=Agg
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
mkdir -p "$HF_DATASETS_CACHE" "$MPLCONFIGDIR" "$XDG_CACHE_HOME"

echo "SFT dataset/tokenizer preflight on $(hostname), job ${SLURM_JOB_ID:-manual}"
exec /work/cxv200006/myenv/bin/python -u scripts/train/check_sft_dataset.py "$@"
