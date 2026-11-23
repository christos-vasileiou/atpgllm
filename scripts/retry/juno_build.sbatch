#!/bin/bash
#SBATCH --job-name=retry_build
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=192G
#SBATCH --time=12:00:00
#SBATCH --output=jobs/retry_build_%j.out
#
# Retry SFT data on juno (CPU and TetraMAX, no GPU): every argument goes to
# scripts/retry/build_retry_sft.py, run under the managed TetraMAX service.
# v5 warm-up (synthetic failures from dataset rows, both circuit kinds):
#   sbatch scripts/retry/juno_build.sbatch --dataset ../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v5 \
#       --split train --per_group 600 --out runs/v5_schedule/retry_warmup --max_model_len 24576
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO" || exit 1
unset BUFFER_SIZE DATASET DATASET_HF_REPO_NAME GRADIENT_ACCUMULATION_STEPS MAX_STEPS METHOD MODEL OUTPUT_DIR \
    PER_DEVICE_TRAIN_BATCH_SIZE REPORT_TO RESUME_FROM RESUME_TRAINING_STATE TRAIN_DATASET USE_DDP \
    USE_DUAL_ADAPTER USE_UNSLOTH USE_VLLM VLLM_MODE SFT_EXTRA_MESSAGES SFT_EXTRA_FRACTION SFT_LEARNING_RATE
source /work/cxv200006/myenv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DATA_PREPROCESSING_ROOT="$(cd "$REPO/../data_preprocessing" && pwd)"
export HF_HOME=/work/cxv200006/.cache/huggingface
export TMAX_LOCK_DIR="$(cd "$REPO/.." && pwd)/.runtime/tmax_locks/${SLURM_JOB_ID:-nojob}_$(hostname -s)"
export TMAX_SERVICE_MODULE=tetramax/vO-2018.06-SP1
export OMP_NUM_THREADS=1
mkdir -p jobs "$TMAX_LOCK_DIR"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Iseconds); $(head -1 "$REPO/PROVENANCE.txt" 2>/dev/null)"
bash scripts/rft/with_tetramax.sh python scripts/retry/build_retry_sft.py --workers 30 --tool_workers 16 "$@"
code=$?
echo "exit $code at $(date -Iseconds)"
exit $code
