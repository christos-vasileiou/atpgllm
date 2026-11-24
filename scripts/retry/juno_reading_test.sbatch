#!/bin/bash
#SBATCH --job-name=reading_test
#SBATCH --partition=h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=jobs/reading_test_%j.out
#
# Plain reading question (scripts/retry/reading_test.py) for each "<name> <adapter|base>" pair, on
# the first tool answers of the gate arm READING_GATE (default retry_ckpt20). One process per model.
#
#   sbatch scripts/retry/juno_reading_test.sbatch base base sft210 runs/sft_v4_ckpt210/checkpoint-210 ...
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO" || exit 1
unset BUFFER_SIZE DATASET DATASET_HF_REPO_NAME GRADIENT_ACCUMULATION_STEPS MAX_STEPS METHOD MODEL OUTPUT_DIR \
    PER_DEVICE_TRAIN_BATCH_SIZE REPORT_TO RESUME_FROM RESUME_TRAINING_STATE TRAIN_DATASET USE_DDP \
    USE_DUAL_ADAPTER USE_UNSLOTH USE_VLLM VLLM_MODE SFT_EXTRA_MESSAGES SFT_EXTRA_FRACTION SFT_LEARNING_RATE
source /work/cxv200006/myenv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DATA_PREPROCESSING_ROOT="$(cd "$REPO/../data_preprocessing" && pwd)"
export HF_HOME=/work/cxv200006/.cache/huggingface HF_HUB_OFFLINE=1
export VLLM_DISABLE_COMPILE_CACHE=1 NCCL_P2P_DISABLE=1
mkdir -p jobs
OUT="${READING_OUT:-runs/retry_sft_v4/reading_test}"
GATE="runs/retry_sft_v4/gate/${READING_GATE:-retry_ckpt20}"
MANIFEST="${GATE_MANIFEST:-runs/retry_sft_v4/gate/fixed_eval_manifest.json}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Iseconds); $(git log -1 --format='%h %s' 2>/dev/null)"
status=0
while [ $# -ge 2 ]; do
    name="$1"; adapter="$2"; shift 2
    adapter_args=()
    [ "$adapter" != "base" ] && adapter_args=(--adapter "$adapter")
    echo "== $name: $adapter"
    python scripts/retry/reading_test.py --name "$name" "${adapter_args[@]}" --gate "$GATE" \
        --manifest "$MANIFEST" --out "$OUT" || status=1
done
echo "reading test finished at $(date -Iseconds), status $status"
exit $status
