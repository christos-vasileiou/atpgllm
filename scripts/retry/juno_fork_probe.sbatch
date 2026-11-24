#!/bin/bash
#SBATCH --job-name=fork_probe
#SBATCH --partition=h100,h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=jobs/fork_probe_%j.out
#
# Teacher-forced probe of the detect-or-miss reading (scripts/retry/fork_probe.py)
# for each "<arm name> <adapter>" pair, on the retry rows and on that arm's gate
# slots (runs/retry_sft_v4/gate/<arm>). One process per checkpoint.
#
#   sbatch scripts/retry/juno_fork_probe.sbatch retry_ckpt20 runs/retry_sft_v4/sft/checkpoint-20 \
#       retry_ckpt10 runs/retry_sft_v4/sft/checkpoint-10 round11 runs/rft_v4_a100/round_11/sft/checkpoint-10
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
OUT="${PROBE_OUT:-runs/retry_sft_v4/fork_probe}"
EXTRA="${PROBE_EXTRA:-runs/retry_sft_v4/data/extra.jsonl}"
MANIFEST="${GATE_MANIFEST:-runs/retry_sft_v4/gate/fixed_eval_manifest.json}"
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Iseconds); $(git log -1 --format='%h %s' 2>/dev/null)"
status=0
while [ $# -ge 2 ]; do
    arm="$1"; adapter="$2"; shift 2
    gate="runs/retry_sft_v4/gate/$arm"
    gate_args=()
    [ -d "$gate" ] && gate_args=(--gate "$gate")
    echo "== $arm: $adapter"
    python scripts/retry/fork_probe.py --name "$arm" --adapter "$adapter" --extra "$EXTRA" \
        --manifest "$MANIFEST" "${gate_args[@]}" --out "$OUT" || status=1
done
echo "fork probe finished at $(date -Iseconds), status $status"
exit $status
