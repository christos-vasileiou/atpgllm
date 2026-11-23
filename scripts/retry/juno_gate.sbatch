#!/bin/bash
#SBATCH --job-name=retry_gate
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
#SBATCH --time=2-00:00:00
#SBATCH --output=jobs/retry_gate_%j.out
#
# Gate before relaunching GDPO: greedy evaluation of checkpoints on the 250
# fault fixed set (the GDPO fixed evaluation, same examples hash) with up to
# MAX_TOOL_ROUNDS calls, so a checkpoint that reads the simulator's answer can
# call again. Each argument pair is "<arm name> <adapter>"; results go to
# runs/retry_sft_v4/gate/<arm>. After the arms, gate_report.py compares every
# arm with the baseline arm (the first one evaluated, or GATE_BASELINE).
#
#   sbatch scripts/retry/juno_gate.sbatch round11 runs/rft_v4_a100/round_11/sft/checkpoint-10
#   sbatch --dependency=afterok:<sft> scripts/retry/juno_gate.sbatch \
#       retry_ckpt10 runs/retry_sft_v4/sft/checkpoint-10 retry_ckpt20 runs/retry_sft_v4/sft/checkpoint-20
set -uo pipefail
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO" || exit 1
# The juno login shell exports run settings that sbatch passes on.
unset BUFFER_SIZE DATASET DATASET_HF_REPO_NAME GRADIENT_ACCUMULATION_STEPS MAX_STEPS METHOD MODEL OUTPUT_DIR \
    PER_DEVICE_TRAIN_BATCH_SIZE REPORT_TO RESUME_FROM RESUME_TRAINING_STATE TRAIN_DATASET USE_DDP \
    USE_DUAL_ADAPTER USE_UNSLOTH USE_VLLM VLLM_MODE SFT_EXTRA_MESSAGES SFT_EXTRA_FRACTION SFT_LEARNING_RATE
source /work/cxv200006/myenv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DATA_PREPROCESSING_ROOT="$(cd "$REPO/../data_preprocessing" && pwd)"
export HF_HOME=/work/cxv200006/.cache/huggingface
export TMAX_LOCK_DIR="$(cd "$REPO/.." && pwd)/.runtime/tmax_locks/${SLURM_JOB_ID:-nojob}_$(hostname -s)"
export TMAX_SERVICE_MODULE=tetramax/vO-2018.06-SP1
export TMAX_REWARD_PROFILE=staged
export VLLM_DISABLE_COMPILE_CACHE=1 NCCL_P2P_DISABLE=1
mkdir -p jobs "$TMAX_LOCK_DIR"
MANIFEST="${GATE_MANIFEST:-runs/retry_sft_v4/gate/fixed_eval_manifest.json}"
MAX_TOOL_ROUNDS="${MAX_TOOL_ROUNDS:-8}"
GATE="runs/retry_sft_v4/gate"
ngpu=$(nvidia-smi -L | wc -l)
gpus=$(seq -s, 0 $((ngpu - 1)))
echo "job ${SLURM_JOB_ID:-local} on $(hostname) at $(date -Iseconds); $(head -1 "$REPO/PROVENANCE.txt" 2>/dev/null)"

arms=()
while [ $# -ge 2 ]; do
    arm="$1"; adapter="$2"; shift 2
    out="$GATE/$arm"
    mkdir -p "$out/logs"
    echo "== $arm: $adapter ($MAX_TOOL_ROUNDS tool rounds, greedy)"
    bash scripts/rft/with_tetramax.sh python scripts/rft/fanout.py --gpus "$gpus" --log_dir "$out/logs" -- \
        python scripts/rft/evaluate.py --adapter "$adapter" --manifest "$MANIFEST" --out "$out" \
        --max_tool_rounds "$MAX_TOOL_ROUNDS" --shard "{shard}" --num_shards "{num_shards}" || exit 1
    python scripts/rft/evaluate.py --out "$out" --summarize > "$out/summary.log" 2>&1
    arms+=("$arm")
done

baseline="${GATE_BASELINE:-round11}"
for arm in "${arms[@]}"; do
    [ "$arm" = "$baseline" ] && continue
    [ -d "$GATE/$baseline" ] || continue
    python scripts/retry/gate_report.py "$GATE/$arm" --baseline "$GATE/$baseline" --manifest "$MANIFEST" \
        --max_k "$MAX_TOOL_ROUNDS" --out "$GATE/report_${arm}_vs_${baseline}.json" > /dev/null
    echo "report: $GATE/report_${arm}_vs_${baseline}.json"
done
for arm in "${arms[@]}"; do
    python scripts/retry/gate_report.py "$GATE/$arm" --manifest "$MANIFEST" --max_k "$MAX_TOOL_ROUNDS" \
        --out "$GATE/report_${arm}.json" > /dev/null
    # Do the edits read the answer? Model edits against random edits of the same size (CPU).
    python scripts/retry/edit_quality.py "$GATE/$arm" --workers 30 --out "$GATE/edit_quality_${arm}.json" > /dev/null
done
# GATE_REQUIRE=<arm>: the job fails unless that arm passes the bar against the
# baseline, so a job queued with --dependency=afterok on this one (the GDPO run
# that starts from that arm's checkpoint) starts only after a pass.
if [ -n "${GATE_REQUIRE:-}" ]; then
    python scripts/retry/gate_report.py "$GATE/$GATE_REQUIRE" --baseline "$GATE/$baseline" --manifest "$MANIFEST" \
        --max_k "$MAX_TOOL_ROUNDS" --require > /dev/null
    code=$?
    echo "gate for $GATE_REQUIRE: $([ $code -eq 0 ] && echo PASSED || echo FAILED) at $(date -Iseconds)"
    exit $code
fi
echo "done at $(date -Iseconds)"
