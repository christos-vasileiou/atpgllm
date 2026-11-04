#!/bin/bash
# The reasoning pilot: benchmark the starting model, SFT with gate by gate
# demonstrations and retries, then benchmark each saved checkpoint. Every part
# skips itself when finished, so this can simply be started again. The live RFT
# run on eng-tamale3 is not touched; this runs on the host it is started on.
#
#   tmux new -d -s reasoning_pilot 'scripts/star/run_reasoning_pilot.sh'
set -uo pipefail
umask 002
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
RUN=${RUN:-runs/reasoning_pilot}
START=${START:-runs/rft_v4_a100/round_4/sft/checkpoint-20}
ROWS=${ROWS:-$RUN/sft.jsonl}
STEPS=${STEPS:-30}
GPUS=${GPUS:-0,1,2,3}
MANIFEST=${MANIFEST:-runs/rft_v4_a100/eval/robust_eval_manifest.json}

NV=/proj/trela/christos/myenv/lib/python3.11/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cudnn/lib:$NV/cublas/lib:$NV/cuda_runtime/lib:$NV/cusparselt/lib:$NV/nccl/lib:$NV/nvjitlink/lib:/home/eng/c/cxv200006/usr/lib64:/home/eng/c/cxv200006/usr/lib:/proj/trela/christos/usr/lib:/usr/local/cuda/include:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
source /proj/trela/christos/myenv/bin/activate
export HF_HOME=/proj/trela/christos/.cache/huggingface/
export PYTHONPATH="$REPO"
export VLLM_DISABLE_COMPILE_CACHE=1
# Benchmarks score with the staged reward that GDPO will train on. SFT itself uses no reward.
export TMAX_REWARD_PROFILE=staged
PY=$(command -v python)
log() { echo "[pilot $(date '+%F %T')] $*"; }

benchmark() {  # benchmark <adapter> <tag>
    local adapter=$1 tag=$2 out="$RUN/bench_$2"
    if [ -f "$RUN/reports/$tag/metrics.json" ]; then log "$tag already reported"; return 0; fi
    mkdir -p "$out"
    log "benchmark $tag on $adapter"
    FAULT_SIM_BACKEND=fast CUDA_VISIBLE_DEVICES=${GPUS%%,*} "$PY" -c "import sys; from pathlib import Path; sys.path.insert(0, 'scripts/eval'); from evaluate_model import resolve_merged_bf16_dir; print(resolve_merged_bf16_dir(Path('$adapter')))" || return 1
    FAULT_SIM_BACKEND=fast "$PY" scripts/rft/fanout.py --gpus "$GPUS" --log_dir "$out/logs" -- \
        "$PY" scripts/rft/evaluate.py --adapter "$adapter" --manifest "$MANIFEST" --out "$out" \
        --shard '{shard}' --num_shards '{num_shards}' --num_samples 16 --temperature 0.7 --top_p 0.95 \
        --max_tool_rounds 5 --max_generated_tokens 16384 --batch_problems 16 || return 1
    "$PY" scripts/rft/evaluate.py --out "$out" --summarize > "$out/summary.log" || return 1
    FAULT_SIM_BACKEND=fast "$PY" scripts/rft/report_eval.py --run_dir "$RUN" --tag "$tag" --eval_dir "$out" \
        --reference "$MANIFEST.reference.json" --no_history || return 1
    "$PY" scripts/star/wandb_reports.py --run_dir "$RUN" || log "W&B upload failed, tables stay on disk"
}

if [ ! -f "$ROWS.report.json" ]; then log "rows are not built: $ROWS"; exit 1; fi
benchmark "$START" pilot_start || { log "start benchmark failed"; exit 1; }

SFT="$RUN/sft"
if [ ! -d "$SFT/checkpoint-$STEPS" ]; then
    log "SFT for $STEPS steps"
    export OUTPUT_DIR="$SFT" MAX_STEPS="$STEPS" SFT_EXTRA_MESSAGES="$ROWS" SFT_EXTRA_FRACTION=0.5 SFT_LEARNING_RATE=2e-5
    export WANDB_RUN_GROUP=reasoning_pilot WANDB_NAME=reasoning_pilot/sft
    if ls -d "$SFT"/checkpoint-* >/dev/null 2>&1; then
        export RESUME_FROM=latest RESUME_TRAINING_STATE=True
    else
        export RESUME_FROM="$START" RESUME_TRAINING_STATE=False
    fi
    CUDA_VISIBLE_DEVICES=$GPUS bash scripts/train/run_training_code.sh scripts/train/configs/sft_granite_4.2_8b_v4_rft.conf > "$RUN/sft.log" 2>&1 || { log "SFT failed, see $RUN/sft.log"; exit 1; }
fi
for step in 10 20 30; do
    [ -d "$SFT/checkpoint-$step" ] && benchmark "$SFT/checkpoint-$step" "pilot_$step"
done
log "pilot finished"
