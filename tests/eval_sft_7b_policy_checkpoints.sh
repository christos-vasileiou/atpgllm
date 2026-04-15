#!/usr/bin/env bash
# Evaluate policy LoRA checkpoints under sft_7b_exper2 with pass@k (vLLM, TP=4).
#
# Usage:
#   ./eval_sft_7b_policy_checkpoints.sh
#   EXP_ROOT=/path/to/sft_7b_exper2 EVAL_RESULTS_DIR=/path/to/results ./eval_sft_7b_policy_checkpoints.sh
#
# Optional:
#   DRY_RUN=1  — print commands only
#   CUDA_VISIBLE_DEVICES=0,1,2,3  — must expose 4 GPUs for tp_size=4 (default: unset, use all visible)
#   EVAL_PROMPT_BATCH_SIZE=8     — fused prompt batch for evaluate_model.py (default: 8)
#   GENERATION_MICRO_BATCH_SIZE=8 — HF backend only; passed through for consistency (default: 8)
#   GPU_MEMORY_UTILIZATION=0.55  — vLLM fraction of VRAM to reserve (default: 0.55; raise if GPUs are idle)
#   WANDB_RUN_NAME / --wandb_run_name — optional; default name is derived from --adapter (experiment + checkpoint)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_model.py"
POLICY_FOLDER="${POLICY_FOLDER:-sft_7b_exper2}"
EXP_ROOT="${EXP_ROOT:-${SCRIPT_DIR}/${POLICY_FOLDER}}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${SCRIPT_DIR}/eval_results_${POLICY_FOLDER}_policy}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-16}"
GENERATION_MICRO_BATCH_SIZE="${GENERATION_MICRO_BATCH_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

if [[ ! -f "$EVAL_SCRIPT" ]]; then
  echo "error: evaluate_model.py not found at $EVAL_SCRIPT" >&2
  exit 1
fi
if [[ ! -d "$EXP_ROOT" ]]; then
  echo "error: experiment root not found: $EXP_ROOT" >&2
  exit 1
fi

mkdir -p "$EVAL_RESULTS_DIR"

mapfile -t CHECKPOINTS < <(
  find "$EXP_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' -printf '%f\n' | sort -V
)

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "error: no checkpoint-* directories under $EXP_ROOT" >&2
  exit 1
fi

echo "Experiment root: $EXP_ROOT"
echo "Checkpoints: ${CHECKPOINTS[*]}"
echo "Output dir:    $EVAL_RESULTS_DIR"
echo "tp_size:       4 (use 4 visible GPUs)"
echo "prompt_batch:  $EVAL_PROMPT_BATCH_SIZE  (EVAL_PROMPT_BATCH_SIZE)"
echo "gpu_mem_util:  $GPU_MEMORY_UTILIZATION  (GPU_MEMORY_UTILIZATION)"
echo ""

for name in "${CHECKPOINTS[@]}"; do
  policy="${EXP_ROOT}/${name}"
  if [[ ! -d "$policy" ]]; then
    echo "skip: no policy adapter at $policy" >&2
    continue
  fi
  out_json="${EVAL_RESULTS_DIR}/${name}_passatk_n16_t0.7_topp0.95_b${EVAL_PROMPT_BATCH_SIZE}.json"
  out_stdout="${EVAL_RESULTS_DIR}/${name}_stdout.log"
  cmd=(
    python "$EVAL_SCRIPT"
    --adapter "$policy"
    --backend vllm
    --tp_size 4
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --n 16
    --k 1 2 4 8 16
    --temperature 0.7
    --top_p 0.95
    --max_new_tokens 16384
    --max_eval_samples 512
    --eval_prompt_batch_size "$EVAL_PROMPT_BATCH_SIZE"
    --generation_micro_batch_size "$GENERATION_MICRO_BATCH_SIZE"
    --report_to wandb
    --output_file "$out_json"
  )
  echo "=== ${name} ==="
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    printf '> %q\n' "$out_stdout"
    continue
  fi
  echo "stdout -> $out_stdout"
  "${cmd[@]}" >"$out_stdout"
done

echo "Done. Results under: $EVAL_RESULTS_DIR"
