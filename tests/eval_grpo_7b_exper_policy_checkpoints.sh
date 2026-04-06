#!/usr/bin/env bash
# Evaluate policy LoRA checkpoints under grpo_7b_exper2 with pass@k (vLLM, TP=4).
#
# Usage:
#   ./eval_grpo_7b_exper2_policy_checkpoints.sh
#   EXP_ROOT=/path/to/grpo_7b_exper2 EVAL_RESULTS_DIR=/path/to/results ./eval_grpo_7b_exper2_policy_checkpoints.sh
#
# Optional:
#   DRY_RUN=1  — print commands only
#   CUDA_VISIBLE_DEVICES=0,1,2,3  — must expose 4 GPUs for tp_size=4 (default: unset, use all visible)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_model.py"
EXP_ROOT="${EXP_ROOT:-${SCRIPT_DIR}/grpo_7b_exper2}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${SCRIPT_DIR}/eval_results_grpo_7b_exper2_policy}"
DRY_RUN="${DRY_RUN:-0}"

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
echo ""

for name in "${CHECKPOINTS[@]}"; do
  policy="${EXP_ROOT}/${name}/combined/policy"
  if [[ ! -d "$policy" ]]; then
    echo "skip: no policy adapter at $policy" >&2
    continue
  fi
  out_json="${EVAL_RESULTS_DIR}/${name}_passatk_n10_t0.7_topp0.95.json"
  cmd=(
    python "$EVAL_SCRIPT"
    --adapter "$policy"
    --backend vllm
    --tp_size 4
    --n 16
    --k 1 2 4 8 16
    --temperature 0.7
    --top_p 0.95
    --output_file "$out_json"
  )
  echo "=== ${name} ==="
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '%q ' "${cmd[@]}"
    echo
    continue
  fi
  "${cmd[@]}"
done

echo "Done. Results under: $EVAL_RESULTS_DIR"
