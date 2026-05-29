#!/usr/bin/env bash
# Evaluate policy LoRA checkpoints under grpo_7b_exper2 with pass@k (vLLM, TP auto).
#
# Usage:
#   ./eval_grpo_7b_policy_checkpoints.sh
#   EXP_ROOT=/path/to/grpo_7b_exper2 EVAL_RESULTS_DIR=/path/to/results ./eval_grpo_7b_policy_checkpoints.sh
#
# Optional:
#   DRY_RUN=1  — print commands only
#   CUDA_VISIBLE_DEVICES=0,1,2,3  — tp_size = number of listed devices; if unset, all GPUs from nvidia-smi -L
#   EVAL_PROMPT_BATCH_SIZE=8     — fused prompt batch for evaluate_model.py (default: 8)
#   GENERATION_MICRO_BATCH_SIZE=8 — HF backend only; passed through for consistency (default: 8)
#   GPU_MEMORY_UTILIZATION=0.55  — vLLM fraction of VRAM to reserve (default: 0.55; raise if GPUs are idle)
#   SAMPLING_METHOD=random       — random | best_of_n | mcts | evolutionary (see sampling_strategies.py)
#   NUM_SAMPLES=16               — --n completions / search budget per problem (max k must be <= n)
#   PASS_AT_K="1 2 4 8 16"       — space-separated pass@k values
#   TEMPERATURE=0.7  TOP_P=0.95  MAX_NEW_TOKENS=16384  MAX_EVAL_SAMPLES=512
#   THRESHOLD_MODE=fault_detected — fault_detected | positive_reward | full_accuracy
#   WANDB_RUN_NAME / --wandb_run_name — optional; default name is derived from --adapter (e.g. …_checkpoint-N_policy)
#
# Examples:
#   SAMPLING_METHOD=best_of_n NUM_SAMPLES=32 ./eval_grpo_7b_policy_checkpoints.sh
#   SAMPLING_METHOD=mcts NUM_SAMPLES=16 EVAL_RESULTS_DIR=./eval_results_grpo_7b_mcts ./eval_grpo_7b_policy_checkpoints.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_model.py"
POLICY_FOLDER="${POLICY_FOLDER:-grpo_7b_exper2}"
EXP_ROOT="${EXP_ROOT:-${SCRIPT_DIR}/${POLICY_FOLDER}}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-${SCRIPT_DIR}/eval_results_${POLICY_FOLDER}_policy}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-16}"
GENERATION_MICRO_BATCH_SIZE="${GENERATION_MICRO_BATCH_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
SAMPLING_METHOD="${SAMPLING_METHOD:-random}"
NUM_SAMPLES="${NUM_SAMPLES:-50}"
PASS_AT_K="${PASS_AT_K:-1 2 4 8 16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-512}"
THRESHOLD_MODE="${THRESHOLD_MODE:-fault_detected}"
read -ra K_VALUES <<< "$PASS_AT_K"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _TP_CUDA_DEVS <<< "$CUDA_VISIBLE_DEVICES"
  TP_SIZE="${#_TP_CUDA_DEVS[@]}"
else
  if ! command -v nvidia-smi &>/dev/null; then
    echo "error: CUDA_VISIBLE_DEVICES unset and nvidia-smi not in PATH; set CUDA_VISIBLE_DEVICES or install drivers" >&2
    exit 1
  fi
  TP_SIZE=$(nvidia-smi -L 2>/dev/null | wc -l)
  TP_SIZE="${TP_SIZE//[[:space:]]/}"
fi
if [[ -z "$TP_SIZE" || "$TP_SIZE" -lt 1 ]]; then
  echo "error: could not determine GPU count for tp_size (got: ${TP_SIZE:-empty})" >&2
  exit 1
fi

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
echo "tp_size:       $TP_SIZE  (from CUDA_VISIBLE_DEVICES if set, else nvidia-smi -L)"
echo "prompt_batch:  $EVAL_PROMPT_BATCH_SIZE  (EVAL_PROMPT_BATCH_SIZE)"
echo "gpu_mem_util:  $GPU_MEMORY_UTILIZATION  (GPU_MEMORY_UTILIZATION)"
echo "sampling:      $SAMPLING_METHOD  (SAMPLING_METHOD)"
echo "num_samples:   $NUM_SAMPLES  (NUM_SAMPLES / --n)"
echo "pass@k:        ${K_VALUES[*]}  (PASS_AT_K)"
echo "temperature:   $TEMPERATURE  top_p: $TOP_P"
echo ""

case "$SAMPLING_METHOD" in
  random|best_of_n|mcts|evolutionary) ;;
  *)
    echo "error: SAMPLING_METHOD must be random, best_of_n, mcts, or evolutionary (got: $SAMPLING_METHOD)" >&2
    exit 1
    ;;
esac
for k in "${K_VALUES[@]}"; do
  if [[ "$k" -gt "$NUM_SAMPLES" ]]; then
    echo "error: pass@${k} requires NUM_SAMPLES >= ${k} (got NUM_SAMPLES=$NUM_SAMPLES)" >&2
    exit 1
  fi
done
echo ""

# New layout: checkpoint-N/policy; legacy: checkpoint-N/combined/policy
# (see dual_adapter_grpo_trainer._resolve_dual_adapter_dirs).
resolve_grpo_policy_adapter() {
  local ckpt_dir="$1"
  local candidate
  for candidate in "${ckpt_dir}/policy" "${ckpt_dir}/combined/policy"; do
    if [[ -f "${candidate}/adapter_config.json" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

for name in "${CHECKPOINTS[@]}"; do
  ckpt_dir="${EXP_ROOT}/${name}"
  if ! policy="$(resolve_grpo_policy_adapter "$ckpt_dir")"; then
    echo "skip: no policy adapter under $ckpt_dir (checked policy/ and combined/policy)" >&2
    continue
  fi
  out_json="${EVAL_RESULTS_DIR}/${name}_passatk_${SAMPLING_METHOD}_n${NUM_SAMPLES}_t${TEMPERATURE}_topp${TOP_P}_b${EVAL_PROMPT_BATCH_SIZE}.json"
  out_stdout="${EVAL_RESULTS_DIR}/${name}_${SAMPLING_METHOD}_n${NUM_SAMPLES}_stdout.log"
  cmd=(
    python "$EVAL_SCRIPT"
    --adapter "$policy"
    --backend vllm
    --tp_size "$TP_SIZE"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --sampling_method "$SAMPLING_METHOD"
    --n "$NUM_SAMPLES"
    --k "${K_VALUES[@]}"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --max_eval_samples "$MAX_EVAL_SAMPLES"
    --threshold_mode "$THRESHOLD_MODE"
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
