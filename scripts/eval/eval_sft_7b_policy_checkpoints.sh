#!/usr/bin/env bash
# Evaluate policy LoRA checkpoints under sft_7b_exper2 with pass@k (vLLM, TP auto).
#
# Usage:
#   ./eval_sft_7b_policy_checkpoints.sh
#   POLICY_FOLDER=sft_7b_exper2 ./eval_sft_7b_policy_checkpoints.sh
#   POLICY_FOLDER=sft_7b_exper2/checkpoint-40 ./eval_sft_7b_policy_checkpoints.sh
#   EXP_ROOT=/path/to/sft_7b_exper2 EVAL_RESULTS_DIR=/path/to/results ./eval_sft_7b_policy_checkpoints.sh
#
# POLICY_FOLDER / EXP_ROOT path shapes (same two modes for both):
#   <exp>                         — discover and eval all checkpoint-* under that folder
#   <exp>/checkpoint-N            — eval that single checkpoint
# POLICY_FOLDER is resolved under runs/ then tests/ when EXP_ROOT is unset.
#
# Optional:
#   DRY_RUN=1  — print commands only
#   CUDA_VISIBLE_DEVICES=0,1,2,3  — tp_size = number of listed devices; if unset, all GPUs from nvidia-smi -L
#   EVAL_PROMPT_BATCH_SIZE=8     — fused prompt batch for evaluate_model.py (default: 8)
#   GENERATION_MICRO_BATCH_SIZE=8 — HF backend only; passed through for consistency (default: 8)
#   GPU_MEMORY_UTILIZATION=0.55  — vLLM fraction of VRAM to reserve (default: 0.55; raise if GPUs are idle)
#   SAMPLING_METHOD=greedy       — model-based: greedy (default LLM+tools) | best_of_n | mcts | evolutionary;
#                                  model-free: random (PI/PO bitvectors; ≠ greedy). See sampling_strategies.py
#   NUM_COMPLETIONS=16           — --num_completions: completions per problem for pass@k, the
#                                  pass@k pool (max k must be <= it). Alias: NUM_SAMPLES.
#   SEARCH_BUDGET=3              — --budget: per-completion search width for mcts/evolutionary
#                                  only (default: 3; orthogonal to NUM_COMPLETIONS)
#   BEST_OF_N_WIDTH=3            — --n: per-completion i.i.d. samples for best_of_n only
#                                  (default: 3; the best is kept; orthogonal to NUM_COMPLETIONS)
#   PASS_AT_K="1 2 4 8 16"       — space-separated pass@k values
#   TEMPERATURE=0.7  TOP_P=0.95  MAX_NEW_TOKENS=16384  MAX_EVAL_SAMPLES=512
#   THRESHOLD_MODE=fault_detected — fault_detected | positive_reward | full_accuracy
#   MAX_PROMPT_LENGTH=4096       — max prompt token length for eval buffering (default: 4096)
#   EVAL_DATASET=chrivasileiou/asap7-language-of-test-v2 — test split source; must match
#                                  TRAIN_DATASET in configs/sft.conf
#   MERGE_DEQUANT=1              — serve the QLoRA-faithful merged bf16 export via
#                                  --merge_dequant (adapter trained on NF4 base); 0 = clean
#                                  bf16 base + dynamic LoRA (NOT faithful to training)
#   WANDB_RUN_NAME / --wandb_run_name — optional; default name is derived from --adapter (experiment + checkpoint)
#
# Examples:
#   SAMPLING_METHOD=best_of_n NUM_COMPLETIONS=32 BEST_OF_N_WIDTH=4 ./eval_sft_7b_policy_checkpoints.sh
#   SAMPLING_METHOD=mcts NUM_COMPLETIONS=16 SEARCH_BUDGET=50 EVAL_RESULTS_DIR=./eval_results_sft_7b_mcts ./eval_sft_7b_policy_checkpoints.sh

set -euo pipefail

# activate virtual environment (prefer Slurm /work path, fall back to shared /proj path)
if [ -f "/work/cxv200006/myenv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source /work/cxv200006/myenv/bin/activate
elif [ -f "/proj/trela/christos/myenv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source /proj/trela/christos/myenv/bin/activate
fi
echo "Python Path: $(which python)"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EVAL_SCRIPT="${SCRIPT_DIR}/evaluate_model.py"
POLICY_FOLDER="${POLICY_FOLDER:-sft_7b_exper2}"
POLICY_FOLDER="${POLICY_FOLDER%/}"
# Prefer runs/, fall back to legacy tests/ (POLICY_FOLDER may be exp or exp/ckpt).
EXP_ROOT="${EXP_ROOT:-}"
EXP_ROOT="${EXP_ROOT%/}"
if [ -z "$EXP_ROOT" ]; then
  if [ -d "$REPO_ROOT/runs/$POLICY_FOLDER" ]; then
    EXP_ROOT="$REPO_ROOT/runs/$POLICY_FOLDER"
  else
    EXP_ROOT="$REPO_ROOT/tests/$POLICY_FOLDER"
  fi
fi
EXP_ROOT="${EXP_ROOT%/}"
if [[ ! -d "$EXP_ROOT" ]]; then
  echo "error: path not found: $EXP_ROOT (POLICY_FOLDER=$POLICY_FOLDER)" >&2
  exit 1
fi

# Normalize path shapes so EXP_ROOT is always the experiment dir and CHECKPOINTS
# lists checkpoint-* basenames.
CHECKPOINTS=()
_target_base="$(basename "$EXP_ROOT")"
if [[ "$_target_base" =~ ^checkpoint- ]]; then
  CHECKPOINTS=("$_target_base")
  EXP_ROOT="$(dirname "$EXP_ROOT")"
else
  mapfile -t CHECKPOINTS < <(
    find "$EXP_ROOT" -maxdepth 1 -mindepth 1 -type d -name 'checkpoint-*' ! -name '*_merged_bf16' -printf '%f\n' | sort -V
  )
fi
unset _target_base

EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-$REPO_ROOT/runs/eval_results_$(basename "$EXP_ROOT")_policy}"
DRY_RUN="${DRY_RUN:-0}"
EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-16}"
GENERATION_MICRO_BATCH_SIZE="${GENERATION_MICRO_BATCH_SIZE:-16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
SAMPLING_METHOD="${SAMPLING_METHOD:-greedy}"
# pass@k pool: completions per problem (max k must be <= it).
# NUM_SAMPLES is kept as a backward-compatible alias for NUM_COMPLETIONS.
NUM_COMPLETIONS="${NUM_COMPLETIONS:-${NUM_SAMPLES:-50}}"
PASS_AT_K="${PASS_AT_K:-1 2 4 8 16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"
MAX_EVAL_SAMPLES="${MAX_EVAL_SAMPLES:-512}"
THRESHOLD_MODE="${THRESHOLD_MODE:-fault_detected}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-4096}"
# Must match TRAIN_DATASET in configs/sft.conf (test split of the same database).
EVAL_DATASET="${EVAL_DATASET:-chrivasileiou/asap7-language-of-test-v2}"
MERGE_DEQUANT="${MERGE_DEQUANT:-1}"
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
mkdir -p "$EVAL_RESULTS_DIR"

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
echo "sampling:        $SAMPLING_METHOD  (SAMPLING_METHOD)"
echo "num_completions: $NUM_COMPLETIONS  (NUM_COMPLETIONS / --num_completions)"
if [[ "$SAMPLING_METHOD" == "mcts" || "$SAMPLING_METHOD" == "evolutionary" ]]; then
  echo "search_budget:   ${SEARCH_BUDGET:-50}  (SEARCH_BUDGET / --budget)"
fi
if [[ "$SAMPLING_METHOD" == "best_of_n" ]]; then
  echo "best_of_n_width: ${BEST_OF_N_WIDTH:-4}  (BEST_OF_N_WIDTH / --n)"
fi
echo "pass@k:        ${K_VALUES[*]}  (PASS_AT_K)"
echo "temperature:   $TEMPERATURE  top_p: $TOP_P"
echo "dataset:       $EVAL_DATASET  (EVAL_DATASET)"
echo "merge_dequant: $MERGE_DEQUANT  (MERGE_DEQUANT; 1 = QLoRA-faithful merged bf16 serving)"
echo ""

case "$SAMPLING_METHOD" in
  greedy|random|best_of_n|mcts|evolutionary) ;;
  *)
    echo "error: SAMPLING_METHOD must be greedy|best_of_n|mcts|evolutionary (model-based) or random (model-free) (got: $SAMPLING_METHOD)" >&2
    exit 1
    ;;
esac
if [[ -n "${SEARCH_BUDGET:-}" ]] && [[ "$SAMPLING_METHOD" != "mcts" && "$SAMPLING_METHOD" != "evolutionary" ]]; then
  echo "error: SEARCH_BUDGET applies only to mcts/evolutionary (got SAMPLING_METHOD=$SAMPLING_METHOD)" >&2
  exit 1
fi
if [[ -n "${BEST_OF_N_WIDTH:-}" ]] && [[ "$SAMPLING_METHOD" != "best_of_n" ]]; then
  echo "error: BEST_OF_N_WIDTH applies only to best_of_n (got SAMPLING_METHOD=$SAMPLING_METHOD)" >&2
  exit 1
fi
case "$SAMPLING_METHOD" in
  mcts|evolutionary)
    SEARCH_BUDGET="${SEARCH_BUDGET:-50}"
    if [[ "$SEARCH_BUDGET" -lt 1 ]]; then
      echo "error: SEARCH_BUDGET ($SEARCH_BUDGET) must be >= 1" >&2
      exit 1
    fi
    ;;
  best_of_n)
    BEST_OF_N_WIDTH="${BEST_OF_N_WIDTH:-4}"
    if [[ "$BEST_OF_N_WIDTH" -lt 1 ]]; then
      echo "error: BEST_OF_N_WIDTH ($BEST_OF_N_WIDTH) must be >= 1" >&2
      exit 1
    fi
    ;;
esac
for k in "${K_VALUES[@]}"; do
  if [[ "$k" -gt "$NUM_COMPLETIONS" ]]; then
    echo "error: pass@${k} requires NUM_COMPLETIONS >= ${k} (got NUM_COMPLETIONS=$NUM_COMPLETIONS)" >&2
    exit 1
  fi
done
echo ""

for name in "${CHECKPOINTS[@]}"; do
  policy="${EXP_ROOT}/${name}"
  if [[ ! -f "${policy}/adapter_config.json" ]]; then
    echo "skip: no adapter_config.json at $policy" >&2
    continue
  fi
  _out_tag="nc${NUM_COMPLETIONS}"
  if [[ "$SAMPLING_METHOD" == "mcts" || "$SAMPLING_METHOD" == "evolutionary" ]]; then
    _out_tag="${_out_tag}_sb${SEARCH_BUDGET}"
  elif [[ "$SAMPLING_METHOD" == "best_of_n" ]]; then
    _out_tag="${_out_tag}_bon${BEST_OF_N_WIDTH}"
  fi
  clean_path=${policy%/}
  wandb_run_name="${clean_path#"${clean_path%/*/*}"/}_$SAMPLING_METHOD"
  wandb_run_name="${wandb_run_name//\//_}"
  out_json="${EVAL_RESULTS_DIR}/${name}_passatk_${SAMPLING_METHOD}_${_out_tag}_t${TEMPERATURE}_topp${TOP_P}_b${EVAL_PROMPT_BATCH_SIZE}.json"
  out_stdout="${EVAL_RESULTS_DIR}/${name}_${SAMPLING_METHOD}_${_out_tag}_stdout.log"
  cmd=(
    python "$EVAL_SCRIPT"
    --adapter "$policy"
    --dataset "$EVAL_DATASET"
    --backend vllm
    --tp_size "$TP_SIZE"
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION"
    --sampling_method "$SAMPLING_METHOD"
    --num_completions "$NUM_COMPLETIONS"
    --k "${K_VALUES[@]}"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_new_tokens "$MAX_NEW_TOKENS"
    --max_eval_samples "$MAX_EVAL_SAMPLES"
    --threshold_mode "$THRESHOLD_MODE"
    --eval_prompt_batch_size "$EVAL_PROMPT_BATCH_SIZE"
    --generation_micro_batch_size "$GENERATION_MICRO_BATCH_SIZE"
    --max_prompt_length "$MAX_PROMPT_LENGTH"
    --report_to wandb
    --output_file "$out_json"
    --wandb_run_name "$wandb_run_name"
  )
  if [[ "$SAMPLING_METHOD" == "mcts" || "$SAMPLING_METHOD" == "evolutionary" ]]; then
    cmd+=( --budget "$SEARCH_BUDGET" )
  elif [[ "$SAMPLING_METHOD" == "best_of_n" ]]; then
    cmd+=( --n "$BEST_OF_N_WIDTH" )
  fi
  if [[ "$MERGE_DEQUANT" == "1" ]]; then
    cmd+=( --merge_dequant )
  fi
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
