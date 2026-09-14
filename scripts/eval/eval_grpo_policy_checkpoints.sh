#!/usr/bin/env bash
# Evaluate GRPO LoRA checkpoints, regardless of base model family or size.
# Usage: ./eval_grpo_policy_checkpoints.sh CHECKPOINT_OR_EXPERIMENT
# Paths may be absolute, relative to cwd/repo, or relative to runs/ (legacy tests/).
# With no argument, CHECKPOINT, EXP_ROOT, then POLICY_FOLDER are accepted.
# The evaluator infers the base model from adapter_config.json.
#
# Environment reference (values/choices are case-sensitive).
# Copy the desired export/unset/launch lines into a shell, removing leading "# ".
# The export block below uses the current defaults; edit values as needed.
# Choices separated by "|" in comments mean select ONE value, not a shell pipe.
#
# Sampling and pass@k:
# export SAMPLING_METHOD=greedy          # greedy|random|best_of_n|mcts|evolutionary
# export NUM_COMPLETIONS=50             # Integer >= 1; independent completions per problem
# export PASS_AT_K="1 2 4 8 16"          # Space-separated integers: 1 <= k <= NUM_COMPLETIONS
# export TEMPERATURE=0.7                # Float >= 0; 0 = deterministic token selection
# export TOP_P=0.95                     # Float: 0 < TOP_P <= 1; 1 disables nucleus filtering
# export THRESHOLD_MODE=fault_detected   # fault_detected|positive_reward|full_accuracy
#
# greedy = independent LLM generations with optional tool calls (temperature applies).
# random = model-free, uniformly sampled PI/PO bitvectors; skips loading the LLM.
# best_of_n = retain the best of N candidates per independent completion.
# mcts/evolutionary = search with a separate budget per independent completion.
# Thresholds: fault_detected = predicted input detects the fault;
# positive_reward = sum of rewards > 0; full_accuracy = all accuracy metrics == 1.
#
# Data, generation limits, and batching:
# export EVAL_DATASET=chrivasileiou/asap7-language-of-test-v2  # Hugging Face dataset ID; test split
# export MAX_EVAL_SAMPLES=512            # Integer >= 1, or -1 for all eligible eval samples
# export MAX_NEW_TOKENS=16384            # Integer >= 1; new-token limit per generation
# export MAX_PROMPT_LENGTH=4096          # Integer >= 1; prompt-token limit for eval buffering
# export EVAL_PROMPT_BATCH_SIZE=16       # Integer >= 1; prompts per fused generation batch
# export GENERATION_MICRO_BATCH_SIZE=16  # Integer >= 1; HF-only, no effect with this vLLM launcher
# export MAX_TOOL_ROUNDS=1               # Integer >= 0; 0 disables tool-call rounds
# export SEED=42                        # Integer in [0, 4294967295]; random seed
# export SIM_CONFIG=sim_config.json     # Simulator config path; default falls back to package data
# MAX_TOOL_ROUNDS, SEED, and SIM_CONFIG are read directly by evaluate_model.py.
#
# Serving, reporting, and preview:
# export GPU_MEMORY_UTILIZATION=0.85    # Float: 0 < value <= 1; vLLM GPU-memory fraction
# export MERGE_DEQUANT=1                # 0|1; 1 = NF4-base + adapter merged bf16 export (cached)
#                                      #      0 = clean base model + dynamic LoRA
# export REPORT_TO=wandb                # wandb|none; wandb project is atpg-eval
# export DRY_RUN=0                      # 0|1; 1 prints commands without running evaluation
#
# Optional overrides (copy only the ones you need; shown values are examples):
# export CUDA_VISIBLE_DEVICES=0,1       # Comma-separated CUDA device IDs/UUIDs; unset = all visible GPUs
# export TP_SIZE=2                      # Integer >= 1; must fit visible GPUs and model parallelism
#                                      # Default: device count from CUDA_VISIBLE_DEVICES/nvidia-smi
#                                      # random uses 1; dry run without nvidia-smi falls back to 1
# export EVAL_RESULTS_DIR=./eval_results # Directory for per-checkpoint JSON metrics and stdout logs
#                                      # Default: <repo>/runs/eval_results_<experiment>_policy
# export WANDB_RUN_NAME="my-eval"        # Any nonempty display name; default derived per checkpoint
#
# Strategy-specific settings: use only the block matching SAMPLING_METHOD.
# A budget/width left set with an incompatible strategy causes an error.
#
# Best-of-N (exact spelling: best_of_n):
# export SAMPLING_METHOD=best_of_n
# export BEST_OF_N_WIDTH=4              # Integer >= 1; default 4; candidates per completion
# unset SEARCH_BUDGET
#
# MCTS or evolutionary search:
# export SAMPLING_METHOD=mcts           # mcts|evolutionary
# export SEARCH_BUDGET=50               # Integer >= 1; default 50; search width per completion
# unset BEST_OF_N_WIDTH
#
# Return to the default sampling strategy:
# export SAMPLING_METHOD=greedy         # greedy|random
# unset SEARCH_BUDGET BEST_OF_N_WIDTH
#
# Legacy aliases (optional; no defaults for paths):
# export NUM_SAMPLES=50                 # Same values as NUM_COMPLETIONS; ignored when that is set
# export EXP_ROOT=/path/to/experiment   # Checkpoint/experiment path; ignored when CHECKPOINT is set
# export POLICY_FOLDER=experiment      # Same path forms; ignored when CHECKPOINT or EXP_ROOT is set
# A positional path takes precedence over CHECKPOINT, EXP_ROOT, and POLICY_FOLDER.
#
# Python environment: activate your virtualenv/Conda environment before launching.
# VIRTUAL_ENV / CONDA_PREFIX are managed by activation; either skips cluster-env fallback.
# PATH selects python. No environment variable switches the fixed vLLM backend here.
#
# Copyable checkpoint selection and launch (paths relative to the atpgllm repo):
# export CHECKPOINT=runs/grpo_granite_4.2_8b/checkpoint-40
# ./scripts/eval/eval_grpo_policy_checkpoints.sh "$CHECKPOINT"
# CHECKPOINT accepts a checkpoint or experiment directory; no default.
# A GRPO checkpoint resolves policy/ or combined/policy/; either can be supplied directly.

set -euo pipefail

die() { echo "error: $*" >&2; exit 1; }
usage() {
  echo "Usage: $(basename "${BASH_SOURCE[0]}") CHECKPOINT_OR_EXPERIMENT"
  echo "Evaluate GRPO adapters; an experiment directory evaluates all checkpoint-* children."
}
if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi
[[ $# -le 1 ]] || die "expected one checkpoint or experiment path"
TARGET="${1:-${CHECKPOINT:-${EXP_ROOT:-${POLICY_FOLDER:-}}}}"
[[ -n "$TARGET" ]] || { usage >&2; exit 1; }

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
# Resolve without changing the caller's working directory.
for candidate in "$TARGET" "$REPO_ROOT/$TARGET" "$REPO_ROOT/runs/$TARGET" "$REPO_ROOT/tests/$TARGET"; do
  if [[ -d "$candidate" ]]; then
    TARGET="$(cd "$candidate" && pwd)"
    break
  fi
done
[[ -d "$TARGET" ]] || die "path not found: $TARGET"

resolve_adapter() {
  local candidate
  for candidate in "$1/policy" "$1/combined/policy"; do
    if [[ -f "$candidate/adapter_config.json" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

# An explicit policy path selects that exact adapter, including legacy layouts.
DIRECT_ADAPTER=""
if [[ "${TARGET##*/}" == "policy" ]]; then
  [[ -f "$TARGET/adapter_config.json" ]] || die "no adapter_config.json at $TARGET"
  DIRECT_ADAPTER="$TARGET"
  TARGET="$(dirname "$TARGET")"
  [[ "${TARGET##*/}" != "combined" ]] || TARGET="$(dirname "$TARGET")"
fi

CHECKPOINTS=()
if [[ -n "$DIRECT_ADAPTER" ]] || resolve_adapter "$TARGET" >/dev/null || [[ "${TARGET##*/}" == checkpoint-* ]]; then
  CHECKPOINTS=("$TARGET")
  EXPERIMENT="$(basename "$(dirname "$TARGET")")"
else
  mapfile -t CHECKPOINTS < <(
    find "$TARGET" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' ! -name '*_merged_bf16' | sort -V
  )
  EXPERIMENT="${TARGET##*/}"
fi
[[ ${#CHECKPOINTS[@]} -gt 0 ]] || die "no checkpoint-* directories under $TARGET"

# Preserve the existing evaluation defaults across all models.
DRY_RUN="${DRY_RUN:-0}"
EVAL_RESULTS_DIR="${EVAL_RESULTS_DIR:-$REPO_ROOT/runs/eval_results_${EXPERIMENT}_policy}"
SAMPLING_METHOD="${SAMPLING_METHOD:-greedy}"
NUM_COMPLETIONS="${NUM_COMPLETIONS:-${NUM_SAMPLES:-50}}"
PASS_AT_K="${PASS_AT_K:-1 2 4 8 16}"
TEMPERATURE="${TEMPERATURE:-0.7}"
TOP_P="${TOP_P:-0.95}"
EVAL_PROMPT_BATCH_SIZE="${EVAL_PROMPT_BATCH_SIZE:-16}"
MERGE_DEQUANT="${MERGE_DEQUANT:-1}"
read -ra K_VALUES <<< "$PASS_AT_K"

positive_integer() { [[ "$1" =~ ^[1-9][0-9]*$ ]]; }
positive_integer "$NUM_COMPLETIONS" || die "NUM_COMPLETIONS must be a positive integer"
[[ ${#K_VALUES[@]} -gt 0 ]] || die "PASS_AT_K must contain at least one value"
for k in "${K_VALUES[@]}"; do
  positive_integer "$k" || die "PASS_AT_K values must be positive integers"
  [[ "$k" -le "$NUM_COMPLETIONS" ]] || die "pass@$k requires NUM_COMPLETIONS >= $k"
done
[[ "$MERGE_DEQUANT" == 0 || "$MERGE_DEQUANT" == 1 ]] || die "MERGE_DEQUANT must be 0 or 1"
case "$SAMPLING_METHOD" in
  greedy|random|best_of_n|mcts|evolutionary) ;;
  *) die "SAMPLING_METHOD must be greedy|random|best_of_n|mcts|evolutionary" ;;
esac
if [[ -n "${SEARCH_BUDGET:-}" && "$SAMPLING_METHOD" != mcts && "$SAMPLING_METHOD" != evolutionary ]]; then
  die "SEARCH_BUDGET applies only to mcts/evolutionary"
fi
if [[ -n "${BEST_OF_N_WIDTH:-}" && "$SAMPLING_METHOD" != best_of_n ]]; then
  die "BEST_OF_N_WIDTH applies only to best_of_n"
fi
SEARCH_ARGS=()
OUT_TAG="nc${NUM_COMPLETIONS}"
case "$SAMPLING_METHOD" in
  mcts|evolutionary)
    SEARCH_BUDGET="${SEARCH_BUDGET:-50}"
    positive_integer "$SEARCH_BUDGET" || die "SEARCH_BUDGET must be a positive integer"
    SEARCH_ARGS=(--budget "$SEARCH_BUDGET")
    OUT_TAG+="_sb${SEARCH_BUDGET}"
    ;;
  best_of_n)
    BEST_OF_N_WIDTH="${BEST_OF_N_WIDTH:-4}"
    positive_integer "$BEST_OF_N_WIDTH" || die "BEST_OF_N_WIDTH must be a positive integer"
    SEARCH_ARGS=(--n "$BEST_OF_N_WIDTH")
    OUT_TAG+="_bon${BEST_OF_N_WIDTH}"
    ;;
esac

# Explicit TP_SIZE wins; otherwise use the visible GPUs. Dry runs need no GPU.
if [[ -z "${TP_SIZE:-}" ]]; then
  if [[ "$SAMPLING_METHOD" == random ]]; then
    TP_SIZE=1
  elif [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    [[ -n "$CUDA_VISIBLE_DEVICES" && "$CUDA_VISIBLE_DEVICES" != "-1" ]] || die "no visible CUDA devices"
    IFS=',' read -ra CUDA_DEVICES <<< "$CUDA_VISIBLE_DEVICES"
    TP_SIZE="${#CUDA_DEVICES[@]}"
  elif command -v nvidia-smi >/dev/null 2>&1; then
    TP_SIZE="$(nvidia-smi -L | wc -l)"
    TP_SIZE="${TP_SIZE//[[:space:]]/}"
  elif [[ "$DRY_RUN" == 1 ]]; then
    TP_SIZE=1
  else
    die "cannot detect GPUs; set CUDA_VISIBLE_DEVICES or TP_SIZE"
  fi
fi
positive_integer "$TP_SIZE" || die "TP_SIZE must be a positive integer"

# Use an active environment, otherwise retain the cluster environment fallback.
if [[ "$DRY_RUN" != 1 && -z "${VIRTUAL_ENV:-}" && -z "${CONDA_PREFIX:-}" ]]; then
  for activate in /work/cxv200006/myenv/bin/activate /proj/trela/christos/myenv/bin/activate; do
    if [[ -f "$activate" ]]; then
      # shellcheck source=/dev/null
      source "$activate"
      break
    fi
  done
fi

evaluated=0
for checkpoint in "${CHECKPOINTS[@]}"; do
  if [[ -n "$DIRECT_ADAPTER" ]]; then
    adapter="$DIRECT_ADAPTER"
  elif adapter="$(resolve_adapter "$checkpoint")"; then
    :
  else
    [[ ${#CHECKPOINTS[@]} -gt 1 ]] || die "no GRPO policy adapter (policy/ or combined/policy) at $checkpoint"
    echo "skip: no GRPO adapter at $checkpoint" >&2
    continue
  fi
  name="${checkpoint##*/}"
  out_json="$EVAL_RESULTS_DIR/${name}_passatk_${SAMPLING_METHOD}_${OUT_TAG}_t${TEMPERATURE}_topp${TOP_P}_b${EVAL_PROMPT_BATCH_SIZE}.json"
  out_stdout="$EVAL_RESULTS_DIR/${name}_${SAMPLING_METHOD}_${OUT_TAG}_stdout.log"
  cmd=(
    python "$SCRIPT_DIR/evaluate_model.py"
    --adapter "$adapter"
    --dataset "${EVAL_DATASET:-chrivasileiou/asap7-language-of-test-v2}"
    --backend vllm
    --tp_size "$TP_SIZE"
    --gpu_memory_utilization "${GPU_MEMORY_UTILIZATION:-0.85}"
    --sampling_method "$SAMPLING_METHOD"
    --num_completions "$NUM_COMPLETIONS"
    --k "${K_VALUES[@]}"
    --temperature "$TEMPERATURE"
    --top_p "$TOP_P"
    --max_new_tokens "${MAX_NEW_TOKENS:-16384}"
    --max_eval_samples "${MAX_EVAL_SAMPLES:-512}"
    --threshold_mode "${THRESHOLD_MODE:-fault_detected}"
    --eval_prompt_batch_size "$EVAL_PROMPT_BATCH_SIZE"
    --generation_micro_batch_size "${GENERATION_MICRO_BATCH_SIZE:-16}"
    --max_prompt_length "${MAX_PROMPT_LENGTH:-4096}"
    --report_to "${REPORT_TO:-wandb}"
    --output_file "$out_json"
    --wandb_run_name "${WANDB_RUN_NAME:-${EXPERIMENT}_${name}_policy_${SAMPLING_METHOD}}"
    "${SEARCH_ARGS[@]}"
  )
  [[ "$MERGE_DEQUANT" != 1 ]] || cmd+=(--merge_dequant)
  echo "Evaluating: $adapter (tp_size=$TP_SIZE)"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '%q ' "${cmd[@]}"
    printf '> %q\n' "$out_stdout"
  else
    mkdir -p "$EVAL_RESULTS_DIR"
    echo "stdout -> $out_stdout"
    "${cmd[@]}" >"$out_stdout"
  fi
  evaluated=$((evaluated + 1))
done
[[ "$evaluated" -gt 0 ]] || die "no evaluable GRPO adapters under $TARGET"
echo "Done. $evaluated checkpoint(s); results under: $EVAL_RESULTS_DIR"
