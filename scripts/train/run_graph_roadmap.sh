#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?"usage: bash scripts/train/run_graph_roadmap.sh <config.conf>"}
source "$CONFIG"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [ -f /proj/trela/christos/myenv/bin/activate ]; then
    source /proj/trela/christos/myenv/bin/activate
fi

DATASET=${DATASET:-chrivasileiou/asap7-language-of-test-v2}
SIM_CONFIG=${SIM_CONFIG:-atpgllm/training/data/sim_config.json}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-64}
MAX_STEPS=${MAX_STEPS:-20000}
SAVE_EVERY=${SAVE_EVERY:-1000}
DRY_RUN=${DRY_RUN:-False}
# This checkout's atpgllm, not the editable install (the primary checkout):
# python puts scripts/train, not the repo root, first on sys.path.
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
[ -n "${DATA_PREPROCESSING_ROOT:-}" ] && export DATA_PREPROCESSING_ROOT
[ -n "${CUDA_VISIBLE_DEVICES:-}" ] && export CUDA_VISIBLE_DEVICES
# The simulator behind the tool, the retry examples and the rewards. TetraMAX
# runs locally under the shared seat pool; a config may name a site profile
# that puts tmax on PATH (non-interactive shells do not source it).
FAULT_SIM_BACKEND=${FAULT_SIM_BACKEND:-fast}
export FAULT_SIM_BACKEND
if [ "$FAULT_SIM_BACKEND" = tetramax ]; then
    export TMAX_LOCK_DIR="${TMAX_LOCK_DIR:-$(cd "$REPO_ROOT/.." && pwd)/.runtime/tetramax}"
    if ! command -v "${TMAX_BIN:-tmax}" >/dev/null 2>&1 && [ -n "${SYNOPSYS_PROFILE:-}" ]; then
        set +u; source "$SYNOPSYS_PROFILE" >/dev/null; set -u
    fi
    # Libraries tmax64 links against that a host may lack (eng-tamale2 has no
    # libXss, libtiff.so.3, libmng or libpng12 installed).
    if [ -n "${TMAX_LIBRARY_PATH:-}" ]; then
        export LD_LIBRARY_PATH="$TMAX_LIBRARY_PATH${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi
    _REPO_ROOT="$REPO_ROOT"
    source scripts/train/_tetramax_lifecycle.sh
    # One 3-valued simulation before any GPU work (binary, license, library).
    TMAX_MANAGE_SERVICE=True preflight_tetramax_simulation
fi

case "${STAGE:?STAGE is required}" in
    graph_pretrain)
        CMD=(python -m atpgllm.graph.scripts.train_graph_encoder
            --dataset "$DATASET" --sim-config "$SIM_CONFIG"
            --output-dir "$OUTPUT_DIR"
            --batch-size "${BATCH_SIZE:-8}" --grad-accum "${GRAD_ACCUM:-1}"
            --max-steps "$MAX_STEPS" --save-every "$SAVE_EVERY")
        if [ -n "${EVAL_SPLIT:-}" ]; then
            CMD+=(--eval-split "$EVAL_SPLIT")
        else
            CMD+=(--split-manifest "${SPLIT_MANIFEST:?SPLIT_MANIFEST or EVAL_SPLIT is required}")
        fi
        [ -n "${EVAL_EVERY:-}" ] && CMD+=(--eval-every "$EVAL_EVERY")
        [ -n "${EVAL_FAULTS_PER_DESIGN:-}" ] && CMD+=(--eval-faults-per-design "$EVAL_FAULTS_PER_DESIGN")
        [ -n "${EVAL_GRAPHS_PER_DIFFICULTY:-}" ] && CMD+=(--eval-graphs-per-difficulty "$EVAL_GRAPHS_PER_DIFFICULTY")
        [ -n "${LOAD:-}" ] && CMD+=(--load "$LOAD")
        [ -n "${MAX_ROWS_PER_CIRCUIT:-}" ] && CMD+=(--max-rows-per-circuit "$MAX_ROWS_PER_CIRCUIT")
        [ -n "${EVAL_ROWS_PER_CIRCUIT:-}" ] && CMD+=(--eval-rows-per-circuit "$EVAL_ROWS_PER_CIRCUIT")
        [ -n "${REVISION:-}" ] && CMD+=(--revision "$REVISION")
        [ -n "${RESUME:-}" ] && CMD+=(--resume "$RESUME")
        ;;
    graph_text_alignment)
        CMD=(python -m atpgllm.graph.scripts.train_stage1
            --dataset "$DATASET" --sim-config "$SIM_CONFIG"
            --output-dir "$OUTPUT_DIR"
            --per-device-train-batch-size "${BATCH_SIZE:-8}"
            --grad-accum "${GRAD_ACCUM:-1}" --max-steps "$MAX_STEPS"
            --save-every "$SAVE_EVERY"
            --graph-policy "${GRAPH_POLICY:-frozen}")
        if [ -n "${RESUME:-}" ]; then
            CMD+=(--resume "$RESUME")
        else
            CMD+=(--graph-ckpt "${GRAPH_CHECKPOINT:?GRAPH_CHECKPOINT is required}")
        fi
        [ -n "${DESCRIPTIONS_JSON:-}" ] && CMD+=(--descriptions-json "$DESCRIPTIONS_JSON")
        [ -n "${GRAPH_INIT:-}" ] && CMD+=(--graph-init "$GRAPH_INIT")
        [ "${FREEZE_TEXT:-False}" = True ] && CMD+=(--freeze-text)
        [ -n "${REVISION:-}" ] && CMD+=(--revision "$REVISION")
        [ -n "${LOAD:-}" ] && CMD+=(--load "$LOAD")
        [ -n "${MAX_ROWS_PER_CIRCUIT:-}" ] && CMD+=(--max-rows-per-circuit "$MAX_ROWS_PER_CIRCUIT")
        [ -n "${EVAL_SPLIT:-}" ] && CMD+=(--eval-split "$EVAL_SPLIT")
        [ -n "${EVAL_EVERY:-}" ] && CMD+=(--eval-every "$EVAL_EVERY")
        [ -n "${EVAL_ROWS_PER_CIRCUIT:-}" ] && CMD+=(--eval-rows-per-circuit "$EVAL_ROWS_PER_CIRCUIT")
        [ -n "${EVAL_FAULTS_PER_CIRCUIT:-}" ] && CMD+=(--eval-faults-per-circuit "$EVAL_FAULTS_PER_CIRCUIT")
        [ -n "${EVAL_PAIRS:-}" ] && CMD+=(--eval-pairs "$EVAL_PAIRS")
        [ -n "${SELECT_METRIC:-}" ] && CMD+=(--select-metric "$SELECT_METRIC")
        [ -n "${MAX_TEXT_LEN:-}" ] && CMD+=(--max-text-len "$MAX_TEXT_LEN")
        [ -n "${MAX_ANSWER_LEN:-}" ] && CMD+=(--max-answer-len "$MAX_ANSWER_LEN")
        [ -n "${NUM_WORKERS:-}" ] && CMD+=(--num-workers "$NUM_WORKERS")
        [ -n "${LR:-}" ] && CMD+=(--lr "$LR")
        [ -n "${SEED:-}" ] && CMD+=(--seed "$SEED")
        ;;
    multimodal_sft|multimodal_grpo)
        METHOD=${STAGE#multimodal_}
        # NPROC_PER_NODE > 1: SFT data-parallel, one rank per GPU (torchrun).
        NPROC_PER_NODE=${NPROC_PER_NODE:-1}
        if [ "$NPROC_PER_NODE" -gt 1 ]; then
            LAUNCH=(torchrun --standalone --nproc-per-node "$NPROC_PER_NODE")
        else
            LAUNCH=(python)
        fi
        CMD=("${LAUNCH[@]}" scripts/train/multimodal_training.py
            --method "$METHOD" --dataset "$DATASET"
            --sim-config "$SIM_CONFIG" --output-dir "$OUTPUT_DIR"
            --alignment-ckpt "${ALIGNMENT_CHECKPOINT:?ALIGNMENT_CHECKPOINT is required}"
            --llm "${LLM:-ibm-granite/granite-4.2-8b}"
            --graph-policy "${GRAPH_POLICY:-full}"
            --batch-size "${BATCH_SIZE:-1}" --grad-accum "${GRAD_ACCUM:-16}"
            --max-steps "$MAX_STEPS" --save-every "$SAVE_EVERY"
            --max-seq-len "${MAX_SEQ_LEN:-4096}"
            --max-prompt-length "${MAX_PROMPT_LENGTH:-2048}"
            --max-completion-length "${MAX_COMPLETION_LENGTH:-2048}"
            --lora-r "${LORA_R:-16}" --lora-alpha "${LORA_ALPHA:-32}"
            --num-generations "${NUM_GENERATIONS:-8}"
            --beta "${GRPO_BETA:-0.03}")
        [ -n "${RESUME:-}" ] && CMD+=(--resume "$RESUME")
        [ -n "${ADAPTER:-}" ] && CMD+=(--adapter "$ADAPTER")
        [ -n "${REVISION:-}" ] && CMD+=(--revision "$REVISION")
        [ -n "${LOAD:-}" ] && CMD+=(--load "$LOAD")
        [ -n "${MAX_ROWS_PER_CIRCUIT:-}" ] && CMD+=(--max-rows-per-circuit "$MAX_ROWS_PER_CIRCUIT")
        [ "${TOOL_FORMAT:-True}" = False ] && CMD+=(--no-tool-format)
        [ -n "${RETRY_FRACTION:-}" ] && CMD+=(--retry-fraction "$RETRY_FRACTION")
        [ -n "${EVAL_RETRY_FRACTION:-}" ] && CMD+=(--eval-retry-fraction "$EVAL_RETRY_FRACTION")
        [ -n "${EXAMPLE_WORKERS:-}" ] && CMD+=(--example-workers "$EXAMPLE_WORKERS")
        [ -n "${TOOL_VIEW:-}" ] && CMD+=(--tool-view "$TOOL_VIEW")
        [ -n "${NETLIST_VIEW:-}" ] && CMD+=(--netlist-view "$NETLIST_VIEW")
        [ -n "${PREFIX_AT:-}" ] && CMD+=(--prefix-at "$PREFIX_AT")
        [ -n "${TARGET:-}" ] && CMD+=(--target "$TARGET")
        [ "${GRAPH_OFF:-False}" = True ] && CMD+=(--graph-off)
        [ -n "${STOP_AFTER:-}" ] && CMD+=(--stop-after "$STOP_AFTER")
        [ -n "${FUSION:-}" ] && CMD+=(--fusion "$FUSION")
        [ -n "${ANCHOR_GATE_LR:-}" ] && CMD+=(--anchor-gate-lr "$ANCHOR_GATE_LR")
        [ -n "${HINTS:-}" ] && CMD+=(--hints "$HINTS")
        [ "${FREEZE_GRAPH_BN:-False}" = True ] && CMD+=(--freeze-graph-bn)
        [ -n "${MAX_TOOL_CALLS:-}" ] && CMD+=(--max-tool-calls "$MAX_TOOL_CALLS")
        [ -n "${ADVANTAGE:-}" ] && CMD+=(--advantage "$ADVANTAGE")
        [ -n "${TEMPERATURE:-}" ] && CMD+=(--temperature "$TEMPERATURE")
        [ -n "${PROBE_MAX_NEW_TOKENS:-}" ] && CMD+=(--probe-max-new-tokens "$PROBE_MAX_NEW_TOKENS")
        [ -n "${EVAL_SPLIT:-}" ] && CMD+=(--eval-split "$EVAL_SPLIT")
        [ -n "${EVAL_EVERY:-}" ] && CMD+=(--eval-every "$EVAL_EVERY")
        [ -n "${EVAL_EXAMPLES:-}" ] && CMD+=(--eval-examples "$EVAL_EXAMPLES")
        [ -n "${PROBE_EXAMPLES:-}" ] && CMD+=(--probe-examples "$PROBE_EXAMPLES")
        [ -n "${PROBE_EVERY:-}" ] && CMD+=(--probe-every "$PROBE_EVERY")
        [ -n "${LR:-}" ] && CMD+=(--lr "$LR")
        [ -n "${GRAPH_LR:-}" ] && CMD+=(--graph-lr "$GRAPH_LR")
        [ -n "${GRAPH_ENCODER_LR:-}" ] && CMD+=(--graph-encoder-lr "$GRAPH_ENCODER_LR")
        [ -n "${WARMUP_STEPS:-}" ] && CMD+=(--warmup-steps "$WARMUP_STEPS")
        [ -n "${WEIGHT_DECAY:-}" ] && CMD+=(--weight-decay "$WEIGHT_DECAY")
        [ -n "${LR_SCHEDULE:-}" ] && CMD+=(--lr-schedule "$LR_SCHEDULE")
        [ -n "${ATTN_IMPLEMENTATION:-}" ] && CMD+=(--attn-implementation "$ATTN_IMPLEMENTATION")
        [ -n "${SAVE_TOTAL_LIMIT:-}" ] && CMD+=(--save-total-limit "$SAVE_TOTAL_LIMIT")
        [ -n "${MICRO_BATCH_TOKENS:-}" ] && CMD+=(--micro-batch-tokens "$MICRO_BATCH_TOKENS")
        [ -n "${EVAL_ROWS_PER_CIRCUIT:-}" ] && CMD+=(--eval-rows-per-circuit "$EVAL_ROWS_PER_CIRCUIT")
        [ -n "${LOG_EVERY:-}" ] && CMD+=(--log-every "$LOG_EVERY")
        [ -n "${SEED:-}" ] && CMD+=(--seed "$SEED")
        if [ "$METHOD" = grpo ] && [ -z "${RESUME:-}" ]; then
            CMD+=(--sft-ckpt "${SFT_CHECKPOINT:?SFT_CHECKPOINT is required}")
        fi
        ;;
    *)
        echo "unknown STAGE=$STAGE" >&2
        exit 2
        ;;
esac

printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'
[ "$DRY_RUN" = True ] || exec "${CMD[@]}"
