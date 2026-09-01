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

case "${STAGE:?STAGE is required}" in
    graph_pretrain)
        CMD=(python -m atpgllm.graph.scripts.train_graph_encoder
            --dataset "$DATASET" --sim-config "$SIM_CONFIG"
            --output-dir "$OUTPUT_DIR"
            --batch-size "${BATCH_SIZE:-8}" --grad-accum "${GRAD_ACCUM:-1}"
            --max-steps "$MAX_STEPS" --save-every "$SAVE_EVERY")
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
        ;;
    multimodal_sft|multimodal_grpo)
        METHOD=${STAGE#multimodal_}
        CMD=(python scripts/train/multimodal_training.py
            --method "$METHOD" --dataset "$DATASET"
            --sim-config "$SIM_CONFIG" --output-dir "$OUTPUT_DIR"
            --alignment-ckpt "${ALIGNMENT_CHECKPOINT:?ALIGNMENT_CHECKPOINT is required}"
            --llm "${LLM:-Qwen/Qwen2.5-7B-Instruct}"
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
