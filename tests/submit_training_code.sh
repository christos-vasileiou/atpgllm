#!/bin/bash
# Freeze a training config at SLURM *submit* time, then queue run_training_code.sh.
#
# Why: sbatch only runs the launcher when the job starts. If you pass
# configs/grpo.conf directly, that file may change while the job waits in the
# queue. This wrapper copies the config immediately and submits the copy.
#
# Usage (from libatpgllm/tests):
#   ./submit_training_code.sh configs/grpo.conf
#   ./submit_training_code.sh configs/sft.conf
#
# Direct / interactive runs (no queue delay) may use either:
#   ./run_training_code.sh configs/sft.conf
#   ./submit_training_code.sh configs/sft.conf   # also freezes at submit time

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_ARG="${1:-}"

if [ -z "$CONFIG_ARG" ]; then
    echo "ERROR: no configuration file provided."
    echo "Usage: $0 <config_file>   (e.g. configs/grpo.conf)"
    exit 1
fi

CONFIG_FILE="$CONFIG_ARG"
if [ ! -f "$CONFIG_FILE" ]; then
    [ -f "$SCRIPT_DIR/$CONFIG_ARG" ] && CONFIG_FILE="$SCRIPT_DIR/$CONFIG_ARG"
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: configuration file not found: $CONFIG_ARG"
    exit 1
fi

LAUNCH_CONFIG_DIR="$SCRIPT_DIR/jobs/launch_configs"
mkdir -p "$LAUNCH_CONFIG_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
BASE="$(basename "$CONFIG_FILE")"
FROZEN="$LAUNCH_CONFIG_DIR/submit_${STAMP}_${BASE}"

cp "$CONFIG_FILE" "$FROZEN"
FROZEN_AT="$(date -Iseconds 2>/dev/null || date)"

cat > "${FROZEN}.meta" <<EOF
launch_config_original=${CONFIG_FILE}
launch_config_frozen_at=${FROZEN_AT}
launch_config_frozen_by=submit
submit_host=$(hostname)
submit_user=${USER:-unknown}
submit_cwd=$(pwd)
EOF

echo "Frozen config at submit time:"
echo "  original: $CONFIG_FILE"
echo "  frozen:   $FROZEN"
echo "  meta:     ${FROZEN}.meta"

# ---------------------------------------------------------------------------
# Build sbatch resource overrides from the (frozen) config.
#
# #SBATCH directives inside run_training_code.sh are static, so the topology
# (partition, node count, GPUs/node, CPUs, memory) is driven here instead:
# sbatch CLI flags take precedence over #SBATCH lines. This is what makes the
# SAME launcher work for single-node (4xH100: PARTITION=h100, NUM_NODES=1,
# GPUS_PER_NODE=4) and multi-node (H200: NUM_NODES=2/3, GPUS_PER_NODE=2).
# ---------------------------------------------------------------------------
# shellcheck source=/dev/null
source "$FROZEN"
PARTITION="${PARTITION:-h200}"
NUM_NODES="${NUM_NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-2}"
CPUS_PER_TASK="${CPUS_PER_TASK:-32}"
MEM="${MEM:-128G}"
JOB_NAME="$(basename "${OUTPUT_DIR:-${METHOD:-train}}")"

SBATCH_FLAGS=(
    --partition="$PARTITION"
    --nodes="$NUM_NODES"
    --ntasks-per-node=1
    --gres=gpu:"$GPUS_PER_NODE"
    --cpus-per-task="$CPUS_PER_TASK"
    --mem="$MEM"
    --job-name="$JOB_NAME"
)

RESERVATION="${RESERVATION:-}"
if [ -n "$RESERVATION" ]; then
    SBATCH_FLAGS+=(--reservation="$RESERVATION")
else
    echo "No reservation specified"
fi

echo "sbatch resource overrides (from config):"
echo "  ${SBATCH_FLAGS[*]}"

cd "$SCRIPT_DIR"
exec sbatch "${SBATCH_FLAGS[@]}" run_training_code.sh "$FROZEN"
