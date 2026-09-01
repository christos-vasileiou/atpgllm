#!/usr/bin/env bash
set -euo pipefail

CONFIG=${1:?"usage: submit_graph_hpo.sh <config.yaml> <stage> <split.json> [promotion.json]"}
STAGE=${2:?stage is required}
SPLIT=${3:?split manifest is required}
PROMOTION=${4:-}
: "${OPTUNA_STORAGE:?Export OPTUNA_STORAGE=postgresql+psycopg://... before submitting}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FREEZE_DIR="$REPO_ROOT/jobs/graph_hpo/launch"
mkdir -p "$FREEZE_DIR" "$REPO_ROOT/jobs/graph_hpo"

STAMP="$(date +%Y%m%d_%H%M%S)"
FROZEN_CONFIG="$FREEZE_DIR/${STAMP}_$(basename "$CONFIG")"
FROZEN_SPLIT="$FREEZE_DIR/${STAMP}_$(basename "$SPLIT")"
cp "$CONFIG" "$FROZEN_CONFIG"
cp "$SPLIT" "$FROZEN_SPLIT"

FROZEN_PROMOTION=
if [ -n "$PROMOTION" ]; then
    FROZEN_PROMOTION="$FREEZE_DIR/${STAMP}_$(basename "$PROMOTION")"
    cp "$PROMOTION" "$FROZEN_PROMOTION"
fi

ARRAY=${HPO_ARRAY:-0-15}
PARTITION=${HPO_PARTITION:-h200}
CPUS=${HPO_CPUS_PER_TASK:-16}
MEM=${HPO_MEM:-64G}
TIME_LIMIT=${HPO_TIME_LIMIT:-2-00:00:00}

export HPO_CONFIG="$FROZEN_CONFIG"
export HPO_STAGE="$STAGE"
export HPO_SPLIT_MANIFEST="$FROZEN_SPLIT"
export HPO_PROMOTION_MANIFEST="$FROZEN_PROMOTION"

echo "Submitting graph HPO stage=$STAGE array=$ARRAY partition=$PARTITION"
exec sbatch \
    --chdir="$REPO_ROOT" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:1 \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    --array="$ARRAY" \
    --job-name="hpo-${STAGE}" \
    --output="jobs/graph_hpo/${STAGE}_%A_%a.out" \
    --error="jobs/graph_hpo/${STAGE}_%A_%a.err" \
    --export=ALL \
    "$SCRIPT_DIR/run_graph_hpo_worker.sh"
