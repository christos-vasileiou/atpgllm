#!/usr/bin/env bash
set -euo pipefail

PLAN=${1:?"usage: submit_graph_hpo_plan.sh <screening-plan.json>"}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
FREEZE_DIR="$REPO_ROOT/jobs/graph_hpo/launch"
mkdir -p "$FREEZE_DIR" "$REPO_ROOT/jobs/graph_hpo"

STAMP="$(date +%Y%m%d_%H%M%S)"
FROZEN="$FREEZE_DIR/${STAMP}_$(basename "$PLAN")"
cp "$PLAN" "$FROZEN"

if [ -f /proj/trela/christos/myenv/bin/activate ]; then
    source /proj/trela/christos/myenv/bin/activate
fi
COUNT=$(python -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["entries"]))' "$FROZEN")
if [ "$COUNT" -lt 1 ]; then
    echo "Plan contains no entries." >&2
    exit 2
fi

PARTITION=${HPO_PARTITION:-h200}
GPUS=${HPO_GPUS_PER_TASK:-2}
CPUS=${HPO_CPUS_PER_TASK:-32}
MEM=${HPO_MEM:-128G}
TIME_LIMIT=${HPO_TIME_LIMIT:-4-00:00:00}
MAX_PARALLEL=${HPO_MAX_PARALLEL:-2}
export HPO_PLAN="$FROZEN"

exec sbatch \
    --chdir="$REPO_ROOT" \
    --partition="$PARTITION" \
    --nodes=1 \
    --ntasks=1 \
    --gres=gpu:"$GPUS" \
    --cpus-per-task="$CPUS" \
    --mem="$MEM" \
    --time="$TIME_LIMIT" \
    --array="0-$((COUNT - 1))%${MAX_PARALLEL}" \
    --job-name=graph-screen \
    --output="jobs/graph_hpo/screen_%A_%a.out" \
    --error="jobs/graph_hpo/screen_%A_%a.err" \
    --export=ALL \
    "$SCRIPT_DIR/run_graph_hpo_plan_entry.sh"
