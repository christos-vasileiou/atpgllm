#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [ -f /work/cxv200006/myenv/bin/activate ]; then
    source /work/cxv200006/myenv/bin/activate
elif [ -f /proj/trela/christos/myenv/bin/activate ]; then
    source /proj/trela/christos/myenv/bin/activate
fi

: "${HPO_CONFIG:?HPO_CONFIG is required}"
: "${HPO_STAGE:?HPO_STAGE is required}"
: "${HPO_SPLIT_MANIFEST:?HPO_SPLIT_MANIFEST is required}"
: "${OPTUNA_STORAGE:?OPTUNA_STORAGE PostgreSQL URL is required}"

CMD=(python -m atpgllm.graph.scripts.search_graph_pipeline worker
    --config "$HPO_CONFIG"
    --profile "${HPO_PROFILE:-production}"
    --stage "$HPO_STAGE"
    --split-manifest "$HPO_SPLIT_MANIFEST"
    --storage "$OPTUNA_STORAGE"
    --trials "${HPO_TRIALS_PER_WORKER:-1}"
    --seed "${HPO_SEED:-42}"
    --device "${HPO_DEVICE:-cuda}")

if [ -n "${HPO_PROMOTION_MANIFEST:-}" ]; then
    CMD+=(--promotion-manifest "$HPO_PROMOTION_MANIFEST")
fi
if [ -n "${HPO_WORKER_TIMEOUT:-}" ]; then
    CMD+=(--timeout "$HPO_WORKER_TIMEOUT")
fi

echo "stage=$HPO_STAGE array_task=${SLURM_ARRAY_TASK_ID:-local} host=$(hostname)"
exec "${CMD[@]}"
