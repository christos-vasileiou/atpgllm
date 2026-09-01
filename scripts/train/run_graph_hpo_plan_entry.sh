#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
if [ -f /work/cxv200006/myenv/bin/activate ]; then
    source /work/cxv200006/myenv/bin/activate
elif [ -f /proj/trela/christos/myenv/bin/activate ]; then
    source /proj/trela/christos/myenv/bin/activate
fi

: "${HPO_PLAN:?HPO_PLAN is required}"
exec python -m atpgllm.graph.scripts.search_graph_pipeline execute-plan \
    --plan "$HPO_PLAN"
