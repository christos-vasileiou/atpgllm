#!/bin/bash
# Run a command with a managed TetraMAX service, then stop the service.
#
#   scripts/rft/with_tetramax.sh python scripts/rft/fanout.py --gpus 0,1,2,3 --log_dir <dir> -- ...
#
# Every process the command starts reaches the service through TMAX_SERVER_FILE.
# TMAX_MAX_CONCURRENT (default 16) is the service's simulation workers, shared
# by all clients, as in the GDPO configs.
set -euo pipefail
_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$_REPO_ROOT"
mkdir -p jobs
export PYTHONPATH="$_REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export FAULT_SIM_BACKEND=tetramax TMAX_MANAGE_SERVICE=True
export TMAX_MAX_CONCURRENT="${TMAX_MAX_CONCURRENT:-16}" TMAX_TIMEOUT_S="${TMAX_TIMEOUT_S:-120}"
export TMAX_ACQUIRE_TIMEOUT_S="${TMAX_ACQUIRE_TIMEOUT_S:-120}"
# shellcheck source=../train/_tetramax_lifecycle.sh
source scripts/train/_tetramax_lifecycle.sh
# A signal must still run the EXIT trap so the service never outlives the command.
trap 'exit 143' TERM
trap 'exit 130' INT
trap stop_tetramax_service EXIT
start_tetramax_service
preflight_tetramax_simulation
"$@"
