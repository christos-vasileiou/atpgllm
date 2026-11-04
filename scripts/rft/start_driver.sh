#!/bin/bash
# Start the RFT driver under sg trela; called by watchdog.sh inside an
# interactive bash, whose environment it passes on.
#   scripts/rft/start_driver.sh <plan.json> <driver log> [venv]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLAN="$1"
LOG="$2"
VENV="${3:-/proj/trela/christos/myenv}"
# sg is setuid and drops LD_LIBRARY_PATH, which tmax64 needs (~/usr/lib64),
# so the inner shell sets it again. tee -i survives a Ctrl-C, so the driver
# and its children can still log while they shut down.
exec sg trela -c "export LD_LIBRARY_PATH='$LD_LIBRARY_PATH'; . '$VENV/bin/activate'; cd '$REPO'; umask 002; python -u scripts/rft/run_rft.py --plan '$PLAN' 2>&1 | tee -i -a '$LOG'"
