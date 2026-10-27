#!/bin/bash
# Start the W&B mirror of an RFT run (scripts/rft/wandb_log.py) under sg trela,
# detached; called by watchdog.sh. It syncs every RFT_WANDB_EVERY seconds
# (default 900) and exits after the run finishes or halts.
#   scripts/rft/start_logger.sh <run_dir> <log> [venv]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="$1"
LOG="$2"
VENV="${3:-/proj/trela/christos/myenv}"
EVERY="${RFT_WANDB_EVERY:-900}"
PROJECT="${RFT_WANDB_PROJECT:-rft-training}"
setsid nohup sg trela -c ". '$VENV/bin/activate'; cd '$REPO'; umask 002; exec python -u scripts/rft/wandb_log.py --run_dir '$RUN_DIR' --project '$PROJECT' --every $EVERY >> '$LOG' 2>&1" </dev/null >/dev/null 2>&1 &
