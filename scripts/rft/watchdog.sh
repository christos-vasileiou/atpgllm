#!/bin/bash
# Keep the RFT driver (scripts/rft/run_rft.py) running across reboots.
#
# cron entries (crontab -e), the first for boot, the second for anything else:
#   @reboot      /proj/.../libatpgllm_hard-faults/scripts/rft/watchdog.sh boot
#   */30 * * * * /proj/.../libatpgllm_hard-faults/scripts/rft/watchdog.sh
#
# The driver is started in tmux session $RFT_TMUX_SESSION unless it is already
# running, the plan has finished, or the driver halted after repeated failures
# of one stage (state.json "halted"; clear it to allow restarts). It runs
# through an interactive bash so it gets the same environment as a login
# terminal (TetraMAX on PATH, its license, HF_HOME, LD_LIBRARY_PATH), under
# sg trela so new files belong to group trela.
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLAN="${RFT_PLAN:-$REPO/scripts/rft/configs/rft_v4_a100.json}"
SESSION="${RFT_TMUX_SESSION:-rft_v4}"
VENV="${RFT_VENV:-/proj/trela/christos/myenv}"
LOG="$HOME/.rft_watchdog.log"
say() { echo "[watchdog $(date '+%F %T')] $*" >> "$LOG"; }

if [ "${1:-}" = "boot" ]; then
    # NFS and the NVIDIA driver can come up after cron starts.
    for _ in $(seq 1 60); do
        [ -f "$PLAN" ] && nvidia-smi >/dev/null 2>&1 && break
        sleep 30
    done
fi
if [ ! -f "$PLAN" ] || ! nvidia-smi >/dev/null 2>&1; then
    say "plan or GPUs unavailable; trying again later"
    exit 0
fi
RUN_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["run_dir"])' "$PLAN")"
case "$RUN_DIR" in /*) ;; *) RUN_DIR="$REPO/$RUN_DIR" ;; esac

exec 9>"$HOME/.rft_watchdog.lock"
flock -n 9 || exit 0

if [ -f "$RUN_DIR/driver.pid" ]; then
    pid="$(cat "$RUN_DIR/driver.pid")"
    if kill -0 "$pid" 2>/dev/null && grep -q run_rft.py "/proc/$pid/cmdline" 2>/dev/null; then
        exit 0
    fi
fi
if [ -f "$RUN_DIR/state.json" ]; then
    status="$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print("finished" if s.get("finished") else "halted" if s.get("halted") else "")' "$RUN_DIR/state.json")"
    if [ -n "$status" ]; then
        exit 0
    fi
fi

stamp="$(date +%Y%m%d_%H%M)"
mkdir -p "$RUN_DIR/logs"
driver_log="$RUN_DIR/logs/driver_$stamp.log"
say "starting the driver (log $driver_log)"
cmd="bash -ic '$REPO/scripts/rft/start_driver.sh $PLAN $driver_log $VENV' </dev/null; exec bash"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" "$cmd"
else
    tmux new-session -d -s "$SESSION" "$cmd"
fi
