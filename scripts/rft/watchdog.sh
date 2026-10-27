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
# cron's umask would hide new run directories from the group.
umask 002
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PLAN="${RFT_PLAN:-$REPO/scripts/rft/configs/rft_v4_a100.json}"
SESSION="${RFT_TMUX_SESSION:-rft_v4}"
VENV="${RFT_VENV:-/proj/trela/christos/myenv}"
LOG="$HOME/.rft_watchdog.log"
say() { echo "[watchdog $(date '+%F %T')] $*" >> "$LOG"; chgrp trela "$LOG" 2>/dev/null; }

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
chgrp trela "$HOME/.rft_watchdog.lock" 2>/dev/null
flock -n 9 || exit 0

alive() {  # alive <pid file> <script name>
    [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null && grep -q "$2" "/proc/$(cat "$1")/cmdline" 2>/dev/null
}
status=""
if [ -f "$RUN_DIR/state.json" ]; then
    status="$(python3 -c 'import json,sys; s=json.load(open(sys.argv[1])); print("finished" if s.get("finished") else "halted" if s.get("halted") else "")' "$RUN_DIR/state.json")"
fi
mkdir -p "$RUN_DIR/logs"
chgrp trela "$RUN_DIR" "$RUN_DIR/logs" 2>/dev/null
chmod g+s "$RUN_DIR" "$RUN_DIR/logs" 2>/dev/null
# The W&B mirror (RFT_WANDB=0 turns it off) runs beside the driver, starting
# in the same pass as a fresh driver, and stops by itself after a final sync
# once the run finishes or halts.
if [ -z "$status" ] && [ "${RFT_WANDB:-1}" != 0 ] \
        && ! alive "$RUN_DIR/wandb_logger.pid" wandb_log.py; then
    say "starting the W&B logger (log $RUN_DIR/logs/wandb_log.log)"
    # 9>&-: the detached logger must not inherit the lock, or every later
    # pass finds it held and exits without restarting a dead driver.
    "$REPO/scripts/rft/start_logger.sh" "$RUN_DIR" "$RUN_DIR/logs/wandb_log.log" "$VENV" 9>&-
fi
if alive "$RUN_DIR/driver.pid" run_rft.py || [ -n "$status" ]; then
    exit 0
fi

stamp="$(date +%Y%m%d_%H%M)"
driver_log="$RUN_DIR/logs/driver_$stamp.log"
say "starting the driver (log $driver_log)"
cmd="bash -ic '$REPO/scripts/rft/start_driver.sh $PLAN $driver_log $VENV' </dev/null; exec bash"
if tmux has-session -t "$SESSION" 2>/dev/null; then
    tmux new-window -t "$SESSION" "$cmd"
else
    tmux new-session -d -s "$SESSION" "$cmd"
fi
