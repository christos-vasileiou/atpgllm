#!/usr/bin/env bash
# Sourced by run_training_code.sh after topology resolution. The service runs
# on the batch host; multi-node clients use the shared credentials file.
TMAX_SERVICE_PID=""

start_tetramax_service() {
    case "${TMAX_MANAGE_SERVICE:-False}" in
        True|true|1) ;;
        False|false|0) return 0 ;;
        *) echo "ERROR: TMAX_MANAGE_SERVICE must be True or False."; return 1 ;;
    esac
    if [ "${FAULT_SIM_BACKEND:-}" != tetramax ]; then
        echo "ERROR: TMAX_MANAGE_SERVICE=True requires FAULT_SIM_BACKEND=tetramax."
        return 1
    fi

    local project_root service_python bind_host advertise_host deadline
    project_root="$(cd "$_REPO_ROOT/.." && pwd)"
    service_python="${SIM_PYTHON:-python}"
    export TMAX_LOCK_DIR="${TMAX_LOCK_DIR:-$project_root/.runtime/tetramax}"
    TMAX_LOCK_DIR="$("$service_python" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$TMAX_LOCK_DIR")" || return 1
    # The coordinator always writes server.json in its pool directory. Resolve
    # this once so srun clients on other nodes receive an absolute shared path.
    export TMAX_SERVER_FILE="$TMAX_LOCK_DIR/server.json"
    unset TMAX_SERVER_URL TMAX_SERVER_TOKEN
    local timeout="${TMAX_SERVICE_STARTUP_TIMEOUT_S:-60}"
    if ! [[ "$timeout" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: TMAX_SERVICE_STARTUP_TIMEOUT_S must be a positive integer."
        return 1
    fi
    bind_host=127.0.0.1
    advertise_host=127.0.0.1
    if [ "${MULTINODE:-false}" == true ]; then
        bind_host=0.0.0.0
        advertise_host="${TMAX_SERVICE_ADVERTISE_HOST:-$(hostname -f)}"
    fi
    local -a command=("$service_python" -u "$project_root/data_preprocessing/tetramax_service.py"
        --host "$bind_host" --advertise-host "$advertise_host"
        --workers "${TMAX_SERVICE_WORKERS:-${TMAX_MAX_CONCURRENT:-16}}"
        --queue-size "${TMAX_QUEUE_SIZE:-128}" --port "${TMAX_SERVICE_PORT:-0}")
    printf 'TetraMAX service command: '; printf '%q ' "${command[@]}"; printf '\n'
    if [ "${DRY_RUN:-False}" == True ]; then
        echo "DRY RUN: would wait up to ${timeout}s for authenticated TetraMAX health, then stop the service on exit."
        return 0
    fi
    if [ -n "${TMAX_SERVICE_MODULE:-}" ]; then
        if ! type module >/dev/null 2>&1 && [ -f /etc/profile.d/lmod.sh ]; then
            source /etc/profile.d/lmod.sh
        fi
        if ! type module >/dev/null 2>&1; then
            echo "ERROR: module command unavailable; load TetraMAX or set TMAX_BIN."
            return 1
        fi
        module load "$TMAX_SERVICE_MODULE" || return 1
    fi
    TMAX_SERVICE_LOG=$(mktemp "$_REPO_ROOT/jobs/tetramax_${SLURM_JOB_ID:-local}_XXXXXX.log") || return 1
    echo "Starting TetraMAX service; log: $TMAX_SERVICE_LOG"
    "${command[@]}" >"$TMAX_SERVICE_LOG" 2>&1 &
    TMAX_SERVICE_PID=$!
    deadline=$((SECONDS + timeout))
    while kill -0 "$TMAX_SERVICE_PID" 2>/dev/null; do
        # Wait for THIS process to announce startup before reading credentials.
        # A stale file or an existing coordinator must not satisfy readiness.
        if grep -q '^TetraMAX service ready;' "$TMAX_SERVICE_LOG" && \
            "$service_python" - "$TMAX_SERVER_FILE" <<'PY'
import json
import sys
from pathlib import Path
from urllib import request

try:
    credentials = json.loads(Path(sys.argv[1]).read_text())
    req = request.Request(credentials['url'].rstrip('/') + '/health',
                          headers={'Authorization': 'Bearer ' + credentials['token']})
    with request.build_opener(request.ProxyHandler({})).open(req, timeout=1) as response:
        health = json.load(response)
    if health.get('status') != 'ok' or not health.get('simulator'):
        sys.exit(1)
except (OSError, ValueError, KeyError):
    sys.exit(1)
PY
        then
            echo "TetraMAX service is ready (PID $TMAX_SERVICE_PID); credentials: $TMAX_SERVER_FILE"
            return 0
        fi
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "ERROR: TetraMAX service did not become ready within ${timeout}s."
            tail -n 30 "$TMAX_SERVICE_LOG"
            return 1
        fi
        sleep 1
    done
    echo "ERROR: TetraMAX service exited before becoming ready. See $TMAX_SERVICE_LOG"
    tail -n 30 "$TMAX_SERVICE_LOG"
    return 1
}

stop_tetramax_service() {
    [ -n "$TMAX_SERVICE_PID" ] || return 0
    echo "Stopping TetraMAX service (PID: $TMAX_SERVICE_PID)..."
    kill "$TMAX_SERVICE_PID" 2>/dev/null || true
    local deadline=$((SECONDS + 30))
    while kill -0 "$TMAX_SERVICE_PID" 2>/dev/null; do
        if [ "$SECONDS" -ge "$deadline" ]; then
            echo "TetraMAX service did not stop within 30s; terminating it."
            kill -KILL "$TMAX_SERVICE_PID" 2>/dev/null || true
            break
        fi
        sleep 1
    done
    wait "$TMAX_SERVICE_PID" 2>/dev/null || true
    TMAX_SERVICE_PID=""
}
