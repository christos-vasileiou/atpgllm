#!/usr/bin/env bash
#SBATCH --job-name=tetramax-service
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=tetramax-service-%j.out
set -euo pipefail
PROJECT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd)
cd "$PROJECT_ROOT"
if ! type module >/dev/null 2>&1; then
    if [[ -f /etc/profile.d/lmod.sh ]]; then
        source /etc/profile.d/lmod.sh
    fi
fi
module load tetramax/vO-2018.06-SP1
export TMAX_MAX_CONCURRENT="${TMAX_MAX_CONCURRENT:-16}"
export TMAX_LOCK_DIR="${TMAX_LOCK_DIR:-$PROJECT_ROOT/.runtime/tetramax}"
export TMAX_TIMEOUT_S="${TMAX_TIMEOUT_S:-120}"
export TMAX_ACQUIRE_TIMEOUT_S="${TMAX_ACQUIRE_TIMEOUT_S:-120}"
SIM_PYTHON="${SIM_PYTHON:-/work/cxv200006/myenv/bin/python}"
if [[ "${TMAX_RECONFIGURE_DRAINED:-0}" == 1 ]]; then
    "$SIM_PYTHON" data_preprocessing/tetramax_pool.py resume
fi
exec "$SIM_PYTHON" data_preprocessing/tetramax_service.py \
    --host 0.0.0.0 --workers "${TMAX_SERVICE_WORKERS:-16}" \
    --queue-size "${TMAX_QUEUE_SIZE:-128}" --port "${TMAX_SERVICE_PORT:-8766}"
