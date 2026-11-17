#!/bin/bash
#SBATCH --job-name=hint_probe_llm
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=runs/hint_arms_eval/logs/%x_%j.out
# A served language model on the rows of the hint arms, with the rich hint and
# the simulator in the loop, no training (scripts/eval/hint_probe_llm.py).
# From /work/cxv200006/atpg_retest_seq/libatpgllm:
#
#   sbatch scripts/eval/juno_hint_probe_llm.sbatch                 # gpt-oss-120b, both views
#   VIEWS=ports LIMIT=6 sbatch -t 01:30:00 scripts/eval/juno_hint_probe_llm.sbatch   # smoke
#   TAG=gptoss_plan VIEWS=ports PLAN=1 EFFORT=medium sbatch scripts/eval/juno_hint_probe_llm.sbatch
#       # second probe (11-15): the decoder's first pattern and ranked changes in the prompt
#
# The rows it writes go to OUT (rollout_<TAG>_<view>.jsonl) and are scored by
#   STEP=report OUT=$OUT BASE="runs/hint_arms_eval runs/hint_arms_eval_long" CALLS=8 \
#       sbatch -p dev --gres=none -t 00:20:00 scripts/eval/juno_hint_arms_eval.sbatch
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT=${OUT:-runs/hint_arms_eval_probe}
SERVED=${SERVED:-gpt-oss-120b}
WEIGHTS=${WEIGHTS:-/work/cxv200006/models/gpt-oss-120b}
TAG=${TAG:-gptoss}
VIEWS=${VIEWS:-ports netlist}
CALLS=${CALLS:-8}
PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 ))
HINTS=${HINTS:-runs/bridges_retest_seq/hints}
export SIM_CONFIG=atpgllm/training/data/sim_config.json
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
export PYG_HOME="${TMPDIR:-/tmp}/pyg_${SLURM_JOB_ID:-$$}"
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
if [ ! -d ../data_preprocessing ]; then
    export DATA_PREPROCESSING_ROOT="${DATA_PREPROCESSING_ROOT:-/work/cxv200006/transformers_atpg/data_preprocessing_seq-v5}"
fi
echo "[probe] model=$SERVED views=$VIEWS host=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
head -1 PROVENANCE.txt
mkdir -p "$OUT"
[ -f "$OUT/rows.jsonl" ] || cp runs/hint_arms_eval/rows.jsonl "$OUT/rows.jsonl"
LOG="$OUT/vllm_${SLURM_JOB_ID:-local}.log"
vllm serve "$WEIGHTS" --served-model-name "$SERVED" --port "$PORT" --max-model-len "${MAX_MODEL_LEN:-98304}" \
    --gpu-memory-utilization "${GPU_UTIL:-0.92}" > "$LOG" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for i in $(seq 1 180); do
    if curl -s -m 5 "http://localhost:$PORT/v1/models" | grep -q "$SERVED"; then break; fi
    if ! kill -0 $SERVER 2>/dev/null; then echo "[probe] the server stopped"; tail -30 "$LOG"; exit 1; fi
    sleep 10
done
curl -s -m 5 "http://localhost:$PORT/v1/models" | grep -q "$SERVED" || { echo "[probe] the server did not come up"; tail -30 "$LOG"; exit 1; }
echo "[probe] server up $(date -Is)"
set +u
[ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
module load tetramax/vO-2018.06-SP1
set -u
export FAULT_SIM_BACKEND=tetramax TMAX_LOCK_DIR="$(cd .. && pwd)/.runtime/tetramax_${SLURM_JOB_ID:-local}"
export TMAX_MAX_CONCURRENT=${TMAX_MAX_CONCURRENT:-8}
mkdir -p "$TMAX_LOCK_DIR"
_REPO_ROOT="$PWD" TMAX_SERVICE_MODULE=tetramax/vO-2018.06-SP1
source scripts/train/_tetramax_lifecycle.sh
TMAX_MANAGE_SERVICE=True preflight_tetramax_simulation
for view in $VIEWS; do
    python scripts/eval/hint_probe_llm.py --out "$OUT" --name "${TAG}_${view}" --view "$view" --hints "$HINTS" \
        --base-url "http://localhost:$PORT/v1" --model "$SERVED" --max-tool-calls "$CALLS" \
        --reasoning-effort "${EFFORT:-high}" --max-turn-tokens "${TURN_TOKENS:-16384}" --workers "${WORKERS:-8}" ${PLAN:+--plan} ${LIMIT:+--limit $LIMIT}
    echo "[probe] view $view done $(date -Is)"
done
echo "[probe] done $(date -Is)"
