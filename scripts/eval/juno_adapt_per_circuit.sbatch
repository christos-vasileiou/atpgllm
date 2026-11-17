#!/bin/bash
#SBATCH --job-name=adapt_circuit
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=300G
#SBATCH --time=04:00:00
#SBATCH --output=runs/adapt_per_circuit/logs/%x_%j.out
# Per-circuit adaptation of the graph model from its own simulator search
# (scripts/eval/adapt_per_circuit.py). CPU only, fast simulator.
# From /work/cxv200006/atpg_retest_seq/libatpgllm, after mkdir -p runs/adapt_per_circuit/logs:
#   OUT=runs/adapt_per_circuit/a1 STEPS=30 sbatch scripts/eval/juno_adapt_per_circuit.sbatch
#   STEP=report OUT=runs/adapt_per_circuit/a1 sbatch --cpus-per-task=2 --mem=16G -t 00:10:00 ...
#   SMOKE=1 OUT=runs/adapt_per_circuit/smoke_job sbatch --cpus-per-task=4 --mem=48G -t 00:30:00 ...
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
OUT=${OUT:?set OUT}
LABELS=${LABELS:-../data/freeset/retest_labels_v5}
CHECKPOINT=${CHECKPOINT:-runs/bridges_retest_seq/r3s/best.pt}
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 FAULT_SIM_BACKEND=fast OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SIM_CONFIG=atpgllm/training/data/sim_config.json
if [ ! -d ../data_preprocessing ]; then
    export DATA_PREPROCESSING_ROOT="${DATA_PREPROCESSING_ROOT:-/work/cxv200006/transformers_atpg/data_preprocessing_seq-v5}"
fi
CACHE="${TMPDIR:-/tmp}/pyg_${SLURM_JOB_ID:-$$}"
echo "[adapt] step=${STEP:-run} out=$OUT host=$(hostname) start=$(date -Is)"
head -1 PROVENANCE.txt
if [ "${STEP:-run}" = report ]; then
    python scripts/eval/adapt_per_circuit.py report --out "$OUT"
    echo "[adapt] done $(date -Is)"; exit 0
fi
COMMON="--checkpoint $CHECKPOINT --labels $LABELS --out $OUT --test ${TEST:-16} --adapt ${ADAPT:-64}
        --attempts ${ATTEMPTS:-16} --steps ${STEPS:-30} --batch-size ${BATCH:-16} --lr ${LR:-1e-4}"
mkdir -p "$OUT/logs"
if [ "${SMOKE:-0}" = 1 ]; then
    PYG_HOME="$CACHE/smoke" python scripts/eval/adapt_per_circuit.py run $COMMON --shards 64 --shard 0 --limit 2
    python scripts/eval/adapt_per_circuit.py report --out "$OUT"
    echo "[adapt] smoke done $(date -Is)"; exit 0
fi
SHARDS=${SHARDS:-32}
pids=()
for shard in $(seq 0 $((SHARDS - 1))); do
    PYG_HOME="$CACHE/$shard" python scripts/eval/adapt_per_circuit.py run $COMMON --shards "$SHARDS" --shard "$shard" \
        > "$OUT/logs/run_$shard.log" 2>&1 &
    pids+=($!)
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
echo "[adapt] shards failed: $failed"
tail -q -n 2 "$OUT"/logs/run_*.log | grep -i -E "error|Traceback" | head -5 || true
python scripts/eval/adapt_per_circuit.py report --out "$OUT"
echo "[adapt] done $(date -Is)"
