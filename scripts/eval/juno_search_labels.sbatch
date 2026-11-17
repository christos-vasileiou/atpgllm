#!/bin/bash
#SBATCH --job-name=search_labels
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
#SBATCH --time=06:00:00
#SBATCH --output=runs/search_labels/logs/%x_%j.out
# Compact the simulator search into the graph model (scripts/eval/search_labels.py).
# From /work/cxv200006/atpg_retest_seq/libatpgllm, after mkdir -p runs/search_labels/logs:
#
#   STEP=search sbatch scripts/eval/juno_search_labels.sbatch            # CPU: search on training problems
#   STEP=merge BLEND=1.0 INTO=runs/search_labels/r1_hard sbatch -p dev -t 00:30:00 ...
#   STEP=train LABELS_NEW=runs/search_labels/r1_hard RUN=runs/search_labels/r1_hard_model \
#       sbatch -p h200 --gres=gpu:1 -t 04:00:00 ...                     # fine-tune the graph model
# Then hints for validation (juno_retest_seq_hints.sbatch with CHECKPOINT and OUT) and the
# decoder pass of juno_hint_arms_eval.sbatch (HINTS, OUT, ATTEMPTS=16) score the new model
# on the 603 rows of the hint arms.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STEP=${STEP:?set STEP=search, merge or train}
OUT=${OUT:-runs/search_labels/r1}
LABELS=${LABELS:-../data/freeset/retest_labels_v5}
HINTS=${HINTS:-runs/bridges_retest_seq/hints}
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 FAULT_SIM_BACKEND=fast OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export SIM_CONFIG=atpgllm/training/data/sim_config.json
if [ ! -d ../data_preprocessing ]; then
    export DATA_PREPROCESSING_ROOT="${DATA_PREPROCESSING_ROOT:-/work/cxv200006/transformers_atpg/data_preprocessing_seq-v5}"
fi
CACHE="${TMPDIR:-/tmp}/pyg_${SLURM_JOB_ID:-$$}"
echo "[search-labels] step=$STEP host=$(hostname) start=$(date -Is)"
head -1 PROVENANCE.txt
case "$STEP" in
    search)
        mkdir -p "$OUT/logs"
        pids=()
        for shard in $(seq 0 $(( ${SHARDS:-32} - 1 ))); do
            PYG_HOME="$CACHE/$shard" python scripts/eval/search_labels.py search --hints "$HINTS" --labels "$LABELS" \
                --out "$OUT" --shard "$shard" --limit "${LIMIT:-4000}" --attempts "${ATTEMPTS:-16}" \
                > "$OUT/logs/search_$shard.log" 2>&1 &
            pids+=($!)
        done
        failed=0
        for pid in "${pids[@]}"; do wait "$pid" || failed=$((failed + 1)); done
        cat "$OUT"/rows.s*.jsonl | wc -l
        tail -q -n 1 "$OUT"/logs/search_*.log | head -40
        echo "[search-labels] shards failed: $failed" ;;
    merge)
        python scripts/eval/search_labels.py merge --labels "$LABELS" --out "$OUT" --blend "${BLEND:-1.0}" \
            --into "${INTO:?set INTO}" ${ONLY_LATER:+--only-later} ;;
    train)
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        export PYG_HOME="$CACHE"
        python -m atpgllm.graph.scripts.train_retest --labels "${LABELS_NEW:?set LABELS_NEW}" \
            --steps "${STEPS:-4000}" --batch-size 64 --lr "${LR:-5e-5}" --max-eval "${MAX_EVAL:-4000}" \
            --eval-every 1000 --text-weight 1 --state-cells \
            --init "${INIT:-runs/bridges_retest_seq/r3s/best.pt}" --out "${RUN:?set RUN}" ;;
    *) echo "STEP must be search, merge or train"; exit 1 ;;
esac
echo "[search-labels] done $(date -Is)"
