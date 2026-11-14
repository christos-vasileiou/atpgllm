#!/bin/bash
#SBATCH --job-name=retest_seq_hints
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=08:00:00
#SBATCH --output=runs/bridges_retest_seq/hints_%j.out
# Plain and rich hints for every v5 problem, from the graph model trained on
# both kinds of circuit. CPU only: one process per shard, one thread each.
#
#   cd /work/cxv200006/atpg_retest_seq/libatpgllm
#   sbatch scripts/train/juno_retest_seq_hints.sbatch
#
# A run that was cut off continues when submitted again: finished problems
# are skipped per shard file. Each process gets its own PyG cache folder:
# torch_geometric writes a generated module there at start, and processes
# that share the folder read each other's half-written file (4 of 32 shards
# failed that way on 2026-11-12). A shard that fails is run once more.
set -euo pipefail
_REPO_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$_REPO_ROOT"
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
export PYTHONPATH="$_REPO_ROOT" FAULT_SIM_BACKEND=fast OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export DATA_PREPROCESSING_ROOT="${DATA_PREPROCESSING_ROOT:-/work/cxv200006/transformers_atpg/data_preprocessing_seq-v5}"
LABELS="${LABELS:-../data/freeset/retest_labels_v5}"
CHECKPOINT="${CHECKPOINT:-runs/bridges_retest_seq/r3s/best.pt}"
OUT="${OUT:-runs/bridges_retest_seq/hints}"
SHARDS="${SHARDS:-${SLURM_CPUS_PER_TASK:-32}}"
echo "[hints] $(hostname) start=$(date -Is) checkpoint=$CHECKPOINT labels=$LABELS shards=$SHARDS"
head -1 PROVENANCE.txt
mkdir -p "$OUT/logs"
CACHE="${TMPDIR:-/tmp}/pyg_${SLURM_JOB_ID:-$$}"
one_shard() {   # split shard
    PYG_HOME="$CACHE/$2" python -m atpgllm.graph.scripts.retest_hints --checkpoint "$CHECKPOINT" \
        --labels "$LABELS" --split "$1" --out "$OUT" --shards "$SHARDS" --shard "$2" --device cpu \
        > "$OUT/logs/${1}_${2}.log" 2>&1
}
for split in ${SPLITS:-validation train}; do
    pids=()
    for shard in $(seq 0 $((SHARDS - 1))); do
        one_shard "$split" "$shard" &
        pids+=($!)
    done
    again=()
    for shard in $(seq 0 $((SHARDS - 1))); do
        wait "${pids[$shard]}" || again+=("$shard")
    done
    for shard in "${again[@]}"; do
        echo "[hints] $split shard $shard failed, running it again"
        tail -3 "$OUT/logs/${split}_${shard}.log"
        if ! one_shard "$split" "$shard"; then
            echo "[hints] $split shard $shard failed twice"; tail -8 "$OUT/logs/${split}_${shard}.log"; exit 1
        fi
    done
    echo "[hints] $split done $(date -Is): $(cat "$OUT"/${split}_*.jsonl | wc -l) problems"
    python - "$OUT" "$split" <<'PY'
import glob, json, sys
out, split = sys.argv[1:]
t = {}
for path in glob.glob(f"{out}/summary_{split}_*.json"):
    for k, v in json.load(open(path)).items():
        t[k] = t.get(k, 0) + v
n = max(1, t.get("problems", 0))
print(f"[hints] {split}: problems {t.get('problems')} rejected {t.get('rejected')} "
      f"same inputs {t.get('same_inputs', 0) / n:.3f} same vector {t.get('same_vector', 0) / n:.3f} "
      f"value agreement {t.get('shared_same_value', 0) / max(1, t.get('shared', 0)):.3f}")
PY
done
echo "[hints] done $(date -Is)"
