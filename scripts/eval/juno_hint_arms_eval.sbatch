#!/bin/bash
#SBATCH --job-name=hint_arms_eval
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=96G
#SBATCH --time=20:00:00
#SBATCH --output=runs/hint_arms_eval/logs/%x_%j.out
# Hint arms on v5: rollouts with the simulator in the loop, the graph model
# alone on the same rows, and the report (scripts/eval/hint_arms_eval.py).
# From /work/cxv200006/atpg_retest_seq/libatpgllm, after mkdir -p runs/hint_arms_eval/logs:
#
#   STEP=rows sbatch -p dev --gres=none -t 01:00:00 scripts/eval/juno_hint_arms_eval.sbatch
#   STEP=decoder sbatch -p normal --gres=none scripts/eval/juno_hint_arms_eval.sbatch
#   STEP=rollout NAME=plain RUN=runs/multimodal_sft_hint_v5_plain SHARD=0 SHARDS=3 sbatch \
#       scripts/eval/juno_hint_arms_eval.sbatch
#   STEP=report sbatch -p dev --gres=none -t 00:20:00 scripts/eval/juno_hint_arms_eval.sbatch
#
# Second pass with wider limits (11-14), in its own folder over the first one:
# the rows that ran into the token budget or the call limit, again with
# TOKENS=16384 CALLS=8 STOP_AT_FAULTS=1, and the report over both folders.
#   STEP=rollout NAME=plain RUN=... OUT=runs/hint_arms_eval_long REDO_FROM=runs/hint_arms_eval \
#       TOKENS=16384 CALLS=8 STOP_AT_FAULTS=1 SHARD=0 SHARDS=3 sbatch scripts/eval/juno_hint_arms_eval.sbatch
#   STEP=report OUT=runs/hint_arms_eval_long BASE=runs/hint_arms_eval CALLS=8 sbatch -p dev ...
# A latent arm takes GRAPH=real, zero or shuffled (the graph it is given).
#
# rollout and decoder simulate with TetraMAX (x stays unknown, as in training).
# Every pass skips finished rows, so a job cut off by its time limit can be
# submitted again.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STEP=${STEP:?set STEP=rows, rollout, decoder or report}
OUT=${OUT:-runs/hint_arms_eval}
# Not DATASET and not SIM_CONFIG from the caller: the juno login shell
# exports both for the text pipeline (DATASET=freeset).
EVAL_DATASET=${EVAL_DATASET:-../data/freeset/dataset.freeset.asap7sc7p5t_28.rvt.tt.v5}
export SIM_CONFIG=atpgllm/training/data/sim_config.json
LABELS=${LABELS:-../data/freeset/retest_labels_v5}
HINTS=${HINTS:-runs/bridges_retest_seq/hints}
CALLS=${CALLS:-3}
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
# Its own PyG cache folder: jobs that start together must not share one.
export PYG_HOME="${TMPDIR:-/tmp}/pyg_${SLURM_JOB_ID:-$$}"
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
if [ ! -d ../data_preprocessing ]; then
    export DATA_PREPROCESSING_ROOT="${DATA_PREPROCESSING_ROOT:-/work/cxv200006/transformers_atpg/data_preprocessing_seq-v5}"
fi
echo "[hint-eval] step=$STEP name=${NAME:-} shard=${SHARD:-0}/${SHARDS:-1} host=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
head -1 PROVENANCE.txt
mkdir -p "$OUT"
tetramax() {
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
}
case "$STEP" in
    rows)
        export FAULT_SIM_BACKEND=fast
        python scripts/eval/hint_arms_eval.py rows --dataset "$EVAL_DATASET" --per-circuit "${PER_CIRCUIT:-3}" --out "$OUT" ;;
    rollout)
        RUN=${RUN:?set RUN to the output directory of the arm}
        CKPT=${CKPT:-step40}
        tetramax
        python scripts/eval/hint_arms_eval.py rollout --name "${NAME:?set NAME}" --out "$OUT" \
            --checkpoint "$RUN/multimodal_sft_${CKPT}.pt" --adapter "$RUN/adapter_${CKPT}" \
            --max-tool-calls "$CALLS" --max-completion-tokens "${TOKENS:-4096}" \
            --graph-condition "${GRAPH:-real}" ${STOP_AT_FAULTS:+--stop-at-faults} \
            ${REDO_FROM:+--redo-from $REDO_FROM} ${REDO_NAME:+--redo-name $REDO_NAME} \
            --shard "${SHARD:-0}" --shards "${SHARDS:-1}" ${LIMIT:+--limit $LIMIT} ;;
    decoder)
        # SIM=fast scores with the fast simulator (a check of its agreement with TetraMAX).
        if [ "${SIM:-tetramax}" = fast ]; then export FAULT_SIM_BACKEND=fast; else tetramax; fi
        python scripts/eval/hint_arms_eval.py decoder --out "$OUT" --hints "$HINTS" --labels "$LABELS" \
            --attempts "${ATTEMPTS:-$CALLS}" --shard "${SHARD:-0}" --shards "${SHARDS:-1}" ;;
    report)
        export FAULT_SIM_BACKEND=fast
        python scripts/eval/hint_arms_eval.py report --out "$OUT" --max-tool-calls "$CALLS" ${BASE:+--base $BASE} ${METHODS:+--methods $METHODS} ;;
    *) echo "STEP must be rows, rollout, decoder or report"; exit 1 ;;
esac
echo "[hint-eval] done $(date -Is)"
