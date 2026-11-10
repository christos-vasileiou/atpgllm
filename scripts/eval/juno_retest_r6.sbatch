#!/bin/bash
#SBATCH --job-name=retest_r6
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=1-00:00:00
#SBATCH --output=runs/retest_r6/%x_%j.out
# BRIDGES re-test R6 on juno, after both R5 arms finished. From
# /work/cxv200006/atpg_retest/libatpgllm (runs/retest_r6 holds
# generate_text.jsonl from the October 21st comparison, same rows):
#
#   a=$(STEP=model MODEL=retest sbatch --parsable scripts/eval/juno_retest_r6.sbatch)
#   b=$(STEP=model MODEL=ports_only HINTS=runs/bridges_retest/hints sbatch --parsable scripts/eval/juno_retest_r6.sbatch)
#   (and MODEL=anchored, MODEL=hint once those arms have trained)
#   STEP=score sbatch --dependency=afterok:$a:$b -t 06:00:00 scripts/eval/juno_retest_r6.sbatch
#
# model: loss on all 704 rows, then greedy tool calls on PER_CIRCUIT rows per
# circuit (the re-test model also runs graph_only, the R4 decoders). score:
# TetraMAX on every vector, then runs/retest_r6/report.md with gate G3.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
STEP=${STEP:?set STEP=model or STEP=score}
# Early check of a run that is still training (11-10): copy a step
# checkpoint aside, since the run prunes old steps, and point RUN_DIR at it:
#   STEP=model MODEL=retest RUN_DIR=runs/retest_early/retest_step40 CKPT=step40 \
#     OUT=runs/retest_early/step40 PASSES=loss SHUFFLE_SEEDS=2 sbatch ...
OUT=${OUT:-runs/retest_r6}
CKPT=${CKPT:-best}
PASSES=${PASSES:-loss generate}
PER_CIRCUIT=${PER_CIRCUIT:-8}
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1
echo "[r6] step=$STEP model=${MODEL:-} host=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
head -1 PROVENANCE.txt
if [ "$STEP" = model ]; then
    case "${MODEL:?set MODEL=retest, ports_only, anchored or hint}" in
        retest) RUN=runs/multimodal_sft_retest ;;
        ports_only) RUN=runs/multimodal_sft_retest_ports_only ;;
        anchored) RUN=runs/multimodal_sft_retest_anchored ;;
        hint) RUN=runs/multimodal_sft_retest_hint ;;
        *) echo "MODEL must be retest, ports_only, anchored or hint"; exit 1 ;;
    esac
    # HINTS on the ports-only model adds the untrained hint check (condition
    # "hint"); on the hint arm it replaces the path the arm trained with.
    HINT_ARGS=${HINTS:+--hints $HINTS}
    RUN=${RUN_DIR:-$RUN}
    CK="--checkpoint $RUN/multimodal_sft_${CKPT}.pt --adapter $RUN/adapter_${CKPT}"
    mkdir -p "$OUT"
    # The tool call is scored later; only the prompt-side tool view is built here.
    export FAULT_SIM_BACKEND=fast
    if [[ " $PASSES " == *" loss "* ]]; then
        python scripts/eval/retest_r6.py loss --model "$MODEL" $CK --out $OUT $HINT_ARGS \
            ${SHUFFLE_SEEDS:+--shuffle-seeds $SHUFFLE_SEEDS}
    fi
    if [[ " $PASSES " == *" generate "* ]]; then
        python scripts/eval/retest_r6.py generate --model "$MODEL" $CK --out $OUT --per-circuit "$PER_CIRCUIT" $HINT_ARGS
    fi
    if [ "$MODEL" = retest ] && [[ " $PASSES " == *" generate "* ]]; then
        python scripts/eval/retest_r6.py graph_only --out $OUT --per-circuit "$PER_CIRCUIT" --device cuda
    fi
else
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
    python scripts/eval/retest_r6.py score --out $OUT
fi
echo "[r6] done $(date -Is)"
