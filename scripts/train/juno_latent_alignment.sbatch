#!/bin/bash
#SBATCH --job-name=latent_align
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=runs/latent_alignment/logs/%x_%j.out
# Go or no-go test of the latent channel into Granite (scripts/train/latent_alignment.py):
# Granite frozen, only the graph side learns, target the graph model's hint.
# From /work/cxv200006/atpg_retest_seq/libatpgllm, after mkdir -p runs/latent_alignment/logs:
#
#   FUSION=prefix TRAIN=projector sbatch scripts/train/juno_latent_alignment.sbatch
#   FUSION=anchored TRAIN=projector sbatch scripts/train/juno_latent_alignment.sbatch
#   FUSION=both TRAIN=graph sbatch scripts/train/juno_latent_alignment.sbatch
# Order test (11-14): the dataset's pattern vote as the target, from the trained
# graph stack and from an untrained one (OUT must differ):
#   FUSION=both TRAIN=graph TARGET=labels INIT=trained OUT=runs/latent_alignment/labels_trained sbatch ...
#   FUSION=both TRAIN=graph TARGET=labels INIT=random OUT=runs/latent_alignment/labels_random sbatch ...
#
# Needs the hint files of runs/bridges_retest_seq/hints (juno_retest_seq_hints.sbatch).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
FUSION=${FUSION:?set FUSION=prefix, anchored or both}
TRAIN=${TRAIN:?set TRAIN=projector or graph}
TARGET=${TARGET:-hint}
INIT=${INIT:-trained}
OUT=${OUT:-runs/latent_alignment/${FUSION}_${TRAIN}}
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
# Its own PyG cache folder: jobs that start together must not share one.
export PYG_HOME="${TMPDIR:-/tmp}/pyg_${SLURM_JOB_ID:-$$}"
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 FAULT_SIM_BACKEND=fast
# The juno login shell exports a SIM_CONFIG for the text pipeline.
export SIM_CONFIG=atpgllm/training/data/sim_config.json
if [ ! -d ../data_preprocessing ]; then
    export DATA_PREPROCESSING_ROOT="${DATA_PREPROCESSING_ROOT:-/work/cxv200006/transformers_atpg/data_preprocessing_seq-v5}"
fi
echo "[align] fusion=$FUSION train=$TRAIN host=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
head -1 PROVENANCE.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
COMMON="--alignment ${ALIGNMENT:-runs/bridges_retest_seq/r3s/best.pt} --labels ${LABELS:-../data/freeset/retest_labels_v5}
        --hints ${HINTS:-runs/bridges_retest_seq/hints} --fusion $FUSION --train $TRAIN
        --target $TARGET --init $INIT"
# Twenty steps first: a broken setup fails in minutes.
python scripts/train/latent_alignment.py $COMMON --steps 20 --eval-every 20 --max-eval 24 --decode 4 \
    --out "$OUT/smoke_${SLURM_JOB_ID:-local}"
echo "[align] smoke passed $(date -Is)"
python scripts/train/latent_alignment.py $COMMON --steps "${STEPS:-3000}" --batch-size "${BATCH:-8}" \
    --eval-every "${EVAL_EVERY:-500}" --max-eval "${MAX_EVAL:-400}" --decode "${DECODE:-150}" --out "$OUT"
echo "[align] done $(date -Is)"
