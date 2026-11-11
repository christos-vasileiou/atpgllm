#!/bin/bash
#SBATCH --job-name=v5_schedule
#SBATCH --partition=h100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:4
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@900
#SBATCH --output=jobs/v5_schedule_%j.out
#
# The v5 schedule (scripts/rft/run_schedule.py) on the juno h100 node. A job
# ends at the 2-day limit; Slurm sends TERM 15 min before, the driver stops its
# stage and exits, and the next job of the chain resumes every stage where it
# stopped. Submit a chain from the repo root (see scripts/rft/submit_juno_schedule.sh):
#   sbatch scripts/rft/juno_schedule.sbatch [schedule json]
#   sbatch --dependency=afterany:<previous> scripts/rft/juno_schedule.sbatch [schedule json]
set -uo pipefail
SCHEDULE="${1:-scripts/rft/configs/schedule_v5_juno_h100.json}"
REPO="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$REPO" || exit 1

# The juno login shell exports run settings (OUTPUT_DIR=sft_finetuned_model, MODEL,
# METHOD, ...) and sbatch passes them on; the configs must not inherit them.
# (2026-11-11: BUFFER_SIZE DATASET GRADIENT_ACCUMULATION_STEPS MAX_STEPS METHOD MODEL OUTPUT_DIR
# PER_DEVICE_TRAIN_BATCH_SIZE REPORT_TO RESUME_FROM RESUME_TRAINING_STATE TRAIN_DATASET USE_* VLLM_MODE.)
unset BUFFER_SIZE DATASET DATASET_HF_REPO_NAME GRADIENT_ACCUMULATION_STEPS MAX_STEPS METHOD MODEL OUTPUT_DIR \
    PER_DEVICE_TRAIN_BATCH_SIZE REPORT_TO RESUME_FROM RESUME_TRAINING_STATE TRAIN_DATASET USE_DDP \
    USE_DUAL_ADAPTER USE_UNSLOTH USE_VLLM VLLM_MODE SFT_EXTRA_MESSAGES SFT_EXTRA_FRACTION SFT_LEARNING_RATE
source /work/cxv200006/myenv/bin/activate
export PYTHONPATH="$REPO${PYTHONPATH:+:$PYTHONPATH}"
export DATA_PREPROCESSING_ROOT="$(cd "$REPO/../data_preprocessing" && pwd)"
export HF_HOME=/work/cxv200006/.cache/huggingface
# Separate TetraMAX seat pool per job (the shared juno pools are bound to other hosts).
export TMAX_LOCK_DIR="$(cd "$REPO/.." && pwd)/.runtime/tmax_locks/${SLURM_JOB_ID:-nojob}_$(hostname -s)"
export TMAX_SERVICE_MODULE=tetramax/vO-2018.06-SP1
export VLLM_DISABLE_COMPILE_CACHE=1
mkdir -p jobs "$TMAX_LOCK_DIR"

state="$(python -c "import json,sys,pathlib; p=pathlib.Path(json.load(open('$SCHEDULE'))['run_dir']) / 'state.json'; p=p if p.is_absolute() else pathlib.Path('$REPO')/p; print(json.load(open(p)).get('finished','') if p.exists() else '')")"
if [ -n "$state" ]; then
    echo "schedule already finished at $state"
    exit 0
fi
echo "job $SLURM_JOB_ID on $(hostname) at $(date -Iseconds); repo $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || cat "$REPO/PROVENANCE.txt" 2>/dev/null | head -1)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

# In the background so the TERM trap runs while python works; python stops its
# stage on TERM (run_rft.on_signal) and exits 143.
python scripts/rft/run_schedule.py --schedule "$SCHEDULE" &
child=$!
trap 'kill -TERM $child 2>/dev/null' TERM
wait $child
code=$?
# wait returns early when the trap fires; wait again for the driver to finish its shutdown.
if kill -0 $child 2>/dev/null; then wait $child; code=$?; fi
echo "driver exit $code at $(date -Iseconds)"
exit $code
