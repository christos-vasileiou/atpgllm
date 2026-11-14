#!/bin/bash
#SBATCH --job-name=retest_r5
#SBATCH --partition=h200
#SBATCH --nodes=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=180G
#SBATCH --time=2-00:00:00
#SBATCH --output=runs/retest_r5_logs/%x_%j.out
# BRIDGES re-test R5 (Stage C SFT, ports prompt, answer-only target) on one
# juno h200 node (2 GPUs). From /work/cxv200006/atpg_retest/libatpgllm:
#
#   mkdir -p runs/retest_r5_logs
#   ARM=graph sbatch scripts/train/juno_retest_stage_c.sbatch      # real graph
#   ARM=ports sbatch scripts/train/juno_retest_stage_c.sbatch      # ports-only control
#   ARM=anchored sbatch scripts/train/juno_retest_stage_c.sbatch   # input-bit anchors (11-10)
#   ARM=hint sbatch scripts/train/juno_retest_stage_c.sbatch       # text hint control (11-10);
#                                         # needs runs/bridges_retest/hints for train and validation
#   ARM=hint_plain sbatch scripts/train/juno_retest_stage_c.sbatch # hint arm on v5 (11-12);
#   ARM=hint_rich sbatch scripts/train/juno_retest_stage_c.sbatch  # needs runs/bridges_retest_seq/hints
#   ARM=graph SMOKE=1 sbatch --gres=gpu:1 -t 01:00:00 scripts/train/juno_retest_stage_c.sbatch
#
# STOP_AFTER=40 stops an arm after step 40 with the 300-step schedule kept,
# for the comparison of all arms at step 40 (rule of 11-10). Submitting the
# arm again without STOP_AFTER continues it to step 300.
#   ARM=anchored STOP_AFTER=40 sbatch -t 12:00:00 scripts/train/juno_retest_stage_c.sbatch
#
# A run cut off by the 2-day limit continues from its latest step checkpoint
# when submitted again (for example with --dependency=afterany:<job>); a run
# that already saved its last step exits at once.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ARM=${ARM:?set ARM=graph, ports, anchored, hint, hint_plain or hint_rich}
case "$ARM" in
    graph) export ARM_CONF=multimodal_sft_retest.conf; OUT=runs/multimodal_sft_retest ;;
    ports) export ARM_CONF=multimodal_sft_retest_ports_only.conf; OUT=runs/multimodal_sft_retest_ports_only ;;
    anchored) export ARM_CONF=multimodal_sft_retest_anchored.conf; OUT=runs/multimodal_sft_retest_anchored ;;
    hint) export ARM_CONF=multimodal_sft_retest_hint.conf; OUT=runs/multimodal_sft_retest_hint ;;
    # v5, both kinds of circuit, hints from the graph model trained on both (11-12).
    hint_plain) export ARM_CONF=multimodal_sft_hint_v5_plain.conf; OUT=runs/multimodal_sft_hint_v5_plain ;;
    hint_rich) export ARM_CONF=multimodal_sft_hint_v5_rich.conf; OUT=runs/multimodal_sft_hint_v5_rich ;;
    *) echo "ARM must be graph, ports, anchored, hint, hint_plain or hint_rich"; exit 1 ;;
esac
set +u
[ -f /etc/profile.d/lmod.sh ] && source /etc/profile.d/lmod.sh
module load tetramax/vO-2018.06-SP1
set -u
source "${PYTHON_VENV:-/work/cxv200006/myenv}/bin/activate"
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 WANDB_MODE=offline
# Its own PyG cache folder (the trainer adds one per rank below it): arms that
# start together on different nodes must not share ~/.cache/pyg.
export PYG_HOME="${TMPDIR:-/tmp}/pyg_${SLURM_JOB_ID:-$$}"
# A tree without a sibling data_preprocessing (the v5 arms) names its checkout.
if [ ! -d ../data_preprocessing ]; then
    export DATA_PREPROCESSING_ROOT="${DATA_PREPROCESSING_ROOT:-/work/cxv200006/transformers_atpg/data_preprocessing_seq-v5}"
fi
# A seat pool per job: pool.json binds a pool to one host, and the two arms
# run on different nodes.
export TMAX_LOCK_DIR="$(cd .. && pwd)/.runtime/tetramax_${SLURM_JOB_ID:-local}"
export TMAX_MAX_CONCURRENT=${TMAX_MAX_CONCURRENT:-8}
mkdir -p "$TMAX_LOCK_DIR"
echo "[r5] arm=$ARM host=$(hostname) GPUs=${CUDA_VISIBLE_DEVICES:-unset} start=$(date -Is)"
head -1 PROVENANCE.txt
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "${SMOKE:-0}" = 1 ]; then
    export SMOKE_MAX_STEPS=2 SMOKE_GRAD_ACCUM=2 SMOKE_EVAL_EVERY=2 SMOKE_SAVE_EVERY=2
    export SMOKE_EVAL_EXAMPLES=16 SMOKE_PROBE_EXAMPLES=2
    export SMOKE_OUTPUT_DIR=runs/retest_r5_smoke_${ARM}_${SLURM_JOB_ID:-local}
else
    max_steps=$(grep -E '^MAX_STEPS=' "scripts/train/configs/$ARM_CONF" | cut -d= -f2)
    if [ -f "$OUT/multimodal_sft_step${max_steps}.pt" ]; then
        echo "[r5] $OUT already saved step $max_steps; nothing to do"
        exit 0
    fi
    # ls exits 2 when no checkpoint exists yet; under pipefail that would end the job.
    latest=$(ls "$OUT"/multimodal_sft_step*.pt 2>/dev/null | sed -E 's/.*step([0-9]+)\.pt/\1/' | sort -n | tail -1 || true)
    if [ -n "$latest" ]; then
        export RESUME_FROM="$OUT/multimodal_sft_step${latest}.pt"
        echo "[r5] resuming from $RESUME_FROM"
    fi
fi
ngpu=$(nvidia-smi -L | wc -l)
if [ "$ngpu" -gt 1 ]; then
    # On 11-08 the first NCCL collective hung on both arms until the 4 hour
    # timeout. Check the transport in a few minutes, and fall back to no GPU
    # peer-to-peer (the two H200 NVL cards share no NVLink) before giving up.
    if ! timeout 240 torchrun --standalone --nproc-per-node "$ngpu" scripts/train/nccl_check.py; then
        echo "[r5] NCCL check failed with defaults; retrying with NCCL_P2P_DISABLE=1"
        export NCCL_P2P_DISABLE=1
        if ! timeout 240 torchrun --standalone --nproc-per-node "$ngpu" scripts/train/nccl_check.py; then
            echo "[r5] NCCL check failed with NCCL_P2P_DISABLE=1 too; not training"
            exit 1
        fi
    fi
fi
bash scripts/train/run_graph_roadmap.sh scripts/train/configs/multimodal_sft_retest_juno.conf
echo "[r5] done $(date -Is)"
