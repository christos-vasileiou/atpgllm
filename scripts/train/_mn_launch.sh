#!/bin/bash
# =============================================================================
# _mn_launch.sh  --  per-node role launcher for MULTI-NODE runs.
#
# This helper is invoked by run_training_code.sh through `srun` (one task per
# node). It is NOT meant to be run by hand.
#
#   srun ... _mn_launch.sh vllm                  # dedicated vLLM generation node
#   srun ... _mn_launch.sh train "${CMD_ARGS[@]}"   # one accelerate rank per training node
#
# All configuration is passed through the environment (MN_* variables), which
# `srun` propagates to every node (we set SLURM_EXPORT_ENV=ALL in the caller).
# The remaining positional arguments of the "train" role are forwarded verbatim
# to training_code.py, so quoting is preserved.
#
# Why a separate file (instead of `srun bash -c '...'`)?
#   * $SLURM_PROCID must be expanded *per task* (each node its own machine rank);
#     a here-string would be expanded once by the batch shell.
#   * "$@" keeps CMD_ARGS correctly quoted.
# =============================================================================
set -euo pipefail

ROLE="${1:-}"
shift || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# --- activate virtual environment (same resolution as run_training_code.sh) ---
if [ -f "/work/cxv200006/myenv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source /work/cxv200006/myenv/bin/activate
elif [ -f "/proj/trela/christos/myenv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source /proj/trela/christos/myenv/bin/activate
fi

# Reduce CUDA fragmentation on long 8k-seq LoRA runs (matches the launcher).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# --- Cross-node NCCL (multi-node only) --------------------------------------
# If multi-node DDP hangs at startup ("NCCL ... timeout" / no progress) or the
# GRPO vLLM weight-sync stalls, the ranks probably selected the wrong network
# interface. Pin the high-speed fabric explicitly (find it with `ip -o link`):
#   export NCCL_SOCKET_IFNAME=ib0        # or eth0, bond0, ...
#   export GLOO_SOCKET_IFNAME="$NCCL_SOCKET_IFNAME"
# Debug with: export NCCL_DEBUG=INFO
# Left unset here so NCCL auto-detects; uncomment above only if needed.

case "$ROLE" in
  vllm)
    echo "[_mn_launch vllm] node=$(hostname) CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} " \
         "port=${MN_VLLM_PORT} TP=${MN_VLLM_TP} DP=${MN_VLLM_DP} max_model_len=${MN_VLLM_MAXLEN}"
    # GRPO MUST use `trl vllm-serve` (NOT `vllm serve`): TRL's GRPOTrainer needs
    # the custom weight-sync endpoints (/get_world_size, /init_communicator,
    # /update_named_param, /reset_prefix_cache). --host 0.0.0.0 so the training
    # nodes can reach this server across the cluster network.
    exec trl vllm-serve \
        --model "$MN_MODEL" \
        --host 0.0.0.0 \
        --port "$MN_VLLM_PORT" \
        --gpu-memory-utilization "$MN_VLLM_UTIL" \
        --tensor-parallel-size "$MN_VLLM_TP" \
        --data-parallel-size "$MN_VLLM_DP" \
        --max-model-len "$MN_VLLM_MAXLEN" \
        --enable_prefix_caching True
    ;;
  train)
    # One task per training node (--ntasks-per-node=1). SLURM_PROCID is then the
    # 0-based node rank across the training nodes -> accelerate --machine_rank.
    # accelerate spawns num_processes/num_machines local workers per node, each
    # pinned to one GPU via LOCAL_RANK (see training_code.py::_get_device_map).
    echo "[_mn_launch train] node=$(hostname) SLURM_PROCID=${SLURM_PROCID:-0} " \
         "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-(unset)} " \
         "machines=${MN_NUM_MACHINES} procs=${MN_NUM_PROCESSES} main=${MN_MAIN_IP}:${MN_MAIN_PORT}"
    exec accelerate launch \
        --multi_gpu \
        --num_machines "$MN_NUM_MACHINES" \
        --num_processes "$MN_NUM_PROCESSES" \
        --machine_rank "${SLURM_PROCID:-0}" \
        --main_process_ip "$MN_MAIN_IP" \
        --main_process_port "$MN_MAIN_PORT" \
        --mixed_precision bf16 \
        "$SCRIPT_DIR/training_code.py" "$@"
    ;;
  *)
    echo "ERROR: _mn_launch.sh unknown role '${ROLE}' (expected 'vllm' or 'train')" >&2
    exit 2
    ;;
esac
