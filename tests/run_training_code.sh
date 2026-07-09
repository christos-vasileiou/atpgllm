#!/bin/bash
#SBATCH --job-name=atpg_train           # Job name
#SBATCH --output=jobs/training_%j.out   # Standard output file (%j will be replaced with job ID)
#SBATCH --error=jobs/training_%j.err    # Standard error file
#SBATCH --nodes=1                       # DEFAULT: 1 node (overridden by submit_training_code.sh)
#SBATCH --ntasks-per-node=1             # ONE launcher task per node (accelerate/srun spawn the ranks)
#SBATCH --cpus-per-task=32              # CPUs per node task
#SBATCH --mem=128G
#SBATCH --partition=h200
#SBATCH --gres=gpu:2
#
# ---------------------------------------------------------------------------
# The #SBATCH resource directives above are DEFAULTS for a direct
#   sbatch run_training_code.sh <config>
# For multi-node runs prefer submit_training_code.sh: it reads PARTITION,
# NUM_NODES, GPUS_PER_NODE, CPUS_PER_TASK and MEM from the config file and
# passes them to sbatch as CLI overrides (which take precedence over the
# #SBATCH lines here). That keeps this file's directives static while the
# topology is driven entirely by the config.
# ---------------------------------------------------------------------------

# Export the exact path of the Slurm log so Python can find it
export SLURM_LOG_FILE="jobs/training_${SLURM_JOB_ID}.out"
export SLURM_ERROR_FILE="jobs/training_${SLURM_JOB_ID}.err"

# activate virtual environment (prefer Slurm /work path, fall back to shared /proj path)
if [ -f "/work/cxv200006/myenv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source /work/cxv200006/myenv/bin/activate
elif [ -f "/proj/trela/christos/myenv/bin/activate" ]; then
    # shellcheck source=/dev/null
    source /proj/trela/christos/myenv/bin/activate
fi
echo "Python Path: $(which python)"

# Reduce CUDA fragmentation on long 8k-seq LoRA runs (especially DDP resume).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

start_time=$(date +%s)

# =============================================================================
# Load Run Configuration
# =============================================================================
# Config is passed as the FIRST positional argument and sourced here (not via
# submit-time env exports). For SLURM, prefer submit_training_code.sh so the
# config is frozen when you queue the job, not when it eventually starts:
#   ./submit_training_code.sh configs/grpo.conf
# Direct / interactive:
#   ./run_training_code.sh configs/sft.conf
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_LAUNCH_CONFIG_DIR="$_SCRIPT_DIR/jobs/launch_configs"
mkdir -p "$_LAUNCH_CONFIG_DIR" "$_LAUNCH_CONFIG_DIR/snapshots"

CONFIG_ARG="${1:-}"
CONFIG_FILE="$CONFIG_ARG"
if [ -z "$CONFIG_FILE" ]; then
    echo "ERROR: no configuration file provided."
    echo "Usage: sbatch run_training_code.sh <config_file>"
    echo "   or: ./run_training_code.sh <config_file>"
    echo "SLURM (freeze at submit): ./submit_training_code.sh <config_file>"
    exit 1
fi
if [ ! -f "$CONFIG_FILE" ]; then
    [ -f "$_SCRIPT_DIR/$CONFIG_ARG" ] && CONFIG_FILE="$_SCRIPT_DIR/$CONFIG_ARG"
fi
if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: configuration file not found: $CONFIG_ARG"
    exit 1
fi

LAUNCH_CONFIG_ORIGINAL="$CONFIG_FILE"
LAUNCH_CONFIG_FROZEN_BY=""
LAUNCH_CONFIG_FROZEN_AT=""
case "$CONFIG_FILE" in
    */jobs/launch_configs/submit_*)
        LAUNCH_CONFIG_FROZEN_BY="submit"
        if [ -f "${CONFIG_FILE}.meta" ]; then
            # shellcheck source=/dev/null
            source "${CONFIG_FILE}.meta"
            LAUNCH_CONFIG_ORIGINAL="${launch_config_original:-$LAUNCH_CONFIG_ORIGINAL}"
            LAUNCH_CONFIG_FROZEN_AT="${launch_config_frozen_at:-$LAUNCH_CONFIG_FROZEN_AT}"
        fi
        ;;
    */jobs/launch_configs/*)
        LAUNCH_CONFIG_FROZEN_BY="frozen_copy"
        ;;
    *)
        _stamp="$(date +%Y%m%d_%H%M%S)"
        if [ -n "${SLURM_JOB_ID:-}" ]; then
            _frozen="$_LAUNCH_CONFIG_DIR/job_${SLURM_JOB_ID}_$(basename "$CONFIG_FILE")"
            LAUNCH_CONFIG_FROZEN_BY="job_start"
        else
            _frozen="$_LAUNCH_CONFIG_DIR/run_${_stamp}_$(basename "$CONFIG_FILE")"
            LAUNCH_CONFIG_FROZEN_BY="direct"
        fi
        cp "$CONFIG_FILE" "$_frozen"
        LAUNCH_CONFIG_FROZEN_AT="$(date -Iseconds 2>/dev/null || date)"
        CONFIG_FILE="$_frozen"
        echo "NOTE: config frozen at run start ($_frozen). For SLURM submit-time freeze use submit_training_code.sh"
        ;;
esac

echo "Loading configuration from: $CONFIG_FILE"
# shellcheck source=/dev/null
source "$CONFIG_FILE"

# Snapshot sourced values for Weights & Biases (exact launcher state at run time).
write_launch_config_snapshot() {
    local _snap_stamp _snap_path _key _val
    local _launch_keys=(
        METHOD MODEL TRAIN_DATASET OUTPUT_DIR
        RESUME_FROM RESUME_TRAINING_STATE AUTO_SKIP_FROM_RESUME SKIP_BUFFER_SIZE
        PER_DEVICE_TRAIN_BATCH_SIZE GRADIENT_ACCUMULATION_STEPS MAX_STEPS REPORT_TO
        MAX_MODEL_LEN MAX_PROMPT_LENGTH MAX_COMPLETION_LENGTH
        ASSISTANT_ONLY_LOSS BUFFER_SIZE NUM_GENERATIONS STEPS_PER_GENERATION
        NETLIST_DIVERSITY_STRATEGY
        USE_DUAL_ADAPTER USE_DDP USE_VLLM VLLM_MODE PORT
        TENSOR_PARALLEL_SIZE DATA_PARALLEL_SIZE VLLM_GPU_MEM_UTIL
        PARTITION NUM_NODES GPUS_PER_NODE CPUS_PER_TASK MEM
        LORA_RANK LORA_ALPHA LORA_TARGET_MODULES QWEN_MOE TUNE_MOE_ROUTER MOE_MAX_MEMORY_GIB
        WANDB_PROJECT DRY_RUN
    )
    _snap_stamp="$(date +%Y%m%d_%H%M%S)"
    _snap_path="$_LAUNCH_CONFIG_DIR/snapshots/launch_${SLURM_JOB_ID:-local}_${_snap_stamp}.env"
    {
        echo "# launch_config_snapshot v1"
        echo "# launch_config_file=$CONFIG_FILE"
        echo "# launch_config_original=$LAUNCH_CONFIG_ORIGINAL"
        echo "# launch_config_frozen_by=${LAUNCH_CONFIG_FROZEN_BY:-unknown}"
        echo "# launch_config_frozen_at=${LAUNCH_CONFIG_FROZEN_AT:-}"
        echo "# launch_config_snapshotted_at=$(date -Iseconds 2>/dev/null || date)"
        echo "# slurm_job_id=${SLURM_JOB_ID:-}"
        echo "# hostname=$(hostname)"
        for _key in "${_launch_keys[@]}"; do
            _val="${!_key-}"
            printf '%s=%s\n' "$_key" "$_val"
        done
    } > "$_snap_path"
    export LAUNCH_CONFIG_SNAPSHOT="$_snap_path"
    export LAUNCH_CONFIG_FROZEN_FILE="$CONFIG_FILE"
    echo "Launch config snapshot for W&B: $_snap_path"
}
write_launch_config_snapshot

# =============================================================================
# Validate Required Configuration
# =============================================================================
METHOD=${METHOD:-sft}

if [ -z "$TRAIN_DATASET" ]; then
    echo "ERROR: TRAIN_DATASET is not set in $CONFIG_FILE"
    exit 1
fi
# MODEL is required for SFT, optional for GRPO (which can resume from checkpoint).
if [ -z "$MODEL" ] && [ "$METHOD" != "grpo" ]; then
    echo "ERROR: MODEL is not set in $CONFIG_FILE (required for METHOD=$METHOD)"
    exit 1
fi

# When DRY_RUN=True, print the commands that would be executed and skip
# launching the vLLM server and the training process.
DRY_RUN=${DRY_RUN:-False}
if [ "$DRY_RUN" == "True" ]; then
    echo "=============================================="
    echo "DRY RUN MODE (DRY_RUN=True)"
    echo "Commands will be printed but NOT executed."
    echo "=============================================="
fi

# These knobs reach Python ONLY through the environment (no CLI flag exists),
# so the launcher exports the values sourced from the config file to the child.
export LORA_RANK LORA_ALPHA LORA_TARGET_MODULES QWEN_MOE TUNE_MOE_ROUTER MOE_MAX_MEMORY_GIB
[ -n "$WANDB_PROJECT" ] && export WANDB_PROJECT

# =============================================================================
# Topology: single-node vs multi-node
# =============================================================================
# NUM_NODES prefers the live SLURM allocation, then the config value, else 1.
# GPUS_PER_NODE prefers the config value, then what SLURM granted on this node.
# The single-node path is byte-for-byte the previous behavior (incl. the 4xH100
# "3 train + 1 vLLM" carve). Multi-node dedicates ONE whole node to the vLLM
# server (GRPO) and runs homogeneous DDP training on the remaining nodes.
PARTITION="${PARTITION:-h200}"
NUM_NODES="${SLURM_JOB_NUM_NODES:-${NUM_NODES:-1}}"
GPUS_PER_NODE="${GPUS_PER_NODE:-${SLURM_GPUS_ON_NODE:-}}"

MULTINODE=false
if [ "${NUM_NODES:-1}" -gt 1 ] 2>/dev/null; then
    MULTINODE=true
fi

# vLLM server host used to build --vllm_server_url. Single-node keeps localhost;
# multi-node overrides it with the dedicated vLLM node's IP inside run_multi_node.
VLLM_HOST="${VLLM_HOST:-localhost}"

# =============================================================================
# Build Command Arguments
# =============================================================================
build_cmd_args() {
    CMD_ARGS=(--method "$METHOD")

    # Add --model_name only if it's set and not "None"
    if [ -n "$MODEL" ]; then
        echo "MODEL: $MODEL"
        if [ "$MODEL" != "None" ]; then
            CMD_ARGS+=(--model_name "$MODEL")
        fi
    fi

    CMD_ARGS+=(
        --dataset "$TRAIN_DATASET"
        --output_dir "$OUTPUT_DIR"
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
        --max_steps "$MAX_STEPS"
        --report_to "$REPORT_TO"
    )

    # Handle vLLM activation
    if [ -n "$USE_VLLM" ]; then
        if [ "$USE_VLLM" == "True" ]; then
            CMD_ARGS+=(--use_vllm)
            if [ "$METHOD" == "sft" ]; then
                PORT=$PORT
                CMD_ARGS+=(--vllm_server_url "http://${VLLM_HOST:-localhost}:$PORT")
                # Single-node only: pick the last local GPU for vLLM. GPU_ARRAY is
                # unset in the multi-node path (dedicated vLLM node), so guard it.
                if [ "$VLLM_MODE" == "server" ] && [ "${#GPU_ARRAY[@]}" -gt 0 ]; then
                    VLLM_GPU=${GPU_ARRAY[-1]}
                fi
            elif [ "$METHOD" == "grpo" ]; then
                PORT=$PORT
                CMD_ARGS+=(--vllm_server_url "http://${VLLM_HOST:-localhost}:$PORT")
                CMD_ARGS+=(--vllm_mode "$VLLM_MODE")
                # Single-node only: pick the last local GPU for vLLM. GPU_ARRAY is
                # unset in the multi-node path (dedicated vLLM node), so guard it.
                if [ "$VLLM_MODE" == "server" ] && [ "${#GPU_ARRAY[@]}" -gt 0 ]; then
                    VLLM_GPU=${GPU_ARRAY[-1]}
                fi
            fi
        fi
    fi

    # Handle DDP activation
    if [ "$USE_DDP" == "True" ]; then
        CMD_ARGS+=("--use_ddp")
    fi

    if [ "$QWEN_MOE" == "True" ]; then
        CMD_ARGS+=(--qwen_moe)
    fi

    # SFT only: assistant-only loss (mask system/user/tool-response tokens)
    if [ "$METHOD" == "sft" ]; then
        if [ -n "$ASSISTANT_ONLY_LOSS" ] && [ "$ASSISTANT_ONLY_LOSS" == "False" ]; then
            CMD_ARGS+=(--no-assistant_only_loss)
        else
            CMD_ARGS+=(--assistant_only_loss)
        fi
    fi

    # Add --resume_from only if it's set and not "None"
    if [ -n "$RESUME_FROM" ] && [ "$RESUME_FROM" != "None" ]; then
        CMD_ARGS+=(--resume_from "$RESUME_FROM")
    fi

    # Add --resume_training_state to restore optimizer, LR schedule, step,
    # and RNG seeds from the checkpoint specified by --resume_from.
    if [ "$RESUME_TRAINING_STATE" == "True" ]; then
        CMD_ARGS+=(--resume_training_state)
    fi

    # Shared by both SFT and GRPO
    CMD_ARGS+=(
        --max_model_len "$MAX_MODEL_LEN"
        --max_prompt_length "$MAX_PROMPT_LENGTH"
        --skip_buffer_size "${SKIP_BUFFER_SIZE:-0}"
    )

    # Disable the auto-skip-from-resume helper only when explicitly requested.
    # Python defaults to ON, so we just send the negation when needed.
    if [ "$AUTO_SKIP_FROM_RESUME" == "False" ]; then
        CMD_ARGS+=(--no-auto_skip_from_resume)
    fi

    # Add GRPO-specific arguments only for GRPO method
    if [ "$METHOD" == "grpo" ]; then
        CMD_ARGS+=(
            --buffer_size "$BUFFER_SIZE"
            --num_generations "$NUM_GENERATIONS"
            --steps_per_generation "$STEPS_PER_GENERATION"
            --max_completion_length "$MAX_COMPLETION_LENGTH"
        )
        # Buffer ordering strategy for netlist diversity per effective batch.
        if [ -n "$NETLIST_DIVERSITY_STRATEGY" ]; then
            CMD_ARGS+=(--netlist_diversity_strategy "$NETLIST_DIVERSITY_STRATEGY")
        fi
        # This condition checks whether the variable USE_DUAL_ADAPTER is set (not empty) and its value is exactly "True".
        if [ -n "$USE_DUAL_ADAPTER" ] && [ "$USE_DUAL_ADAPTER" == "True" ]; then
            CMD_ARGS+=(--use_dual_adapter)
        fi
    fi
    echo "=============================================="
    echo "Command Arguments"
    echo "=============================================="
    echo "METHOD: $METHOD"
    echo "MODEL: $MODEL"
    echo "TRAIN_DATASET: $TRAIN_DATASET"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    echo "RESUME_FROM: ${RESUME_FROM:-(not set)}"
    echo "RESUME_TRAINING_STATE: $RESUME_TRAINING_STATE"
    echo "PER_DEVICE_TRAIN_BATCH_SIZE: $PER_DEVICE_TRAIN_BATCH_SIZE"
    echo "GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
    echo "MAX_STEPS: $MAX_STEPS"
    echo "MAX_MODEL_LEN: $MAX_MODEL_LEN"
    echo "MAX_PROMPT_LENGTH: $MAX_PROMPT_LENGTH"
    echo "REPORT_TO: $REPORT_TO"
    echo "USE_VLLM: $USE_VLLM"
    echo "VLLM_MODE: ${VLLM_MODE:-(n/a)} (GRPO only)"
    echo "PORT: $PORT (http://${VLLM_HOST:-localhost}:$PORT, otherwise no vLLM server needed)"
    echo "*VLLM_GPU: ${VLLM_GPU:-(auto)} (set when vLLM server mode is used, single-node)"
    echo "USE_DUAL_ADAPTER: $USE_DUAL_ADAPTER"
    echo "USE_DDP: $USE_DDP"
    echo "LORA_RANK: $LORA_RANK"
    echo "LORA_ALPHA: $LORA_ALPHA"
    echo "LORA_TARGET_MODULES: ${LORA_TARGET_MODULES:-'(default: q/k/v/o_proj + gate/up/down_proj)'}"
    echo "QWEN_MOE: $QWEN_MOE"
    echo "TUNE_MOE_ROUTER: $TUNE_MOE_ROUTER"
    echo "MOE_MAX_MEMORY_GIB: ${MOE_MAX_MEMORY_GIB:-'(auto 95% per GPU)'}"
    echo "SKIP_BUFFER_SIZE: ${SKIP_BUFFER_SIZE:-0}"
    echo "AUTO_SKIP_FROM_RESUME: $AUTO_SKIP_FROM_RESUME"
    if [ "$METHOD" == "sft" ]; then
        echo "ASSISTANT_ONLY_LOSS: ${ASSISTANT_ONLY_LOSS:-True}"
    fi
    if [ "$METHOD" == "grpo" ]; then
        echo "--- GRPO-specific ---"
        echo "BUFFER_SIZE: $BUFFER_SIZE"
        echo "NUM_GENERATIONS: $NUM_GENERATIONS"
        echo "STEPS_PER_GENERATION: $STEPS_PER_GENERATION"
        echo "MAX_COMPLETION_LENGTH: $MAX_COMPLETION_LENGTH"
        echo "NETLIST_DIVERSITY_STRATEGY: ${NETLIST_DIVERSITY_STRATEGY:-even_spacing}"
    fi
    echo "=============================================="
    echo "CMD_ARGS: ${CMD_ARGS[*]}"
    echo "=============================================="
}

# =============================================================================
# Multi-node helpers
# =============================================================================
# Resolve a compute-node hostname to a routable IPv4 (falls back to the hostname
# itself when name resolution is unavailable). Used for accelerate's
# --main_process_ip and for the vLLM server URL.
resolve_ip() {
    local host="$1" ip=""
    ip=$(getent hosts "$host" 2>/dev/null | awk '{print $1; exit}')
    [ -z "$ip" ] && ip=$(getent ahostsv4 "$host" 2>/dev/null | awk '{print $1; exit}')
    echo "${ip:-$host}"
}

# Multi-node orchestration:
#   * GRPO server mode -> the LAST allocated node is dedicated to `trl vllm-serve`
#     (uses all its GPUs via TP/DP); the remaining nodes run homogeneous DDP.
#   * SFT / GRPO colocate -> every allocated node is a training node.
# One `srun` task per training node runs accelerate launch with its own
# --machine_rank (SLURM_PROCID). Only global rank 0 talks to the vLLM server
# (TRL server-mode weight sync + generation), and any node can reach it by IP.
run_multi_node() {
    if [ -z "${SLURM_JOB_ID:-}" ]; then
        echo "ERROR: multi-node training (NUM_NODES=$NUM_NODES) requires a SLURM allocation."
        echo "       Submit with: ./submit_training_code.sh <config>   (sets --nodes/--gres/...)."
        return 1
    fi
    if [ -z "${GPUS_PER_NODE:-}" ]; then
        echo "ERROR: GPUS_PER_NODE is empty and SLURM_GPUS_ON_NODE was not set."
        echo "       Set GPUS_PER_NODE in the config (e.g. 2 for H200 nodes)."
        return 1
    fi

    mapfile -t ALL_NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
    local n_alloc=${#ALL_NODES[@]}
    echo "Allocated nodes ($n_alloc): ${ALL_NODES[*]}"

    # accelerate rendezvous port (distinct from the vLLM PORT).
    ACC_PORT=${ACC_PORT:-$((20000 + RANDOM % 20000))}

    local grpo_server=false
    if [ "$METHOD" == "grpo" ] && [ "$USE_VLLM" == "True" ] && [ "$VLLM_MODE" == "server" ]; then
        grpo_server=true
    fi
    if [ "$METHOD" == "sft" ] && [ "$USE_VLLM" == "True" ] && [ "$VLLM_MODE" == "server" ]; then
        echo "WARNING: multi-node SFT with USE_VLLM=server does NOT auto-start a dedicated"
        echo "         vLLM eval server. Training runs as pure multi-node DDP. Start an"
        echo "         external server yourself if the stopping callback needs one."
    fi

    local -a TRAIN_NODES
    VLLM_NODE=""
    if [ "$grpo_server" == "true" ]; then
        if [ "$n_alloc" -lt 2 ]; then
            echo "ERROR: multi-node GRPO server mode needs >=2 nodes (1 dedicated to vLLM)."
            echo "       Use a single-node config for the '1 node = N-1 train + 1 vLLM' layout."
            return 1
        fi
        VLLM_NODE="${ALL_NODES[-1]}"
        TRAIN_NODES=("${ALL_NODES[@]:0:$((n_alloc - 1))}")
        VLLM_HOST="$(resolve_ip "$VLLM_NODE")"
    else
        TRAIN_NODES=("${ALL_NODES[@]}")
        VLLM_HOST="localhost"
    fi

    local num_train_nodes=${#TRAIN_NODES[@]}
    local total_procs=$((num_train_nodes * GPUS_PER_NODE))
    local head_node="${TRAIN_NODES[0]}"
    local head_ip; head_ip="$(resolve_ip "$head_node")"
    local train_nodelist; train_nodelist="$(IFS=,; echo "${TRAIN_NODES[*]}")"

    # Rebuild CMD_ARGS now that VLLM_HOST points at the dedicated vLLM node.
    build_cmd_args

    echo ""
    echo "=============================================="
    echo "MULTI-NODE TOPOLOGY"
    echo "=============================================="
    echo "METHOD:          $METHOD"
    echo "Total nodes:     $n_alloc"
    echo "GPUs / node:     $GPUS_PER_NODE"
    if [ "$grpo_server" == "true" ]; then
        echo "vLLM node:       $VLLM_NODE ($VLLM_HOST)  [dedicated, ${GPUS_PER_NODE} GPU]"
    fi
    echo "Training nodes:  $num_train_nodes  -> $train_nodelist"
    echo "Training procs:  $total_procs  (accelerate --num_processes = nodes x gpus/node)"
    echo "Rendezvous:      $head_ip:$ACC_PORT  (accelerate main process)"
    echo "=============================================="

    # ---- export MN_* config for _mn_launch.sh (srun propagates the env) ----
    export SLURM_EXPORT_ENV=ALL
    export MN_MODEL="$MODEL"
    export MN_NUM_MACHINES="$num_train_nodes"
    export MN_NUM_PROCESSES="$total_procs"
    export MN_MAIN_IP="$head_ip"
    export MN_MAIN_PORT="$ACC_PORT"

    # ---- GRPO: start the dedicated vLLM server on its own node ----
    if [ "$grpo_server" == "true" ]; then
        # The whole node serves vLLM, so keep TP*DP == GPUS_PER_NODE. Default
        # (TP=1) turns the node's GPUs into that many data-parallel generation
        # engines -> more rollout throughput, the usual RL bottleneck.
        local tp=${TENSOR_PARALLEL_SIZE:-1}
        local dp=${DATA_PARALLEL_SIZE:-1}
        if [ "$dp" -le 1 ]; then
            dp=$(( GPUS_PER_NODE / tp ))
        fi
        if [ $(( tp * dp )) -ne "$GPUS_PER_NODE" ]; then
            echo "WARNING: TP($tp) * DP($dp) != GPUS_PER_NODE($GPUS_PER_NODE); forcing DP=$(( GPUS_PER_NODE / tp ))."
            dp=$(( GPUS_PER_NODE / tp ))
        fi
        [ "$dp" -lt 1 ] && dp=1
        export MN_VLLM_PORT="$PORT"
        export MN_VLLM_UTIL="${VLLM_GPU_MEM_UTIL:-0.9}"
        export MN_VLLM_TP="$tp"
        export MN_VLLM_DP="$dp"
        export MN_VLLM_MAXLEN="$MAX_MODEL_LEN"

        echo ""
        echo "vLLM (dedicated node $VLLM_NODE): trl vllm-serve --model $MODEL --host 0.0.0.0 --port $PORT \\"
        echo "     --tensor-parallel-size $tp --data-parallel-size $dp --gpu-memory-utilization $MN_VLLM_UTIL \\"
        echo "     --max-model-len $MAX_MODEL_LEN --enable_prefix_caching True"
        echo "Launch: srun --overlap --nodes=1 --nodelist=$VLLM_NODE --ntasks=1 _mn_launch.sh vllm &"
        if [ "$DRY_RUN" != "True" ]; then
            srun --overlap --nodes=1 --nodelist="$VLLM_NODE" --ntasks=1 \
                "$_SCRIPT_DIR/_mn_launch.sh" vllm &
            VLLM_PID=$!
            echo "Waiting for vLLM server at http://$VLLM_HOST:$PORT (srun PID $VLLM_PID)..."
            local _t=0 _to=1800
            while ! curl -s "http://$VLLM_HOST:$PORT/health" >/dev/null 2>&1; do
                if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                    echo "ERROR: vLLM srun step exited before becoming ready."
                    return 1
                fi
                if [ "$_t" -ge "$_to" ]; then
                    echo "ERROR: vLLM server did not become ready within ${_to}s."
                    return 1
                fi
                sleep 5; _t=$((_t + 5))
            done
            echo "vLLM server is ready at http://$VLLM_HOST:$PORT (~${_t}s)."
        fi
    fi

    # ---- launch DDP training across the training nodes ----
    echo ""
    echo "Training: srun --overlap --nodes=$num_train_nodes --nodelist=$train_nodelist \\"
    echo "     --ntasks=$num_train_nodes --ntasks-per-node=1 _mn_launch.sh train ${CMD_ARGS[*]}"
    echo "  per node -> accelerate launch --multi_gpu --num_machines $num_train_nodes \\"
    echo "     --num_processes $total_procs --machine_rank \$SLURM_PROCID \\"
    echo "     --main_process_ip $head_ip --main_process_port $ACC_PORT training_code.py ${CMD_ARGS[*]}"
    local rc=0
    if [ "$DRY_RUN" != "True" ]; then
        srun --overlap --nodes="$num_train_nodes" --nodelist="$train_nodelist" \
             --ntasks="$num_train_nodes" --ntasks-per-node=1 \
             "$_SCRIPT_DIR/_mn_launch.sh" train "${CMD_ARGS[@]}"
        rc=$?
    fi
    return $rc
}

# =============================================================================
# vLLM lifecycle (shared) + cleanup
# =============================================================================
VLLM_PID=""

find_vllm_pid_for_port() {
    local port="$1"
    # Prefer lsof if available (most precise for port ownership).
    if command -v lsof >/dev/null 2>&1; then
        lsof -t -i :"$port" -sTCP:LISTEN 2>/dev/null | head -n1
    # Fallback to ss if lsof is missing.
    elif command -v ss >/dev/null 2>&1; then
        ss -ltnp "sport = :$port" 2>/dev/null | awk 'NR>1 {gsub(/pid=/,"",$NF); split($NF,a,","); print a[1]; exit}'
    # Last resort: any vllm serve process (not port-specific).
    elif command -v pgrep >/dev/null 2>&1; then
        pgrep -f "vllm serve" | head -n1
    fi
}

# Always try to shut down the vLLM server on exit (including OOM, Ctrl-C, etc.).
# In multi-node runs VLLM_PID is the backgrounded `srun ... _mn_launch.sh vllm`
# step; killing it tears down the remote server too.
cleanup() {
    if [ -n "$VLLM_PID" ]; then
        echo "Stopping vLLM server (PID: $VLLM_PID)..."
        if kill -0 "$VLLM_PID" 2>/dev/null; then
            kill "$VLLM_PID" 2>/dev/null || true
            wait "$VLLM_PID" 2>/dev/null || true
        fi
    fi
}

trap cleanup EXIT INT TERM

# =============================================================================
# Dispatch: multi-node vs single-node
# =============================================================================
echo ""
echo "=============================================="
echo "Launch mode: $([ "$MULTINODE" == "true" ] && echo MULTI-NODE || echo SINGLE-NODE)"
echo "PARTITION=$PARTITION NUM_NODES=$NUM_NODES GPUS_PER_NODE=${GPUS_PER_NODE:-'(local detect)'}"
echo "=============================================="

if [ "$MULTINODE" == "true" ]; then
    run_multi_node
    EXIT_CODE=$?
else
    # =========================================================================
    # SINGLE-NODE PATH  (unchanged behavior, incl. 4xH100 "3 train + 1 vLLM")
    # =========================================================================

    # -------------------------------------------------------------------------
    # GPU Configuration and Setup
    # -------------------------------------------------------------------------
    echo "Allocated GPU:"
    echo $CUDA_VISIBLE_DEVICES
    nvidia-smi

    # When CUDA_VISIBLE_DEVICES is not set, auto-detect all available GPUs
    if [ -z "$CUDA_VISIBLE_DEVICES" ]; then
        _detected=$(nvidia-smi -L 2>/dev/null | grep -c '^GPU ')
        if [ "$_detected" -gt 0 ]; then
            export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((_detected - 1)))
        fi
    fi

    IFS=',' read -ra GPU_ARRAY <<< "$CUDA_VISIBLE_DEVICES"

    IS_MIG=false
    NUM_GPUS=${#GPU_ARRAY[@]}
    ORIGINAL_CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES"

    # Determine number of GPUs from CUDA_VISIBLE_DEVICES
    if [ -n "$CUDA_VISIBLE_DEVICES" ]; then
        # Check if any token starts with "MIG-" or "GPU-"
        if echo "$CUDA_VISIBLE_DEVICES" | grep -qE '(MIG-|GPU-)'; then
            IS_MIG=true
            NUM_MIGS=$(nvidia-smi -L | grep -c 'MIG')

            # -----------------------------------------------------------------
            # NCCL identifies GPUs by PCI Bus ID.  MIG instances on the SAME
            # physical GPU share one Bus ID, so NCCL rejects them as
            #   "Duplicate GPU detected: rank X and rank Y both on CUDA device …"
            #
            # Workaround: keep only ONE MIG UUID per physical GPU.
            # We parse `nvidia-smi -L` to group MIG UUIDs under their parent
            # GPU and pick the first allocated instance from each.
            # -----------------------------------------------------------------
            SELECTED_MIGS=$(
                nvidia-smi -L | awk -v allocated="$CUDA_VISIBLE_DEVICES" '
                BEGIN {
                    split(allocated, a, ",")
                    for (i in a) wanted[a[i]] = 1
                    gpu = -1
                }
                /^GPU / { gpu++ }
                /MIG-/ {
                    match($0, /(MIG-[a-f0-9-]+)/, m)
                    if (m[1] in wanted && !(gpu in seen)) {
                        print m[1]
                        seen[gpu] = 1
                    }
                }
                ' | paste -sd, -
            )

            if [ -n "$SELECTED_MIGS" ]; then
                export CUDA_VISIBLE_DEVICES="$SELECTED_MIGS"
            fi
            NUM_GPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

            echo ""
            echo "=============================================="
            echo "MIG (Multi-Instance GPU) Detected"
            echo "=============================================="
            echo "Total MIG instances (allocated): $NUM_MIGS"
            echo "DDP processes (1 per physical GPU): $NUM_GPUS"
            echo "Selected UUIDs: $CUDA_VISIBLE_DEVICES"
            echo "=============================================="
        fi
    fi

    # Check if the number of GPUs is sufficient for the vLLM server mode
    if [ $NUM_GPUS -lt 2 ] && [ "$USE_VLLM" == "True" ] && [ "$VLLM_MODE" == "server" ]; then
        echo "WARNING: $METHOD server mode requires at least 2 GPUs (1 for vLLM, 1 for training)."
        echo "Available GPUs: ${#GPU_ARRAY[@]}"
        echo "Falling back to colocate mode."
        VLLM_MODE="colocate"
        for i in "${!CMD_ARGS[@]}"; do
            if [[ "${CMD_ARGS[$i]}" == "--vllm_mode" ]]; then
                CMD_ARGS[$((i+1))]="colocate"
                break
            fi
        done
    fi

    echo ""
    echo "=============================================="
    echo "GPU Setup Summary"
    echo "=============================================="
    echo "IS_MIG: $IS_MIG"
    echo "NUM_GPUS: $NUM_GPUS"
    echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'(not set)'}"
    echo "=============================================="

    build_cmd_args
    export CMD_ARGS

    # Run the training script — DDP via accelerate or single-process
    if [ "$USE_VLLM" == "True" ] && [ "$VLLM_MODE" == "server" ]; then
        # MAX_MODEL_LEN, MAX_PROMPT_LENGTH, MAX_COMPLETION_LENGTH are set in the
        # method-defaults block above and also passed to Python via CMD_ARGS.
        VLLM_MAX_MODEL_LEN=$MAX_MODEL_LEN

        if [ "$METHOD" == "grpo" ]; then
            # GRPO MUST use `trl vllm-serve` (NOT `vllm serve`).
            # TRL's GRPOTrainer needs custom endpoints (/get_world_size,
            # /init_communicator, /update_named_param, /reset_prefix_cache)
            # for weight synchronisation between the trainer and the vLLM
            # generation server.  The standard `vllm serve` does not expose
            # these endpoints and will fail with a 404 on /get_world_size.
            #
            # Note: `trl vllm-serve` does NOT support --enable-lora /
            # --max-lora-rank.  Instead, TRL pushes updated weights directly
            # to vLLM via the NCCL communicator.

            # Extract the number before 'B' or 'b' in the model string (e.g., Qwen2.5-72B -> 72)
            MODEL_SIZE_STR=$(echo "$MODEL" | grep -ioP '\d+(\.\d+)?(?=b)' | head -n 1)
            MODEL_SIZE=${MODEL_SIZE_STR:-0}

            # Evaluate if the model size is strictly greater than 32
            IS_LARGE_MODEL=$(awk -v size="$MODEL_SIZE" 'BEGIN { print (size >= 32) ? 1 : 0 }')

            TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
            DATA_PARALLEL_SIZE=${DATA_PARALLEL_SIZE:-1}
            VLLM_GPU_MEM_UTIL=${VLLM_GPU_MEM_UTIL:-0.9}
            echo ""
            echo "=============================================="
            echo "$METHOD vLLM Server Setup"
            echo "=============================================="
            echo "vLLM server GPU: $VLLM_GPU (last GPU)"
            echo "Starting vLLM server on GPU $VLLM_GPU (port $PORT)..."
            echo "DATA_PARALLEL_SIZE: $DATA_PARALLEL_SIZE"
            echo "TENSOR_PARALLEL_SIZE: $TENSOR_PARALLEL_SIZE"
            echo "VLLM_MAX_MODEL_LEN: $VLLM_MAX_MODEL_LEN"
            echo "MAX_PROMPT_LENGTH: $MAX_PROMPT_LENGTH"
            echo "MAX_COMPLETION_LENGTH: $MAX_COMPLETION_LENGTH"
            echo "=============================================="
            echo "Running: CUDA_VISIBLE_DEVICES=$VLLM_GPU trl vllm-serve --model $MODEL --port $PORT --gpu_memory_utilization $VLLM_GPU_MEM_UTIL --data-parallel-size $DATA_PARALLEL_SIZE --tensor-parallel-size $TENSOR_PARALLEL_SIZE --max-model-len $VLLM_MAX_MODEL_LEN &"
            if [ "$DRY_RUN" != "True" ]; then
                CUDA_VISIBLE_DEVICES=$VLLM_GPU \
                trl vllm-serve \
                    --model $MODEL \
                    --port $PORT \
                    --gpu-memory-utilization "$VLLM_GPU_MEM_UTIL" \
                    --data-parallel-size "$DATA_PARALLEL_SIZE" \
                    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
                    --max-model-len "$VLLM_MAX_MODEL_LEN" \
                    --enable_prefix_caching True &
                VLLM_PID=$!
                echo "Waiting for TRL vLLM server to become ready (PID: $VLLM_PID)..."
            fi
            COUNT_VLLM_GPUS=$(echo "$VLLM_GPU" | tr ',' '\n' | wc -l)
            NUM_TRAIN_GPUS=$((${#GPU_ARRAY[@]} - $COUNT_VLLM_GPUS))
            TRAINING_GPUS=("${GPU_ARRAY[@]:0:$NUM_TRAIN_GPUS}")
            export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${TRAINING_GPUS[*]}")"
        fi

        if [ "$DRY_RUN" != "True" ]; then
            _vllm_timeout=1800
            _vllm_elapsed=0
            while ! curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; do
                if ! kill -0 "$VLLM_PID" 2>/dev/null; then
                    echo "ERROR: TRL vLLM server process died unexpectedly."
                    exit 1
                fi
                if [ "$_vllm_elapsed" -ge "$_vllm_timeout" ]; then
                    echo "ERROR: TRL vLLM server did not become ready within ${_vllm_timeout}s."
                    kill "$VLLM_PID" 2>/dev/null || true
                    exit 1
                fi
                sleep 2
                _vllm_elapsed=$((_vllm_elapsed + 2))
            done
            echo "$METHOD vLLM server is ready on port $PORT (took ~${_vllm_elapsed}s)."
        fi
        echo "=============================================="
        echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES, TRAINING_GPUS: ${#TRAINING_GPUS[@]}, VLLM_GPU: $VLLM_GPU"
    fi

    echo "=============================================="
    echo "Starting Training"
    echo "=============================================="
    # GRPO+vLLM server: CUDA_VISIBLE_DEVICES is trimmed to training GPUs; NUM_GPUS must match.
    if [ "${#TRAINING_GPUS[@]}" -gt 0 ]; then
        export NUM_GPUS="${#TRAINING_GPUS[@]}"
    fi

    if [ "$USE_DDP" == "True" ]; then
        echo "Command: accelerate launch --multi_gpu --num_processes $NUM_GPUS --mixed_precision bf16 training_code.py ${CMD_ARGS[*]}"
        echo ""
        if [ "$DRY_RUN" != "True" ]; then
            accelerate launch \
                --multi_gpu \
                --num_processes $NUM_GPUS \
                --mixed_precision bf16 \
                training_code.py "${CMD_ARGS[@]}"
        fi
    else
        echo "Command: python training_code.py ${CMD_ARGS[*]}"
        echo ""
        if [ "$DRY_RUN" != "True" ]; then
            python training_code.py "${CMD_ARGS[@]}"
        fi
    fi
    EXIT_CODE=$?
fi

# Capture exit code
if [ "$DRY_RUN" == "True" ]; then
    echo "DRY RUN: skipped training execution."
    EXIT_CODE=0
fi

echo ""
echo "=============================================="
echo "Training Completed"
echo "=============================================="
echo "Exit Code: $EXIT_CODE"
echo "End Time: $(date)"
end_time=$(date +%s)

duration=$((end_time - start_time))
printf "Duration: %d seconds, (%d days, %02d:%02d:%02d)\n" "$duration" "$((duration/86400))" "$(( (duration%86400)/3600 ))" "$(( (duration%3600)/60 ))" "$(( duration%60 ))"
echo "=============================================="

exit $EXIT_CODE
