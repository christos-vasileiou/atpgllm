#!/bin/bash
#SBATCH --job-name=grpo_vllm        # Job name
#SBATCH --output=jobs/training_%j.out   # Standard output file (%j will be replaced with job ID)
#SBATCH --error=jobs/training_%j.err    # Standard error file
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=32              # Request 16 CPUs per task
#SBATCH --mem=128G
#SBATCH --partition=h100
#SBATCH --gres=gpu:4
#SBATCH --reservation=vasileoiou

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

start_time=$(date +%s)
# =============================================================================
# Debugging Information
# =============================================================================


# echo "=============================================="
# echo "Job Information"
# echo "=============================================="
# echo "Job ID: $SLURM_JOB_ID"
# echo "Job Name: $SLURM_JOB_NAME"
# echo "Node: $(hostname)"
# echo "GRES: ${SLURM_GRES:-'(not set)'}"
# echo "Date: $(date)"
# echo "Working Directory: $(pwd)"
# echo ""

# echo "=============================================="
# echo "System Information"
# echo "=============================================="
# echo "Python Version: $(python --version 2>&1)"
# echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
# echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not available')"
# echo "CUDA Version: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Not available')"
# echo ""

# echo "=============================================="
# echo "ML Library Versions"
# echo "=============================================="
# echo "Transformers Version: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'Not available')"
# echo "TRL Version: $(python -c 'import trl; print(trl.__version__)' 2>/dev/null || echo 'Not available')"
# echo "PEFT Version: $(python -c 'import peft; print(peft.__version__)' 2>/dev/null || echo 'Not available')"
# echo ""

# echo "=============================================="
# echo "GPU Information"
# echo "=============================================="
# nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not available"
# echo ""

# echo "=============================================="
# echo "SLURM Environment"
# echo "=============================================="
# echo "SLURM_NTASKS: $SLURM_NTASKS"
# echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
# echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
# echo "SLURM_GPUS: $SLURM_GPUS"
# echo "SLURM_STEP_GPUS: $SLURM_STEP_GPUS"
# echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'(not set)'}"
# echo ""

# =============================================================================
# Validate Required Environment Variables
# =============================================================================
if [ -z "$MODEL" ]; then
    echo "ERROR: MODEL environment variable is not set!"
    echo "Usage: MODEL=<model_name> TRAIN_DATASET=<dataset_path> METHOD=<sft|grpo> sbatch run_test_training_code.sh"
    exit 1
fi

if [ -z "$TRAIN_DATASET" ]; then
    echo "ERROR: TRAIN_DATASET environment variable is not set!"
    echo "Usage: MODEL=<model_name> TRAIN_DATASET=<dataset_path> METHOD=<sft|grpo> sbatch run_test_training_code.sh"
    exit 1
fi

# Default METHOD to 'sft' if not specified
METHOD=${METHOD:-sft}

# MODEL is required for SFT, optional for GRPO (which can resume from checkpoint)
if [ -z "$MODEL" ] && [ "$METHOD" != "grpo" ]; then
    echo "ERROR: MODEL environment variable is not set!"
    echo "Usage: MODEL=<model_name> TRAIN_DATASET=<dataset_path> METHOD=<sft|grpo> sbatch run_test_training_code.sh"
    exit 1
fi
if [ "$METHOD" == "sft" ]; then
    MODEL=${MODEL:-Qwen/Qwen2.5-7B-Instruct}
    TRAIN_DATASET=${TRAIN_DATASET:-chrivasileiou/asap7-language-of-test}
    OUTPUT_DIR=${OUTPUT_DIR:-sft_finetuned_model}
    RESUME_FROM=${RESUME_FROM:-}
    RESUME_TRAINING_STATE=${RESUME_TRAINING_STATE:-False}
    PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
    MAX_STEPS=${MAX_STEPS:-50}
    REPORT_TO=${REPORT_TO:-wandb}
    USE_DUAL_ADAPTER=${USE_DUAL_ADAPTER:-False}
    # SFT does not need a vLLM engine during the training loop itself.
    # The SFT stopping callback *can* use vLLM for faster validation,
    # but on MIG instances (~12 GB) there is not enough VRAM to host
    # both the training model and a vLLM engine (KV cache = 0 GB).
    # Default to False; override with USE_VLLM=True for large GPUs.
    # When USE_VLLM=True, start a persistent vLLM server on a spare GPU
    # *before* launching training with dynamic LoRA loading enabled:
    #   VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
    #       CUDA_VISIBLE_DEVICES=<gpu> vllm serve <model> \
    #       --enable-lora --max-lora-rank 64 --port 8000
    USE_VLLM=${USE_VLLM:-False}
    VLLM_MODE=${VLLM_MODE:-server}
    PORT=${PORT:-8002}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
    MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
    USE_UNSLOTH=${USE_UNSLOTH:-$(python -c "import unsloth" 2>/dev/null && echo True || echo False)}
    USE_DDP=${USE_DDP:-True}
elif [ "$METHOD" == "grpo" ]; then
    MODEL=${MODEL:-}
    TRAIN_DATASET=${TRAIN_DATASET:-chrivasileiou/asap7-language-of-test}
    OUTPUT_DIR=${OUTPUT_DIR:-grpo_finetuned_model}
    RESUME_FROM=${RESUME_FROM:-sft_finetuned_model/checkpoint-150}
    RESUME_TRAINING_STATE=${RESUME_TRAINING_STATE:-False}
    BUFFER_SIZE=${BUFFER_SIZE:-10000}
    PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
    MAX_STEPS=${MAX_STEPS:-50}
    REPORT_TO=${REPORT_TO:-wandb}
    NUM_GENERATIONS=${NUM_GENERATIONS:-8}
    STEPS_PER_GENERATION=${STEPS_PER_GENERATION:-4}
    USE_DUAL_ADAPTER=${USE_DUAL_ADAPTER:-True}
    USE_VLLM=${USE_VLLM:-False}
    VLLM_MODE=${VLLM_MODE:-server}
    PORT=${PORT:-8002}
    USE_UNSLOTH=${USE_UNSLOTH:-False}
    USE_DDP=${USE_DDP:-False}
    MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
    MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
    MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-6144}
    SKIP_BUFFER_SIZE=${SKIP_BUFFER_SIZE:-0}
fi

# LoRA hyper-parameters (read by training_code.py via environment / argparse defaults)
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-}
export LORA_RANK LORA_ALPHA LORA_TARGET_MODULES

# =============================================================================
# GPU Configuration and Setup
# =============================================================================
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
                CMD_ARGS+=(--vllm_server_url "http://localhost:$PORT")
                if [ "$VLLM_MODE" == "server" ]; then
                    VLLM_GPU=${GPU_ARRAY[-1]}
                fi
            elif [ "$METHOD" == "grpo" ]; then
                PORT=$PORT
                CMD_ARGS+=(--vllm_server_url "http://localhost:$PORT")
                CMD_ARGS+=(--vllm_mode "$VLLM_MODE")
                if [ "$VLLM_MODE" == "server" ]; then
                    VLLM_GPU=${GPU_ARRAY[-1]}
                fi
            fi
        fi
    fi

    # Handle DDP activation
    if [ "$USE_DDP" == "True" ]; then
        CMD_ARGS+=("--use_ddp")                
    fi

    # Handle unsloth activation
    if [ -n "$USE_UNSLOTH" ] && [ "$USE_UNSLOTH" == "True" ]; then
        CMD_ARGS+=(--use_unsloth)
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
    )

    # Add GRPO-specific arguments only for GRPO method
    if [ "$METHOD" == "grpo" ]; then
        CMD_ARGS+=(
            --buffer_size "$BUFFER_SIZE"
            --skip_buffer_size "${SKIP_BUFFER_SIZE:-0}"
            --num_generations "$NUM_GENERATIONS"
            --steps_per_generation "$STEPS_PER_GENERATION"
            --max_completion_length "$MAX_COMPLETION_LENGTH"
        )
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
    echo "PORT: $PORT (http://localhost:$PORT, otherwise no vLLM server needed)"
    echo "*VLLM_GPU: ${VLLM_GPU:-(auto)} (set when vLLM server mode is used)"
    echo "USE_DUAL_ADAPTER: $USE_DUAL_ADAPTER"
    echo "USE_UNSLOTH: $USE_UNSLOTH"
    echo "USE_DDP: $USE_DDP"
    echo "LORA_RANK: $LORA_RANK"
    echo "LORA_ALPHA: $LORA_ALPHA"
    echo "LORA_TARGET_MODULES: ${LORA_TARGET_MODULES:-'(default: q/k/v/o_proj + gate/up/down_proj)'}"
    if [ "$METHOD" == "grpo" ]; then
        echo "--- GRPO-specific ---"
        echo "BUFFER_SIZE: $BUFFER_SIZE"
        echo "SKIP_BUFFER_SIZE: ${SKIP_BUFFER_SIZE:-0}"
        echo "NUM_GENERATIONS: $NUM_GENERATIONS"
        echo "STEPS_PER_GENERATION: $STEPS_PER_GENERATION"
        echo "MAX_COMPLETION_LENGTH: $MAX_COMPLETION_LENGTH"
    fi
    echo "=============================================="
    echo "CMD_ARGS: ${CMD_ARGS[*]}"
    echo "=============================================="
}

build_cmd_args
export CMD_ARGS

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
        COUNT_VLLM_GPUS=$(echo "$VLLM_GPU" | tr ',' '\n' | wc -l)
        NUM_TRAIN_GPUS=$((${#GPU_ARRAY[@]} - $COUNT_VLLM_GPUS))
        TRAINING_GPUS=("${GPU_ARRAY[@]:0:$NUM_TRAIN_GPUS}")
        export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${TRAINING_GPUS[*]}")"
    fi

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
    accelerate launch \
        --multi_gpu \
        --num_processes $NUM_GPUS \
        --mixed_precision bf16 \
        training_code.py "${CMD_ARGS[@]}"
else
    echo "Command: python training_code.py ${CMD_ARGS[*]}"
    echo ""
    python training_code.py "${CMD_ARGS[@]}"
fi

# Capture exit code
EXIT_CODE=$?

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
