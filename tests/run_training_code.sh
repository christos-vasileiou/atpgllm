#!/bin/bash
#SBATCH --job-name=unsloth_vllm        # Job name
#SBATCH --output=jobs/training_%j.out   # Standard output file (%j will be replaced with job ID)
#SBATCH --error=jobs/training_%j.err    # Standard error file
#SBATCH --nodes=1                       # Request 1 node
#SBATCH --ntasks=1                      # Run a single task
#SBATCH --cpus-per-task=16              # Request 16 CPUs per task
#SBATCH --mem=128G
#SBATCH --partition=h100
#SBATCH --gres=gpu:nvidia_h100_nvl:2

start_time=$(date +%s)
# =============================================================================
# Debugging Information
# =============================================================================
echo "=============================================="
echo "Job Information"
echo "=============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $(hostname)"
echo "GRES: ${SLURM_GRES:-'(not set)'}"
echo "Node: $(hostname)"
echo "Date: $(date)"
echo "Working Directory: $(pwd)"
echo ""

echo "=============================================="
echo "System Information"
echo "=============================================="
echo "Python Version: $(python --version 2>&1)"
echo "PyTorch Version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not available')"
echo "CUDA Available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'Not available')"
echo "CUDA Version: $(python -c 'import torch; print(torch.version.cuda)' 2>/dev/null || echo 'Not available')"
echo ""

echo "=============================================="
echo "ML Library Versions"
echo "=============================================="
echo "Transformers Version: $(python -c 'import transformers; print(transformers.__version__)' 2>/dev/null || echo 'Not available')"
echo "TRL Version: $(python -c 'import trl; print(trl.__version__)' 2>/dev/null || echo 'Not available')"
echo "PEFT Version: $(python -c 'import peft; print(peft.__version__)' 2>/dev/null || echo 'Not available')"
echo ""

echo "=============================================="
echo "GPU Information"
echo "=============================================="
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv 2>/dev/null || echo "nvidia-smi not available"
echo ""

echo "=============================================="
echo "SLURM Environment"
echo "=============================================="
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM_GPUS: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-'(not set)'}"
echo ""

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

# activate virtual environment (activate is alias)
activate

# MODEL is required for SFT, optional for GRPO (which can resume from checkpoint)
if [ -z "$MODEL" ] && [ "$METHOD" != "grpo" ]; then
    echo "ERROR: MODEL environment variable is not set!"
    echo "Usage: MODEL=<model_name> TRAIN_DATASET=<dataset_path> METHOD=<sft|grpo> sbatch run_test_training_code.sh"
    exit 1
fi
if [ "$METHOD" == "sft" ]; then
    MODEL=${MODEL:-Qwen/Qwen2.5-72B-Instruct}
    TRAIN_DATASET=${TRAIN_DATASET:-chrivasileiou/asap7-language-of-test}
    OUTPUT_DIR=${OUTPUT_DIR:-sft_finetuned_model}
    RESUME_FROM=${RESUME_FROM:-}
    PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
    MAX_STEPS=${MAX_STEPS:-50}
    REPORT_TO=${REPORT_TO:-wandb}
    USE_DUAL_ADAPTER=${USE_DUAL_ADAPTER:-False}
    USE_VLLM=${USE_VLLM:-$(python -c "import vllm" && echo True || echo False)}
    USE_UNSLOTH=${USE_UNSLOTH:-$(python -c "import unsloth" && echo True || echo False)}
elif [ "$METHOD" == "grpo" ]; then
    MODEL=${MODEL:-}
    TRAIN_DATASET=${TRAIN_DATASET:-chrivasileiou/asap7-language-of-test}
    OUTPUT_DIR=${OUTPUT_DIR:-grpo_finetuned_model}
    RESUME_FROM=${RESUME_FROM:-sft_finetuned_model/checkpoint-150}
    BUFFER_SIZE=${BUFFER_SIZE:-10000}
    PER_DEVICE_TRAIN_BATCH_SIZE=${PER_DEVICE_TRAIN_BATCH_SIZE:-2}
    GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
    MAX_STEPS=${MAX_STEPS:-50}
    REPORT_TO=${REPORT_TO:-wandb}
    NUM_GENERATIONS=${NUM_GENERATIONS:-8}
    STEPS_PER_GENERATION=${STEPS_PER_GENERATION:-4}
    USE_DUAL_ADAPTER=${USE_DUAL_ADAPTER:-True}
    USE_VLLM=${USE_VLLM:-$(python -c "import vllm" && echo True || echo False)}
    USE_UNSLOTH=${USE_UNSLOTH:-$(python -c "import unsloth" && echo True || echo False)}
fi

echo "=============================================="
echo "Command Arguments"
echo "=============================================="
echo "METHOD: $METHOD"
echo "MODEL: $MODEL"
echo "TRAIN_DATASET: $TRAIN_DATASET"
echo "OUTPUT_DIR: $OUTPUT_DIR"
echo "RESUME_FROM: $RESUME_FROM"
echo "BUFFER_SIZE: $BUFFER_SIZE"
echo "PER_DEVICE_TRAIN_BATCH_SIZE: $PER_DEVICE_TRAIN_BATCH_SIZE"
echo "GRADIENT_ACCUMULATION_STEPS: $GRADIENT_ACCUMULATION_STEPS"
echo "MAX_STEPS: $MAX_STEPS"
echo "REPORT_TO: $REPORT_TO"
echo "NUM_GENERATIONS: $NUM_GENERATIONS"
echo "STEPS_PER_GENERATION: $STEPS_PER_GENERATION"
echo "USE_DUAL_ADAPTER: $USE_DUAL_ADAPTER"
echo "USE_VLLM: $USE_VLLM"
echo "USE_UNSLOTH: $USE_UNSLOTH"
echo "=============================================="

# =============================================================================
# Build Command Arguments
# =============================================================================
build_cmd_args() {
    CMD_ARGS=(--method "$METHOD")

    # Add --model_name only if it's set and not "None"
    if [ -n "$MODEL" ] && [ "$MODEL" != "None" ]; then
        CMD_ARGS+=(--model_name "$MODEL")
    fi

    CMD_ARGS+=(
        --dataset "$TRAIN_DATASET"
        --output_dir "$OUTPUT_DIR"
        --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE"
        --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS"
        --max_steps "$MAX_STEPS"
        --report_to "$REPORT_TO"
    )

    if [ -n "$USE_VLLM" ] && [ "$USE_VLLM" == "True" ]; then
        CMD_ARGS+=(--use_vllm)
    fi

    if [ -n "$USE_UNSLOTH" ] && [ "$USE_UNSLOTH" == "True" ]; then
        CMD_ARGS+=(--use_unsloth)
    fi

    # Add --resume_from only if it's set and not "None"
    if [ -n "$RESUME_FROM" ] && [ "$RESUME_FROM" != "None" ]; then
        CMD_ARGS+=(--resume_from "$RESUME_FROM")
    fi

    # Add GRPO-specific arguments only for GRPO method
    if [ "$METHOD" == "grpo" ]; then
        CMD_ARGS+=(
            --buffer_size "$BUFFER_SIZE"
            --num_generations "$NUM_GENERATIONS"
            --steps_per_generation "$STEPS_PER_GENERATION"
        )
        # This condition checks whether the variable USE_DUAL_ADAPTER is set (not empty) and its value is exactly "True".
        if [ -n "$USE_DUAL_ADAPTER" ] && [ "$USE_DUAL_ADAPTER" == "True" ]; then
            CMD_ARGS+=(--use_dual_adapter)
        fi
    fi
}

build_cmd_args

export CMD_ARGS

echo "=============================================="
echo "Starting Training"
echo "=============================================="
echo "Command: python training_code.py ${CMD_ARGS[*]}"
echo ""

echo "Allocated GPU:"
echo $CUDA_VISIBLE_DEVICES
nvidia-smi

# Run the training script
python training_code.py "${CMD_ARGS[@]}"

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
