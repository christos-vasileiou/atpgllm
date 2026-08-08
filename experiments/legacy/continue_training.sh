#!/bin/bash
#SBATCH --job-name=orig_llama_training        # Job name
#SBATCH --output=jobs/orig_llama_training_%j.out   # Standard output file (%j will be replaced with job ID)
#SBATCH --error=jobs/orig_llama_training_%j.err    # Standard error file
#SBATCH --nodes=1                      # Request 1 node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=8              # Request 8 CPUs per task (adjust as needed)
#SBATCH --mem=80G                      # Request 64GB of memory (adjust as needed)
#SBATCH --partition=h100               # Specify the GPU partition (adjust based on your cluster)
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3:2 

mkdir -p jobs
# Print some information about the job
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start time: $(date)"

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export CUDA_VISIBLE_DEVICES="0,1"

# Run the training command
torchrun --nproc-per-node=2 ft_train_with_rl.py \
  --is-causal \
  --batch-size 4096 \
  --micro-batch-size 8 \
  --epochs 5 \
  --lr 5e-7 \
  --num-workers 0 \
  --model-max-length 1280 \
  --data-file "../../data/cot_atpg_data_v[1-5].csv" \
  --model-name llama-2 \
  --save-model test-model-v2 \
  --parallel \
  --lora-r 512 \
  --lora-alpha 4096 \
  --grpo-tau 1. \
  --num-generations 8 \
  --adapter-name grpo_adapter \
  --adapter-repo logs/models/best_reward_model/TestModel-2 \
  --initial-ref-update-freq 50 \
  --final-ref-update-freq 50 \
  --wandb

echo "End time: $(date)" 
