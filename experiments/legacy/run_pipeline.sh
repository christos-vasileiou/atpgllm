#!/bin/bash

echo "Starting SFT Training 2x H100..."
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 --num_machines 1 train_sft_70b.py
echo "SFT Training completed"

echo "Starting GRPO Training 2x H100..."
accelerate launch --multi_gpu --num_processes 2 --mixed_precision bf16 --num_machines 1 train_grpo_70b.py
echo "GRPO Training completed"
