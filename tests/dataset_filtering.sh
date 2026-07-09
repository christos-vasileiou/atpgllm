#!/bin/bash
#SBATCH --job-name=asap7-ds-filter
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/asap7-filter-%j.out
#SBATCH --error=logs/asap7-filter-%j.err


set -euo pipefail
mkdir -p jobs

source activate

python dataset_filtering.py


