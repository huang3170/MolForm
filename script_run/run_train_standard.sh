#!/bin/bash

# Standard Training Script
# This script demonstrates how to run standard diffusion model training

export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Standard training (no DPO data provided)
echo "Starting standard training at $(date)"
python scripts/train_diffusion.py \
    configs/training_standard.yml \
    --tag standard_training \
    --name "Standard Training"

echo "Standard training completed at $(date)"
