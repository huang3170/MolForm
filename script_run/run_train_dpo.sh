#!/bin/bash

# DPO Training Script
# This script demonstrates how to run DPO training

export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# DPO training with preference data
echo "Starting DPO training at $(date)"
python scripts/train_diffusion.py \
    configs/training_dpo.yml \
    --tag dpo_training \
    --dpo_data /work/hdd/bdrx/dzhang5/sbdd_data/data/dpo_data/dpo_idx_sort_new.pkl \
    --name "DPO Training"

echo "DPO training completed at $(date)"
