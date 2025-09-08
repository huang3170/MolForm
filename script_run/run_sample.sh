#!/bin/bash



# # Activate conda environment
# conda activate targetdiff_2_5
# echo "Activated environment: $CONDA_DEFAULT_ENV"

# # Go to working directory
# echo "Working directory: $(pwd)"


# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3


# Run the command
echo "Starting training at $(date)"
CUDA_VISIBLE_DEVICES=0 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs/samples 4 0 0 &
CUDA_VISIBLE_DEVICES=1 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs/samples 4 1 0 &
CUDA_VISIBLE_DEVICES=2 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs/samples 4 2 0 &
CUDA_VISIBLE_DEVICES=3 bash scripts/batch_sample_diffusion.sh configs/sampling.yml outputs/samples 4 3 0 &
wait
# Print completion info
echo "Job completed at $(date)"

