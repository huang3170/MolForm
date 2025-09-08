#!/bin/bash


# Set environment variables
export PYTHONPATH=$(pwd):$PYTHONPATH
python scripts/evaluate_diffusion.py ./output --docking_mode vina_dock --protein_root data/test_set

