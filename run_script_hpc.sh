#!/bin/bash

# Check if a python script file was provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <script.py>"
    exit 1
fi

PYTHON_SCRIPT="$1"

# Check if the file exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: File '$PYTHON_SCRIPT' not found!"
    exit 1
fi

echo "=========================================="
echo " Setting up Northeastern HPC Environment"
echo "=========================================="

# Set Cache boundaries to protect Home Quota
export HF_HOME="/scratch/$USER/huggingface_cache"
export TRANSFORMERS_CACHE="/scratch/$USER/huggingface_cache"
export CONDA_PKGS_DIRS="/scratch/$USER/conda_pkgs"
export PIP_CACHE_DIR="/scratch/$USER/pip_cache"

# Load modules
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

# Activate existing Python environment
source activate "/scratch/$USER/6140_env"

echo "=========================================="
echo " Running $PYTHON_SCRIPT"
echo "=========================================="
python "$PYTHON_SCRIPT"

echo "=========================================="
echo " Finished execution of $PYTHON_SCRIPT"
echo "=========================================="
