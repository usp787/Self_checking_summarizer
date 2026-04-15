#!/bin/bash
#SBATCH --job-name=cs6140_v1_350
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=01:40:00
#SBATCH --mem=32GB
#SBATCH --output=slurm_v1_350_%j.out
#SBATCH --error=slurm_v1_350_%j.err

echo "=========================================="
echo "Starting job on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "=========================================="

# Run the wrapper script with the Python file as the argument
/home/xu.x1/6140/Self_checking_summarizer/run_script_hpc.sh /home/xu.x1/6140/Self_checking_summarizer/CS6140_final_v1_350.py

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
