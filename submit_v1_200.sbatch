#!/bin/bash
#SBATCH --job-name=cs6140_v1_200
#SBATCH --partition=gpu
#SBATCH --gres=gpu:h200:1
#SBATCH --time=05:00:00
#SBATCH --mem=32GB
#SBATCH --output=slurm_v1_200_%j.out
#SBATCH --error=slurm_v1_200_%j.err

echo "=========================================="
echo "Starting job on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "=========================================="

# Run the wrapper script with the Python file as the argument
/home/xu.x1/6140/Self_checking_summarizer/run_script_hpc.sh /home/xu.x1/6140/Self_checking_summarizer/CS6140_final_v1_200.py

echo "=========================================="
echo "Job finished at $(date)"
echo "=========================================="
