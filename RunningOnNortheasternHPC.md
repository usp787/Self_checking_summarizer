## SSH into Northeastern HPC
```sh
ssh user.name@login.explorer.northeastern.edu
# use your northastern credentials

# if necessary, hop to the explorer node with previous tmux session
ssh explorer-01 # if you are not on 01 but have a tmux session there, hop to 01 first, then tmux attach
```

## Requesting GPU resources on Northeastern HPC
```sh
srun -p courses-gpu --gres=gpu:1 --time=4:00:00 --mem=32GB --cpus-per-task=4 --pty /bin/bash
```

## Convert a notebook to a Python script for easier execution on HPC
```sh
jupyter nbconvert --to python CS6140_final_baseline.ipynb
```

## General commands for a new HPC session
Do use `tmux` whenever you ask the HPC to run something for you, so that you can safely disconnect without killing your job.
```sh
# Set Cache boundaries to protect Home Quota
export HF_HOME=/scratch/$USER/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs
export PIP_CACHE_DIR=/scratch/$USER/pip_cache

# Load system environments (Crucial order: purge first, THEN load explorer for proxies)
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

# Activate existing Python environment
source activate /scratch/$USER/6140_env

# Run your script
cd /home/xu.x1/6140/Self_checking_summarizer/
# my home contains this git repo
git pull
python CS6140_inference_verification.py
```

## Rebuild /scratch/$USER/6140_env if you need to

```sh
# 1. Setup Caches to protect Home Quota
export HF_HOME=/scratch/$USER/huggingface_cache
export TRANSFORMERS_CACHE=/scratch/$USER/huggingface_cache
export CONDA_PKGS_DIRS=/scratch/$USER/conda_pkgs
export PIP_CACHE_DIR=/scratch/$USER/pip_cache

# 2. Re-create directories (just in case they were also wiped)
mkdir -p /scratch/$USER/huggingface_cache
mkdir -p /scratch/$USER/conda_pkgs
mkdir -p /scratch/$USER/pip_cache

# 3. Load Modules (Critical: explorer handles the proxy, cuda handles the GPU)
module purge
module load explorer anaconda3/2024.06 cuda/12.1.1

# 4. Create the new Conda Environment
conda create --prefix /scratch/$USER/6140_env -c conda-forge python=3.12.4 -y

# 5. Activate the new empty environment
source activate /scratch/$USER/6140_env

# 6. Reinstall exactly what the HPC needs for GPU acceleration (PyTorch 2.5 on CUDA 12.1)
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 7. Install project dependencies from requirements.txt (single source of truth)
cd /home/xu.x1/6140/Self_checking_summarizer/
pip install -r requirements.txt

# 8. Run your script! (It will re-download the model weights automatically)
cd /home/xu.x1/6140/Self_checking_summarizer/
git pull
python CS6140_inference_verification.py
```