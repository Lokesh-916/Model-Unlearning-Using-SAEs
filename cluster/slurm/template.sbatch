#!/bin/bash
# Base template for dsg jobs on gpuws. Copy, rename, set --job-name/--time and the body.
# No --mem: the node advertises RealMemory=1 MB, so ANY --mem request is rejected by Slurm
# (admin item). Memory is not confined either; our jobs peak at ~25 GB host RAM.
#SBATCH --job-name=dsg-template
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"

# --- job body (must be resumable: a restart re-runs this script from the top) ---
python -c "import torch; print(torch.cuda.get_device_name(0))"
