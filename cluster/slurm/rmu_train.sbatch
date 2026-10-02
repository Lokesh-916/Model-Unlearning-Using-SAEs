#!/bin/bash
# Step 7a: RMU dev grid (6 configs) on gemma-2-2b-it / WMDP-Bio. Resumable (grid_state.json +
# trainer state every 50 steps); a restart continues where it stopped.
#SBATCH --job-name=dsg-rmu-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
python cluster/rmu_train.py
