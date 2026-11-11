#!/bin/bash
# Step 7b: TEST eval (bs=1, both views) of the selected RMU vs base, DSG-paper, third-party RMU.
# Harness runs skip when DONE, so a restart only redoes the unfinished run.
#SBATCH --job-name=dsg-rmu-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
python cluster/rmu_eval.py
