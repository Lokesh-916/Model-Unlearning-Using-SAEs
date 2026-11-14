#!/bin/bash
# D1 v2: trains the next unfinished noise alpha (0.05, 0.1, 0.2; 4000 steps) + its DEV eval. Chain 3-4 copies.
#SBATCH --job-name=dsg-d1-v2-train
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/d1_v2.py train --budget-min "$DSG_BUDGET_MIN"
