#!/bin/bash
# DSG figure parity, part A: clamp strength x feature count grid (DEV) and data efficiency (DEV). Resumable.
#SBATCH --job-name=dsg-figs-a
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
python cluster/figparity.py clamp dataeff --budget-min "$DSG_BUDGET_MIN"
