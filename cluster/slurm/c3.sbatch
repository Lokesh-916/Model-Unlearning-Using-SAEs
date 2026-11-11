#!/bin/bash
# Lab C3 layer sweep (MOVED-TO-SERVER) with its pinned exp-branch code (cluster/lab_jobs.py). Resumable.
#SBATCH --job-name=dsg-c3
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
python cluster/lab_jobs.py --group c3 --budget-min "$DSG_BUDGET_MIN"
