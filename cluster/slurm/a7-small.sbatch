#!/bin/bash
# A7 on gpuws: Gemma 3 1B and 4B (base / DSG rule / best fix x clean, dilution, translate). Resumable (DONE runs skipped).
#SBATCH --job-name=dsg-a7-small
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
python cluster/a7_server.py --sizes 1b 4b --budget-min "$DSG_BUDGET_MIN"
