#!/bin/bash
# FP-highlight re-run (session 13) on the re-trained TOFU model of tofu-full-v3 (~15 min).
#SBATCH --job-name=dsg-figs-hl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later8   # session 13: Trainer accumulation fix (f40dde8) + train_version guards
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/figparity.py highlight --budget-min "$DSG_BUDGET_MIN"
