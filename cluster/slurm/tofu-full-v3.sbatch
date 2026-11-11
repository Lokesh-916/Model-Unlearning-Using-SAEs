#!/bin/bash
# A2-tofu-full re-run (session 13): both TOFU fine-tunes with the fixed Trainer (all 4 micro-batches per step),
# then retain / full / full+DSG / full+window-gate metrics. Old models and partials (no train_version 2) are never
# reused. Stops at a checkpoint / between conditions when the budget runs low; chain 2 copies (~2.5 h total).
#SBATCH --job-name=dsg-tofu-full-v3
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later8   # session 13: Trainer accumulation fix (f40dde8) + train_version guards
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/tofu_full.py --budget-min "$DSG_BUDGET_MIN"
