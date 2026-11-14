#!/bin/bash
# A5-muse re-run (session 13): retrain + target fine-tunes with the fixed Trainer (all 4 micro-batches per step),
# official VerbMem / KnowMem / PrivLeak; old partials / DONE (no train_version 2) are redone. Chain 3 copies.
#SBATCH --job-name=dsg-muse-v2
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
python cluster/muse.py --budget-min "$DSG_BUDGET_MIN"
