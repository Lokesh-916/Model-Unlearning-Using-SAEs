#!/bin/bash
# A5-muse (BM1): gemma-2-2b-it MUSE targets (News, Books) + retrain references, DSG and the best gate,
# official VerbMem / KnowMem / PrivLeak. Resumable stages within a <= 3 h budget; chain 3 copies
# (cluster/muse.py: training stops at a checkpoint when the budget is low, finished conditions are skipped).
#SBATCH --job-name=dsg-muse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later4
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/muse.py --budget-min "$DSG_BUDGET_MIN"
