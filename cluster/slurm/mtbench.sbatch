#!/bin/bash
# MT-Bench (BM3): base / DSG / window-w16 gate answers, fixed open judge google/gemma-2-9b-it (same family as the judged
# model: labelled), FastChat single-answer prompts, greedy. Resumable per answer / judgment row; <= 3 h per copy, chain 4.
#SBATCH --job-name=dsg-mtbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later7   # session 11: gemma-2-9b-it judge, window-w16 mode, budget
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/mtbench_open.py --budget-min "$DSG_BUDGET_MIN"
