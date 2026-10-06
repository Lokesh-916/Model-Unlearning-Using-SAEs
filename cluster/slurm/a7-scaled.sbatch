#!/bin/bash
# A7-scaled (MOVED-TO-SERVER, session 20): Gemma 3 1B, base / DSG default / clamp scaled x103; pinned exp/A7-gemma3 code.
#SBATCH --job-name=dsg-a7-scaled
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later9
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/lab_jobs.py --group a7-scaled --offline-sae-shapes --budget-min "$DSG_BUDGET_MIN"
