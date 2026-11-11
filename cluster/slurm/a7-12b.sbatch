#!/bin/bash
# A7 on gpuws: Gemma 3 12B, same conditions as a7. Chain 2 copies (the first builds the 12B cache). Resumable.
#SBATCH --job-name=dsg-a7-12b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later6   # session 10: offline Gemma Scope 2 shapes (job 110)
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/a7_server.py --sizes 12b --budget-min "$DSG_BUDGET_MIN"
