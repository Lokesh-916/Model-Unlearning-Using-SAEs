#!/bin/bash
# D1 v2: DEV selection (lowest forget, utility drop <= 0.02) then TEST of base, DSG, selected student.
#SBATCH --job-name=dsg-d1-v2-test
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
python cluster/d1_v2.py test --budget-min "$DSG_BUDGET_MIN"
