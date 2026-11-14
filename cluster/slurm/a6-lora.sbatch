#!/bin/bash
# Lab A6 LoRA tampering grid, DSG hook/no-hook targets (MOVED-TO-SERVER), pinned exp code. Chain 2-3 copies; each continues.
#SBATCH --job-name=dsg-a6-lora
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
python cluster/lab_jobs.py --group a6-lora --budget-min "$DSG_BUDGET_MIN"
