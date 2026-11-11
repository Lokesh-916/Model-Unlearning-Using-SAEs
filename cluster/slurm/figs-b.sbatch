#!/bin/bash
# DSG figure parity, part B: static vs dynamic, multi-topic (TEST), latency by length, TOFU highlights. Resumable.
#SBATCH --job-name=dsg-figs-b
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
python cluster/figparity.py highlight latency static multitopic --budget-min "$DSG_BUDGET_MIN"
