#!/bin/bash
# DSG figure parity: latency re-run with the interleaved protocol (v2) and host-load logging (~15 min).
#SBATCH --job-name=dsg-figs-lat
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:45:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later3 && source "$HOME/dsg_cluster/slurm/later.sh" && python cluster/figparity.py latency --budget-min "$DSG_BUDGET_MIN"
