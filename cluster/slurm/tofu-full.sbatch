#!/bin/bash
# A2-tofu-full: full TOFU fine-tunes (full, retain), DSG + best gate on top, TOFU metrics. Resumable.
#SBATCH --job-name=dsg-tofu-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
python cluster/tofu_full.py
