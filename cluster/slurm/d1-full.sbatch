#!/bin/bash
# D1-full: full-parameter UNDO distillation (3 noise levels) + harness TEST eval. Resumable (trainer state every 100 steps; finished students and DONE runs are skipped).
#SBATCH --job-name=dsg-d1-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
python cluster/d1_full.py
