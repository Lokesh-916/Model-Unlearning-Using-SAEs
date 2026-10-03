#!/bin/bash
# A6-full relearning (full fine-tune, k 10/100/1000) on the D1 v2 selected student.
#SBATCH --job-name=dsg-d1-v2-a6
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/a6_full.py --targets d1v2
