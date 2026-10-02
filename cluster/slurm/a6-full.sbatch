#!/bin/bash
# A6-full: full fine-tune relearning on every staged target x k in {10,100,1000}. A requeue continues with the next unfinished cell.
#SBATCH --job-name=dsg-a6-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
python cluster/a6_full.py
