#!/bin/bash
# MT-Bench, open judge (fixed: unsloth/Qwen2.5-32B-Instruct-bnb-4bit, greedy, FastChat single-answer prompts). Resumable per answer / judgment row.
#SBATCH --job-name=dsg-mtbench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
python cluster/mtbench_open.py
