#!/bin/bash
# A7-12b: Gemma 3 12B + Gemma Scope 2 (layer 24) harness TEST runs: base / DSG rule / best fix x clean, dilution, translate. DONE runs are skipped on restart.
#SBATCH --job-name=dsg-a7-12b
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
python cluster/a7_12b.py
