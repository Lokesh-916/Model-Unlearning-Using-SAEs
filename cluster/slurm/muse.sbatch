#!/bin/bash
# A5-muse: gemma-2-2b-it MUSE targets (News, Books), DSG, VerbMem / KnowMem / PrivLeak. Finished corpora are skipped on restart.
#SBATCH --job-name=dsg-muse
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
python cluster/muse.py
