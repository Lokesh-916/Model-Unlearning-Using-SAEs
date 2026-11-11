#!/bin/bash
# A2-tofu-full re-evaluation with the corrected TOFU metrics (METRIC_VERSION 2: per-item truth-ratio score,
# option-normalised answer prob on real/world sets). Fine-tuned models are cached; eval only (~1.5 h). Resumable.
#SBATCH --job-name=dsg-tofu-full-v2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later2
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/tofu_full.py
