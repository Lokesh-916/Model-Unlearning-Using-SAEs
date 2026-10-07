#!/bin/bash
# X1-suite on gpuws (session 27), step 1: the retain-only TOFU reference model on gpuws (tofu_full.finetune, ~25 min)
# + ckpt:A2/{tofu_full,tofu_retain} links to the gpuws TOFU-full v3 models. Exit 3 (afterok holds the chain) if the
# retain fine-tune did not finish; resubmit to resume from its checkpoint.
#SBATCH --job-name=dsg-x1suite-models
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later11   # session 27: cluster/x1suite_server.py
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/x1suite_server.py models --budget-min "$DSG_BUDGET_MIN"
