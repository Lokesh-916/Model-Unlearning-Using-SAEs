#!/bin/bash
# D1 v3 test: TEST bs 1 of base, DSG paper, the 3 students (@forget, @utility, 9 hazard-adjacent subjects); chain 2. cluster/d1_v3.py (PREPARED 2026-10-08, README "How to run D1 v3 on gpuws without Claude").
#SBATCH --job-name=dsg-d1-v3-test
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-d1v3   # staged by server.sh stage d1-v3 (STAGE_SCRIPT)
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/d1_v3.py test --budget-min "$DSG_BUDGET_MIN"
