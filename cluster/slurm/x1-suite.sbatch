#!/bin/bash
# X1-suite on gpuws (session 27): the lab's six X1-suite jobs (StreamGuard = X1 CUSUM gate vs DSG: TOFU metrics,
# streaming TOFU QA forget/retain, A3 benign-biology open-ended, B6 leakage/gibberish, paired incl. the gpuws X1
# hard-negative MCQ runs) with the pinned exp/X1-suite code (cluster/lab_jobs.py, group x1-suite; a replica).
# Resumable: DONE jobs are skipped, a killed generation loop resumes from its item chunk; chain 3 copies.
#SBATCH --job-name=dsg-x1-suite
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later11
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/lab_jobs.py --group x1-suite --budget-min "$DSG_BUDGET_MIN"
