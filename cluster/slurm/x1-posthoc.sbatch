#!/bin/bash
# POST-HOC, EXPLORATORY (session 26, 2026-10-07; DEVIATIONS): X1 combined gate (CUSUM) with the conformal threshold
# (N6, alpha 0.05) + DSG comparators, same TEST items and attacks as X1 (experiment PH-X1-conformal, branch
# exp/PH-X1-conformal = 93625f5 + config). Not a claim-rule input. cluster/lab_jobs.py, group x1-posthoc.
# Resumable: DONE jobs are skipped, a killed eval resumes from its 200-item chunk; chain 3 copies.
#SBATCH --job-name=dsg-x1-posthoc
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later9   # lab_jobs.py of session 20 (the X1 replica's runner)
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/lab_jobs.py --group x1-posthoc --budget-min "$DSG_BUDGET_MIN"
