#!/bin/bash
# POST-HOC, EXPLORATORY (session 28; DEVIATIONS 2026-10-07): PH-tofucal (TOFU-retain-DEV calibrated StreamGuard on TOFU v3)
# then PH-union (DSG rho OR CUSUM, conformal alpha 0.05: X1 MCQ set, A3 benign open-ended, B6, TOFU v3, paired), DSG
# comparators in the same experiments. Branch exp/PH-union via cluster/lab_jobs.py, group ph-union. Not claim inputs.
# Resumable: DONE jobs are skipped, a killed eval / generation loop resumes from its item chunk; chain 4 copies.
#SBATCH --job-name=dsg-ph-union
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later11   # lab_jobs.py runner (session 27)
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/lab_jobs.py --group ph-union --budget-min "$DSG_BUDGET_MIN"
