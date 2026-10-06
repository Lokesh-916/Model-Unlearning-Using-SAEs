#!/bin/bash
# X1 TEST replication (session 20): the lab's 33 X1 TEST jobs with pinned exp/X1-combine code (cluster/lab_jobs.py,
# group x1). Resumable: DONE jobs are skipped, a killed eval resumes from its 200-item chunk; chain 4 copies.
#SBATCH --job-name=dsg-x1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later9   # session 20: lab_jobs.py --offline-sae-shapes, stage_lab_jobs LABJOBS_PIN
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/lab_jobs.py --group x1 --budget-min "$DSG_BUDGET_MIN"
