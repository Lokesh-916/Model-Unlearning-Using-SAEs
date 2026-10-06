#!/bin/bash
# MT-Bench (BM3) for the final method: adds mode cusum (X1 combined gate) to the existing base / DSG / window-w16 answers and
# judgments on gpuws; fixed judge google/gemma-2-9b-it (same family: labelled), FastChat prompts, greedy. Resumable; chain 2.
#SBATCH --job-name=dsg-mtbench-x1
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
export DSG_CODE_SNAPSHOT=code-later10   # session 22: stream token_fn + mtbench cusum mode (stage_code_snapshot.sh later10)
source "$HOME/dsg_cluster/slurm/later.sh"
python cluster/mtbench_open.py --budget-min "$DSG_BUDGET_MIN"   # all MODES; finished base/dsg/window rows are skipped
