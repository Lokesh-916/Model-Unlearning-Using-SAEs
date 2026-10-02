#!/bin/bash
# RMU v2 step 2: TEST eval (bs=1, both views) of the selected RMU-v2 vs base, DSG-paper, third-party RMU
# (+ paired vs RMU v1 from its gpuws TEST run). DONE harness runs are skipped on restart.
#SBATCH --job-name=dsg-rmu-v2-eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
# Code: the snapshot code-rmu-v2/ (stage_code_snapshot.sh), so code/ was never changed under a running chain.
CODE="$DSGC/code-rmu-v2"; [ -d "$CODE" ] || CODE="$DSGC/code"
cd "$CODE"; export PYTHONPATH="$CODE" DSG_WORKTREES="$CODE"
echo "=== code dir: $CODE $(cat CODE_COMMIT.json 2>/dev/null)"
python cluster/rmu_v2_eval.py
