#!/bin/bash
# D1-full: full-parameter UNDO distillation (3 noise levels) + harness TEST eval. Resumable (trainer state every 100 steps; finished students and DONE runs are skipped).
#SBATCH --job-name=dsg-d1-full
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
# Optional hard end (GPU promised to someone else): submit with --export=ALL,DSG_END_BY=2026-10-03T09:00
# (plus --deadline/--time-min). The run is stopped 5 min before it; checkpoints make the loss small.
RUN=()
if [ -n "${DSG_END_BY:-}" ]; then
    left=$(( $(date -d "$DSG_END_BY" +%s) - $(date +%s) - 300 ))
    [ "$left" -gt 900 ] || { echo "=== DSG_END_BY $DSG_END_BY: less than 20 min left, not starting"; exit 0; }
    echo "=== DSG_END_BY $DSG_END_BY: stopping after ${left}s at the latest"
    RUN=(timeout --signal=TERM --kill-after=120 "$left")
fi
"${RUN[@]}" python cluster/d1_full.py
