#!/bin/bash
# Q2 attribution graphs (circuit-tracer 0.5.0 in the overlay venv env/q2, built offline from wheels/q2 on first use). Finished items are skipped on restart.
#SBATCH --job-name=dsg-q2-graphs
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
Q2="$DSGC/env/q2"
if [ ! -x "$Q2/bin/python" ]; then
    echo "=== creating overlay venv $Q2 (offline, from $DSGC/wheels/q2)"
    "$DSGC/env/mechunlearn2/bin/python" -m venv --system-site-packages "$Q2"
    "$Q2/bin/pip" install --no-index --find-links "$DSGC/wheels/q2" -r "$DSGC/wheels/q2/requirements-q2.txt"
fi
"$Q2/bin/python" cluster/q2_graphs.py
