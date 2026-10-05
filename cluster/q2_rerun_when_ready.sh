#!/bin/bash
# Run on the LAB PC (background). Session 13 re-run of Q2 on the re-trained TOFU model: waits until the gpuws
# tofu-full re-run has written metrics.json with train_version 2 AND none of our dsg-tofu-full* / dsg-figs-hl /
# dsg-muse* jobs is still queued (the transcoders, 7.9 GB, would otherwise break the 50 GB free-disk floor
# at the MUSE peak). Then stages q2-graphs (server.sh stage refuses if a disk limit would break; nothing is
# submitted then) and submits a held chain (validate -> q2-graphs-v2) behind our last queued job, then starts
# release_when_free.sh if it is not running.
#   nohup cluster/q2_rerun_when_ready.sh >> ~/projects/mechunlearn-project/dsg_results_cluster/q2_rerun_watch.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
EVERY="${EVERY:-900}"
echo "[q2-watch] $(date -Is) waiting for the tofu-full re-run (train_version 2) and the end of the muse re-run"
while true; do
    st=$(ssh -n -o BatchMode=yes -o ConnectTimeout=20 gpuws '
        tv=$(~/dsg_cluster/env/mechunlearn2/bin/python -c "import json;print(json.load(open(\"$HOME/dsg_cluster/results/runs/A2-tofu-full/tofu-metrics/metrics.json\")).get(\"train_version\"))" 2>/dev/null)
        busy=$(squeue -h -u "$USER" -o "%j" | grep -cE "^dsg-(tofu-full|figs-hl|muse)")
        echo "${tv:-none} $busy"' 2>/dev/null) || { echo "[q2-watch] $(date -Is) ssh failed; retry"; sleep "$EVERY"; continue; }
    read -r tv busy <<< "$st"
    if [ "$tv" = "2" ] && [ "$busy" = "0" ]; then break; fi
    if [ "$tv" != "2" ] && [ "$busy" = "0" ]; then
        echo "[q2-watch] $(date -Is) no tofu-full/muse job queued but no train_version-2 TOFU metrics: tofu-full failed? exiting"; exit 1
    fi
    echo "[q2-watch] $(date -Is) train_version=$tv, our tofu/figs-hl/muse jobs queued: $busy"
    sleep "$EVERY"
done
echo "[q2-watch] $(date -Is) tofu-full v2 done, muse left the queue: staging q2-graphs"
cluster/server.sh stage q2-graphs || { echo "[q2-watch] stage refused/failed: chain not submitted"; exit 1; }
last=$(ssh -n -o BatchMode=yes gpuws 'squeue -h -u suraj -o "%i"' | sort -n | tail -1)
cluster/submit_chain.sh ${last:+--after-any $last} --nice 10000 --hold q2-graphs-v2.sbatch \
    || { echo "[q2-watch] submit failed"; exit 1; }
pgrep -f release_when_free.sh > /dev/null \
    || nohup cluster/release_when_free.sh >> "$HOME/projects/mechunlearn-project/dsg_results_cluster/release_watch.log" 2>&1 &
echo "[q2-watch] $(date -Is) submitted held q2 chain; release watcher running"
