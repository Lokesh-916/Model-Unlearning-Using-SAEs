#!/bin/bash
# Run on the LAB PC (background). When the lab jobs D1-train-sameref and D1-train-undo-a0.3 are DONE (their
# weights dsg_cache/models/D1/{sameref_a0.0,undo_a0.3} exist), stage the full a6-baked group (student + d1 + d2;
# DONE d2 jobs are skipped on the server) and submit a held chain behind our last queued gpuws job, then start
# release_when_free.sh. server.sh stage refuses if a disk limit would break (the chain is then not submitted).
#   nohup cluster/a6_baked_d1_when_ready.sh >> ~/projects/mechunlearn-project/dsg_results_cluster/a6_baked_d1_watch.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"
EVERY="${EVERY:-900}"
JOBS="D1-train-sameref D1-train-undo-a0.3"
echo "[a6-d1-watch] $(date -Is) waiting for: $JOBS"
while true; do
    st=$("$DSGX_PY" - $JOBS <<'PY'
import sys
from dsgx.queue import common as q
s = [q.load_state(j)["status"] for j in sys.argv[1:]]
print("FAILED" if "FAILED" in s else ("DONE" if all(x == "DONE" for x in s) else "WAIT"))
PY
)
    [ "$st" = "FAILED" ] && { echo "[a6-d1-watch] $(date -Is) a D1 train job FAILED: nothing staged"; exit 1; }
    [ "$st" = "DONE" ] && break
    sleep "$EVERY"
done
for d in sameref_a0.0 undo_a0.3; do
    [ -d "$DSG_CACHE/models/D1/$d" ] || { echo "[a6-d1-watch] missing $DSG_CACHE/models/D1/$d"; exit 1; }
done
echo "[a6-d1-watch] $(date -Is) D1 weights ready: staging a6-baked (all globs)"
cluster/server.sh stage a6-baked || { echo "[a6-d1-watch] stage refused/failed: chain not submitted"; exit 1; }
last=$(ssh -n -o BatchMode=yes gpuws 'squeue -h -u suraj -o "%i"' | sort -n | tail -1)
cluster/submit_chain.sh ${last:+--after-any $last} --nice 10000 --hold a6-baked.sbatch a6-baked.sbatch a6-baked.sbatch \
    || { echo "[a6-d1-watch] submit failed"; exit 1; }
nohup cluster/release_when_free.sh >> "$HOME/projects/mechunlearn-project/dsg_results_cluster/release_watch.log" 2>&1 &
echo "[a6-d1-watch] $(date -Is) submitted held chain; release watcher started"
