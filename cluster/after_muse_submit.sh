#!/bin/bash
# Run on the LAB PC (background). Keeps gpuws busy after MUSE-v2 without a manual step (session 24):
#  1. waits until none of our dsg-muse* jobs is queued;
#  2. MUSE incomplete (summary.json lacks news + books)? resubmits muse-v2.sbatch behind our chain (resumable; at most
#     MAX_MUSE_RETRY times) and keeps waiting;
#  3. runbook 5.8/5.6: fetch + verify + cleanup x1 (if its status says complete), a7-scaled (if DONE) and muse, which frees
#     the disk the judge needs (cleanup refuses anything not fetched and verified);
#  4. cluster/server.sh stage mtbench-x1 (refuses if free would drop below 50 GB or ours reach 100 GB: nothing submitted),
#     then server.sh submit mtbench-x1 (validate -> 2 x mtbench-x1, afterok) behind the end of our chain;
#  5. starts cluster/q2_rerun_when_ready.sh (Q2 re-run, held chain + release watcher) unless it is already running.
#   nohup cluster/after_muse_submit.sh >> ~/projects/mechunlearn-project/dsg_results_cluster/after_muse_watch.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
EVERY="${EVERY:-600}"; MAX_MUSE_RETRY="${MAX_MUSE_RETRY:-2}"; retries=0
LOGD="$HOME/projects/mechunlearn-project/dsg_results_cluster"
log() { echo "[after-muse] $(date -Is) $*"; }
rq() { ssh -n -o BatchMode=yes -o ConnectTimeout=20 gpuws "$@"; }
PY='~/dsg_cluster/env/mechunlearn2/bin/python'

log "waiting for our dsg-muse* jobs to leave the gpuws queue"
while true; do
    st=$(rq "busy=\$(squeue -h -u \"\$USER\" -o '%j' | grep -c '^dsg-muse'); \
             fin=\$($PY -c \"import json;d=json.load(open('\$HOME/dsg_cluster/results/jobs/muse/summary.json'));print(int({'news','books'} <= set(d.get('finished', []))))\" 2>/dev/null); \
             echo \$busy \${fin:-0}") || { log "ssh failed; retry"; sleep "$EVERY"; continue; }
    read -r busy fin <<< "$st"
    if [ "$busy" = "0" ] && [ "$fin" = "1" ]; then break; fi
    if [ "$busy" = "0" ]; then
        if [ "$retries" -ge "$MAX_MUSE_RETRY" ]; then log "MUSE incomplete after $retries resubmits: stopping (nothing staged)"; exit 1; fi
        tail=$(cluster/chain_tail.sh)
        log "MUSE left the queue incomplete: resubmitting muse-v2.sbatch ${tail:+after $tail}"
        rq "cd ~/dsg_cluster/slurm && ./submit.sh muse-v2.sbatch ${tail:+--dependency=afterany:$tail}" || { log "resubmit failed"; exit 1; }
        retries=$((retries + 1))
    else
        log "our muse jobs queued: $busy"
    fi
    sleep "$EVERY"
done
log "MUSE done (news + books): fetch / verify / cleanup"

x1c=$(rq "$PY -c \"import json;print(int(json.load(open('\$HOME/dsg_cluster/results/jobs/labjobs-x1/status.json')).get('complete', False)))\"" 2>/dev/null)
if [ "$x1c" = "1" ]; then
    cluster/server.sh fetch x1 && cluster/server.sh verify x1 && cluster/server.sh cleanup x1 --yes || log "x1 fetch/cleanup failed (continuing)"
else
    log "x1 not complete on gpuws: not fetched/cleaned (runbook 5.8: resubmit x1.sbatch)"
fi
a7=$(rq "$PY -c \"import json;print(int(json.load(open('\$HOME/dsg_cluster/results/jobs/labjobs-a7-scaled/status.json')).get('complete', False)))\"" 2>/dev/null)
if [ "$a7" = "1" ]; then
    cluster/server.sh fetch a7-scaled && cluster/server.sh verify a7-scaled && cluster/server.sh cleanup a7-scaled --yes || log "a7-scaled fetch/cleanup failed (continuing)"
else
    log "a7-scaled not complete on gpuws: not fetched/cleaned"
fi
cluster/server.sh fetch muse && cluster/server.sh verify muse && cluster/server.sh cleanup muse --yes \
    || log "muse fetch/cleanup failed (continuing; stage below checks the disk itself)"

log "staging mtbench-x1"
cluster/server.sh plan mtbench-x1
n=0
until cluster/server.sh stage mtbench-x1; do  # refused (disk limits) or failed: retry; nothing is submitted meanwhile
    n=$((n + 1)); [ "$n" -ge "${STAGE_TRIES:-12}" ] && { log "stage refused/failed $n times: MT-Bench not submitted"; exit 1; }
    log "stage refused/failed (try $n); retry in $EVERY s"; sleep "$EVERY"
done
tail=$(cluster/chain_tail.sh)
if [ -n "$tail" ]; then cluster/server.sh submit mtbench-x1 --after-any "$tail"; else cluster/server.sh submit mtbench-x1; fi \
    || { log "submit failed"; exit 1; }
log "MT-Bench (cusum) submitted"

if pgrep -f q2_rerun_when_ready.sh > /dev/null; then log "q2 watcher already running"
else nohup cluster/q2_rerun_when_ready.sh >> "$LOGD/q2_rerun_watch.log" 2>&1 & log "q2 watcher started"; fi
