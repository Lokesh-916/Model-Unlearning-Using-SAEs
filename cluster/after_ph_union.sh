#!/bin/bash
# Run on the LAB PC (background). Session 28: when the POST-HOC exploratory ph-union chain (validate -> x1suite-models ->
# 4 x ph-union) has left the gpuws queue, without a manual step:
#  1. labjobs-ph-union status complete? else resubmit ph-union.sbatch behind our chain (resumable; at most MAX_RETRY);
#  2. server.sh fetch ph-union, verify, cleanup ph-union --yes (cleanup refuses anything not fetched and verified);
#  3. results_digest (post-hoc section) and scripts/paper_update.sh: the "exploratory follow-ups" paragraphs of the TMLR
#     and SRW papers switch from "pending" to the numbers (\resGpuPhUnionStatus / \resGpuPhTofucalStatus = done);
#  4. commits numbers.tex in the paper repo and numbers.tex + main.pdf in paper-srw (local commits, never pushed).
#   nohup cluster/after_ph_union.sh >> ~/projects/mechunlearn-project/dsg_results_cluster/after_ph_union.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"
EVERY="${EVERY:-600}"; MAX_RETRY="${MAX_RETRY:-2}"; retries=0
P="$HOME/projects/mechunlearn-project"
log() { echo "[after-ph-union] $(date -Is) $*"; }
rq() { ssh -n -o BatchMode=yes -o ConnectTimeout=20 gpuws "$@"; }
PY='~/dsg_cluster/env/mechunlearn2/bin/python'

log "waiting for our dsg-ph-union / dsg-x1suite-models jobs to leave the gpuws queue"
while true; do
    st=$(rq "busy=\$(squeue -h -u \"\$USER\" -o '%j' | grep -cE '^dsg-(ph-union|x1suite-models)'); \
             fin=\$($PY -c \"import json;print(int(json.load(open('\$HOME/dsg_cluster/results/jobs/labjobs-ph-union/status.json')).get('complete', False)))\" 2>/dev/null); \
             echo \$busy \${fin:-0}") || { log "ssh failed; retry"; sleep "$EVERY"; continue; }
    read -r busy fin <<< "$st"
    if [ "$busy" = "0" ] && [ "$fin" = "1" ]; then break; fi
    if [ "$busy" = "0" ]; then
        if [ "$retries" -ge "$MAX_RETRY" ]; then log "ph-union incomplete after $retries resubmits: stopping (runbook: check the job logs)"; exit 1; fi
        tail=$(cluster/chain_tail.sh)
        log "ph-union left the queue incomplete: resubmitting ph-union.sbatch ${tail:+after $tail}"
        rq "cd ~/dsg_cluster/slurm && ./submit.sh ph-union.sbatch ${tail:+--dependency=afterany:$tail}" || { log "resubmit failed"; exit 1; }
        retries=$((retries + 1))
    else
        log "our ph-union jobs queued: $busy"
    fi
    sleep "$EVERY"
done
log "ph-union complete: fetch / verify / cleanup"
cluster/server.sh fetch ph-union && cluster/server.sh verify ph-union && cluster/server.sh cleanup ph-union --yes \
    || { log "fetch/verify/cleanup failed: stopping before the paper update"; exit 1; }
source scripts/env.sh
CUDA_VISIBLE_DEVICES= "$DSGX_PY" -m dsgx.analysis.results_digest | tail -n 1
scripts/paper_update.sh || { log "paper_update failed"; exit 1; }
git -C "$P/paper" add numbers.tex && git -C "$P/paper" commit -q -m "numbers.tex regenerated: post-hoc exploratory PH-union / PH-tofucal results (gpuws)" \
    && log "paper: numbers.tex committed (not pushed)"
git -C "$P/paper-srw" add numbers.tex main.pdf && git -C "$P/paper-srw" commit -q -m "numbers.tex re-synced: post-hoc exploratory PH-union / PH-tofucal results" \
    && log "paper-srw: committed"
log "done"
