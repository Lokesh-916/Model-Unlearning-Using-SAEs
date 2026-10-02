#!/bin/bash
# Run on the LAB PC. Copy a finished job's run dirs from gpuws to dsg_results_cluster, verify
# them by sha256, then write a "fetched" marker on the server (cleanup.sh requires it).
#   fetch_results.sh <job>        (job = name of slurm/jobs/<job>.conf)
# Never copies private/ (hazardous text lives only there).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
JOB="${1:?usage: fetch_results.sh <job>}"
source "$HERE/jobs/$JOB.conf"            # RESULT_PATHS, LARGE_INPUTS, FETCH_EXTRA
DEST="${DSG_RESULTS_CLUSTER:-$HOME/projects/mechunlearn-project/dsg_results_cluster}"
RS="rsync --bwlimit=50000 --partial -a"
log() { ssh -o BatchMode=yes gpuws "printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" \"$1\" \"$2\" >> ~/dsg_cluster/COMMAND_LOG.md"; }

avail=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
echo "lab PC free: ${avail} GB (must stay > 60)"
[ "$avail" -gt 65 ] || { echo "REFUSING: lab PC disk too full"; exit 1; }

running=$(ssh gpuws "squeue -h -u \$USER -o '%j %T'" | grep -E "^dsg-$JOB" || true)
[ -z "$running" ] || { echo "job still queued/running:"; echo "$running"; echo "fetch after it finishes"; exit 1; }

mkdir -p "$DEST/logs"
paths="$RESULT_PATHS ${FETCH_EXTRA:-}"
log "rsync (pull from lab PC) ${paths} logs/dsg-$JOB*" "fetch_results.sh $JOB"
for p in $paths; do
    if ssh gpuws "test -e ~/dsg_cluster/$p"; then
        mkdir -p "$DEST/$(dirname "$p")"
        $RS "gpuws:dsg_cluster/$p" "$DEST/$(dirname "$p")/"
    else
        echo "warning: $p does not exist on the server (skipped)"
    fi
done
$RS --include "dsg-$JOB*" --exclude '*' "gpuws:dsg_cluster/logs/" "$DEST/logs/"

# Verify: same file list and sha256 on both sides.
listing='for p in '"$paths"'; do [ -e "$p" ] && find "$p" -type f ! -name ".lock" -print0; done | sort -z | xargs -0 -r sha256sum'
remote=$(ssh gpuws "cd ~/dsg_cluster && $listing")
local=$(cd "$DEST" && bash -c "$listing")
if [ "$remote" != "$local" ]; then
    echo "VERIFY FAILED: local and remote file lists differ"; diff <(echo "$remote") <(echo "$local") | head; exit 1
fi
n=$(echo "$remote" | grep -c . || true)
h=$(echo "$remote" | sha256sum | cut -c1-16)
ssh gpuws "mkdir -p ~/dsg_cluster/results/.fetched && echo '$h $n files $(date -Is) -> $DEST' > ~/dsg_cluster/results/.fetched/$JOB"
log "write results/.fetched/$JOB ($n files, listing $h)" "fetch verified"
echo "fetched and verified $n files (listing sha $h) into $DEST"
