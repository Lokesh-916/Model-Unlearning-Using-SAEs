#!/bin/bash
# Submit one dsg job on gpuws after checking disk and showing the queue (rules 4, 5, 8).
#   submit.sh <job.sbatch> [extra sbatch args, e.g. --dependency=afterok:123]
# Refuses if one of our dsg-* jobs is already queued/running, unless the new job is chained
# with --dependency=afterok:<id> or (to start even if that job fails) --dependency=afterany:<id>.
set -euo pipefail
DSGC="$HOME/dsg_cluster"
[ $# -ge 1 ] || { echo "usage: submit.sh <job.sbatch> [sbatch args]"; exit 2; }
script="$1"; shift
[ -f "$script" ] || { echo "no such file: $script"; exit 2; }

avail_gb=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
ours_gb=$(du -sBG "$DSGC" 2>/dev/null | cut -f1 | tr -dc 0-9)
echo "disk: ${avail_gb} GB free on /, ours ${ours_gb} GB (limits: free >= 50, ours < 100)"
if [ "$avail_gb" -lt 50 ] || [ "$ours_gb" -ge 100 ]; then
    echo "REFUSING: disk limits violated. Clean up first (cleanup.sh) and tell the user."; exit 1
fi

echo "--- squeue (all users)"
squeue -o "%.7i %.9u %.20j %.2t %.10M %.12l %.6C %b %R"
mine=$(squeue -h -u "$USER" -o "%i %j" | awk '$2 ~ /^dsg-/' || true)
if [ -n "$mine" ] && ! printf '%s\n' "$@" | grep -qE -- '--dependency=after(ok|any):[0-9]+'; then
    echo "REFUSING: one of our dsg jobs is already queued/running:"; echo "$mine"
    echo "Chain the new job with --dependency=afterok:<id> (or afterany:<id>) or wait."; exit 1
fi

out=$(sbatch --parsable "$@" "$script")
jid=${out%%;*}
printf -- '- %s | `sbatch %s %s` -> job %s | submit (via submit.sh)\n' "$(date '+%F %T')" "$*" "$script" "$jid" >> "$DSGC/COMMAND_LOG.md"
echo "submitted $script as job $jid"
squeue -j "$jid" -o "%.7i %.20j %.2t %.10M %.12l %R"
