#!/bin/bash
# Run on gpuws. Delete a job's large inputs after confirming its results were fetched (rule 5).
#   cleanup.sh <job> [--dry-run]
# Requires results/.fetched/<job> (written by fetch_results.sh) and re-checks that no result
# file changed since the fetch. Never touches env/, code/, slurm/, logs/ or results/.
set -euo pipefail
DSGC="$HOME/dsg_cluster"
JOB="${1:?usage: cleanup.sh <job> [--dry-run]}"; DRY="${2:-}"
source "$DSGC/slurm/jobs/$JOB.conf"
marker="$DSGC/results/.fetched/$JOB"
[ -f "$marker" ] || { echo "REFUSING: $marker missing; run fetch_results.sh $JOB on the lab PC first"; exit 1; }
cd "$DSGC"
paths="$RESULT_PATHS ${FETCH_EXTRA:-}"
lst='find . -type f ! -name ".lock" -print0 | sort -z | xargs -0 -r sha256sum'
listing=""
for p in $paths; do
    [ -e "$p" ] || continue
    listing+="== $p"$'\n'"$(cd "$DSGC/$p" && bash -c "$lst")"$'\n'
done
h=$(echo "$listing" | sha256sum | cut -c1-16)
[ "$h" = "$(cut -d' ' -f1 "$marker")" ] || { echo "REFUSING: results changed since fetch ($h vs marker); fetch again"; exit 1; }
echo "results verified as fetched: $(cat "$marker")"
for p in $LARGE_INPUTS; do
    case "$p" in ""|/*|*..*|env|env/*|code|code/*|slurm*|logs*|results*) echo "skip unsafe path: $p"; continue;; esac
    [ -e "$p" ] || { echo "already gone: $p"; continue; }
    sz=$(du -sh "$p" | cut -f1)
    if [ "$DRY" = "--dry-run" ]; then echo "would delete $p ($sz)"; continue; fi
    rm -rf -- "$DSGC/$p"
    printf -- '- %s | `rm -rf ~/dsg_cluster/%s` (%s) | cleanup.sh %s\n' "$(date '+%F %T')" "$p" "$sz" "$JOB" >> "$DSGC/COMMAND_LOG.md"
    echo "deleted $p ($sz)"
done
echo "kept: ${KEEP:-env code}"
du -sh "$DSGC"; df -h "$HOME" | tail -1
