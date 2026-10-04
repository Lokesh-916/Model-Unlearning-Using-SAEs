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
FIND_EXCL=""
for x in ${FETCH_EXCLUDE:-}; do FIND_EXCL+=" ! -path \"*/$x\""; done
lst="find . -type f ! -name \".lock\"$FIND_EXCL -print0 | sort -z | xargs -0 -r sha256sum"
listing=""
for p in $paths; do
    [ -e "$p" ] || continue
    listing+="== $p"$'\n'"$(cd "$DSGC/$p" && bash -c "$lst")"$'\n'
done
h=$(echo "$listing" | sha256sum | cut -c1-16)
[ "$h" = "$(cut -d' ' -f1 "$marker")" ] || { echo "REFUSING: results changed since fetch ($h vs marker); fetch again"; exit 1; }
echo "results verified as fetched: $(cat "$marker")"
# Files excluded from the fetch (FETCH_EXCLUDE, e.g. optimizer state) are deleted only inside finished runs
# (the run dir, i.e. the path minus the pattern, has a DONE marker).
for x in ${FETCH_EXCLUDE:-}; do
    for p in $paths; do
        [ -d "$p" ] || continue
        while IFS= read -r f; do
            rd="${f%/$x}"; [ -f "$rd/DONE" ] || { echo "kept $f (run not DONE)"; continue; }
            sz=$(du -sh "$f" | cut -f1)
            if [ "$DRY" = "--dry-run" ]; then echo "would delete $f ($sz)"; continue; fi
            rm -f -- "$DSGC/$f"
            printf -- '- %s | `rm -f ~/dsg_cluster/%s` (%s, not fetched, run DONE) | cleanup.sh %s\n' "$(date '+%F %T')" "$f" "$sz" "$JOB" >> "$DSGC/COMMAND_LOG.md"
            echo "deleted $f ($sz)"
        done < <(find "$p" -type f -path "*/$x")
    done
done
[ "${PRUNE_ONLY:-}" = 1 ] && { echo "PRUNE_ONLY: large inputs kept"; du -sh "$DSGC"; exit 0; }
# Inputs shared by several jobs: never delete them while one of our dsg jobs is still queued or running
# (e.g. rmu-v2 queued behind a6-full needs the corpus that a6-full's cleanup would remove).
SHARED="private/corpora/bio-forget-corpus.jsonl"
busy=$(squeue -h -u "$USER" -o "%i %j" 2>/dev/null | awk '$2 ~ /^dsg-/' || true)
for p in $LARGE_INPUTS; do
    case "$p" in ""|/*|*..*|env|env/*|code|code/*|slurm*|logs*|results*) echo "skip unsafe path: $p"; continue;; esac
    [ -e "$p" ] || { echo "already gone: $p"; continue; }
    if [ -n "$busy" ] && [[ " $SHARED " == *" $p "* ]]; then echo "kept $p (shared input; queued/running: $(echo $busy))"; continue; fi
    sz=$(du -sh "$p" | cut -f1)
    if [ "$DRY" = "--dry-run" ]; then echo "would delete $p ($sz)"; continue; fi
    rm -rf -- "$DSGC/$p"
    printf -- '- %s | `rm -rf ~/dsg_cluster/%s` (%s) | cleanup.sh %s\n' "$(date '+%F %T')" "$p" "$sz" "$JOB" >> "$DSGC/COMMAND_LOG.md"
    echo "deleted $p ($sz)"
done
echo "kept: ${KEEP:-env code}"
du -sh "$DSGC"; df -h "$HOME" | tail -1
