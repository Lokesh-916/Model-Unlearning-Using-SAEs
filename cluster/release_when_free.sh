#!/bin/bash
# Run on the LAB PC (background). Releases OUR held gpuws jobs (user suraj, reason JobHeldUser) once no other
# user has a job in the Slurm queue (running or pending) on two consecutive checks. Never touches other users'
# jobs; only `scontrol release` on our own job ids. Logs to the server COMMAND_LOG (rule 8) and to stdout.
#   nohup cluster/release_when_free.sh >> ~/projects/mechunlearn-project/dsg_results_cluster/release_watch.log 2>&1 &
#   EVERY=600 (seconds between checks), CLEAR_CHECKS=2 (consecutive clear checks before releasing)
#   WAIT_USER=<name>: wait only until that user has 0 jobs (others are ignored), e.g.
#   WAIT_USER=anish EVERY=120 CLEAR_CHECKS=1 nohup cluster/release_when_free.sh >> ~/release_after_anish.log 2>&1 &
set -uo pipefail
EVERY="${EVERY:-600}"; CLEAR_CHECKS="${CLEAR_CHECKS:-2}"; WAIT_USER="${WAIT_USER:-}"; ME=suraj
SSH=(timeout 120 ssh -n -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 -o ServerAliveCountMax=3 gpuws)
clear=0
echo "[release-watch] $(date -Is) start: every ${EVERY}s, release after ${CLEAR_CHECKS} clear checks, waiting for: ${WAIT_USER:-any other user}"
while true; do
    q=$("${SSH[@]}" 'squeue -h -o "%i %u %t %r"' 2>/dev/null) || {
        echo "[release-watch] $(date -Is) ssh failed; retry"; sleep "$EVERY"; continue; }
    [ -n "$q" ] || { echo "[release-watch] $(date -Is) empty squeue reply; retry"; sleep "$EVERY"; continue; }
    if [ -n "$WAIT_USER" ]; then others=$(awk -v u="$WAIT_USER" '$2 == u' <<< "$q")
    else others=$(awk -v me="$ME" '$2 != me' <<< "$q"); fi
    held=$(awk -v me="$ME" '$2 == me && $3 == "PD" && $4 == "JobHeldUser" {print $1}' <<< "$q" | sort -n | tr '\n' ' ')
    if [ -z "${held// /}" ]; then echo "[release-watch] $(date -Is) no held jobs of ours; exit"; exit 0; fi
    if [ -n "$others" ]; then
        clear=0; echo "[release-watch] $(date -Is) ${WAIT_USER:-other users}' jobs present ($(wc -l <<< "$others")); held: $held"
    else
        clear=$((clear + 1)); echo "[release-watch] $(date -Is) queue clear of ${WAIT_USER:-other users} ($clear/$CLEAR_CHECKS); held: $held"
        if [ "$clear" -ge "$CLEAR_CHECKS" ]; then
            "${SSH[@]}" "scontrol release $held && printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'scontrol release $held' 'release_when_free.sh: no job of ${WAIT_USER:-another user} in the queue' >> ~/dsg_cluster/COMMAND_LOG.md" \
                && { echo "[release-watch] $(date -Is) released: $held"; exit 0; }
            echo "[release-watch] $(date -Is) release failed; retry"
        fi
    fi
    sleep "$EVERY"
done
