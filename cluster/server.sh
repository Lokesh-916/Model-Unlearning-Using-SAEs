#!/bin/bash
# One lab-PC entry point for every gpuws operation (CLAUDE.md rules 1-10). Run from this repo.
#
#   cluster/server.sh status                 # our jobs, GPU, disk (ours / free)
#   cluster/server.sh sync                   # sync code + slurm tooling, CPU import check on the login node
#   cluster/server.sh plan  <job>            # what <job> stages, its size, disk after staging (no changes)
#   cluster/server.sh stage <job>            # copy <job>'s inputs (refuses if limits would be broken)
#   cluster/server.sh submit <job>           # validate.sbatch -> <job> sbatch chain (afterok)
#   cluster/server.sh check <job>            # queue state + last log lines (metrics only) of <job>
#   cluster/server.sh fetch <job>            # pull results (+ validate) to dsg_results_cluster, sha256-verified
#   cluster/server.sh verify <job>           # local file count + summary headline of fetched results
#   cluster/server.sh cleanup <job> [--yes]  # delete <job>'s large inputs on the server (dry-run without --yes)
#   cluster/server.sh run <job>              # plan + stage + submit in one go (asks before staging)
#
# Jobs are described by cluster/slurm/jobs/<job>.conf:
#   STAGE         lines "LOCAL_SRC  REMOTE_DEST" (REMOTE_DEST relative to ~/dsg_cluster)
#   STAGE_SCRIPT  optional extra script run on the lab PC to stage (e.g. stage_rmu.sh)
#   CHAIN         sbatch files submitted in order after validate.sbatch, each afterok the previous
#   RESULT_PATHS / FETCH_EXTRA / LARGE_INPUTS / KEEP   used by fetch_results.sh and cleanup.sh
#   EST_DISK_GB / EST_HOURS                            shown by plan
# Testing without the server: DSG_FAKE_SERVER=/some/dir runs "remote" commands locally with HOME=that dir.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
CMD="${1:-help}"; JOB="${2:-}"
FAKE="${DSG_FAKE_SERVER:-}"
RS_OPTS=(--bwlimit=50000 --partial -a)
MIN_FREE_GB=50; MAX_OURS_GB=100; LAB_MIN_FREE_GB=60
DEST_LOCAL="${DSG_RESULTS_CLUSTER:-$HOME/projects/mechunlearn-project/dsg_results_cluster}"

remote() {  # run a shell command on gpuws (or in the fake server dir)
    # -n: never read stdin, or ssh swallows the rest of a `stage_lines | while read` loop (only line 1 staged)
    if [ -n "$FAKE" ]; then HOME="$FAKE" bash -c "$1" < /dev/null; else ssh -n -o BatchMode=yes gpuws "$1"; fi
}
push() {    # push SRC to ~/dsg_cluster/DEST
    local src="$1" dest="$2"
    if [ -n "$FAKE" ]; then mkdir -p "$FAKE/dsg_cluster/$dest"; rsync "${RS_OPTS[@]}" "$src" "$FAKE/dsg_cluster/$dest/";
    else remote "mkdir -p ~/dsg_cluster/$dest"; rsync "${RS_OPTS[@]}" "$src" "gpuws:dsg_cluster/$dest/" < /dev/null; fi
}
rlog() {    # append to the server command log (rule 8)
    remote "mkdir -p ~/dsg_cluster && printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" \"$1\" \"$2\" >> ~/dsg_cluster/COMMAND_LOG.md"
}
conf() {
    [ -n "$JOB" ] || { echo "usage: server.sh $CMD <job>"; ls "$HERE/slurm/jobs" | sed 's/\.conf$//;s/^/  job: /'; exit 2; }
    STAGE=""; STAGE_SCRIPT=""; CHAIN=""; EST_DISK_GB="?"; EST_HOURS="?"; RESULT_PATHS=""; FETCH_EXTRA=""; LARGE_INPUTS=""; KEEP=""
    # shellcheck disable=SC1090
    source "$HERE/slurm/jobs/$JOB.conf"
}
stage_lines() { printf '%s\n' "$STAGE" | sed 's/#.*//' | awk 'NF==2'; }
stage_gb() {    # total size of the STAGE sources, GB (rounded up)
    local tot=0 s
    while read -r src _; do
        [ -e "$src" ] || { echo "MISSING on lab PC: $src" >&2; continue; }
        s=$(du -sbL "$src" | cut -f1); tot=$((tot + s))
    done < <(stage_lines)
    echo $(( (tot + 1073741823) / 1073741824 ))
}
peak_gb() {     # max(staged size, EST_DISK_GB): the job's peak footprint on the server
    local e="${EST_DISK_GB%.*}"; [[ "$e" =~ ^[0-9]+$ ]] || e=0
    [ "$1" -gt "$e" ] && echo "$1" || echo "$e"
}
disk() {        # prints "free_gb ours_gb"
    remote 'echo "$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9) $(du -sBG "$HOME/dsg_cluster" 2>/dev/null | cut -f1 | tr -dc 0-9)"'
}

case "$CMD" in
status)
    remote 'echo "--- our jobs"; (squeue -u "$USER" -o "%.7i %.22j %.2t %.10M %.12l %R" 2>/dev/null || echo "(no squeue)");
            echo "--- GPU"; (nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)");
            echo "--- disk"; df -h "$HOME" | tail -1; du -sh "$HOME/dsg_cluster" 2>/dev/null || true'
    ;;
sync)
    [ -n "$FAKE" ] && { echo "sync is not simulated"; exit 0; }
    "$HERE/sync_code.sh"
    remote 'source ~/dsg_cluster/env.sh && cd ~/dsg_cluster/code && CUDA_VISIBLE_DEVICES= python cluster/import_check.py | tail -3'
    rlog "CUDA_VISIBLE_DEVICES= python cluster/import_check.py" "server.sh sync: CPU import check"
    ;;
plan)
    conf
    read -r free ours < <(disk)
    need=$(stage_gb); peak=$(peak_gb "$need")
    echo "job $JOB: stage ${need} GB, est. peak incl. outputs ${peak} GB, runtime ~${EST_HOURS} h"
    echo "server now: ${free} GB free on /, ours ${ours} GB"
    echo "at peak: free $((free - peak)) GB (must stay >= $MIN_FREE_GB), ours $((ours + peak)) GB (must stay < $MAX_OURS_GB)"
    stage_lines | while read -r src dest; do printf '  %-8s %s -> %s\n' "$(du -shL "$src" 2>/dev/null | cut -f1)" "$src" "$dest"; done
    [ -n "$STAGE_SCRIPT" ] && echo "  + $STAGE_SCRIPT"
    echo "chain: validate.sbatch ${CHAIN}"
    ;;
stage)
    conf
    read -r free ours < <(disk)
    need=$(stage_gb); peak=$(peak_gb "$need")
    if [ $((free - peak)) -lt $MIN_FREE_GB ] || [ $((ours + peak)) -ge $MAX_OURS_GB ]; then
        echo "REFUSING to stage $JOB: peak ${peak} GB; free ${free} -> $((free - peak)) (min $MIN_FREE_GB), ours ${ours} -> $((ours + peak)) (max $MAX_OURS_GB)."
        echo "Clean up a finished job first (server.sh cleanup <job> --yes) and tell the team."; exit 1
    fi
    lab=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
    [ "$lab" -gt "$LAB_MIN_FREE_GB" ] || { echo "REFUSING: lab PC free ${lab} GB <= $LAB_MIN_FREE_GB"; exit 1; }
    rlog "rsync stage $JOB (${need} GB)" "server.sh stage $JOB"
    stage_lines | while read -r src dest; do
        [ -e "$src" ] || { echo "skip (not on the lab PC): $src"; continue; }
        echo "stage $src -> $dest"; push "$src" "$dest"
    done
    [ -n "$STAGE_SCRIPT" ] && { [ -n "$FAKE" ] && echo "(fake) skip $STAGE_SCRIPT" || bash "$REPO/$STAGE_SCRIPT"; }
    remote "mkdir -p ~/dsg_cluster/results/.staged && date -Is > ~/dsg_cluster/results/.staged/$JOB"
    read -r free ours < <(disk); echo "staged $JOB; server free ${free} GB, ours ${ours} GB"
    ;;
submit)
    conf
    [ -n "$CHAIN" ] || { echo "no CHAIN in $JOB.conf"; exit 2; }
    remote "test -f ~/dsg_cluster/results/.staged/$JOB" || [ -z "$STAGE" ] || { echo "REFUSING: $JOB not staged (server.sh stage $JOB)"; exit 1; }
    if [ -n "$FAKE" ]; then echo "(fake) would submit: validate.sbatch -> $CHAIN"; exit 0; fi
    prev=$(remote "cd ~/dsg_cluster/slurm && ./submit.sh validate.sbatch" | tee /dev/stderr | sed -n 's/^submitted .* as job \([0-9]*\)$/\1/p')
    [ -n "$prev" ] || { echo "validate submit failed"; exit 1; }
    for sb in $CHAIN; do
        prev=$(remote "cd ~/dsg_cluster/slurm && ./submit.sh $sb --dependency=afterok:$prev" | tee /dev/stderr | sed -n 's/^submitted .* as job \([0-9]*\)$/\1/p')
        [ -n "$prev" ] || { echo "submit of $sb failed"; exit 1; }
    done
    echo "chain submitted; last job $prev. Check: cluster/server.sh check $JOB"
    ;;
check)
    conf
    remote "squeue -u \$USER -o '%.7i %.22j %.2t %.10M %.12l %R' 2>/dev/null | grep -E 'JOBID|dsg-' || true;
            for f in \$(ls -t ~/dsg_cluster/logs/dsg-* 2>/dev/null | head -4); do echo \"--- \$f\"; grep -E '^(===|\[precheck\]|VALIDATE|SANITY|\[.*\] (done|cfg|step|selected|eval|summary|resumed)|Traceback|[A-Za-z]+Error)' \$f | tail -6; done"
    ;;
fetch)
    conf
    [ -n "$FAKE" ] && { echo "(fake) would run fetch_results.sh validate and $JOB"; exit 0; }
    "$HERE/slurm/fetch_results.sh" validate || true
    "$HERE/slurm/fetch_results.sh" "$JOB"
    ;;
verify)
    conf
    for p in $RESULT_PATHS; do
        d="$DEST_LOCAL/${p#results/}"
        printf '  %-60s %s files\n' "$d" "$( [ -d "$d" ] && find "$d" -type f | wc -l || echo MISSING)"
    done
    for s in "$DEST_LOCAL"/jobs/"$JOB"/summary.json; do [ -f "$s" ] && python3 -c "import json,sys;d=json.load(open(sys.argv[1]));print('  headline:', json.dumps(d.get('headline', {k: d[k] for k in list(d)[:4]}))[:400])" "$s"; done
    ;;
cleanup)
    conf
    if [ "${3:-}" = "--yes" ]; then remote "~/dsg_cluster/slurm/cleanup.sh $JOB"; else remote "~/dsg_cluster/slurm/cleanup.sh $JOB --dry-run"; echo "(dry run; add --yes to delete)"; fi
    ;;
run)
    "$0" plan "$JOB"
    read -r -p "stage and submit $JOB now? [y/N] " ok; [ "$ok" = "y" ] || exit 0
    "$0" stage "$JOB" && "$0" submit "$JOB"
    ;;
*)
    sed -n '2,22p' "$0"; exit 2 ;;
esac
