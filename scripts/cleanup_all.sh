#!/usr/bin/env bash
# End-of-project cleanup, lab PC + gpuws. DRY RUN by default: prints every check and every action, changes nothing.
#
#   scripts/cleanup_all.sh              # dry run (default; same as --dry-run)
#   scripts/cleanup_all.sh --yes        # act, but only if every check passes
#   options: --skip-server  --skip-lab  (do one side only; the checks for that side still run)
#
# Checks (all must pass before --yes acts):
#   1. lab queue: nothing running / waiting / failed / blocked
#   2. gpuws: none of our dsg-* jobs queued or running
#   3. every server result file is on the lab PC (rsync --dry-run --size-only per conf RESULT_PATHS, minus
#      FETCH_EXCLUDE; plus the dsg-* Slurm logs and results/_superseded) -> nothing left to fetch
#   4. git: every local branch of the main repo and the paper repo is pushed (no branch ahead of origin, no
#      branch missing on origin); every worktree has no uncommitted tracked changes; paper-srw / presentation /
#      release (no remote) are committed (warning only: they exist only on this PC, back them up)
# Actions with --yes, in this order:
#   lab:    stop our watchers (they would otherwise resubmit server jobs)
#   server: copy ~/dsg_cluster/COMMAND_LOG.md to the lab PC, rm -rf ~/dsg_cluster,
#           then (last, it ends our access) remove our key line from ~/.ssh/authorized_keys on gpuws
#   lab:    stop the dsg-* tmux sessions (baselines_DSG/scripts/tmux_down.sh), remove our crontab lines
#   lab:    list large caches as OPTIONAL deletions (sizes + commands; never deleted by this script)
# Never touched: other users' files, crontab lines that are not ours, $DSG_RESULTS, dsg_results_cluster, git repos.
set -uo pipefail
P="${P:-$HOME/projects/mechunlearn-project}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${DSG_RESULTS_CLUSTER:-$P/dsg_results_cluster}"
KEYPUB="${DSG_GPUWS_KEY:-$HOME/.ssh/id_ed25519_gpuws.pub}"
ACT=0; DO_SERVER=1; DO_LAB=1
for a in "$@"; do case "$a" in
    --yes) ACT=1;; --dry-run) ACT=0;; --skip-server) DO_SERVER=0;; --skip-lab) DO_LAB=0;;
    -h|--help) sed -n 2,23p "$0"; exit 0;;
    *) echo "unknown option $a"; exit 2;; esac; done
# shellcheck disable=SC1091
source "$REPO/scripts/env.sh" >/dev/null 2>&1 || true
PY="${DSGX_PY:-$HOME/miniconda3/envs/mechunlearn2/bin/python}"
fails=0; warns=0
ok()   { echo "  ok    $*"; }
bad()  { echo "  FAIL  $*"; fails=$((fails+1)); }
warn() { echo "  warn  $*"; warns=$((warns+1)); }
act()  { local d="$1"; shift; if [ $ACT = 1 ]; then echo "  run   $d"; "$@"; else echo "  would $d"; fi; }
g()    { ssh -n -o BatchMode=yes -o ConnectTimeout=15 gpuws "$@"; }

echo "== cleanup_all.sh ($([ $ACT = 1 ] && echo ACT || echo DRY RUN), $(date '+%F %T'))"

echo "-- 1. lab queue"
line=$(cd "$REPO" && "$PY" -m dsgx.queue.status --chat 2>/dev/null | grep -m1 -E '^done [0-9]+')
if [ -z "$line" ]; then bad "could not read the lab queue status"
else
    busy=$(echo "$line" | grep -oE '(running|waiting|failed|blocked) [0-9]+' | awk '$2>0' | tr '\n' ' ')
    [ -z "$busy" ] && ok "queue idle: $line" || bad "queue not finished: $busy"
fi

echo "-- 2. gpuws jobs"
SERVER_UP=0
if g true 2>/dev/null; then
    SERVER_UP=1
    if g 'test -d ~/dsg_cluster'; then
        q=$(g "squeue -h -u \$USER -o '%i %j %T'" | awk '$2 ~ /^dsg-/')
        [ -z "$q" ] && ok "no dsg-* job queued" || bad "dsg-* jobs still queued: $(echo "$q" | tr '\n' ';')"
    else ok "~/dsg_cluster already removed"; fi
else
    [ $DO_SERVER = 1 ] && bad "gpuws not reachable (ssh gpuws)" || warn "gpuws not reachable (skipped)"
fi

echo "-- 3. server results fetched to $DEST"
if [ $SERVER_UP = 1 ] && g 'test -d ~/dsg_cluster/results'; then
    dest_of() { case "$1" in results/*) echo "${1#results/}";; *) echo "checkpoints/${1#data/dsg_cache/models/}";; esac; }
    declare -A seen=()
    for c in "$REPO"/cluster/slurm/jobs/*.conf; do
        RESULT_PATHS=""; FETCH_EXTRA=""; FETCH_EXCLUDE=""
        # shellcheck disable=SC1090
        eval "$(grep -E '^(RESULT_PATHS|FETCH_EXTRA|FETCH_EXCLUDE)=' "$c")"
        excl=(--exclude .lock --exclude .fetched/); for x in $FETCH_EXCLUDE; do excl+=(--exclude "$x"); done
        for p in $RESULT_PATHS $FETCH_EXTRA; do
            [ -n "${seen[$p]:-}" ] && continue; seen[$p]=1
            g "test -e ~/dsg_cluster/$p" || continue
            n=$(rsync -a --dry-run --size-only --out-format='%n' "${excl[@]}" "gpuws:dsg_cluster/$p/" "$DEST/$(dest_of "$p")/" \
                < /dev/null 2>&1 | grep -vc '/$')
            [ "$n" = 0 ] && ok "$p" || bad "$p: $n file(s) not on the lab PC (cluster/server.sh fetch $(basename "$c" .conf))"
        done
    done
    for pair in "results/_superseded/:_superseded/" ; do
        s=${pair%%:*}; d=${pair#*:}
        if g "test -d ~/dsg_cluster/$s"; then
            n=$(rsync -a --dry-run --size-only --out-format='%n' "gpuws:dsg_cluster/$s" "$DEST/$d" < /dev/null 2>&1 | grep -vc '/$')
            [ "$n" = 0 ] && ok "$s" || bad "$s: $n file(s) not on the lab PC"
        fi
    done
    n=$(rsync -a --dry-run --size-only --out-format='%n' --include 'dsg-*' --exclude '*' "gpuws:dsg_cluster/logs/" "$DEST/logs/" \
        < /dev/null 2>&1 | grep -vc '/$')
    [ "$n" = 0 ] && ok "logs/dsg-*" || warn "logs: $n Slurm log(s) not on the lab PC (fetched with their job; copy with: rsync -a gpuws:dsg_cluster/logs/ $DEST/logs/)"
    nomark=$(g 'cd ~/dsg_cluster/results/jobs 2>/dev/null && for j in *; do [ -e ../.fetched/$j ] || echo -n "$j "; done')
    [ -z "$nomark" ] || echo "  info  job dirs without a .fetched marker (fine if their files are all above): $nomark"
elif [ $SERVER_UP = 1 ]; then ok "nothing on the server"; fi

echo "-- 4. git pushed / committed"
check_pushed() {   # repo: every local branch exists on origin and is not ahead of it
    local r="$1" b
    git -C "$r" fetch -q origin 2>/dev/null || { bad "$r: git fetch origin failed"; return; }
    while read -r b; do
        if ! git -C "$r" rev-parse -q --verify "refs/remotes/origin/$b" >/dev/null; then bad "$(basename "$r"): branch $b not on origin (git push -u origin $b)"
        else
            a=$(git -C "$r" rev-list --count "origin/$b..$b")
            [ "$a" = 0 ] || bad "$(basename "$r"): branch $b is $a commit(s) ahead of origin"
        fi
    done < <(git -C "$r" for-each-ref --format='%(refname:short)' refs/heads)
    while read -r w; do
        d=$(git -C "$w" status --porcelain --untracked-files=no | wc -l)
        [ "$d" = 0 ] || bad "$w: $d uncommitted tracked change(s)"
        u=$(git -C "$w" status --porcelain | grep -c '^??')
        [ "$u" = 0 ] || warn "$w: $u untracked file(s) (not in git)"
    done < <(git -C "$r" worktree list --porcelain | awk '/^worktree /{print $2}')
}
before=$fails
check_pushed "$P/baselines_DSG"
[ -d "$P/paper/.git" ] && check_pushed "$P/paper"
[ $fails = $before ] && ok "main repo (all worktrees, all branches) and paper repo pushed and clean"
for r in paper-srw presentation release; do
    [ -d "$P/$r/.git" ] || continue
    d=$(git -C "$P/$r" status --porcelain --untracked-files=no | wc -l)
    [ "$d" = 0 ] && warn "$r: committed, but it has no remote (only on this PC: back it up)" \
                 || bad "$r: $d uncommitted change(s)"
done

echo "== checks: $fails failure(s), $warns warning(s)"
if [ $ACT = 1 ] && [ $fails -gt 0 ]; then echo "REFUSING to act: fix the failures above (or use the dry run to see them)"; exit 1; fi

WATCHERS="after_ph_union.sh after_muse_submit.sh q2_rerun_when_ready.sh release_when_free.sh a6_baked_d1_when_ready.sh
          x1_enqueue_when_ready.sh n10_after_x1.sh dsgx.analysis.dashboard"
OURS_CRON='mechunlearn-project|dsg_|dsgx|dsg-'
srv_log_copy() {
    g "printf -- '- %s | rm -rf ~/dsg_cluster; remove our authorized_keys line | cleanup_all.sh (end of project)\n' \"\$(date '+%F %T')\" >> ~/dsg_cluster/COMMAND_LOG.md"
    mkdir -p "$DEST/server_logs"
    rsync -a gpuws:dsg_cluster/COMMAND_LOG.md gpuws:dsg_cluster/CLUSTER_SETUP_REPORT.md "$DEST/server_logs/" < /dev/null \
        && echo "  copied COMMAND_LOG.md, CLUSTER_SETUP_REPORT.md -> $DEST/server_logs/"
}
srv_rm() { g 'rm -rf "$HOME/dsg_cluster"' && g 'test ! -e "$HOME/dsg_cluster"' && echo "  removed ~/dsg_cluster"; }
srv_key() {   # atomic: filtered copy next to the file, exactly $2 lines must go, mode 600, then mv
    ssh -o BatchMode=yes gpuws bash -s -- "$1" "$2" <<'REMOTE'
set -e; blob=$1; m=$2; f=$HOME/.ssh/authorized_keys; t=$(mktemp "$HOME/.ssh/ak.XXXXXX")
grep -vF -- "$blob" "$f" > "$t" || true
a=$(grep -c '' "$f"); b=$(grep -c '' "$t" || true)
if [ $((a - b)) -ne "$m" ]; then rm -f "$t"; echo "  line count mismatch ($a -> $b); authorized_keys left unchanged"; exit 1; fi
chmod 600 "$t"; mv "$t" "$f"; echo "  removed $m line(s); $b line(s) remain"
REMOTE
}
cron_strip() { crontab -l | grep -vE "$OURS_CRON" | crontab -; }

if [ $DO_LAB = 1 ]; then
    echo "-- lab: stop our watchers"
    for w in $WATCHERS; do pgrep -f "$w" >/dev/null && act "pkill -f $w" pkill -f "$w"; done
fi

if [ $DO_SERVER = 1 ] && [ $SERVER_UP = 1 ]; then
    if g 'test -d ~/dsg_cluster && ! test -L ~/dsg_cluster'; then
        echo "-- server: remove ~/dsg_cluster (ours $(g 'du -sh ~/dsg_cluster 2>/dev/null | cut -f1'))"
        act "log the cleanup, copy COMMAND_LOG.md + CLUSTER_SETUP_REPORT.md to $DEST/server_logs/" srv_log_copy || exit 1
        act "ssh gpuws rm -rf ~/dsg_cluster" srv_rm || exit 1
    fi
    echo "-- server: remove our key line from ~/.ssh/authorized_keys (LAST: ends our access)"
    if [ -f "$KEYPUB" ]; then
        blob=$(awk '{print $2}' "$KEYPUB")
        m=$(g "grep -cF -- '$blob' ~/.ssh/authorized_keys" 2>/dev/null); m=${m:-0}
        echo "  key $(awk '{print $1, substr($2,1,16) "...", $3}' "$KEYPUB"): $m matching line(s) on gpuws"
        [ "$m" -ge 1 ] && act "remove the $m line(s) with this key from gpuws ~/.ssh/authorized_keys" srv_key "$blob" "$m"
    else warn "public key $KEYPUB not found: remove our line from gpuws ~/.ssh/authorized_keys by hand"; fi
fi

if [ $DO_LAB = 1 ]; then
    echo "-- lab: stop tmux (dsg-monitor, dsg-queue, dsg-workers)"
    if tmux ls 2>/dev/null | grep -q '^dsg-'; then act "$P/baselines_DSG/scripts/tmux_down.sh" "$P/baselines_DSG/scripts/tmux_down.sh"
    else ok "no dsg-* tmux session"; fi
    echo "-- lab: crontab lines of this project"
    ours=$(crontab -l 2>/dev/null | grep -E "$OURS_CRON" || true)
    if [ -n "$ours" ]; then echo "$ours" | sed 's/^/  line  /'; act "remove these crontab lines (others kept)" cron_strip
    else ok "no crontab line of ours (other lines are never touched)"; fi

    echo "-- lab: OPTIONAL deletions (never done by this script; run the command yourself if you want the space)"
    opt() { [ -e "$1" ] && printf '  %-6s %s\n         %s\n' "$(du -sh "$1" 2>/dev/null | cut -f1)" "$1" "${2:-rm -rf '$1'}"; return 0; }
    opt "$P/dsg_cache"
    opt "$P/dsg_results_cluster/checkpoints"
    opt "$P/checkpoints"
    opt "$P/env_q2_lab"
    opt "$P/wheels"
    opt "$P/dsg_worktrees" "git -C $P/baselines_DSG worktree list   # then: git worktree remove <path> for each dsg_worktrees entry"
    HUB="${HF_HOME:-$HOME/.cache/huggingface}/hub"
    for m in "$HUB"/models--google--gemma-3-* "$HUB"/models--google--gemma-2-9b-it "$HUB"/models--google--gemma-scope-* \
             "$HUB"/models--google--gemma-2-2b-it "$HUB"/datasets--muse-bench--*; do opt "$m"; done
    opt "$HOME/miniconda3/envs/mechunlearn2" "conda env remove -n mechunlearn2"
    opt "$P/dsg_private" "rm -rf '$P/dsg_private'   # hazardous data: delete when the data agreement says so"
    echo "  kept always: ${DSG_RESULTS:-$P/dsg_results}, $DEST (all results), the git repos"
    echo "  yours to remove by hand if wanted: the 'Host gpuws' block in ~/.ssh/config, $KEYPUB and its private key"
fi

echo "== done ($([ $ACT = 1 ] && echo acted || echo 'dry run: nothing changed; re-run with --yes when every check is ok'))"
