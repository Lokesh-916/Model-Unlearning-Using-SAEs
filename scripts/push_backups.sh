#!/bin/bash
# Back up paper-srw, presentation and release to NEW PRIVATE GitHub repos (they have no remote). Never public.
#   scripts/push_backups.sh            # needs the gh CLI, logged in as Lokesh-916 (gh auth login)
#   scripts/push_backups.sh --print    # print the manual commands instead (no gh: create the repos on github.com)
# release is pushed only after its leak scan prints SCAN OK. Re-running is safe (existing remotes are just pushed).
set -euo pipefail
P="${P:-$HOME/projects/mechunlearn-project}"; OWNER="${OWNER:-Lokesh-916}"
declare -A NAME=([paper-srw]=dsg-paper-srw [presentation]=dsg-presentation [release]=dsg-release)
scan() {
    ( source "$P/prep/scripts/env.sh" && cd "$P/release" && \
      "$HOME/miniconda3/envs/mechunlearn2/bin/python" scan_release.py --private-check --private-root "$DSG_PRIVATE" \
        --forget-corpus "$P/baselines_DSG/dynamic_sae_guardrails/evals/unlearning/data/bio-forget-corpus.jsonl" 2>/dev/null | tail -1 ) \
        | grep -qx "SCAN OK"
}
if [ "${1:-}" = "--print" ]; then
    cat <<EOT
# 1) On https://github.com/new create three repos, each with Visibility = PRIVATE, no README/licence/.gitignore:
#      ${NAME[paper-srw]}  ${NAME[presentation]}  ${NAME[release]}
# 2) Leak scan of the release (must print SCAN OK as its last line):
cd $P/release && source $P/prep/scripts/env.sh && python scan_release.py --private-check --private-root \$DSG_PRIVATE --forget-corpus $P/baselines_DSG/dynamic_sae_guardrails/evals/unlearning/data/bio-forget-corpus.jsonl | tail -1
# 3) Push (SSH key as for the main repo):
EOT
    for r in paper-srw presentation release; do
        echo "git -C $P/$r remote add origin git@github.com:$OWNER/${NAME[$r]}.git && git -C $P/$r push -u origin main"
    done
    echo "# 4) Check on github.com that all three show the 'Private' badge."
    exit 0
fi
command -v gh >/dev/null || { echo "gh CLI not installed. Run: scripts/push_backups.sh --print"; exit 1; }
gh auth status >/dev/null 2>&1 || { echo "gh not logged in: gh auth login (as $OWNER), or scripts/push_backups.sh --print"; exit 1; }
scan || { echo "release leak scan did NOT print SCAN OK: nothing pushed"; exit 1; }
echo "release scan: SCAN OK"
for r in paper-srw presentation release; do
    d="$P/$r"; n="${NAME[$r]}"
    if git -C "$d" remote get-url origin >/dev/null 2>&1; then
        echo "$r: origin exists ($(git -C "$d" remote get-url origin)); pushing"; git -C "$d" push -u origin main; continue
    fi
    if gh repo view "$OWNER/$n" >/dev/null 2>&1; then
        [ "$(gh repo view "$OWNER/$n" --json visibility -q .visibility)" = "PRIVATE" ] || { echo "REFUSING: $OWNER/$n exists and is not private"; exit 1; }
        git -C "$d" remote add origin "git@github.com:$OWNER/$n.git"; git -C "$d" push -u origin main
    else
        gh repo create "$OWNER/$n" --private --source "$d" --remote origin --push
    fi
    [ "$(gh repo view "$OWNER/$n" --json visibility -q .visibility)" = "PRIVATE" ] && echo "$r -> $OWNER/$n (PRIVATE)"
done
