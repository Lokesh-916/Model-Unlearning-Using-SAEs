#!/usr/bin/env bash
# List the non-merge commits (all local branches) whose own diff touches the "break" areas
# (attacks B1-B6 and the shared attack transforms, GuardBreak toolkit N1, red-team challenge N4,
# adaptive hardening loop N5, dilution theory checks T1/T2 = experiment T), plus any hashes in
# extra_break_commits.txt, minus the hashes in exclude_break_commits.txt. Output: one full hash per line.
# Usage: select_break_commits.sh <repo>
set -euo pipefail
repo=${1:?repo}; here=$(cd "$(dirname "$0")" && pwd)
PATHS=(experiments/B1 experiments/B2 experiments/B3 experiments/B4 experiments/B5 experiments/B6
       configs/experiments/B1.yaml configs/experiments/B2.yaml configs/experiments/B3.yaml
       configs/experiments/B4.yaml configs/experiments/B5.yaml configs/experiments/B6.yaml
       dsgx/attacks guardbreak
       experiments/N1 experiments/N4 experiments/N5
       configs/experiments/N1.yaml configs/experiments/N4.yaml configs/experiments/N5.yaml
       experiments/T configs/experiments/T.yaml)
{
  git -C "$repo" log --branches --no-merges --format=%H -- "${PATHS[@]}"
  if [[ -f $here/extra_break_commits.txt ]]; then
    { grep -oE '^[0-9a-f]{7,40}' "$here/extra_break_commits.txt" || true; } | while read -r h; do git -C "$repo" rev-parse "$h^{commit}"; done
  fi
} | sort -u | if [[ -f $here/exclude_break_commits.txt ]]; then
  grep -vxF -f <({ grep -oE '^[0-9a-f]{7,40}' "$here/exclude_break_commits.txt" || true; } | while read -r h; do git -C "$repo" rev-parse "$h^{commit}"; done; echo NONE)
else cat; fi
