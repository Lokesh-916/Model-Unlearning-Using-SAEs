#!/usr/bin/env bash
# Create experiment branches (from v2-harness) and their worktrees in $DSG_WORKTREES/<branch>.
#   scripts/make_worktrees.sh exp/A1-baselines [exp/B1-dilution ...]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"
for b in "$@"; do
  case "$b" in exp/*) ;; *) echo "refusing non-exp branch $b"; exit 1;; esac
  git show-ref --verify --quiet "refs/heads/$b" || git branch "$b" v2-harness
  wt="$DSG_WORKTREES/$b"
  if [ -d "$wt" ]; then echo "exists: $wt"; continue; fi
  mkdir -p "$(dirname "$wt")"
  git worktree add "$wt" "$b"
  echo "created $wt on $b"
done
git worktree list
