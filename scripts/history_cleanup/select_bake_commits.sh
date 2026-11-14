#!/usr/bin/env bash
# List the non-merge commits (all local branches) whose own diff touches the "bake" areas
# (Part III open-weights erasure: D1 / d1-full / B1 distillation seed, D2, D3, A6 / a6-full),
# plus any hashes in extra_bake_commits.txt. Output: one full hash per line.
# Usage: select_bake_commits.sh <repo>
set -euo pipefail
repo=${1:?repo}; here=$(cd "$(dirname "$0")" && pwd)
PATHS=(experiments/D1 experiments/D2 experiments/D3 experiments/A6
       configs/experiments/D1.yaml configs/experiments/D2.yaml configs/experiments/D3.yaml configs/experiments/A6.yaml
       cluster/d1_full.py cluster/a6_full.py cluster/slurm/d1-full.sbatch cluster/slurm/a6-full.sbatch
       cluster/slurm/jobs/d1-full.conf cluster/slurm/jobs/a6-full.conf
       novelty_trials/b1_distill_weights)
{
  git -C "$repo" log --branches --no-merges --format=%H -- "${PATHS[@]}"
  if [[ -f $here/extra_bake_commits.txt ]]; then
    { grep -oE '^[0-9a-f]{7,40}' "$here/extra_bake_commits.txt" || true; } | while read -r h; do git -C "$repo" rev-parse "$h^{commit}"; done
  fi
} | sort -u
