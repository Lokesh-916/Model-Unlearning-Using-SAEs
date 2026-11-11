#!/bin/bash
# Run on the LAB PC: copy this worktree's code, env.sh and slurm tooling to gpuws (rule 7),
# and stamp the commit into code/CODE_COMMIT.json (the copy has no .git).
set -euo pipefail
WT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WT"
commit=$(git rev-parse HEAD); branch=$(git rev-parse --abbrev-ref HEAD)
dirty=$([ -n "$(git status --porcelain --untracked-files=no)" ] && echo true || echo false)
ssh -o BatchMode=yes gpuws "printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'rsync code/ env.sh slurm/ (sync_code.sh from lab PC)' 'sync code $branch@${commit:0:7} dirty=$dirty' >> ~/dsg_cluster/COMMAND_LOG.md"
rsync --bwlimit=50000 --partial -a --delete --exclude .git --exclude figures --exclude __pycache__ \
      --exclude CODE_COMMIT.json ./ gpuws:dsg_cluster/code/
rsync --bwlimit=50000 --partial -a cluster/env.sh gpuws:dsg_cluster/env.sh
rsync --bwlimit=50000 --partial -a cluster/slurm/ gpuws:dsg_cluster/slurm/
ssh gpuws "printf '{\"commit\": \"$commit\", \"branch\": \"$branch\", \"dirty\": $dirty, \"synced\": \"$(date -Is)\"}\n' > dsg_cluster/code/CODE_COMMIT.json && chmod +x dsg_cluster/slurm/*.sh"
echo "synced $branch@${commit:0:7} dirty=$dirty"
