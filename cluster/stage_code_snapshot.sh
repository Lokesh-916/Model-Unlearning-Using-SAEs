#!/bin/bash
# Run on the LAB PC: copy this worktree's committed code to gpuws:~/dsg_cluster/code-<name>/ with its own
# CODE_COMMIT.json. A job queued behind a running chain runs from this snapshot, so code/ (used by the
# running chain) is never changed under it. Refuses a dirty tree (the snapshot must match a commit).
#   cluster/stage_code_snapshot.sh <name>
set -euo pipefail
NAME="${1:?usage: stage_code_snapshot.sh <name>}"
WT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$WT"
[ -z "$(git status --porcelain --untracked-files=no)" ] || { echo "REFUSING: uncommitted changes; commit first"; exit 1; }
commit=$(git rev-parse HEAD); branch=$(git rev-parse --abbrev-ref HEAD)
ssh -n -o BatchMode=yes gpuws "printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'rsync code snapshot -> code-$NAME/' 'stage_code_snapshot.sh $branch@${commit:0:7}' >> ~/dsg_cluster/COMMAND_LOG.md"
rsync --bwlimit=50000 --partial -a --delete --exclude .git --exclude figures --exclude __pycache__ \
      --exclude CODE_COMMIT.json ./ "gpuws:dsg_cluster/code-$NAME/" < /dev/null
ssh -n gpuws "printf '{\"commit\": \"$commit\", \"branch\": \"$branch\", \"dirty\": false, \"synced\": \"$(date -Is)\", \"snapshot\": \"code-$NAME\"}\n' > dsg_cluster/code-$NAME/CODE_COMMIT.json"
ssh -n gpuws "source ~/dsg_cluster/env.sh && cd ~/dsg_cluster/code-$NAME && PYTHONPATH=\$PWD CUDA_VISIBLE_DEVICES= python cluster/import_check.py | tail -3"
ssh -n gpuws "printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'CUDA_VISIBLE_DEVICES= python cluster/import_check.py (code-$NAME)' 'CPU import check of the snapshot' >> ~/dsg_cluster/COMMAND_LOG.md"
echo "snapshot code-$NAME = $branch@${commit:0:7}"
