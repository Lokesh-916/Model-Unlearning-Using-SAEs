#!/usr/bin/env bash
# After a lab-PC reboot: bring the tmux sessions back, re-queue jobs the reboot killed (and the jobs
# they blocked), and print the status. Safe to run any time; it never re-queues code errors,
# leakage failures or canary drift (see python -m dsgx.queue.doctor).
#   scripts/reboot_recover.sh [--dry-run]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
# The scheduler always runs from the baselines_DSG checkout (v2-harness), never from this worktree.
QUEUE_REPO="${DSG_QUEUE_REPO:-$DSG_PROJECT/baselines_DSG}"
cd "$REPO"
nvidia-smi -L >/dev/null 2>&1 || { echo "GPU driver not ready (nvidia-smi failed); wait a minute and retry"; exit 1; }
"$QUEUE_REPO/scripts/tmux_up.sh"
sleep 15   # let the scheduler reconcile jobs that died with the machine
if [ "${1:-}" = "--dry-run" ]; then
  "$DSGX_PY" -m dsgx.queue.doctor --apply --dry-run || true
else
  "$DSGX_PY" -m dsgx.queue.doctor --apply || true
fi
"$DSGX_PY" -m dsgx.queue.status --chat | head -25
