#!/usr/bin/env bash
# Start (or restore after a reboot) the three tmux sessions of MASTER_PLAN 4.4.
# Safe to re-run: existing sessions are kept, and the scheduler lock prevents duplicates.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
L="$DSG_RESULTS/logs"
touch "$L/slot_gpu0.log" "$L/slot_gpu1.log" "$L/slot_cpu.log" "$L/scheduler.log"

if ! tmux has-session -t dsg-queue 2>/dev/null; then
  tmux new-session -d -s dsg-queue -n supervisor -c "$REPO"
  tmux send-keys -t dsg-queue:supervisor \
    "source scripts/env.sh; while true; do \"\$DSGX_PY\" -m dsgx.queue.scheduler 2>>\"$L/scheduler.stderr.log\"; echo \"[supervisor] scheduler exited \$?; restart in 30 s\"; sleep 30; done" Enter
fi
if ! tmux has-session -t dsg-workers 2>/dev/null; then
  tmux new-session -d -s dsg-workers -n gpu0 -c "$REPO" "tail -n 50 -F $L/slot_gpu0.log"
  tmux new-window -t dsg-workers -n gpu1 -c "$REPO" "tail -n 50 -F $L/slot_gpu1.log"
  tmux new-window -t dsg-workers -n cpu -c "$REPO" "tail -n 50 -F $L/slot_cpu.log"
fi
if ! tmux has-session -t dsg-monitor 2>/dev/null; then
  tmux new-session -d -s dsg-monitor -n status -c "$REPO" \
    "bash -c 'source scripts/env.sh; watch -n 120 \"\$DSGX_PY\" -m dsgx.queue.status'"
  tmux new-window -t dsg-monitor -n gpu "watch -n 10 nvidia-smi"
fi
tmux ls
