#!/usr/bin/env bash
# Stop the tmux sessions. Running jobs are detached and keep running unless --kill-jobs.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
for s in dsg-monitor dsg-workers dsg-queue; do
  tmux kill-session -t "$s" 2>/dev/null && echo "stopped $s" || true
done
# Let the scheduler release its lock (it exits on SIGHUP/SIGTERM).
LOCK="$DSG_RESULTS/queue/scheduler.lock"
if [ -f "$LOCK" ]; then
  PID="$(cat "$LOCK")"; kill "$PID" 2>/dev/null || true; sleep 2
  kill -0 "$PID" 2>/dev/null || rm -f "$LOCK"
fi
cd "$REPO"
"$DSGX_PY" - "$@" <<'PY'
import os, signal, sys
from dsgx.queue import common as q
kill = "--kill-jobs" in sys.argv
for jid, job in q.load_jobs().items():
    st = q.load_state(jid)
    if st["status"] == q.RUNNING and q.pid_alive(st.get("pid")):
        if kill:
            os.killpg(int(st["pgid"]), signal.SIGTERM)
            print(f"killed running job {jid}")
        else:
            print(f"still running (detached): {jid} pid {st['pid']}; tmux_up.sh re-attaches")
PY
