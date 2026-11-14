#!/usr/bin/env bash
# Enqueue the X1 combination wave on the lab PC as soon as its baked inputs exist, i.e. behind the
# D1 OOM-fix jobs (D1-train-undo-a0.{1,3,5} DONE -> dsg_cache/models/D1/undo_a*). Exits without
# enqueueing (and says why) if one of them ends FAILED, so a missing D1 candidate is never dropped silently.
#   nohup scripts/x1_enqueue_when_ready.sh >> $DSG_RESULTS/logs/x1_enqueue.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"
JOBS="D1-train-undo-a0.1 D1-train-undo-a0.3 D1-train-undo-a0.5"
EVERY="${EVERY:-600}"
echo "[x1-watch] $(date -Is) waiting for: $JOBS"
while true; do
  st=$("$DSGX_PY" - $JOBS <<'PY'
import sys
from dsgx.queue import common as q
s = [q.load_state(j)["status"] for j in sys.argv[1:]]
print("FAILED" if "FAILED" in s else ("DONE" if all(x == "DONE" for x in s) else "WAIT"))
PY
)
  if [ "$st" = "FAILED" ]; then
    echo "[x1-watch] $(date -Is) a D1 undo job FAILED: not enqueueing X1 (fix D1, then rerun this script)"; exit 1
  fi
  if [ "$st" = "DONE" ]; then
    echo "[x1-watch] $(date -Is) D1 checkpoints ready"
    "$DSGX_PY" -m dsgx.combine --dry-run
    "$DSGX_PY" -m dsgx.combine --enqueue
    rc=$?
    echo "[x1-watch] $(date -Is) combine --enqueue exit $rc"; exit $rc
  fi
  sleep "$EVERY"
done
