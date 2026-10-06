#!/usr/bin/env bash
# Re-run N10 audit cards on real data once the X1 TEST wave has finished (session 17).
# N10 reads $DSG_RESULTS/summary.json, written by final_report, so the order must be:
#   every X1-* job (not X1-screen) DONE -> final_report --interim (labpc) -> re-queue N10-cards.
# T-T3 and A8-tables need no watcher: their queue jobs depend on all X1 jobs directly.
# Exits without re-queueing (and says why) if an X1 job ends FAILED or BLOCKED.
#   nohup scripts/n10_after_x1.sh >> $DSG_RESULTS/logs/n10_after_x1.log 2>&1 &
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"
EVERY="${EVERY:-600}"
echo "[n10-watch] $(date -Is) waiting for every X1 job (excluding X1-screen) to be DONE"
while true; do
  st=$("$DSGX_PY" - <<'PY'
from dsgx.queue import common as q
s = [q.load_state(j).get("status") for j in q.load_jobs() if j.startswith("X1-") and not j.startswith("X1-screen")]
bad = {"FAILED", "BLOCKED", "HUNG"} & set(s)
print("BAD:" + ",".join(sorted(bad)) if bad else ("DONE" if s and all(x == "DONE" for x in s) else "WAIT"))
PY
)
  case "$st" in
    BAD*) echo "[n10-watch] $(date -Is) X1 has $st jobs: not re-queueing N10 (doctor, then rerun this script)"; exit 1 ;;
    DONE)
      echo "[n10-watch] $(date -Is) X1 done; writing summary.json"
      CUDA_VISIBLE_DEVICES= "$DSGX_PY" -m dsgx.analysis.final_report --interim --hardware labpc --no-figures || exit 1
      "$DSGX_PY" -m dsgx.queue.doctor --requeue N10-cards
      echo "[n10-watch] $(date -Is) N10-cards re-queued"; exit 0 ;;
  esac
  sleep "$EVERY"
done
