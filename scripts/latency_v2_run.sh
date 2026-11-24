#!/usr/bin/env bash
# Full latency run on the lab PC (tmux lat-watch). Resumes cell by cell; waits if another compute process uses the GPU.
cd "$(dirname "$0")/.." && source scripts/env.sh
export PYTHONPATH="$DSG_PROJECT/streamguard-clean/src" DSG_DEBUG=0
for i in $(seq 1 40); do
  "$DSGX_PY" scripts/latency_v2.py bench --json docs/LATENCY_V2.json && { echo FINISHED; exit 0; }
  echo "stopped (attempt $i) $(date -Is)"; sleep 300
done
exit 1
