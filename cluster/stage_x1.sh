#!/bin/bash
# Run on the LAB PC (STAGE_SCRIPT of jobs/x1.conf). X1 TEST replication on gpuws (session 20):
#   1. the lab A1-dev bio DEV runs' config.json / metrics.json / DONE only (no items): X1's dsg-faithful runs resolve
#      their config with select_config(A1-dev) exactly as on the lab (n20 / rp95 / m500). They are a configuration
#      input, never reported from gpuws (results/runs/A1-dev is not in RESULT_PATHS; SOURCE.json says so);
#   2. the 33 X1 TEST jobs (X1-000..031 + X1-attack-success) at one commit (93625f5, see stage_lab_jobs.sh
#      LABJOBS_PIN), as a replica: the lab keeps its own X1 run.
set -euo pipefail
P="${DSG_PROJECT:-$HOME/projects/mechunlearn-project}"
R="${DSG_RESULTS:-$P/dsg_results}/runs/A1-dev"
HERE="$(cd "$(dirname "$0")" && pwd)"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/x1-a1dev.XXXX")"; trap 'rm -rf "$TMP"' EXIT
n=0
for d in "$R"/A1-dev__*__bio-*__dev__*; do
    [ -f "$d/DONE" ] || continue
    mkdir -p "$TMP/A1-dev/$(basename "$d")"; cp "$d/config.json" "$d/metrics.json" "$d/DONE" "$TMP/A1-dev/$(basename "$d")/"; n=$((n + 1))
done
printf '{"source": "labpc A1-dev DEV runs (config.json, metrics.json, DONE only)", "use": "select_config input of X1 dsg-faithful; never reported on gpuws", "runs": %d, "copied": "%s"}\n' "$n" "$(date -Is)" > "$TMP/A1-dev/SOURCE.json"
ssh -n -o BatchMode=yes gpuws "printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'rsync results/runs/A1-dev/ ($n bio DEV run configs + metrics)' 'stage_x1.sh: X1 dsg-faithful select_config input' >> ~/dsg_cluster/COMMAND_LOG.md; mkdir -p ~/dsg_cluster/results/runs"
rsync --bwlimit=50000 --partial -a "$TMP/A1-dev" gpuws:dsg_cluster/results/runs/ < /dev/null
echo "staged $n A1-dev bio DEV runs (select input)"
LABJOBS_PIN=93625f5daf04740f62a1f9dcd842f90838f586f3 LABJOBS_REPLICA=1 "$HERE/stage_lab_jobs.sh" x1 'X1-0[0-9][0-9]' X1-attack-success
