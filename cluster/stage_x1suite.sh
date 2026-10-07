#!/bin/bash
# Run on the LAB PC (STAGE_SCRIPT of jobs/x1-suite.conf): stage the lab's six X1-suite jobs (exp/X1-suite b933e20) as
# a gpuws replica, then set the run order to the session-27 request: (a) TOFU first, then (c) benign-open, leak, paired.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LABJOBS_REPLICA=1 "$HERE/stage_lab_jobs.sh" x1-suite 'X1-suite-*'
ORDER="X1-suite-tofu-metrics X1-suite-tofu-qa-forget X1-suite-tofu-qa-retain X1-suite-benign-open X1-suite-leak X1-suite-paired"
ssh -n -o BatchMode=yes gpuws "printf '%s\n' $ORDER > ~/dsg_cluster/labjobs/x1-suite/ORDER && printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'write labjobs/x1-suite/ORDER' 'stage_x1suite.sh: run order (a) TOFU, then (c)' >> ~/dsg_cluster/COMMAND_LOG.md && cat ~/dsg_cluster/labjobs/x1-suite/ORDER"
