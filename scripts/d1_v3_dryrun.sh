#!/bin/bash
# D1 v3 dry run (lab PC; submits NOTHING). README "How to run D1 v3 on gpuws without Claude".
#   scripts/d1_v3_dryrun.sh            # 1) tiny CPU tests  2) real-config CPU plan  3) server disk plan  4) submit chain (dry)
#   scripts/d1_v3_dryrun.sh --server   # after `cluster/server.sh stage d1-v3`: the same CPU plan on the gpuws login node
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"; cd "$HERE"
PY="${PY:-$HOME/miniconda3/envs/mechunlearn2/bin/python}"
if [ "${1:-}" = "--server" ]; then
    ssh -n -o BatchMode=yes gpuws "printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'CUDA_VISIBLE_DEVICES= python cluster/d1_v3.py plan (code-d1v3, login node)' 'd1_v3_dryrun.sh --server: CPU plan, no GPU' >> ~/dsg_cluster/COMMAND_LOG.md"
    ssh -n -o BatchMode=yes gpuws 'source ~/dsg_cluster/env.sh && cd ~/dsg_cluster/code-d1v3 && PYTHONPATH=$PWD CUDA_VISIBLE_DEVICES= python cluster/d1_v3.py plan 2>/dev/null | tail -45'
    exit 0
fi
[ -f scripts/env.sh ] && source scripts/env.sh
SCR="$(mktemp -d)"; trap 'rm -rf "$SCR"' EXIT
echo "== 1/4 tiny CPU tests (about 2-3 min)"
CUDA_VISIBLE_DEVICES= "$PY" -m pytest -q tests/test_prep_d1v3.py 2>&1 | tail -2
echo "== 2/4 real-config CPU plan (inputs, item counts, DSG tau 0.5458, TEST configs, estimate; writes only to a temp dir)"
DSG_RESULTS="$SCR" PYTHONPATH="$HERE" CUDA_VISIBLE_DEVICES= "$PY" cluster/d1_v3.py plan 2>/dev/null | tail -45
echo "== 3/4 server disk plan (read-only ssh)"
cluster/server.sh plan d1-v3
echo "== 4/4 chain that would be submitted (dry)"
cluster/submit_chain.sh --dry-run $(. cluster/slurm/jobs/d1-v3.conf; echo $CHAIN | sed 's/\.sbatch/.sbatch+/g') 2>&1 | tail -15
echo "dry run done: nothing staged, nothing submitted"
