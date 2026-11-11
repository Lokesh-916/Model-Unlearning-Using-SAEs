#!/bin/bash
# Run on the LAB PC. Print the id of the LAST job of our gpuws chain: the queued job of ours that no other
# queued job of ours depends on (highest id if several; nothing if our queue is empty). Watchers append new
# chains with `submit_chain.sh --after-any $(cluster/chain_tail.sh)`, so they join the end of the one linear
# chain instead of branching off the highest job id (session 13: 134 depends on 159, so 159 is not the end).
set -euo pipefail
ssh -n -o BatchMode=yes gpuws 'squeue -h -u "$USER" -o "%i %E"' | awk '
    { ids[$1] = 1; n = split($2, d, ","); for (i = 1; i <= n; i++) { sub(/^after[a-z]*:/, "", d[i]); sub(/\(.*/, "", d[i]); dep[d[i]] = 1 } }
    END { for (j in ids) if (!(j in dep)) print j }' | sort -n | tail -1
