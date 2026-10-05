#!/bin/bash
# Run on the LAB PC. Submit one long chain of our gpuws jobs behind a running one, through slurm/submit.sh
# (disk check, COMMAND_LOG, refusal rules). The first job is always validate.sbatch (sanity-gpuws, exact).
#   cluster/submit_chain.sh [--after-any ID] [--nice N] [--hold] [--dry-run] SPEC...
# --hold submits every job held (JobHeldUser); release with `scontrol release <ids>` (cluster/release_when_free.sh).
# SPEC = file.sbatch    -> --dependency=afterok:<validate>,afterany:<previous>  (an unrelated failure or a
#                          time limit upstream does not cancel it; every job resumes / skips finished work)
#        file.sbatch+   -> --dependency=afterok:<previous>  (needs the previous one to succeed)
# Example (session 8):
#   cluster/submit_chain.sh --after-any 97 --nice 10000 figs-b.sbatch d1-v2-train.sbatch d1-v2-train.sbatch \
#       d1-v2-train.sbatch d1-v2-train.sbatch d1-v2-test.sbatch d1-v2-a6.sbatch+ c3.sbatch ...
set -euo pipefail
AFTER=""; NICE=""; DRY=""
while [ $# -gt 0 ]; do
    case "$1" in
    --after-any) AFTER="$2"; shift 2;;
    --nice) NICE="--nice=$2"; shift 2;;
    --hold) NICE="$NICE --hold"; shift;;
    --dry-run) DRY=1; shift;;
    *) break;;
    esac
done
[ $# -ge 1 ] || { sed -n '2,12p' "$0"; exit 2; }
sub() {  # sbatch-file dependency -> prints job id
    local out
    if [ -n "$DRY" ]; then echo "(dry) submit.sh $1 $NICE $2" >&2; echo $((RANDOM + 1000)); return; fi
    out=$(ssh -n -o BatchMode=yes gpuws "cd ~/dsg_cluster/slurm && ./submit.sh $1 $NICE $2" | tee /dev/stderr)
    echo "$out" | sed -n 's/^submitted .* as job \([0-9]*\)$/\1/p'
}
dep=""; [ -n "$AFTER" ] && dep="--dependency=afterany:$AFTER"
V=$(sub validate.sbatch "$dep"); [ -n "$V" ] || { echo "validate submit failed"; exit 1; }
prev=$V; ids="validate=$V"
for s in "$@"; do
    if [[ "$s" == *+ ]] || [ "$prev" = "$V" ]; then f="${s%+}"; d="--dependency=afterok:$prev"; else f="$s"; d="--dependency=afterok:$V,afterany:$prev"; fi
    j=$(sub "$f" "$d"); [ -n "$j" ] || { echo "submit of $f failed; chain so far: $ids"; exit 1; }
    prev=$j; ids+=" ${f%.sbatch}=$j"
done
echo "CHAIN: $ids"
