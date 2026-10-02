# Sourced at the top of every dsg sbatch job (see template.sbatch).
# 1) environment, 2) job header, 3) GPU-memory precheck (rule 4).
set -euo pipefail
source "$HOME/dsg_cluster/env.sh"
cd "$DSGC/code"
echo "=== dsg job ${SLURM_JOB_NAME:-?} (${SLURM_JOB_ID:-?}) on $(hostname) at $(date -Is)"
echo "=== code: $(cat CODE_COMMIT.json 2>/dev/null || echo 'no CODE_COMMIT.json')"
echo "=== cpus=${SLURM_CPUS_PER_TASK:-?} gres=${SLURM_JOB_GPUS:-${CUDA_VISIBLE_DEVICES:-?}} restarts=${SLURM_RESTART_COUNT:-0}"

MIN_FREE_MIB=${MIN_FREE_MIB:-40960}
free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo "=== GPU free memory: ${free_mib} MiB (need >= ${MIN_FREE_MIB})"
if [ -z "$free_mib" ] || [ "$free_mib" -lt "$MIN_FREE_MIB" ]; then
    echo "[precheck] ABORT: less than 40 GB of GPU memory is free (${free_mib:-unknown} MiB)."
    echo "[precheck] Someone is probably using the GPU outside Slurm. Nothing was run; resubmit later."
    # Non-zero (75 = EX_TEMPFAIL) so that --dependency=afterok chains do not start.
    exit 75
fi
