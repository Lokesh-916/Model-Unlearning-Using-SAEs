# Shared environment for all dsgx scripts (sourced). Paths per MASTER_PLAN 3.2.
export DSG_PROJECT="${DSG_PROJECT:-/home/amaloch/projects/mechunlearn-project}"
export DSG_CACHE="${DSG_CACHE:-$DSG_PROJECT/dsg_cache}"
export DSG_RESULTS="${DSG_RESULTS:-$DSG_PROJECT/dsg_results}"
export DSG_PRIVATE="${DSG_PRIVATE:-$DSG_PROJECT/dsg_private}"
export DSG_WORKTREES="${DSG_WORKTREES:-$DSG_PROJECT/dsg_worktrees}"
export DSGX_PY="${DSGX_PY:-/home/amaloch/miniconda3/envs/mechunlearn2/bin/python}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Everything the queue needs is in the local HF cache; offline mode avoids slow revalidation.
# Set DSGX_OFFLINE=0 for jobs that must download (e.g. new models or datasets).
if [ "${DSGX_OFFLINE:-1}" = "1" ]; then export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1; fi
export DSG_DEBUG="${DSG_DEBUG:-0}"
mkdir -p "$DSG_CACHE" "$DSG_RESULTS/logs" "$DSG_PRIVATE"
