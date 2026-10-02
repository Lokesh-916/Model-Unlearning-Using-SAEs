# ~/dsg_cluster/env.sh — sourced by every job on gpuws. Offline; all paths inside ~/dsg_cluster.
export DSGC="$HOME/dsg_cluster"
export PATH="$DSGC/env/mechunlearn2/bin:$PATH"
export DSGX_PY="$DSGC/env/mechunlearn2/bin/python"
export HF_HOME="$DSGC/hf_cache"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export DSG_PROJECT="$DSGC"
export DSG_CACHE="$DSGC/data/dsg_cache"
export DSG_RESULTS="$DSGC/results"
export DSG_PRIVATE="$DSGC/private"
export DSG_LEGACY_ROOT="$DSGC/data/legacy"
export DSG_WORKTREES="$DSGC/code"
export DSG_DEBUG=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Slurm does not confine CPUs on this node (no task/cgroup): cap threads ourselves.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" TOKENIZERS_PARALLELISM=false
# Never use /tmp (RAM-backed) for large files.
export TMPDIR="$DSGC/tmp"; mkdir -p "$TMPDIR"
