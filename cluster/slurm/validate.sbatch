#!/bin/bash
# sanity-gpuws gate (Bio N=20, seed 0, bs=1, DSG-subset view): must match the gpuws targets EXACTLY
# (161/538, MMLU-u 0.9971, tau 0.5458, same 20 features), which also confirms same-GPU determinism.
# Run first in every server chain; a mismatch exits non-zero and afterok cancels the rest.
#SBATCH --job-name=dsg-validate
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/home/suraj/dsg_cluster/logs/%x-%j.out
#SBATCH --open-mode=append
#SBATCH --requeue
#SBATCH --kill-on-invalid-dep=yes
source "$HOME/dsg_cluster/slurm/preamble.sh"
python - <<'PY'
import sys, torch, transformers, sae_lens
print("env:", sys.version.split()[0], "torch", torch.__version__, "cuda", torch.version.cuda,
      "cudnn", torch.backends.cudnn.version(), "transformers", transformers.__version__, "sae_lens", sae_lens.__version__)
print("gpu:", torch.cuda.get_device_name(0), "free/total GB", [round(x / 2**30, 1) for x in torch.cuda.mem_get_info()])
PY
# --force: the cluster has no earlier sanity run; a restart simply recomputes (~minutes).
python -m dsgx.checks.sanity --force
python cluster/validate_check.py
