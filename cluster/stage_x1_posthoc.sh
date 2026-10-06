#!/bin/bash
# Stage the POST-HOC exploratory PH-X1-conformal group on gpuws (conf x1-posthoc): branch, snapshot, specs, ORDER.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/mechunlearn2/bin/python}
CUDA_VISIBLE_DEVICES= "$PY" cluster/x1_posthoc.py build
CUDA_VISIBLE_DEVICES= "$PY" cluster/x1_posthoc.py stage
