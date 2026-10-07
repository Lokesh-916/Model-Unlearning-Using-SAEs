#!/bin/bash
# Stage the POST-HOC exploratory ph-union group on gpuws (conf ph-union): configs on exp/PH-union, snapshot, specs, ORDER.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-$HOME/miniconda3/envs/mechunlearn2/bin/python}
CUDA_VISIBLE_DEVICES= "$PY" cluster/ph_union.py build
CUDA_VISIBLE_DEVICES= "$PY" cluster/ph_union.py stage
