#!/bin/bash
# Run on the LAB PC (network). Downloads what a later server job needs into the HF cache, so that
# `cluster/server.sh stage <job>` can copy it. Refuses if lab-PC free disk would drop below 60 GB.
#   cluster/fetch_models.sh a7-12b      gemma-3-12b-it (~24 GB, licence must be accepted on the HF
#                                       account: `huggingface-cli login` once) + Gemma Scope 2 12B layer 24 SAE
#   cluster/fetch_models.sh mtbench     unsloth/Qwen2.5-32B-Instruct-bnb-4bit (~19 GB, the fixed open judge)
#   cluster/fetch_models.sh q2-graphs   mwhanna/gemma-scope-transcoders (~8 GB, circuit-tracer "gemma" set)
#   cluster/fetch_models.sh muse        MUSE-News + MUSE-Books datasets (~0.2 GB) + muse_bench code tarball
#   cluster/fetch_models.sh list        sizes and what is already present
# After use and cleanup on the server, delete big downloads from the lab PC too if space is needed:
#   huggingface-cli delete-cache   (interactive) or rm -rf ~/.cache/huggingface/hub/models--<org>--<name>
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
source "$REPO/scripts/env.sh"
export HF_HUB_OFFLINE=0 HF_DATASETS_OFFLINE=0 TRANSFORMERS_OFFLINE=0
W="$DSG_PROJECT/wheels"
need() {  # GB
    local free; free=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
    echo "lab PC free ${free} GB; download ~$1 GB"
    [ $((free - $1)) -gt 60 ] || { echo "REFUSING: lab PC would drop below 60 GB free"; exit 1; }
}
snap() {  # repo [allow-pattern]
    "$DSGX_PY" - "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo = sys.argv[1]
allow = sys.argv[2:] or None
p = snapshot_download(repo, allow_patterns=allow)
print("downloaded", repo, "->", p)
PY
}
case "${1:-list}" in
a7-12b)
    need 26
    snap google/gemma-3-12b-it
    snap google/gemma-scope-2-12b-it "resid_post/layer_24_width_16k_l0_medium/*" ;;
mtbench)
    need 20
    snap unsloth/Qwen2.5-32B-Instruct-bnb-4bit ;;
q2-graphs)
    need 9
    snap mwhanna/gemma-scope-transcoders
    ls "$W/q2" | wc -l | xargs echo "wheels/q2 files:" ;;
muse)
    need 1
    "$DSGX_PY" - <<'PY'
from datasets import get_dataset_config_names, load_dataset
for repo in ("muse-bench/MUSE-News", "muse-bench/MUSE-Books"):
    for c in get_dataset_config_names(repo):
        d = load_dataset(repo, c)
        print(repo, c, {s: len(d[s]) for s in d})
PY
    mkdir -p "$W/muse"
    curl -fsSL -o "$W/muse/muse_bench-main.tar.gz" https://github.com/swj0419/muse_bench/archive/refs/heads/main.tar.gz
    sha256sum "$W/muse/muse_bench-main.tar.gz" | tee "$W/muse/SHA256SUMS"
    echo "next: python -m cluster.muse --inspect (prints configs/splits/columns, no text)" ;;
list)
    for r in models--google--gemma-3-12b-it models--google--gemma-scope-2-12b-it models--unsloth--Qwen2.5-32B-Instruct-bnb-4bit \
             models--mwhanna--gemma-scope-transcoders datasets--muse-bench--MUSE-News datasets--muse-bench--MUSE-Books; do
        p="$HOME/.cache/huggingface/hub/$r"
        printf '  %-55s %s\n' "$r" "$( [ -d "$p" ] && du -sh "$p" | cut -f1 || echo 'not downloaded')"
    done
    printf '  %-55s %s\n' "wheels/q2" "$(du -sh "$W/q2" 2>/dev/null | cut -f1 || echo missing)" ;;
*) sed -n '2,13p' "$0"; exit 2 ;;
esac
