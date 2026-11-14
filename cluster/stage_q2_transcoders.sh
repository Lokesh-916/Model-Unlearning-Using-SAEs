#!/bin/bash
# Run on the LAB PC (STAGE_SCRIPT of q2-graphs). Streams mwhanna/gemma-scope-transcoders (config.yaml +
# layer_*.safetensors, 7.9 GB; the 4 GB features/ visualisation files are not needed) to gpuws hf_cache/hub ONE
# FILE AT A TIME: download into the lab HF cache, rsync the repo dir (blobs + snapshot symlinks), check the size
# on the server, delete the local blob. The lab PC never holds more than one layer (~0.3 GB), so it stays above
# its 60 GB floor (67 GB free on 2026-10-03; a full download would leave 59 GB). Re-runs skip files already on
# the server with the right size. Every server command is logged (rule 8).
set -euo pipefail
REPO_ID="mwhanna/gemma-scope-transcoders"; D="models--mwhanna--gemma-scope-transcoders"
HUB="$HOME/.cache/huggingface/hub"; PY="${DSGX_PY:-$HOME/miniconda3/envs/mechunlearn2/bin/python}"
ssh -n -o BatchMode=yes gpuws "mkdir -p ~/dsg_cluster/hf_cache/hub && printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'rsync $D (per file, stage_q2_transcoders.sh)' 'stage Q2 transcoders' >> ~/dsg_cluster/COMMAND_LOG.md"
free=$(ssh -n gpuws "df -BG --output=avail ~ | tail -1 | tr -dc 0-9")
[ $((free - 8)) -ge 50 ] || { echo "REFUSING: server free ${free} GB would drop below 50"; exit 1; }
HF_HUB_OFFLINE=0 "$PY" - <<'PY' > /tmp/q2_tc_files.$$
from huggingface_hub import HfApi
for f in HfApi().list_repo_tree("mwhanna/gemma-scope-transcoders", recursive=True):
    if getattr(f, "size", None) and (f.path == "config.yaml" or f.path.startswith("layer_")):
        print(f.path, f.size)
PY
n=0
while read -r path size; do
    have=$(ssh -n gpuws "stat -L -c %s ~/dsg_cluster/hf_cache/hub/$D/snapshots/*/$path 2>/dev/null | head -1" || true)
    if [ "$have" = "$size" ]; then n=$((n + 1)); continue; fi
    lab=$(df -BG --output=avail "$HOME" | tail -1 | tr -dc 0-9)
    [ "$lab" -gt 61 ] || { echo "REFUSING: lab PC free ${lab} GB"; exit 1; }
    local_path=$(HF_HUB_OFFLINE=0 "$PY" -c "from huggingface_hub import hf_hub_download as h; print(h('$REPO_ID', '$path'))")
    rsync --bwlimit=50000 --partial -a "$HUB/$D" gpuws:dsg_cluster/hf_cache/hub/ < /dev/null
    have=$(ssh -n gpuws "stat -L -c %s ~/dsg_cluster/hf_cache/hub/$D/snapshots/*/$path | head -1")
    [ "$have" = "$size" ] || { echo "SIZE MISMATCH on server for $path ($have vs $size)"; exit 1; }
    [ "$path" = "config.yaml" ] || rm -f "$(readlink -f "$local_path")"   # keep the tiny config locally
    n=$((n + 1)); echo "  $path ok ($size bytes; $n done)"
done < /tmp/q2_tc_files.$$
rm -f /tmp/q2_tc_files.$$
ssh -n gpuws "printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'du -sh hf_cache/hub/$D' 'transcoders staged ($n files)' >> ~/dsg_cluster/COMMAND_LOG.md; du -sh ~/dsg_cluster/hf_cache/hub/$D"
echo "transcoders: $n files on gpuws (lab copies of the layer files deleted)"
