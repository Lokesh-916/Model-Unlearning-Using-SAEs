#!/bin/bash
# Stage today's inputs (validation + RMU) from the lab PC to gpuws. Read-only on the lab PC side.
set -euo pipefail
RS="rsync --bwlimit=50000 --partial -a"
H=~/.cache/huggingface; P=~/projects/mechunlearn-project
A=$P/baselines_DSG/artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it
SAE_REV=fd571b47c1c64851e9b1989792367b9babb4af63
SAE_SUB=layer_3/width_16k/average_l0_142
ACT=gemma-2-2b-it__gemma-scope-2b-pt-res__layer_3-width_16k-average_l0_142__bio-forget-corpus__wikitext__s0__n1024__L1024
R=gpuws:dsg_cluster
ssh gpuws "mkdir -p dsg_cluster/hf_cache/hub/models--google--gemma-scope-2b-pt-res/{refs,snapshots/$SAE_REV/$SAE_SUB} dsg_cluster/hf_cache/datasets dsg_cluster/data/dsg_cache/actcache dsg_cluster/data/legacy/artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it/gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities dsg_cluster/private/corpora dsg_cluster/data/legacy/dynamic_sae_guardrails/evals/unlearning && chmod 700 dsg_cluster/private && ln -sfn ~/dsg_cluster/private/corpora dsg_cluster/data/legacy/dynamic_sae_guardrails/evals/unlearning/data"
# 1. base model (blobs + snapshot symlinks)
$RS $H/hub/models--google--gemma-2-2b-it $R/hf_cache/hub/
# 2. SAE: only the layer-3 16k l0_142 params (dereferenced into the snapshot)
$RS $H/hub/models--google--gemma-scope-2b-pt-res/refs/ $R/hf_cache/hub/models--google--gemma-scope-2b-pt-res/refs/
$RS -L $H/hub/models--google--gemma-scope-2b-pt-res/snapshots/$SAE_REV/$SAE_SUB/ $R/hf_cache/hub/models--google--gemma-scope-2b-pt-res/snapshots/$SAE_REV/$SAE_SUB/
# 3. datasets: WMDP MCQ, MMLU (all 57 subjects, no auxiliary_train), wikitext
for d in datasets--cais--wmdp datasets--cais--mmlu datasets--Salesforce--wikitext; do $RS $H/hub/$d $R/hf_cache/hub/; done
$RS $H/datasets/cais___wmdp $H/datasets/Salesforce___wikitext $R/hf_cache/datasets/
$RS --exclude auxiliary_train $H/datasets/cais___mmlu $R/hf_cache/datasets/
# 4. Bio activation cache (layer 3, seed 0)
$RS --exclude .lock $P/dsg_cache/actcache/$ACT $R/data/dsg_cache/actcache/
# 5. legacy small files: DSG-subset question ids, baseline metrics, sparsity txt (no pickles)
$RS $A/data $R/data/legacy/artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it/
$RS $A/gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities/feature_sparsity_{forget,retain}.txt $R/data/legacy/artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it/gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities/
# 6. hazardous forget corpus -> private (never printed; checksum only)
$RS $P/baselines_DSG/dynamic_sae_guardrails/evals/unlearning/data/bio-forget-corpus.jsonl $R/private/corpora/
