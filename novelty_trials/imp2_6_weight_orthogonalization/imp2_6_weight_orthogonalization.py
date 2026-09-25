"""
Idea 6 (round 2, "main contribution if possible"): edit the model's WEIGHTS
instead of intervening on activations at runtime, using the SAE's own
decoder directions to decide what to remove, while trying to keep utility
(MMLU) intact.

User's idea: "instead of acting on the activations can we do changes to the
weight effectively while maintaining utility score good (main contribution
if possible) using sae's." DSG's own intervention is entirely a runtime hook:
every forward pass, encode the residual stream with the SAE, check a
whole-sequence gate, and conditionally overwrite feature activations before
decoding back. This has three costs the paper never has to pay if the
guardrail lived in the weights instead: (1) it requires shipping/running the
SAE alongside the model forever, (2) it requires the gate to correctly
distinguish forget from retain inputs at runtime (round 1's dilution attack
and this round's ideas 2/3 are all about that gate), and (3) it adds
inference latency (measured elsewhere in this project).

This tries a training-free, purely static alternative directly analogous to
"directional ablation" / "refusal-direction removal" techniques: use the SAE
decoder vectors for DSG's own top-20 selected forget features as a subspace,
and permanently project that subspace OUT of every weight matrix that can
write into the residual stream up to and including layer 3 (the token
embedding, and each of blocks 0-3's attention output projection W_O and MLP
output projection W_out) -- so the model's own forward pass can never
reconstruct a component along those directions in the first place, by
construction, with zero runtime cost and zero gate to fool or miscalibrate.

Concretely, for the k=20 decoder directions {d_1...d_20} (normalized), build
an orthonormal basis Q for their span (via QR decomposition) and apply the
projection P = I - Q Q^T to the OUTPUT dimension of every write-component:
  W_new = W_old @ P
This zeroes exactly the components along the selected features' directions
and leaves everything orthogonal to that subspace untouched -- the same
"redirect around a subspace, change nothing else" logic DSG's own decoder
reconstruction already relies on, just applied once, permanently, to the
weights instead of every forward pass.
"""
import json
import os
import sys

import numpy as np
import torch
from functools import partial

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import calculate_MCQ_metrics
from evals.unlearning.utils.feature_activation import get_top_features_percentile

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
TARGET_LAYER = 3  # edit embeddings + blocks 0..TARGET_LAYER inclusive (every component that can
                   # write a contribution into resid_post of block 3, DSG's own intervention point)
N_FEATURES = 20
RATIO_PERCENTILE = 95

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SAE_RELEASE_AND_ID = f"{SAE_NAME}_{SAE_BLOCK}"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, SAE_RELEASE_AND_ID, "results", "sparsities")
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

device = "cuda"

print("[1] Deriving DSG's original top-20 feature set + their decoder directions...")
fs_fgt = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
fs_ret = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
sel_orig, perc_orig = get_top_features_percentile(
    fs_fgt, fs_ret, ratio_percentile=RATIO_PERCENTILE, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features = sel_orig[:N_FEATURES].tolist()
print(f"  features: {features}")

sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)
decoder_dirs = sae.W_dec[features].detach().float()  # [20, d_model]
decoder_dirs = decoder_dirs / decoder_dirs.norm(dim=1, keepdim=True).clamp_min(1e-8)
d_model = decoder_dirs.shape[1]
del sae
torch.cuda.empty_cache()

print("\n[2] Building the orthogonal-complement projection for the 20-feature subspace...")
Q, _ = torch.linalg.qr(decoder_dirs.T)  # [d_model, k] orthonormal basis for the span
rank = Q.shape[1]
print(f"  subspace rank after QR: {rank} (20 requested; lower means some near-duplicate directions)")
P = torch.eye(d_model, device=device, dtype=torch.float32) - Q @ Q.T  # [d_model, d_model]

print("\n[3] Loading model (bf16, to keep memory reasonable) and applying the permanent weight edit "
      f"(embeddings + attn.W_O/mlp.W_out for blocks 0..{TARGET_LAYER})...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)

with torch.no_grad():
    # Each edit upcasts just that one tensor to float32 for the projection matmul,
    # then casts back to bf16 -- avoids holding the whole ~2.6B-param model in float32.
    model.embed.W_E.data = (model.embed.W_E.data.float() @ P).to(torch.bfloat16)
    for layer in range(TARGET_LAYER + 1):
        block = model.blocks[layer]
        # W_O: [n_heads, d_head, d_model] -- project the OUTPUT (last) dimension
        block.attn.W_O.data = torch.einsum("hde,ef->hdf", block.attn.W_O.data.float(), P).to(torch.bfloat16)
        # W_out: [d_mlp, d_model] -- project the OUTPUT (last) dimension
        block.mlp.W_out.data = (block.mlp.W_out.data.float() @ P).to(torch.bfloat16)

model.reset_hooks()

print("\n[4] Evaluating the permanently weight-edited model (NO runtime hook, NO gate, NO SAE at eval time)...")
edited_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct",
                               split="all", verbose=False)
    edited_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
mmlu_edited = float(np.mean([edited_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n" + "=" * 70)
print("FINAL SUMMARY: static weight orthogonalization vs DSG's runtime clamp vs unedited base model")
print("=" * 70)
print(f"{'config':>36} {'WMDP-bio':>12} {'MMLU_avg':>12}")
print(f"{'DSG runtime clamp (cached)':>36} {29.368:>11.3f}% {99.412:>11.3f}%")
print(f"{'weight-orthogonalized (this script)':>36} {edited_results['wmdp-bio']*100:>11.3f}% {mmlu_edited*100:>11.3f}%")

out = {
    "features": features,
    "subspace_rank": rank,
    "target_layer": TARGET_LAYER,
    "edited_results": edited_results,
    "mmlu_edited": mmlu_edited,
}
out_path = "novelty_trials/imp2_6_weight_orthogonalization/imp2_6_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved:", out_path)
print("DONE")
