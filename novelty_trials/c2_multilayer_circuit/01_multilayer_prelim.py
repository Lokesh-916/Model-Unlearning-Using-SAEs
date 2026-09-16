"""
Multi-layer DSG intervention: apply the SAME validated whole-sequence-average
gating + clamp mechanism (anthropic_clamp_resid_SAE_features in the committed
intervention.py) independently at TWO layers simultaneously (layer 3, the
paper's main config, and layer 8, chosen because it has a Gemma Scope SAE
checkpoint with a matching average_l0=142, keeping the two SAEs' sparsity
comparable). Tests whether catching forget-relevant activity at an additional
depth recovers WMDP accuracy that a wider feature set at a single layer could
not (diagnosed bottleneck: feature recall, not clamp strength).
"""
import gc
import os
import sys

import numpy as np
import torch
from functools import partial

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import calculate_MCQ_metrics
from evals.unlearning.utils.feature_activation import get_top_features_percentile, save_feature_sparsity

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
MULTIPLIER = 500
RATIO_PERCENTILE = 95
N_FEATURES = 20

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

# layer 3: paper's main config (already cached)
L3_BLOCK = "layer_3/width_16k/average_l0_142"
L3_SAE_ID = f"{SAE_NAME}_{L3_BLOCK}"
# layer 8: matching average_l0=142 (keeps SAE sparsity comparable across layers)
L8_BLOCK = "layer_8/width_16k/average_l0_142"
L8_SAE_ID = f"{SAE_NAME}_{L8_BLOCK}"

device = "cuda"

print("[1] Loading model + both SAEs...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae_l3, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=L3_BLOCK, device=device)
sae_l8, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=L8_BLOCK, device=device)

print("\n[2] Computing feature sparsity for layer 8 (layer 3 already cached)...")
save_feature_sparsity(
    model, sae_l8, ARTIFACTS_FOLDER, L8_SAE_ID,
    dataset_size=1024, seq_len=1024, batch_size=4, dataset_fraction=100,
    fgt_set="bio-forget-corpus", retain_set="wikitext",
)
gc.collect()
torch.cuda.empty_cache()

SPARSITY_DIR_L3 = os.path.join(ARTIFACTS_FOLDER, L3_SAE_ID, "results", "sparsities")
SPARSITY_DIR_L8 = os.path.join(ARTIFACTS_FOLDER, L8_SAE_ID, "results", "sparsities")

print("\n[3] Deriving feature sets + thresholds independently for each layer...")
fs_fgt_l3 = np.loadtxt(os.path.join(SPARSITY_DIR_L3, "feature_sparsity_forget.txt"), dtype=float)
fs_ret_l3 = np.loadtxt(os.path.join(SPARSITY_DIR_L3, "feature_sparsity_retain.txt"), dtype=float)
sel_l3, perc_l3 = get_top_features_percentile(
    fs_fgt_l3, fs_ret_l3, ratio_percentile=RATIO_PERCENTILE, folder_name=SPARSITY_DIR_L3,
    n_features_lst=[N_FEATURES],
)
features_l3 = sel_l3[:N_FEATURES]
threshold_l3 = perc_l3[str(N_FEATURES)]
print(f"Layer 3: {len(features_l3)} features, threshold={threshold_l3:.4f}")

fs_fgt_l8 = np.loadtxt(os.path.join(SPARSITY_DIR_L8, "feature_sparsity_forget.txt"), dtype=float)
fs_ret_l8 = np.loadtxt(os.path.join(SPARSITY_DIR_L8, "feature_sparsity_retain.txt"), dtype=float)
sel_l8, perc_l8 = get_top_features_percentile(
    fs_fgt_l8, fs_ret_l8, ratio_percentile=RATIO_PERCENTILE, folder_name=SPARSITY_DIR_L8,
    n_features_lst=[N_FEATURES],
)
features_l8 = sel_l8[:N_FEATURES]
threshold_l8 = perc_l8[str(N_FEATURES)]
print(f"Layer 8: {len(features_l8)} features, threshold={threshold_l8:.4f}")


def clamp_hook(resid, hook, sae, features_to_ablate, multiplier, activation_threshold):
    """Exact replica of the validated anthropic_clamp_resid_SAE_features logic
    (committed intervention.py), minus debug prints, parameterized per-layer."""
    feature_activations = sae.encode(resid)
    feature_activations[:, 0, :] = 0.0
    reconstruction = sae.decode(feature_activations)
    error = resid - reconstruction

    target_features = feature_activations[:, :, features_to_ablate]
    activation_mask = (target_features > 0).sum(dim=2) > 0
    batch_activation_rates = activation_mask.sum(dim=1) / activation_mask.shape[1]
    active_batches = batch_activation_rates > activation_threshold

    final_mask = activation_mask.unsqueeze(2) & active_batches.unsqueeze(1).unsqueeze(2)
    feature_activations[:, :, features_to_ablate] = torch.where(
        final_mask, torch.full_like(target_features, -multiplier),
        feature_activations[:, :, features_to_ablate],
    )
    modified_reconstruction = sae.decode(feature_activations)
    return modified_reconstruction + error


hook_l3 = partial(clamp_hook, sae=sae_l3, features_to_ablate=features_l3,
                   multiplier=MULTIPLIER, activation_threshold=threshold_l3)
hook_l8 = partial(clamp_hook, sae=sae_l8, features_to_ablate=features_l8,
                   multiplier=MULTIPLIER, activation_threshold=threshold_l8)


def install_multilayer_hooks():
    model.reset_hooks()
    model.add_hook(sae_l3.cfg.metadata.hook_name, hook_l3)
    model.add_hook(sae_l8.cfg.metadata.hook_name, hook_l8)


print("\n[4] Evaluating single-layer-3-only baseline (sanity check, should match cached 29.368%/99.412%)...")
model.reset_hooks()
model.add_hook(sae_l3.cfg.metadata.hook_name, hook_l3)
single_l3 = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct", split="all")
    single_l3[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_l3 = float(np.mean([single_l3[d] for d in DATASET_NAMES if d != "wmdp-bio"]))
print(f"Layer-3-only: WMDP-bio={single_l3['wmdp-bio']*100:.3f}%  MMLU_avg={mmlu_l3*100:.3f}%")

print("\n[5] Evaluating MULTI-LAYER (layer 3 + layer 8 combined) intervention...")
install_multilayer_hooks()
multi_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct", split="all")
    multi_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_multi = float(np.mean([multi_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n" + "=" * 70)
print("FINAL SUMMARY: multi-layer (layer 3 + layer 8) vs single-layer-3 vs paper baseline")
print("=" * 70)
print(f"{'config':>28} {'WMDP-bio':>12} {'MMLU_avg':>12}")
print(f"{'paper/original cached':>28} {29.368:>11.3f}% {99.412:>11.3f}%")
print(f"{'layer-3-only (this run)':>28} {single_l3['wmdp-bio']*100:>11.3f}% {mmlu_l3*100:>11.3f}%")
print(f"{'layer-3 + layer-8 (multi)':>28} {multi_results['wmdp-bio']*100:>11.3f}% {mmlu_multi*100:>11.3f}%")
print("DONE")
