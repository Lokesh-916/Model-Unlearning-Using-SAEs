"""
Idea 4 (round 2): per-feature variable clamp values, replacing DSG's single
scalar multiplier=-500 applied identically to every selected feature.

User's idea: "we are currently clamping feature values with -500 ... is
there any other (variable value clamping for each feature instead of one
single value clamping to every feature) way." Prior branches (C3's residual
projection, A2's calibrated mean-ablation) already showed that *gentler*
replacements for the clamp value collapse forgetting almost entirely -- the
extreme magnitude is doing real, necessary work, not just being a crude
hack. This experiment keeps that lesson in mind: instead of making clamping
gentler, it keeps the same *total* suppression budget (the per-feature
clamp values average out to -500, matching the baseline exactly) but
redistributes it according to each feature's own natural activation range on
the forget corpus, on the hypothesis that DSG's flat -500 is proportionally
far more extreme for features with a small natural range (max activation
~10) than for features with a large one (max activation ~48) -- a 5x
disparity in how many "natural units" of suppression each feature actually
gets, purely as a side effect of using one shared constant.

clamp_j = -500 * (max_forget_activation_j / mean_j(max_forget_activation_j))

Reuses the cached layer-3 act_fgt.pkl (to compute each feature's max
forget-corpus activation) and DSG's own top-20 feature selection + threshold
(so the feature set and gate are identical to the baseline -- only the
per-feature clamp *magnitude* changes).
"""
import gc
import json
import os
import pickle
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
HOOK_NAME = "blocks.3.hook_resid_post"
BASE_MULTIPLIER = 500
N_FEATURES = 20
RATIO_PERCENTILE = 95

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SAE_RELEASE_AND_ID = f"{SAE_NAME}_{SAE_BLOCK}"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, SAE_RELEASE_AND_ID, "results", "sparsities")
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

device = "cuda"

print("[1] Deriving DSG's original top-20 feature set (same features, only clamp magnitude changes)...")
fs_fgt = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
fs_ret = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
sel_orig, perc_orig = get_top_features_percentile(
    fs_fgt, fs_ret, ratio_percentile=RATIO_PERCENTILE, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features = sel_orig[:N_FEATURES].tolist()
gate_threshold = perc_orig[str(N_FEATURES)]
print(f"  features: {features}")
print(f"  gate threshold (same as baseline): {gate_threshold:.4f}")

print("\n[2] Computing each feature's max forget-corpus activation from cached act_fgt.pkl...")
with open(os.path.join(SPARSITY_DIR, "act_fgt.pkl"), "rb") as f:
    act_fgt = pickle.load(f)
per_seq_max = np.stack([seq[0][:, features].max(axis=0) for seq in act_fgt])  # [n_seq, n_features]
max_forget_activation = per_seq_max.max(axis=0)  # [n_features]
del act_fgt
gc.collect()
print(f"  per-feature max activation: {max_forget_activation.tolist()}")

mean_max = float(max_forget_activation.mean())
per_feature_multiplier = BASE_MULTIPLIER * (max_forget_activation / mean_max)
print(f"  mean of per-feature max activations: {mean_max:.3f}")
print(f"  per-feature clamp multipliers (mean should be {BASE_MULTIPLIER}): {per_feature_multiplier.tolist()}")
print(f"  mean of assigned multipliers: {per_feature_multiplier.mean():.3f}, "
      f"min={per_feature_multiplier.min():.1f}, max={per_feature_multiplier.max():.1f}")

per_feature_multiplier_t = torch.tensor(per_feature_multiplier, dtype=torch.float32, device=device)


def per_feature_clamp_hook(resid, hook, sae, features_to_ablate, per_feature_mult, activation_threshold):
    """Identical mechanics to anthropic_clamp_resid_SAE_features, except the
    clamp value is a per-feature vector (-per_feature_mult[j] for feature j)
    instead of one shared scalar multiplier."""
    feature_activations = sae.encode(resid)
    feature_activations[:, 0, :] = 0.0
    reconstruction = sae.decode(feature_activations)
    error = resid - reconstruction

    target_features = feature_activations[:, :, features_to_ablate]
    activation_mask = (target_features > 0).sum(dim=2) > 0
    batch_activation_rates = activation_mask.sum(dim=1) / activation_mask.shape[1]
    active_batches = batch_activation_rates > activation_threshold

    final_mask = activation_mask.unsqueeze(2) & active_batches.unsqueeze(1).unsqueeze(2)
    clamp_values = -per_feature_mult.view(1, 1, -1).expand_as(target_features)
    feature_activations[:, :, features_to_ablate] = torch.where(
        final_mask, clamp_values, feature_activations[:, :, features_to_ablate]
    )
    modified_reconstruction = sae.decode(feature_activations)
    return modified_reconstruction + error


print("\n[3] Loading model + SAE for evaluation...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

print("\n[4] Evaluating per-feature variable clamp intervention...")
hook = partial(per_feature_clamp_hook, sae=sae, features_to_ablate=features,
               per_feature_mult=per_feature_multiplier_t, activation_threshold=gate_threshold)
model.reset_hooks()
model.add_hook(HOOK_NAME, hook)
varclamp_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct",
                               split="all", verbose=False)
    varclamp_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_varclamp = float(np.mean([varclamp_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n" + "=" * 70)
print("FINAL SUMMARY: per-feature variable clamp vs DSG's flat -500 clamp")
print("=" * 70)
print(f"{'config':>32} {'WMDP-bio':>12} {'MMLU_avg':>12}")
print(f"{'DSG flat -500 clamp (cached)':>32} {29.368:>11.3f}% {99.412:>11.3f}%")
print(f"{'per-feature variable clamp':>32} {varclamp_results['wmdp-bio']*100:>11.3f}% {mmlu_varclamp*100:>11.3f}%")

out = {
    "features": features,
    "max_forget_activation": max_forget_activation.tolist(),
    "per_feature_multiplier": per_feature_multiplier.tolist(),
    "varclamp_results": varclamp_results,
    "mmlu_varclamp": mmlu_varclamp,
}
out_path = "novelty_trials/imp2_4_per_feature_clamp/imp2_4_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved:", out_path)
print("DONE")
