"""
Idea 1 (round 2): replace DSG's Fisher-info-style feature ranking (mean squared
activation, forget/retain ratio, computed over raw per-token activation
magnitude) with a document-level co-occurrence statistic: for each SAE
feature, does it fire (activation > 0 at any token) more often in forget-corpus
sequences than in retain-corpus sequences, measured with a chi-squared test of
independence on the 2x2 (fires-in-sequence x forget-or-retain) contingency
table.

This targets a different failure mode than C1 (round 1)'s attribution-patching
attempt. C1's attribution score is a *signed, single-token, single-question*
quantity (d log p(correct answer) / d feature) that can cancel out across
questions and measures causal relevance to one MCQ letter rather than general
topical presence. Chi-squared co-occurrence, like DSG's own method, is a
*non-negative, whole-sequence* presence statistic -- it stays closer to what
DSG's method measures (does this feature carry forget-domain content at all)
while asking a different statistical question: not "how large is the
activation, on average" (sensitive to a few very high outlier activations
dominating the mean) but "how consistently distinctive is firing-at-all
between the two corpora" (a document-frequency / association-strength view,
robust to activation-magnitude outliers).

Reuses the already-cached layer-3 activations (act_fgt.pkl / act_ret.pkl,
feature_sparsity_{forget,retain}.txt) from the main DSG run, so this script
does no forward passes over the forget/retain corpora -- only the final
WMDP-bio + MMLU evaluation requires the model.
"""
import gc
import json
import os
import pickle
import sys

import numpy as np
from functools import partial

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

import torch
from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import calculate_MCQ_metrics
from evals.unlearning.utils.feature_activation import get_top_features_percentile
from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
HOOK_NAME = "blocks.3.hook_resid_post"
MULTIPLIER = 500
N_FEATURES = 20
FORGET_RATE_PERCENTILE = 95   # mirrors DSG's forget_percentile=5 (keep only top 5% forget-firing features)
RATIO_PERCENTILE = 90         # mirrors DSG's ratio_percentile constraint (forget/retain-rate ratio)

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SAE_RELEASE_AND_ID = f"{SAE_NAME}_{SAE_BLOCK}"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, SAE_RELEASE_AND_ID, "results", "sparsities")
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

device = "cuda"

print("[1] Loading cached layer-3 activations...")
with open(os.path.join(SPARSITY_DIR, "act_fgt.pkl"), "rb") as f:
    act_fgt = pickle.load(f)  # list of [1, seq, d_sae]
with open(os.path.join(SPARSITY_DIR, "act_ret.pkl"), "rb") as f:
    act_ret = pickle.load(f)
d_sae = act_fgt[0].shape[-1]
print(f"  {len(act_fgt)} forget sequences, {len(act_ret)} retain sequences, d_sae={d_sae}")

print("\n[2] Building per-sequence firing matrix and chi-squared association scores...")
# fires[i, j] = 1 if feature j fires (activation > 0) anywhere in sequence i
fires_fgt = np.stack([(seq[0] > 0).any(axis=0) for seq in act_fgt])  # [n_fgt, d_sae]
fires_ret = np.stack([(seq[0] > 0).any(axis=0) for seq in act_ret])  # [n_ret, d_sae]
del act_fgt, act_ret
gc.collect()

n_fgt = fires_fgt.shape[0]
n_ret = fires_ret.shape[0]
n_total = n_fgt + n_ret

a = fires_fgt.sum(axis=0).astype(np.float64)          # fires in forget
b = n_fgt - a                                          # doesn't fire in forget
c = fires_ret.sum(axis=0).astype(np.float64)          # fires in retain
d = n_ret - c                                          # doesn't fire in retain

# chi-squared statistic for each 2x2 table, signed positive only when the
# feature fires disproportionately more in forget than retain (excludes
# retain-skewed features from ever ranking highly, mirroring DSG's own
# forget-vs-retain asymmetry).
eps = 1e-12
expected_a = (a + b) * (a + c) / n_total
expected_b = (a + b) * (b + d) / n_total
expected_c = (c + d) * (a + c) / n_total
expected_d = (c + d) * (b + d) / n_total
chi2 = (
    (a - expected_a) ** 2 / (expected_a + eps)
    + (b - expected_b) ** 2 / (expected_b + eps)
    + (c - expected_c) ** 2 / (expected_c + eps)
    + (d - expected_d) ** 2 / (expected_d + eps)
)
forget_rate = a / n_fgt
retain_rate = c / n_ret
chi2_signed = np.where(forget_rate > retain_rate, chi2, 0.0)

print("\n[3] Selecting top features: chi-squared co-occurrence score, filtered by the same two-constraint")
print("    structure DSG itself uses (top forget-percentile AND top ratio-percentile), just computed on")
print("    firing rates instead of squared activation magnitudes...")
fs_fgt = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
fs_ret = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
ratio_rate = forget_rate / (retain_rate + eps)
forget_rate_threshold = np.percentile(forget_rate, FORGET_RATE_PERCENTILE)
ratio_threshold = np.percentile(ratio_rate, RATIO_PERCENTILE)
eligible = np.where((forget_rate >= forget_rate_threshold) & (ratio_rate >= ratio_threshold))[0]
ranked_eligible = eligible[np.argsort(-chi2_signed[eligible])]
features_new = ranked_eligible[:N_FEATURES]
print(f"  candidates after forget-percentile + ratio-percentile filter: {len(eligible)}/{d_sae}")
print(f"  selected features: {features_new.tolist()}")
print(f"  their chi2 scores: {chi2_signed[features_new].tolist()}")
print(f"  their forget firing rate: {forget_rate[features_new].tolist()}")
print(f"  their retain firing rate: {retain_rate[features_new].tolist()}")

print("\n[4] Calibrating whole-sequence rho(x) threshold for the new feature set (reusing cached act_ret.pkl)...")
with open(os.path.join(SPARSITY_DIR, "act_ret.pkl"), "rb") as f:
    act_ret_full = pickle.load(f)
distrib = []
for el in act_ret_full:
    buff = el[:, :, features_new] > 0
    buff2 = (buff.sum(axis=2) > 0)
    distrib.append(buff2.sum() / buff2.shape[1])
del act_ret_full
gc.collect()
distrib = np.asarray(distrib)
threshold_new = float(np.percentile(distrib, 95))
print(f"  threshold: {threshold_new:.4f}")

print("\n[5] Loading model for evaluation...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
model.reset_hooks()

print("\n[6] Evaluating new (co-occurrence-selected) feature set...")
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)
hook_new = partial(anthropic_clamp_resid_SAE_features, sae=sae, features_to_ablate=features_new.tolist(),
                    multiplier=MULTIPLIER, activation_threshold=threshold_new)
model.add_hook(HOOK_NAME, hook_new)
new_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct", split="all")
    new_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_new = float(np.mean([new_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n[7] Re-deriving DSG's original (Fisher-info-style) feature set for a direct comparison...")
sel_orig, perc_orig = get_top_features_percentile(
    fs_fgt, fs_ret, ratio_percentile=95, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features_orig = sel_orig[:N_FEATURES]
threshold_orig = perc_orig[str(N_FEATURES)]
overlap = len(set(features_orig.tolist()) & set(features_new.tolist()))
print(f"Overlap between DSG's top-20 and co-occurrence top-20: {overlap}/20")

out = {
    "features_new": features_new.tolist(),
    "threshold_new": threshold_new,
    "features_dsg_original": features_orig.tolist(),
    "threshold_dsg_original": float(threshold_orig),
    "overlap_count": overlap,
    "new_results": new_results,
    "mmlu_new": mmlu_new,
}
out_path = "novelty_trials/imp2_1_cooccurrence_feature_selection/imp2_1_results_v2.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("FINAL SUMMARY: chi-squared co-occurrence selection vs DSG's original selection")
print("=" * 70)
print(f"{'config':>28} {'WMDP-bio':>12} {'MMLU_avg':>12}")
print(f"{'DSG original (cached)':>28} {29.368:>11.3f}% {99.412:>11.3f}%")
print(f"{'Co-occurrence-selected':>28} {new_results['wmdp-bio']*100:>11.3f}% {mmlu_new*100:>11.3f}%")
print(f"Feature overlap with DSG's top-20: {overlap}/20")
print("\nSaved:", out_path)
print("DONE")
