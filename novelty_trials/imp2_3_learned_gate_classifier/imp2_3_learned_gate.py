"""
Idea 3 (round 2): learned (logistic-regression) input gate, replacing DSG's
hand-picked "fraction of positions where ANY of the top-k features fires,
thresholded at a fixed percentile" rho(x) rule.

User's idea: "currently... can we improve input classifier." DSG's own gate
is not a general classifier -- it's a single hand-designed statistic (OR
across features, at the position level, then a whole-sequence average,
thresholded at a percentile calibrated on retain data). All 20 selected
features are treated identically (a position counts if *any* of them fires,
regardless of which one or how strongly). This tries the natural
improvement: fit a real classifier -- logistic regression -- on top of a
per-sequence summary of the same 20 features' activations (mean activation
per feature across all non-BOS positions), so features that are individually
more discriminative between forget and retain get more weight, instead of an
unweighted OR.

Trained on the same cached layer-3 activations DSG's own gate is calibrated
on (act_fgt.pkl / act_ret.pkl, 275 forget + 275 retain sequences), using
DSG's own top-20 feature selection (so the *feature set* is identical between
the two gates -- this isolates the comparison to gate design, not feature
choice). Calibrated to the same target false-positive rate DSG uses (5% on
the retain distribution) for a fair comparison.
"""
import gc
import json
import os
import pickle
import sys

import numpy as np
import torch
from functools import partial
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import calculate_MCQ_metrics
from evals.unlearning.utils.feature_activation import get_top_features_percentile

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
HOOK_NAME = "blocks.3.hook_resid_post"
MULTIPLIER = 500
N_FEATURES = 20
RATIO_PERCENTILE = 95
TARGET_FPR_PERCENTILE = 95  # same convention DSG uses: threshold = 95th percentile of the retain-side gate score

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SAE_RELEASE_AND_ID = f"{SAE_NAME}_{SAE_BLOCK}"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, SAE_RELEASE_AND_ID, "results", "sparsities")
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

device = "cuda"

print("[1] Deriving DSG's original top-20 feature set (same features, both gates use these)...")
fs_fgt = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
fs_ret = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
sel_orig, perc_orig = get_top_features_percentile(
    fs_fgt, fs_ret, ratio_percentile=RATIO_PERCENTILE, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features = sel_orig[:N_FEATURES].tolist()
threshold_dsg = perc_orig[str(N_FEATURES)]
print(f"  features: {features}")
print(f"  DSG's own calibrated threshold (for reference): {threshold_dsg:.4f}")

print("\n[2] Loading cached layer-3 activations and building per-sequence mean-activation features...")
with open(os.path.join(SPARSITY_DIR, "act_fgt.pkl"), "rb") as f:
    act_fgt = pickle.load(f)
with open(os.path.join(SPARSITY_DIR, "act_ret.pkl"), "rb") as f:
    act_ret = pickle.load(f)

def seq_mean_features(seq, feat_idx):
    # seq: [1, seq_len, d_sae]; exclude BOS (position 0), mean over remaining positions.
    # NB: seq[0, 1:, feat_idx] mixes a slice with a fancy (list) index, which numpy
    # resolves by moving the fancy-indexed axis to the FRONT -> shape (len(feat_idx), seq_len-1),
    # not (seq_len-1, len(feat_idx)) as the naive reading suggests. Index in two steps instead.
    return seq[0, 1:, :][:, feat_idx].mean(axis=0)

X_fgt = np.stack([seq_mean_features(s, features) for s in act_fgt])
X_ret = np.stack([seq_mean_features(s, features) for s in act_ret])
del act_fgt, act_ret
gc.collect()

X = np.concatenate([X_fgt, X_ret], axis=0)
y = np.concatenate([np.ones(len(X_fgt)), np.zeros(len(X_ret))])
print(f"  training set: {X.shape[0]} sequences x {X.shape[1]} features ({len(X_fgt)} forget / {len(X_ret)} retain)")

print("\n[3] Fitting logistic regression gate...")
clf = LogisticRegression(class_weight="balanced", max_iter=2000)
clf.fit(X, y)
train_acc = clf.score(X, y)
print(f"  training accuracy: {train_acc*100:.2f}%")
print(f"  learned weights: {clf.coef_[0].tolist()}")
print(f"  learned bias: {clf.intercept_[0]:.4f}")

retain_scores = clf.decision_function(X_ret)
gate_threshold = float(np.percentile(retain_scores, TARGET_FPR_PERCENTILE))
forget_scores = clf.decision_function(X_fgt)
forget_trigger_rate = float((forget_scores > gate_threshold).mean())
retain_trigger_rate = float((retain_scores > gate_threshold).mean())
print(f"  gate threshold (95th pctile of retain scores): {gate_threshold:.4f}")
print(f"  forget-set trigger rate at this threshold: {forget_trigger_rate*100:.2f}%")
print(f"  retain-set trigger rate at this threshold (should be ~5%): {retain_trigger_rate*100:.2f}%")

W = torch.tensor(clf.coef_[0], dtype=torch.float32, device=device)
B = float(clf.intercept_[0])
GATE_THRESHOLD = gate_threshold


def learned_gate_clamp_hook(resid, hook, sae, features_to_ablate, multiplier):
    """Same clamp mechanics as anthropic_clamp_resid_SAE_features, but the
    per-sequence active/inactive decision comes from the learned logistic
    regression gate (mean activation of the 20 features -> linear score ->
    threshold), not from "fraction of positions where any feature fires"."""
    feature_activations = sae.encode(resid)
    feature_activations[:, 0, :] = 0.0
    reconstruction = sae.decode(feature_activations)
    error = resid - reconstruction

    target_features = feature_activations[:, :, features_to_ablate]  # [batch, seq, k]
    mean_feats = target_features[:, 1:, :].mean(dim=1).float()  # [batch, k], exclude BOS
    score = mean_feats @ W + B  # [batch]
    active_batches = score > GATE_THRESHOLD

    activation_mask = (target_features > 0).sum(dim=2) > 0  # position-level mask, same as DSG (only clamp where the feature actually fires)
    final_mask = activation_mask & active_batches.unsqueeze(1)
    feature_activations[:, :, features_to_ablate] = torch.where(
        final_mask.unsqueeze(2), torch.full_like(target_features, -multiplier),
        feature_activations[:, :, features_to_ablate],
    )
    modified_reconstruction = sae.decode(feature_activations)
    return modified_reconstruction + error


print("\n[4] Loading model + SAE for evaluation...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

print("\n[5] Evaluating learned-gate intervention...")
hook = partial(learned_gate_clamp_hook, sae=sae, features_to_ablate=features, multiplier=MULTIPLIER)
model.reset_hooks()
model.add_hook(HOOK_NAME, hook)
learned_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct",
                               split="all", verbose=False)
    learned_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_learned = float(np.mean([learned_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n" + "=" * 70)
print("FINAL SUMMARY: learned logistic-regression gate vs DSG's original OR-threshold gate")
print("=" * 70)
print(f"{'config':>32} {'WMDP-bio':>12} {'MMLU_avg':>12}")
print(f"{'DSG original gate (cached)':>32} {29.368:>11.3f}% {99.412:>11.3f}%")
print(f"{'learned logistic-regr. gate':>32} {learned_results['wmdp-bio']*100:>11.3f}% {mmlu_learned*100:>11.3f}%")

out = {
    "features": features,
    "gate_threshold": GATE_THRESHOLD,
    "train_acc": train_acc,
    "forget_trigger_rate_on_calibration_set": forget_trigger_rate,
    "retain_trigger_rate_on_calibration_set": retain_trigger_rate,
    "learned_results": learned_results,
    "mmlu_learned": mmlu_learned,
}
out_path = "novelty_trials/imp2_3_learned_gate_classifier/imp2_3_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved:", out_path)
print("DONE")
