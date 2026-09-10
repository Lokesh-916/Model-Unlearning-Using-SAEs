import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import modify_and_calculate_metrics

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
MULTIPLIER = 500
N_FEATS = 20
FORGET_PERCENTILE = 5      # paper default, not varied here
RETAIN_PERCENTILE = 100    # paper default, not varied here

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, "gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities")
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

P_RATIO_SWEEP = [75, 80, 85, 90, 95]
P_DYN_SWEEP = [60, 70, 80, 90, 95, 97]

device = "cuda"


def select_features(forget_sparsity, retain_sparsity, ratio_percentile, n_feats=N_FEATS):
    forget_score = forget_sparsity ** 2
    retain_score = retain_sparsity ** 2
    importance_ratio = forget_score / (retain_score + 1e-21)

    forget_threshold = np.percentile(forget_score, FORGET_PERCENTILE)
    retain_threshold = np.percentile(retain_score, RETAIN_PERCENTILE)
    ratio_threshold = np.percentile(importance_ratio, ratio_percentile)

    selected = np.where(
        (forget_score >= forget_threshold)
        & (retain_score <= retain_threshold)
        & (importance_ratio >= ratio_threshold)
    )[0]
    sel_ind_out = selected[np.argsort(-forget_score[selected])]
    return sel_ind_out[:n_feats]


def compute_threshold(act_ret, features, p_dyn):
    distrib = []
    for el in act_ret:
        buff = el[:, :, features] > 0
        buff2 = buff.sum(axis=2) > 0
        distrib.append(buff2.sum() / buff2.shape[1])
    distrib = np.asarray(distrib)
    return float(np.percentile(distrib, p_dyn)), distrib


print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

print("[2] Loading cached Bio forget/retain sparsity (unchanged, main-experiment data)...")
forget_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
retain_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)

print("[3] Loading cached act_ret.pkl (original WikiText activations, ~18GB, once)...")
with open(os.path.join(SPARSITY_DIR, "act_ret.pkl"), "rb") as f:
    act_ret = pickle.load(f)
print(f"Loaded {len(act_ret)} retain batches")


def run_eval(features, threshold, label):
    metrics = modify_and_calculate_metrics(
        model, mcq_batch_size=1, artifacts_folder=ARTIFACTS_FOLDER, sae=sae,
        dataset_names=DATASET_NAMES, intervention_method="clamp_feature_activation",
        features_to_ablate=features, multiplier=MULTIPLIER, activation_threshold=threshold,
    )
    row = {d: metrics[d]["mean_correct"] for d in DATASET_NAMES}
    mmlu_avg = float(np.mean([row[d] for d in DATASET_NAMES if d != "wmdp-bio"]))
    print(f">>> {label}: WMDP-bio={row['wmdp-bio']*100:.3f}%  MMLU_avg={mmlu_avg*100:.3f}%")
    return {"wmdp-bio": row["wmdp-bio"], "mmlu_avg": mmlu_avg}


results = {"p_ratio_sweep": {}, "p_dyn_sweep": {}}

print("\n" + "=" * 70 + "\nP_RATIO SWEEP (p_dyn fixed at 95)\n" + "=" * 70)
for p_ratio in P_RATIO_SWEEP:
    feats = select_features(forget_sparsity, retain_sparsity, p_ratio)
    threshold, _ = compute_threshold(act_ret, feats, 95)
    print(f"\np_ratio={p_ratio}: n_features_selected={len(feats)}  threshold={threshold:.4f}")
    r = run_eval(feats, threshold, f"p_ratio={p_ratio}")
    r["n_features_selected"] = int(len(feats))
    r["threshold"] = threshold
    results["p_ratio_sweep"][str(p_ratio)] = r

print("\n" + "=" * 70 + "\nP_DYN SWEEP (p_ratio fixed at 95, main feature set)\n" + "=" * 70)
main_feats = select_features(forget_sparsity, retain_sparsity, 95)
print(f"Main features (p_ratio=95): {main_feats.tolist()}")
_, distrib = compute_threshold(act_ret, main_feats, 95)
for p_dyn in P_DYN_SWEEP:
    threshold = float(np.percentile(distrib, p_dyn))
    print(f"\np_dyn={p_dyn}: threshold={threshold:.4f}")
    r = run_eval(main_feats, threshold, f"p_dyn={p_dyn}")
    r["threshold"] = threshold
    results["p_dyn_sweep"][str(p_dyn)] = r

out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_threshold_ablations_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70 + "\nSUMMARY: p_ratio sweep\n" + "=" * 70)
print(f"{'p_ratio':>8} {'n_feats':>8} {'WMDP-bio':>10} {'MMLU avg':>10}")
for p_ratio in P_RATIO_SWEEP:
    r = results["p_ratio_sweep"][str(p_ratio)]
    print(f"{p_ratio:>8} {r['n_features_selected']:>8} {r['wmdp-bio']*100:>9.3f}% {r['mmlu_avg']*100:>9.3f}%")

print("\n" + "=" * 70 + "\nSUMMARY: p_dyn sweep\n" + "=" * 70)
print(f"{'p_dyn':>8} {'WMDP-bio':>10} {'MMLU avg':>10}")
for p_dyn in P_DYN_SWEEP:
    r = results["p_dyn_sweep"][str(p_dyn)]
    print(f"{p_dyn:>8} {r['wmdp-bio']*100:>9.3f}% {r['mmlu_avg']*100:>9.3f}%")

print("\nSaved:", out_path)
print("DONE")
