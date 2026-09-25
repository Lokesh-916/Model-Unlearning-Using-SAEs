"""
Idea 2 (round 2): AND-gated multi-layer intervention.

User's idea: "currently we are using sae at layer 3 can we use that any other
layer or multiple layers so that improvements will come." Round 1
(`imp-c2-multilayer-circuit`) already tried adding a second layer's gate
(layer 8) but applied each layer's whole-sequence-average gate INDEPENDENTLY
(OR semantics: clamp fires at layer 3 if layer 3's own rho exceeds its
threshold, regardless of layer 8, and vice versa). That compounded each
layer's own retain false-positive rate: WMDP-bio improved 29.4% -> 27.7%, but
MMLU dropped 99.4% -> 97.9% -- a real Pareto trade-off.

This tries the fix that diagnosis suggests: gate on the CONJUNCTION of both
layers' triggers (AND semantics) instead of the union. A forget-domain input
should activate forget-relevant features at BOTH layer 3 (earlier/more
lexical) and layer 8 (deeper/more semantic); requiring agreement between two
independent depths should be harder to false-trigger on ordinary retain text
than either gate alone, while still catching forget-domain content both
layers agree on.

`calculate_MCQ_metrics` is called with `mcq_batch_size=1` (as in every other
script in this repo), so it scores exactly one question per forward pass,
sequentially, in a single deterministic hook-call order (layer 3's hook
always fires before layer 8's, since layer 3 precedes layer 8 in the
network). This script exploits that: pass 1 installs read-only hooks that
just record each layer's own per-question trigger; pass 2 installs clamp
hooks that read back the pre-computed joint (AND) decision for the
corresponding question index and only clamp when both layers agreed.

Reuses round 1's cached layer-3 and layer-8 feature sparsity files (both
already computed by the main run and by imp-c2), so this script does no new
sparsity computation -- only two evaluation forward passes per dataset.
"""
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
MULTIPLIER = 500
RATIO_PERCENTILE = 95
N_FEATURES = 20

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

L3_BLOCK = "layer_3/width_16k/average_l0_142"
L8_BLOCK = "layer_8/width_16k/average_l0_142"
L3_SAE_ID = f"{SAE_NAME}_{L3_BLOCK}"
L8_SAE_ID = f"{SAE_NAME}_{L8_BLOCK}"

device = "cuda"

print("[1] Loading model + both SAEs...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae_l3, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=L3_BLOCK, device=device)
sae_l8, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=L8_BLOCK, device=device)
HOOK_L3 = sae_l3.cfg.metadata.hook_name
HOOK_L8 = sae_l8.cfg.metadata.hook_name

print("\n[2] Deriving feature sets + thresholds independently for each layer (cached sparsities)...")
SPARSITY_DIR_L3 = os.path.join(ARTIFACTS_FOLDER, L3_SAE_ID, "results", "sparsities")
SPARSITY_DIR_L8 = os.path.join(ARTIFACTS_FOLDER, L8_SAE_ID, "results", "sparsities")

fs_fgt_l3 = np.loadtxt(os.path.join(SPARSITY_DIR_L3, "feature_sparsity_forget.txt"), dtype=float)
fs_ret_l3 = np.loadtxt(os.path.join(SPARSITY_DIR_L3, "feature_sparsity_retain.txt"), dtype=float)
sel_l3, perc_l3 = get_top_features_percentile(
    fs_fgt_l3, fs_ret_l3, ratio_percentile=RATIO_PERCENTILE, folder_name=SPARSITY_DIR_L3,
    n_features_lst=[N_FEATURES],
)
features_l3 = sel_l3[:N_FEATURES].tolist()
threshold_l3 = perc_l3[str(N_FEATURES)]
print(f"Layer 3: threshold={threshold_l3:.4f}")

fs_fgt_l8 = np.loadtxt(os.path.join(SPARSITY_DIR_L8, "feature_sparsity_forget.txt"), dtype=float)
fs_ret_l8 = np.loadtxt(os.path.join(SPARSITY_DIR_L8, "feature_sparsity_retain.txt"), dtype=float)
sel_l8, perc_l8 = get_top_features_percentile(
    fs_fgt_l8, fs_ret_l8, ratio_percentile=RATIO_PERCENTILE, folder_name=SPARSITY_DIR_L8,
    n_features_lst=[N_FEATURES],
)
features_l8 = sel_l8[:N_FEATURES].tolist()
threshold_l8 = perc_l8[str(N_FEATURES)]
print(f"Layer 8: threshold={threshold_l8:.4f}")


def batch_active(resid, sae, features_to_ablate, activation_threshold):
    feature_activations = sae.encode(resid)
    feature_activations[:, 0, :] = 0.0
    target_features = feature_activations[:, :, features_to_ablate]
    activation_mask = (target_features > 0).sum(dim=2) > 0
    rate = activation_mask.sum(dim=1) / activation_mask.shape[1]
    return rate > activation_threshold, activation_mask


def record_hook(resid, hook, sae, features_to_ablate, activation_threshold, record_list):
    active, _ = batch_active(resid, sae, features_to_ablate, activation_threshold)
    record_list.append(active.detach().cpu())
    return resid


def and_gated_clamp_hook(resid, hook, sae, features_to_ablate, multiplier, joint_list, counter):
    _, activation_mask = batch_active(resid, sae, features_to_ablate, 0.0)  # threshold unused here; we use joint_list
    joint_active = joint_list[counter[0]].to(resid.device)
    feature_activations = sae.encode(resid)
    feature_activations[:, 0, :] = 0.0
    reconstruction = sae.decode(feature_activations)
    error = resid - reconstruction

    target_features = feature_activations[:, :, features_to_ablate]
    final_mask = activation_mask & joint_active.unsqueeze(1)
    feature_activations[:, :, features_to_ablate] = torch.where(
        final_mask.unsqueeze(2), torch.full_like(target_features, -multiplier),
        feature_activations[:, :, features_to_ablate],
    )
    modified_reconstruction = sae.decode(feature_activations)
    return modified_reconstruction + error


def evaluate_dataset_and_gated(dataset_name):
    # Pass 1: record each layer's own trigger, per question, unmodified residual stream.
    l3_record, l8_record = [], []
    model.reset_hooks()
    model.add_hook(HOOK_L3, partial(record_hook, sae=sae_l3, features_to_ablate=features_l3,
                                     activation_threshold=threshold_l3, record_list=l3_record))
    model.add_hook(HOOK_L8, partial(record_hook, sae=sae_l8, features_to_ablate=features_l8,
                                     activation_threshold=threshold_l8, record_list=l8_record))
    calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct",
                           split="all", verbose=False)
    model.reset_hooks()
    assert len(l3_record) == len(l8_record), "hook call count mismatch between layers"
    joint_list = [l3.bool() & l8.bool() for l3, l8 in zip(l3_record, l8_record)]

    # Pass 2: clamp both layers, gated on the precomputed joint (AND) decision.
    # Layer 3's hook fires first per question and only reads counter (doesn't advance it);
    # layer 8's hook fires second and advances the shared counter after use.
    counter = [0]

    def l3_hook(resid, hook):
        return and_gated_clamp_hook(resid, hook, sae_l3, features_l3, MULTIPLIER, joint_list, counter)

    def l8_hook(resid, hook):
        out = and_gated_clamp_hook(resid, hook, sae_l8, features_l8, MULTIPLIER, joint_list, counter)
        counter[0] += 1
        return out

    model.reset_hooks()
    model.add_hook(HOOK_L3, l3_hook)
    model.add_hook(HOOK_L8, l8_hook)
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct",
                               split="all", verbose=False)
    model.reset_hooks()
    assert counter[0] == len(joint_list), "pass-2 hook call count mismatch with pass-1"
    n_joint_active = sum(bool(j.item()) for j in joint_list)
    return m["mean_correct"], n_joint_active, len(joint_list)


print("\n[3] Evaluating AND-gated multi-layer (layer 3 AND layer 8) intervention...")
and_results = {}
and_trigger_rates = {}
for dataset_name in DATASET_NAMES:
    acc, n_active, n_total = evaluate_dataset_and_gated(dataset_name)
    and_results[dataset_name] = acc
    and_trigger_rates[dataset_name] = (n_active, n_total)
    print(f"  {dataset_name}: {acc*100:.3f}%  (joint-gate fired on {n_active}/{n_total} questions)")

mmlu_and = float(np.mean([and_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n" + "=" * 70)
print("FINAL SUMMARY: AND-gated multi-layer vs OR-gated multi-layer (imp-c2) vs single-layer-3 baseline")
print("=" * 70)
print(f"{'config':>32} {'WMDP-bio':>12} {'MMLU_avg':>12}")
print(f"{'single-layer-3 (cached baseline)':>32} {29.368:>11.3f}% {99.412:>11.3f}%")
print(f"{'OR-gated L3+L8 (imp-c2)':>32} {27.695:>11.3f}% {97.941:>11.3f}%")
print(f"{'AND-gated L3+L8 (this script)':>32} {and_results['wmdp-bio']*100:>11.3f}% {mmlu_and*100:>11.3f}%")
print("DONE")

import json
out = {
    "and_results": and_results,
    "mmlu_and": mmlu_and,
    "and_trigger_rates": and_trigger_rates,
    "threshold_l3": float(threshold_l3),
    "threshold_l8": float(threshold_l8),
}
with open("novelty_trials/imp2_2_and_gated_multilayer/imp2_2_results.json", "w") as f:
    json.dump(out, f, indent=2)
print("Saved: novelty_trials/imp2_2_and_gated_multilayer/imp2_2_results.json")
