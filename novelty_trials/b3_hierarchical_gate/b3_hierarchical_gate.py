"""
B3: hierarchical gate protecting neighboring knowledge.

Kill test: does DSG's single gate (trained/calibrated against generic WikiText
as the only retain signal) collapse accuracy on BENIGN same-domain content
(college biology, virology, anatomy) that it was never checked against, even
though it holds up fine on the paper's own unrelated retain set (history, CS,
geography, aging)?

Fix: a two-level gate. A domain-general detector ("is this biology at all?")
is calibrated against WikiText, same as before. A hazard-specific detector
("is this DANGEROUS biology, as opposed to benign biology?") is calibrated
contrastively against a benign-biology/medicine corpus instead of WikiText.
The clamp intervention only touches the hazard-specific features, and only
fires when BOTH gates are active.
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

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import calculate_MCQ_metrics
from evals.unlearning.utils.feature_activation import get_top_features_percentile, get_feature_activation_sparsity
from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features
import dsg_utils.dataset_utils as dataset_utils

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
HOOK_NAME = "blocks.3.hook_resid_post"
MULTIPLIER = 500
N_FEATURES = 20
BENIGN_BIO_SEQ_LEN = 256

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SAE_RELEASE_AND_ID = f"{SAE_NAME}_{SAE_BLOCK}"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, SAE_RELEASE_AND_ID, "results", "sparsities")
ORIGINAL_RETAIN_NAMES = ["high_school_us_history", "college_computer_science",
                          "high_school_geography", "human_aging"]
NEIGHBOR_NAMES = ["college_biology", "virology", "anatomy"]
BENIGN_BIO_SUBJECTS = ["college_biology", "virology", "anatomy", "clinical_knowledge",
                        "medical_genetics", "professional_medicine", "college_medicine"]

device = "cuda"

print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

print("\n[2] Re-deriving DSG's original single-gate feature set + threshold...")
fs_fgt = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
fs_ret = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
sel_orig, perc_orig = get_top_features_percentile(
    fs_fgt, fs_ret, ratio_percentile=95, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features_orig = sel_orig[:N_FEATURES]
threshold_orig = perc_orig[str(N_FEATURES)]
print(f"Original: {len(features_orig)} features, threshold={threshold_orig:.4f}")

hook_orig = partial(anthropic_clamp_resid_SAE_features, sae=sae, features_to_ablate=features_orig.tolist(),
                     multiplier=MULTIPLIER, activation_threshold=threshold_orig)

print("\n[3] KILL TEST: evaluating the single-gate baseline on neighboring benign-biology subjects...")
model.reset_hooks()
model.add_hook(HOOK_NAME, hook_orig)
kill_test = {}
for dataset_name in ["wmdp-bio"] + ORIGINAL_RETAIN_NAMES + NEIGHBOR_NAMES:
    # neighbor subjects have no pre-generated "correct"-subset question-id file
    # (that file only exists for the original pipeline's datasets), so evaluate
    # on the full test set for them instead.
    tm = None if dataset_name in NEIGHBOR_NAMES else "correct"
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric=tm, split="all")
    kill_test[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()

print("\n[4] Building benign-biology/medicine corpus for the hazard-specific contrastive gate...")
benign_texts = []
for subj in BENIGN_BIO_SUBJECTS:
    ds = load_dataset("cais/mmlu", subj, split="test")
    for ex in ds:
        choices_str = " ".join(ex["choices"])
        benign_texts.append(f"{ex['question']} {choices_str}")
print(f"  {len(benign_texts)} benign-bio question texts across {len(BENIGN_BIO_SUBJECTS)} subjects")

benign_tokens = dataset_utils.tokenize_and_concat_dataset(
    model.tokenizer, benign_texts, seq_len=BENIGN_BIO_SEQ_LEN,
).to(device)
print(f"  benign-bio tokens shape: {benign_tokens.shape}")

print("\n[5] Collecting SAE activations on the benign-bio corpus...")
feature_sparsity_benign, act_benign = get_feature_activation_sparsity(
    benign_tokens, model, sae, batch_size=4, layer=3, hook_name=HOOK_NAME, mask_bos_pad_eos_tokens=True,
)
feature_sparsity_benign = feature_sparsity_benign.cpu().numpy()
os.makedirs(os.path.join(SPARSITY_DIR, "..", "benign_bio"), exist_ok=True)
np.savetxt(os.path.join(SPARSITY_DIR, "feature_sparsity_benign_bio.txt"), feature_sparsity_benign, fmt="%f")

print("\n[6] Selecting domain-general features (forget UNION benign-bio, vs WikiText)...")
domain_positive_score = np.maximum(fs_fgt, feature_sparsity_benign)
sel_domain, perc_domain = get_top_features_percentile(
    domain_positive_score, fs_ret, ratio_percentile=95, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features_domain = sel_domain[:N_FEATURES]
threshold_domain = perc_domain[str(N_FEATURES)]  # calibrated on WikiText, same as before
print(f"Domain-general: {len(features_domain)} features, threshold(on WikiText)={threshold_domain:.4f}")

print("\n[7] Selecting hazard-specific features (forget vs benign-bio, contrastive)...")
# reuse get_top_features_percentile's selection logic, but the *threshold* it returns
# is calibrated against WikiText's act_ret.pkl (hardcoded inside that function) -- we
# only want its feature *selection*, then calibrate our own threshold against the
# benign-bio corpus separately below.
sel_hazard, _ = get_top_features_percentile(
    fs_fgt, feature_sparsity_benign, ratio_percentile=95, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features_hazard = sel_hazard[:N_FEATURES]

distrib = []
for el in act_benign:
    buff = el[:, :, features_hazard] > 0
    buff2 = (buff.sum(axis=2) > 0)  # [batch, seq]
    # one rate PER SEQUENCE, not summed across the whole batch dimension --
    # get_top_features_percentile's version of this snippet implicitly assumes
    # batch=1 per element (matching the "bs1" cached corpora elsewhere in this
    # repo); this benign-bio collection used batch_size=4, so summing over the
    # batch dim before dividing by only one sequence's length silently produced
    # rates above 1.0 (impossible for a true fraction), which is exactly what
    # broke the calibration (threshold=1.1937, an unreachable value -> gate
    # never fires).
    for b in range(buff2.shape[0]):
        distrib.append(buff2[b].sum() / buff2.shape[1])
distrib = np.asarray(distrib)
threshold_hazard = float(np.percentile(distrib, 95))
print(f"Hazard-specific: {len(features_hazard)} features, threshold(on benign-bio)={threshold_hazard:.4f}")

overlap = len(set(features_domain.tolist()) & set(features_hazard.tolist()))
print(f"Overlap between domain-general and hazard-specific feature sets: {overlap}/20")

del act_benign
gc.collect()
torch.cuda.empty_cache()


def hierarchical_clamp_hook(resid, hook, sae, domain_features, hazard_features,
                              domain_threshold, hazard_threshold, multiplier):
    feature_activations = sae.encode(resid)
    feature_activations[:, 0, :] = 0.0
    reconstruction = sae.decode(feature_activations)
    error = resid - reconstruction

    domain_target = feature_activations[:, :, domain_features]
    domain_mask = (domain_target > 0).sum(dim=2) > 0
    domain_rate = domain_mask.sum(dim=1) / domain_mask.shape[1]
    domain_active = domain_rate > domain_threshold

    hazard_target = feature_activations[:, :, hazard_features]
    hazard_mask = (hazard_target > 0).sum(dim=2) > 0
    hazard_rate = hazard_mask.sum(dim=1) / hazard_mask.shape[1]
    hazard_active = hazard_rate > hazard_threshold

    both_active = domain_active & hazard_active
    final_mask = hazard_mask.unsqueeze(2) & both_active.unsqueeze(1).unsqueeze(2)

    feature_activations[:, :, hazard_features] = torch.where(
        final_mask, torch.full_like(hazard_target, -multiplier),
        feature_activations[:, :, hazard_features],
    )
    modified_reconstruction = sae.decode(feature_activations)
    return modified_reconstruction + error


hook_b3 = partial(hierarchical_clamp_hook, sae=sae, domain_features=features_domain,
                   hazard_features=features_hazard, domain_threshold=threshold_domain,
                   hazard_threshold=threshold_hazard, multiplier=MULTIPLIER)

print("\n[8] Evaluating the B3 hierarchical two-gate intervention...")
model.reset_hooks()
model.add_hook(HOOK_NAME, hook_b3)
b3_results = {}
for dataset_name in ["wmdp-bio"] + ORIGINAL_RETAIN_NAMES + NEIGHBOR_NAMES:
    tm = None if dataset_name in NEIGHBOR_NAMES else "correct"
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric=tm, split="all")
    b3_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()

out = {
    "kill_test_single_gate": kill_test,
    "b3_hierarchical_gate": b3_results,
    "features_domain": features_domain.tolist(),
    "threshold_domain": threshold_domain,
    "features_hazard": features_hazard.tolist(),
    "threshold_hazard": threshold_hazard,
    "domain_hazard_overlap": overlap,
}
out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_b3_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("FINAL SUMMARY: B3 hierarchical gate vs single-gate baseline (kill test)")
print("=" * 70)
print(f"{'dataset':>28} {'single-gate (kill test)':>25} {'B3 hierarchical':>18}")
for d in ["wmdp-bio"] + ORIGINAL_RETAIN_NAMES + NEIGHBOR_NAMES:
    print(f"{d:>28} {kill_test[d]*100:>24.3f}% {b3_results[d]*100:>17.3f}%")
print(f"\nDomain-general threshold (WikiText-calibrated): {threshold_domain:.4f}")
print(f"Hazard-specific threshold (benign-bio-calibrated): {threshold_hazard:.4f}")
print(f"Domain/hazard feature overlap: {overlap}/20")
print("\nSaved:", out_path)
print("DONE")
