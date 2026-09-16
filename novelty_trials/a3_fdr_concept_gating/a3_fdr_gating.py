"""
A3 (scaled down from the spec's 50 concepts to 4, given this hardware/data):
a bank of per-concept detectors instead of one merged gate, with Bonferroni
multiple-testing control across the active detectors so the combined retain
false-positive rate stays bounded as more concepts are added, instead of
growing with each addition (DSG-union's failure mode).

Concepts: wmdp-bio, wmdp-cyber (both already have cached forget/retain SAE
activations from the main reproduction), plus two MMLU subjects built the
same way as B3's benign-bio corpus (high_school_us_history,
college_computer_science). Two untouched MMLU subjects (high_school_geography,
human_aging) serve as holdout retain checks -- never used as a forget target,
purely to measure false-positive damage.

Compares: naive union (each detector calibrated independently at the plain
95th percentile, DSG's original per-detector convention) vs Bonferroni-
corrected union (each calibrated at the 100*(1-alpha/n) percentile so the
combined false-alarm rate across all n detectors stays <= alpha).
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
from evals.unlearning.utils.feature_activation import get_feature_activation_sparsity
import dsg_utils.dataset_utils as dataset_utils

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
HOOK_NAME = "blocks.3.hook_resid_post"
MULTIPLIER = 500
N_FEATURES_PER_CONCEPT = 10
ALPHA = 0.05
MMLU_CORPUS_SEQ_LEN = 256

BIO_ARTIFACTS = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
BIO_SPARSITY_DIR = os.path.join(BIO_ARTIFACTS, "gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities")
CYBER_ARTIFACTS = "artifacts_dynamic_bs1_cyber/unlearning/gemma-2-2b-it"
CYBER_SPARSITY_DIR = os.path.join(CYBER_ARTIFACTS, "gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities")

CONCEPT_MMLU_SUBJECTS = ["high_school_us_history", "college_computer_science"]
HOLDOUT_RETAIN_SUBJECTS = ["high_school_geography", "human_aging"]

device = "cuda"

print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

print("\n[2] Loading shared WikiText retain reference (reusing Bio's cached act_ret.pkl)...")
with open(os.path.join(BIO_SPARSITY_DIR, "act_ret.pkl"), "rb") as f:
    act_ret_wikitext = pickle.load(f)
fs_ret_wikitext = np.loadtxt(os.path.join(BIO_SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)


def select_features(forget_score_sq, retain_score_sq, n_features, ratio_percentile=95):
    ratio = forget_score_sq / (retain_score_sq + 1e-21)
    forget_thr = np.percentile(forget_score_sq, 5)
    ratio_thr = np.percentile(ratio, ratio_percentile)
    candidates = np.where((forget_score_sq >= forget_thr) & (ratio >= ratio_thr))[0]
    candidates = candidates[np.argsort(-forget_score_sq[candidates])]
    return candidates[:n_features]


def calibrate_threshold(features, act_ret_list, calibration_percentile):
    distrib = []
    for el in act_ret_list:
        buff = el[:, :, features] > 0
        buff2 = (buff.sum(axis=2) > 0)
        for b in range(buff2.shape[0]):
            distrib.append(buff2[b].sum() / buff2.shape[1])
    distrib = np.asarray(distrib)
    return float(np.percentile(distrib, calibration_percentile))


print("\n[3] Building concept detectors...")
concepts = {}

# wmdp-bio (cached)
fs_fgt_bio = np.loadtxt(os.path.join(BIO_SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
concepts["wmdp-bio"] = {"forget_score": fs_fgt_bio ** 2, "eval_dataset": "wmdp-bio"}

# wmdp-cyber (cached)
fs_fgt_cyber = np.loadtxt(os.path.join(CYBER_SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
concepts["wmdp-cyber"] = {"forget_score": fs_fgt_cyber ** 2, "eval_dataset": "wmdp-cyber"}

# MMLU-subject concepts: build a small forget corpus from the subject's own question text
for subj in CONCEPT_MMLU_SUBJECTS:
    ds = load_dataset("cais/mmlu", subj, split="test")
    texts = [f"{ex['question']} {' '.join(ex['choices'])}" for ex in ds]
    tokens = dataset_utils.tokenize_and_concat_dataset(model.tokenizer, texts, seq_len=MMLU_CORPUS_SEQ_LEN).to(device)
    sparsity, _ = get_feature_activation_sparsity(
        tokens, model, sae, batch_size=4, layer=3, hook_name=HOOK_NAME, mask_bos_pad_eos_tokens=True,
    )
    concepts[subj] = {"forget_score": sparsity.cpu().numpy() ** 2, "eval_dataset": subj}
    print(f"  built forget corpus + sparsity for concept '{subj}' ({tokens.shape[0]} sequences)")

retain_score_sq = fs_ret_wikitext ** 2
for name, c in concepts.items():
    c["features"] = select_features(c["forget_score"], retain_score_sq, N_FEATURES_PER_CONCEPT, ratio_percentile=95)
    print(f"  {name}: {len(c['features'])} features selected")

n_concepts = len(concepts)
bonferroni_percentile = 100 * (1 - ALPHA / n_concepts)
print(f"\nNaive per-detector calibration percentile: 95.0")
print(f"Bonferroni-corrected per-detector calibration percentile (n={n_concepts}, alpha={ALPHA}): "
      f"{bonferroni_percentile:.4f}")

print("\n[4] Calibrating thresholds (naive 95th percentile, and Bonferroni-corrected)...")
for name, c in concepts.items():
    c["threshold_naive"] = calibrate_threshold(c["features"], act_ret_wikitext, 95)
    c["threshold_bonferroni"] = calibrate_threshold(c["features"], act_ret_wikitext, bonferroni_percentile)
    print(f"  {name}: naive={c['threshold_naive']:.4f}  bonferroni={c['threshold_bonferroni']:.4f}")

del act_ret_wikitext
gc.collect()
torch.cuda.empty_cache()


def multi_concept_hook(resid, hook, sae, concepts, threshold_key, multiplier):
    feature_activations = sae.encode(resid)
    feature_activations[:, 0, :] = 0.0
    reconstruction = sae.decode(feature_activations)
    error = resid - reconstruction

    for name, c in concepts.items():
        feats = c["features"]
        thr = c[threshold_key]
        target = feature_activations[:, :, feats]
        activation_mask = (target > 0).sum(dim=2) > 0
        rates = activation_mask.sum(dim=1) / activation_mask.shape[1]
        active = rates > thr
        final_mask = activation_mask.unsqueeze(2) & active.unsqueeze(1).unsqueeze(2)
        feature_activations[:, :, feats] = torch.where(
            final_mask, torch.full_like(target, -multiplier), feature_activations[:, :, feats],
        )
    modified_reconstruction = sae.decode(feature_activations)
    return modified_reconstruction + error


def evaluate(threshold_key):
    hook_fn = partial(multi_concept_hook, sae=sae, concepts=concepts, threshold_key=threshold_key, multiplier=MULTIPLIER)
    model.reset_hooks()
    model.add_hook(HOOK_NAME, hook_fn)
    results = {}
    all_datasets = list(dict.fromkeys([c["eval_dataset"] for c in concepts.values()] + HOLDOUT_RETAIN_SUBJECTS))
    for dataset_name in all_datasets:
        tm = "correct" if dataset_name in ("wmdp-bio", "wmdp-cyber") or dataset_name in HOLDOUT_RETAIN_SUBJECTS else None
        artifacts = CYBER_ARTIFACTS if dataset_name == "wmdp-cyber" else BIO_ARTIFACTS
        try:
            m = calculate_MCQ_metrics(model, 1, artifacts, dataset_name=dataset_name, target_metric=tm, split="all")
        except FileNotFoundError:
            m = calculate_MCQ_metrics(model, 1, artifacts, dataset_name=dataset_name, target_metric=None, split="all")
        results[dataset_name] = m["mean_correct"]
        print(f"    {dataset_name}: {m['mean_correct']*100:.3f}%")
    model.reset_hooks()
    return results


print("\n[5] Evaluating NAIVE union (each detector at plain 95th percentile)...")
naive_results = evaluate("threshold_naive")

print("\n[6] Evaluating BONFERRONI-corrected union...")
bonferroni_results = evaluate("threshold_bonferroni")

out = {
    "n_concepts": n_concepts,
    "bonferroni_percentile": bonferroni_percentile,
    "concept_thresholds": {name: {"naive": c["threshold_naive"], "bonferroni": c["threshold_bonferroni"]}
                             for name, c in concepts.items()},
    "naive_results": naive_results,
    "bonferroni_results": bonferroni_results,
}
out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_a3_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("FINAL SUMMARY: A3 naive union vs Bonferroni-corrected union (4 concepts)")
print("=" * 70)
print(f"{'dataset':>28} {'naive union':>14} {'bonferroni union':>18}")
for d in list(dict.fromkeys([c["eval_dataset"] for c in concepts.values()] + HOLDOUT_RETAIN_SUBJECTS)):
    print(f"{d:>28} {naive_results[d]*100:>13.3f}% {bonferroni_results[d]*100:>17.3f}%")
print("\nSaved:", out_path)
print("DONE")
