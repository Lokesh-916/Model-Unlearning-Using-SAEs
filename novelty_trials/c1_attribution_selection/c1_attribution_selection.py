"""
C1: replace DSG's Fisher-info-style feature ranking (average squared
activation, forget/retain ratio) with attribution-patching scores
(activation x gradient of the correct-answer log-prob), then greedy-select
a redundancy-aware feature set (penalizing decoder-vector cosine overlap).
Compares the resulting WMDP-bio/MMLU accuracy at equal feature count (20)
against the paper's original selection.
"""
import gc
import json
import os
import random
import sys

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import (
    convert_wmdp_data_to_prompt, calculate_MCQ_metrics,
)
from evals.unlearning.utils.feature_activation import get_top_features_percentile
from evals.unlearning.utils.var import PRE_WMDP_BIO, PRE_QUESTION_FORMAT
from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features
from functools import partial

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
HOOK_NAME = "blocks.3.hook_resid_post"
MULTIPLIER = 500
N_FEATURES = 20
N_FORGET_SAMPLES = 150
N_RETAIN_SAMPLES = 150
SEED = 0

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SAE_RELEASE_AND_ID = f"{SAE_NAME}_{SAE_BLOCK}"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, SAE_RELEASE_AND_ID, "results", "sparsities")
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]
RETAIN_DATASET_NAMES = DATASET_NAMES[1:]

random.seed(SEED)
device = "cuda"

print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)
d_sae = sae.cfg.d_sae

# Critical: freeze every parameter so backward() only builds a graph node (and
# allocates a .grad tensor) for the one intermediate activation we actually
# need gradients for. Without this, every backward() call allocates gradient
# tensors for all ~2.6B model parameters too, which OOMs the GPU almost
# immediately (this was the actual bug in the first run of this script: only
# 1/150 forget samples and 0/150 retain samples succeeded before every
# subsequent call OOM'd, so that run's "result" was noise from one sample).
model.requires_grad_(False)
sae.requires_grad_(False)

answer_tokens = model.to_tokens([" A", " B", " C", " D"], prepend_bos=False).flatten()


def attribution_hook(resid, hook, captured):
    feature_activations = sae.encode(resid)
    # resid/sae params are frozen, so encode() output has requires_grad=False
    # by default; explicitly mark it as a leaf requiring grad so autograd
    # tracks *only* this tensor onward, not any frozen parameter.
    feature_activations.requires_grad_(True)
    captured["feats"] = feature_activations
    reconstruction = sae.decode(feature_activations)
    error = (resid - reconstruction).detach()
    return reconstruction + error


def compute_attribution(prompt, correct_answer_idx):
    """Returns attribution[d_sae] = f_j * d(log p(correct answer))/d(f_j),
    summed over all non-BOS token positions."""
    model.reset_hooks()
    captured = {}
    model.add_hook(HOOK_NAME, partial(attribution_hook, captured=captured))

    tokens = model.to_tokens(prompt, padding_side="right", prepend_bos=False)
    with torch.set_grad_enabled(True):
        resid = model(tokens, return_type=None, stop_at_layer=model.cfg.n_layers)
        resid = model.ln_final(resid)
        last_resid = resid[0, -1]  # [d_model]
        logits = model.unembed(last_resid.unsqueeze(0).unsqueeze(0))[0, 0]  # [d_vocab]
        softcap = getattr(model.cfg, "output_logits_soft_cap", None)
        if softcap is not None and softcap > 0:
            logits = softcap * torch.tanh(logits / softcap)
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        target_log_prob = log_probs[answer_tokens[correct_answer_idx]]
        model.zero_grad(set_to_none=True)
        target_log_prob.backward()

    feats = captured["feats"].detach()[0]  # [seq, d_sae]
    grad = captured["feats"].grad.detach()[0]  # [seq, d_sae]
    attribution = (feats[1:] * grad[1:]).sum(dim=0).float().cpu().numpy()  # exclude BOS position
    model.reset_hooks()
    del captured, feats, grad, resid, logits, log_probs, target_log_prob, tokens
    return attribution


print("\n[2] Computing attribution scores on WMDP-bio (forget) questions...")
wmdp_ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
forget_idx = random.sample(range(len(wmdp_ds)), min(N_FORGET_SAMPLES, len(wmdp_ds)))
forget_attribution = np.zeros(d_sae, dtype=np.float64)
n_ok = 0
for i in tqdm(forget_idx):
    ex = wmdp_ds[i]
    prompt = convert_wmdp_data_to_prompt(
        ex["question"], ex["choices"], prompt_format="GEMMA_INST_FORMAT", without_question=False,
        pre_question=PRE_WMDP_BIO,
    )
    try:
        forget_attribution += compute_attribution(prompt, int(ex["answer"]))
        n_ok += 1
    except Exception as e:
        print(f"  skip (forget) idx={i}: {e}")
forget_attribution /= max(n_ok, 1)
print(f"  used {n_ok}/{len(forget_idx)} forget questions")

print("\n[3] Computing attribution scores on MMLU retain questions...")
retain_attribution = np.zeros(d_sae, dtype=np.float64)
n_ok = 0
per_subject = N_RETAIN_SAMPLES // len(RETAIN_DATASET_NAMES)
for subject in RETAIN_DATASET_NAMES:
    ds = load_dataset("cais/mmlu", subject, split="test")
    idxs = random.sample(range(len(ds)), min(per_subject, len(ds)))
    pre_q = PRE_QUESTION_FORMAT.format(subject=subject.replace("_", " "))
    for i in tqdm(idxs, desc=subject):
        ex = ds[i]
        prompt = convert_wmdp_data_to_prompt(
            ex["question"], ex["choices"], prompt_format="GEMMA_INST_FORMAT", without_question=False,
            pre_question=pre_q,
        )
        try:
            retain_attribution += compute_attribution(prompt, int(ex["answer"]))
            n_ok += 1
        except Exception as e:
            print(f"  skip (retain) {subject} idx={i}: {e}")
retain_attribution /= max(n_ok, 1)
print(f"  used {n_ok} retain questions across {len(RETAIN_DATASET_NAMES)} subjects")

del model
gc.collect()
torch.cuda.empty_cache()
print("\n[4] Reloading model fresh (post-backward-pass cleanup) for evaluation...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
model.reset_hooks()

print("\n[5] Redundancy-aware greedy selection...")
eps = 1e-12
ratio_score = forget_attribution / (np.abs(retain_attribution) + eps)
# only consider features with positive forget attribution (i.e. genuinely helps predict
# the correct forget-domain answer when active) as candidates
candidates = np.where(forget_attribution > 0)[0]
candidates = candidates[np.argsort(-ratio_score[candidates])][:500]  # top-500 pool for greedy search

W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]
W_dec_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + eps)

selected = []
selected_decoders = []
remaining = list(candidates)
for _ in range(N_FEATURES):
    best_j, best_score = None, -np.inf
    for j in remaining:
        base = forget_attribution[j]
        if selected_decoders:
            sims = torch.stack([torch.abs(torch.dot(W_dec_norm[j], d)) for d in selected_decoders])
            penalty = 1.0 - sims.max().item()
        else:
            penalty = 1.0
        score = base * penalty
        if score > best_score:
            best_score, best_j = score, j
    selected.append(int(best_j))
    selected_decoders.append(W_dec_norm[best_j])
    remaining.remove(best_j)

features_c1 = np.array(selected)
print(f"C1 attribution-selected features: {features_c1.tolist()}")

print("\n[6] Calibrating whole-sequence threshold for the C1 feature set (reusing cached act_ret.pkl)...")
# get_top_features_percentile computes the same rho(x)-percentile threshold for an
# arbitrary feature list if we pass a folder containing act_fgt.pkl/act_ret.pkl --
# reuse the layer-3 cached activations directly here instead (cheaper, no re-selection).
import pickle
with open(os.path.join(SPARSITY_DIR, "act_ret.pkl"), "rb") as f:
    act_ret = pickle.load(f)
distrib = []
for el in act_ret:
    buff = el[:, :, features_c1] > 0
    buff2 = (buff.sum(axis=2) > 0)
    distrib.append(buff2.sum() / buff2.shape[1])
del act_ret
gc.collect()
distrib = np.asarray(distrib)
threshold_c1 = float(np.percentile(distrib, 95))
print(f"C1 threshold: {threshold_c1:.4f}")

print("\n[7] Evaluating C1 (attribution-selected) feature set...")
hook_c1 = partial(anthropic_clamp_resid_SAE_features, sae=sae, features_to_ablate=features_c1.tolist(),
                   multiplier=MULTIPLIER, activation_threshold=threshold_c1)
model.reset_hooks()
model.add_hook(HOOK_NAME, hook_c1)
c1_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct", split="all")
    c1_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_c1 = float(np.mean([c1_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n[8] Re-deriving DSG's original (Fisher-info-style) feature set for a direct comparison...")
fs_fgt = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
fs_ret = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
sel_orig, perc_orig = get_top_features_percentile(
    fs_fgt, fs_ret, ratio_percentile=95, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features_orig = sel_orig[:N_FEATURES]
threshold_orig = perc_orig[str(N_FEATURES)]
overlap = len(set(features_orig.tolist()) & set(features_c1.tolist()))
print(f"Overlap between DSG's top-20 and C1's attribution top-20: {overlap}/20")

out = {
    "features_c1": features_c1.tolist(),
    "threshold_c1": threshold_c1,
    "features_dsg_original": features_orig.tolist(),
    "threshold_dsg_original": float(threshold_orig),
    "overlap_count": overlap,
    "c1_results": c1_results,
    "mmlu_c1": mmlu_c1,
}
out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_c1_attribution_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("FINAL SUMMARY: C1 attribution-based selection vs DSG's original selection")
print("=" * 70)
print(f"{'config':>28} {'WMDP-bio':>12} {'MMLU_avg':>12}")
print(f"{'DSG original (cached)':>28} {29.368:>11.3f}% {99.412:>11.3f}%")
print(f"{'C1 attribution-selected':>28} {c1_results['wmdp-bio']*100:>11.3f}% {mmlu_c1*100:>11.3f}%")
print(f"Feature overlap with DSG's top-20: {overlap}/20")
print("\nSaved:", out_path)
print("DONE")
