"""
C3: replace "clamp the selected latents" with "project the whole residual
stream (including the SAE's own reconstruction error) out of the subspace
spanned by an expanded, redundancy-clustered feature family's decoder
directions." Then measure whether either intervention actually erases the
forget-relevant information, via linear probes trained on post-intervention
residual activations to predict the correct WMDP-bio answer -- a metric DSG
never reports.
"""
import gc
import json
import os
import sys

import numpy as np
import torch
from functools import partial
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import convert_wmdp_data_to_prompt, calculate_MCQ_metrics
from evals.unlearning.utils.feature_activation import get_top_features_percentile
from evals.unlearning.utils.var import PRE_WMDP_BIO
from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
HOOK_NAME = "blocks.3.hook_resid_post"
MULTIPLIER = 500
N_FEATURES = 20
COSINE_THRESHOLD = 0.15
MAX_EXPANDED = 100
N_PROBE_QUESTIONS = 300
SEED = 0

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SAE_RELEASE_AND_ID = f"{SAE_NAME}_{SAE_BLOCK}"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, SAE_RELEASE_AND_ID, "results", "sparsities")
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

device = "cuda"
np.random.seed(SEED)

print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)
model.requires_grad_(False)
sae.requires_grad_(False)

print("\n[2] Re-deriving DSG's original feature set + threshold (the gate stays the same; only the")
print("    intervention mechanism changes)...")
fs_fgt = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
fs_ret = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
sel_orig, perc_orig = get_top_features_percentile(
    fs_fgt, fs_ret, ratio_percentile=95, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features_orig = sel_orig[:N_FEATURES]
threshold_orig = perc_orig[str(N_FEATURES)]
print(f"Original: {len(features_orig)} features, threshold={threshold_orig:.4f}")

print("\n[3] Expanding into a feature family via decoder-cosine clustering...")
W_dec = sae.W_dec.detach().float()  # [d_sae, d_model]
W_dec_norm = W_dec / (W_dec.norm(dim=1, keepdim=True) + 1e-12)
seed_norm = W_dec_norm[features_orig]  # [20, d_model]
sims = seed_norm @ W_dec_norm.T  # [20, d_sae]
max_sim_to_seed = sims.abs().max(dim=0).values  # [d_sae]
expanded_candidates = torch.where(max_sim_to_seed > COSINE_THRESHOLD)[0].cpu().numpy()
expanded_features = np.union1d(features_orig, expanded_candidates)
if len(expanded_features) > MAX_EXPANDED:
    scores = max_sim_to_seed[expanded_features].cpu().numpy()
    top = np.argsort(-scores)[:MAX_EXPANDED]
    expanded_features = expanded_features[top]
print(f"Expanded feature family: {len(expanded_features)} features "
      f"(from {len(features_orig)} seed features, cosine>{COSINE_THRESHOLD})")

print("\n[4] Building the orthonormal projection basis (QR of expanded decoder vectors)...")
decoder_subset = W_dec[expanded_features]  # [n_expanded, d_model]
Q, _ = torch.linalg.qr(decoder_subset.T)  # [d_model, min(n_expanded, d_model)]
U = Q  # orthonormal basis spanning (approximately) the expanded decoder subspace

print("\n[5] Computing a reference mean residual (mu_ref) from WikiText retain tokens...")
wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
wiki_texts = [x["text"] for x in wiki if len(x["text"]) > 200][:100]
resid_sum = torch.zeros(model.cfg.d_model, device=device, dtype=torch.float32)
n_tok = 0
model.reset_hooks()
with torch.no_grad():
    for t in wiki_texts:
        tokens = model.to_tokens(t, prepend_bos=True)[:, :256]
        _, cache = model.run_with_cache(tokens, stop_at_layer=4, names_filter=HOOK_NAME)
        r = cache[HOOK_NAME][0, 1:].float()  # exclude BOS
        resid_sum += r.sum(dim=0)
        n_tok += r.shape[0]
mu_ref = (resid_sum / n_tok)
print(f"mu_ref computed from {n_tok} WikiText tokens, norm={mu_ref.norm().item():.3f}")


def projection_clamp_hook(resid, hook, sae, gate_features, gate_threshold, U, mu_ref):
    """Same whole-sequence rho(x) gate as the original DSG hook, but the
    intervention is a full-residual projection (h <- h - U U^T (h - mu_ref)),
    applied to every token whose gate fires -- not just a clamp on 20 SAE
    latents, so it also removes whatever the SAE's own reconstruction error
    was carrying."""
    feature_activations = sae.encode(resid)
    gate_target = feature_activations[:, :, gate_features]
    activation_mask = (gate_target > 0).sum(dim=2) > 0
    batch_activation_rates = activation_mask.sum(dim=1) / activation_mask.shape[1]
    active_batches = batch_activation_rates > gate_threshold
    final_mask = activation_mask & active_batches.unsqueeze(1)  # [batch, seq]

    centered = resid - mu_ref
    proj = torch.einsum("bsd,de->bse", centered, U)  # [batch, seq, k]
    reconstructed = torch.einsum("bse,de->bsd", proj, U)  # [batch, seq, d_model]
    projected_resid = resid - reconstructed  # h - U U^T (h - mu_ref)

    out = torch.where(final_mask.unsqueeze(2), projected_resid, resid)
    return out


hook_c3 = partial(projection_clamp_hook, sae=sae, gate_features=features_orig,
                   gate_threshold=threshold_orig, U=U, mu_ref=mu_ref)

print("\n[6] Evaluating C3 (residual-projection intervention) on WMDP-bio + MMLU...")
model.reset_hooks()
model.add_hook(HOOK_NAME, hook_c3)
c3_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct", split="all")
    c3_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_c3 = float(np.mean([c3_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n[7] Collecting residual activations for linear-probe leak measurement...")
wmdp_ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
probe_idx = np.random.choice(len(wmdp_ds), min(N_PROBE_QUESTIONS, len(wmdp_ds)), replace=False)

hook_dsg_clamp = partial(anthropic_clamp_resid_SAE_features, sae=sae, features_to_ablate=features_orig.tolist(),
                          multiplier=MULTIPLIER, activation_threshold=threshold_orig)


PROBE_LAYER = 22
PROBE_HOOK_NAME = f"blocks.{PROBE_LAYER}.hook_resid_post"


def collect_residuals(hook_fn):
    """Captures residuals at PROBE_LAYER (late in the network, where the
    correct-answer identity should actually be linearly decodable), while the
    intervention (if any) still runs at layer 3 (HOOK_NAME) upstream. A first
    version of this probed at layer 3 itself, where even the *unintervened*
    model's probe accuracy came out at chance (24.44%, vs 25% chance) --
    that's not evidence of no leak, it's evidence layer 3 is too early in the
    network to have "decided" the answer yet, making the whole comparison
    uninformative. Probing downstream of the intervention, at a layer where
    the base model's own accuracy is clearly above chance, is the version of
    this test that can actually detect a leak."""
    X, y = [], []
    for i in probe_idx:
        ex = wmdp_ds[int(i)]
        prompt = convert_wmdp_data_to_prompt(
            ex["question"], ex["choices"], prompt_format="GEMMA_INST_FORMAT", without_question=False,
            pre_question=PRE_WMDP_BIO,
        )
        tokens = model.to_tokens(prompt, padding_side="right", prepend_bos=False)
        model.reset_hooks()
        if hook_fn is not None:
            model.add_hook(HOOK_NAME, hook_fn)
        with torch.no_grad():
            _, cache = model.run_with_cache(tokens, stop_at_layer=PROBE_LAYER + 1, names_filter=PROBE_HOOK_NAME)
        r = cache[PROBE_HOOK_NAME][0, -1].float().cpu().numpy()
        X.append(r)
        y.append(int(ex["answer"]))
    model.reset_hooks()
    return np.array(X), np.array(y)


print("  collecting: no intervention (upper bound)...")
X_base, y_base = collect_residuals(None)
print("  collecting: DSG original clamp...")
X_dsg, y_dsg = collect_residuals(hook_dsg_clamp)
print("  collecting: C3 residual projection...")
X_c3, y_c3 = collect_residuals(hook_c3)


def probe_accuracy(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)
    clf = LogisticRegression(max_iter=2000)
    clf.fit(Xtr, ytr)
    return clf.score(Xte, yte)


print("\n[8] Training linear probes to predict the correct answer from residual activations...")
acc_base = probe_accuracy(X_base, y_base)
acc_dsg = probe_accuracy(X_dsg, y_dsg)
acc_c3 = probe_accuracy(X_c3, y_c3)
print(f"  no intervention (upper bound): probe accuracy = {acc_base*100:.2f}%")
print(f"  DSG original clamp:            probe accuracy = {acc_dsg*100:.2f}%")
print(f"  C3 residual projection:        probe accuracy = {acc_c3*100:.2f}%")
print(f"  chance level (4-way): 25.00%")

out = {
    "expanded_feature_count": int(len(expanded_features)),
    "c3_results": c3_results,
    "mmlu_c3": mmlu_c3,
    "probe_accuracy": {"no_intervention": acc_base, "dsg_clamp": acc_dsg, "c3_projection": acc_c3, "chance": 0.25},
}
out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_c3_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("FINAL SUMMARY: C3 residual projection + linear-probe leak measurement")
print("=" * 70)
print(f"{'config':>28} {'WMDP-bio':>12} {'MMLU_avg':>12}")
print(f"{'DSG original (cached)':>28} {29.368:>11.3f}% {99.412:>11.3f}%")
print(f"{'C3 residual projection':>28} {c3_results['wmdp-bio']*100:>11.3f}% {mmlu_c3*100:>11.3f}%")
print(f"\nLinear-probe accuracy predicting the correct WMDP-bio answer from residual activations:")
print(f"  no intervention: {acc_base*100:.2f}%   DSG clamp: {acc_dsg*100:.2f}%   "
      f"C3 projection: {acc_c3*100:.2f}%   chance: 25.00%")
print("\nSaved:", out_path)
print("DONE")
