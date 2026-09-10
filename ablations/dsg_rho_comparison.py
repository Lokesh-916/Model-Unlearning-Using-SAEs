import json
import os
import pickle
import sys

import numpy as np
import torch
from scipy.stats import gaussian_kde
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import convert_wmdp_data_to_prompt
from dsg_utils.activation_collection import get_bos_pad_eos_mask

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
LAYER = 3
HOOK_NAME = "blocks.3.hook_resid_post"

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, "gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities")

FORGET_PERCENTILE = 5
RETAIN_PERCENTILE = 100
RATIO_PERCENTILE = 95
N_FEATS = 20

MMLU_SUBJECTS = ["high_school_us_history", "college_computer_science", "high_school_geography", "human_aging"]
N_PER_MMLU_SUBJECT = 40
N_WMDP = 160
N_BOOTSTRAP = 1000

device = "cuda"


def select_features(forget_sparsity, retain_sparsity, ratio_percentile, n_feats=N_FEATS):
    forget_score = forget_sparsity ** 2
    retain_score = retain_sparsity ** 2
    importance_ratio = forget_score / (retain_score + 1e-21)
    forget_threshold = np.percentile(forget_score, FORGET_PERCENTILE)
    retain_threshold = np.percentile(retain_score, RETAIN_PERCENTILE)
    ratio_threshold = np.percentile(importance_ratio, ratio_percentile)
    selected = np.where(
        (forget_score >= forget_threshold) & (retain_score <= retain_threshold) & (importance_ratio >= ratio_threshold)
    )[0]
    sel_ind_out = selected[np.argsort(-forget_score[selected])]
    return sel_ind_out[:n_feats]


print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

print("[2] Selecting main Bio feature set (p_ratio=95, n_feats=20)...")
forget_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
retain_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
FEATURES = select_features(forget_sparsity, retain_sparsity, RATIO_PERCENTILE)
print("Features:", FEATURES.tolist())


def rho_and_rho_raw_for_prompt(prompt):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, stop_at_layer=LAYER + 1, names_filter=HOOK_NAME)
        resid = cache[HOOK_NAME]
        feats = sae.encode(resid)  # [1, seq, d_sae]
    mask = get_bos_pad_eos_mask(tokens, model.tokenizer).to(feats.device)  # [1, seq]
    target = feats[:, :, FEATURES] > 0          # [1, seq, n_feats]
    any_active = (target.any(dim=2)) & mask      # [1, seq]
    valid_len = mask.sum().item()
    rho = any_active.sum().item() / valid_len if valid_len > 0 else 0.0
    rho_raw = any_active.sum().item()
    return rho, rho_raw


print("\n[3] Computing rho/rho_raw for WikiText (reusing cached act_ret.pkl, no forward passes needed)...")
with open(os.path.join(SPARSITY_DIR, "act_ret.pkl"), "rb") as f:
    act_ret = pickle.load(f)
wiki_rho, wiki_rho_raw = [], []
for el in act_ret:
    buff = el[:, :, FEATURES] > 0
    any_active = buff.sum(axis=2) > 0   # [batch, seq] -- BOS already zeroed at collection time
    seq_len = any_active.shape[1]
    for b in range(any_active.shape[0]):
        cnt = int(any_active[b].sum())
        wiki_rho.append(cnt / seq_len)
        wiki_rho_raw.append(cnt)
del act_ret
print(f"WikiText sequences: {len(wiki_rho)}")

print("\n[4] Computing rho/rho_raw for MMLU (pooled across Bio's 4 MMLU subjects)...")
mmlu_rho, mmlu_rho_raw = [], []
for subject in MMLU_SUBJECTS:
    ds = load_dataset("cais/mmlu", subject, split="test")
    n = min(N_PER_MMLU_SUBJECT, len(ds))
    subject_readable = subject.replace("_", " ")
    for i in tqdm(range(n), desc=subject):
        ex = ds[i]
        prompt = convert_wmdp_data_to_prompt(
            ex["question"], ex["choices"], prompt_format="GEMMA_INST_FORMAT", without_question=False,
            pre_question=f"The following are multiple choice questions (with answers) about {subject_readable}.\n\n",
        )
        rho, rho_raw = rho_and_rho_raw_for_prompt(prompt)
        mmlu_rho.append(rho)
        mmlu_rho_raw.append(rho_raw)
print(f"MMLU sequences: {len(mmlu_rho)}")

print("\n[5] Computing rho/rho_raw for WMDP-Bio (eval question set)...")
wmdp_ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
n_wmdp = min(N_WMDP, len(wmdp_ds))
wmdp_rho, wmdp_rho_raw = [], []
for i in tqdm(range(n_wmdp), desc="wmdp-bio"):
    ex = wmdp_ds[i]
    prompt = convert_wmdp_data_to_prompt(
        ex["question"], ex["choices"], prompt_format="GEMMA_INST_FORMAT", without_question=False,
        pre_question="The following are multiple choice questions (with answers) about biology.\n\n",
    )
    rho, rho_raw = rho_and_rho_raw_for_prompt(prompt)
    wmdp_rho.append(rho)
    wmdp_rho_raw.append(rho_raw)
print(f"WMDP-Bio sequences: {len(wmdp_rho)}")


def tvd_kde(sample_a, sample_b, grid_points=400):
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    lo = min(a.min(), b.min())
    hi = max(a.max(), b.max())
    if hi <= lo:
        return 0.0
    grid = np.linspace(lo, hi, grid_points)
    try:
        kde_a = gaussian_kde(a)(grid)
        kde_b = gaussian_kde(b)(grid)
    except np.linalg.LinAlgError:
        # degenerate (near-zero variance) sample; fall back to a fine histogram
        bins = np.linspace(lo, hi, grid_points)
        kde_a, _ = np.histogram(a, bins=bins, density=True)
        kde_b, _ = np.histogram(b, bins=bins, density=True)
        grid = bins[:-1]
    dx = grid[1] - grid[0]
    kde_a = kde_a / (kde_a.sum() * dx)
    kde_b = kde_b / (kde_b.sum() * dx)
    return 0.5 * np.sum(np.abs(kde_a - kde_b)) * dx


def bootstrap_tvd(sample_a, sample_b, n_boot=N_BOOTSTRAP, seed=0):
    rng = np.random.default_rng(seed)
    a = np.asarray(sample_a, dtype=float)
    b = np.asarray(sample_b, dtype=float)
    vals = []
    for _ in range(n_boot):
        ra = rng.choice(a, size=len(a), replace=True)
        rb = rng.choice(b, size=len(b), replace=True)
        vals.append(tvd_kde(ra, rb))
    vals = np.asarray(vals)
    return float(vals.mean()), float(vals.std())


print("\n[6] Bootstrap TVD (KDE, 1000 resamples)...")
print("WikiText vs MMLU, rho:")
tvd_rho_retain_mean, tvd_rho_retain_std = bootstrap_tvd(wiki_rho, mmlu_rho)
print(f"  {tvd_rho_retain_mean:.3f} +/- {tvd_rho_retain_std:.3f}")

print("WikiText vs MMLU, rho_raw:")
tvd_raw_retain_mean, tvd_raw_retain_std = bootstrap_tvd(wiki_rho_raw, mmlu_rho_raw)
print(f"  {tvd_raw_retain_mean:.3f} +/- {tvd_raw_retain_std:.3f}")

print("WikiText vs WMDP-Bio, rho:")
tvd_rho_forget_mean, tvd_rho_forget_std = bootstrap_tvd(wiki_rho, wmdp_rho)
print(f"  {tvd_rho_forget_mean:.3f} +/- {tvd_rho_forget_std:.3f}")

print("WikiText vs WMDP-Bio, rho_raw:")
tvd_raw_forget_mean, tvd_raw_forget_std = bootstrap_tvd(wiki_rho_raw, wmdp_rho_raw)
print(f"  {tvd_raw_forget_mean:.3f} +/- {tvd_raw_forget_std:.3f}")

results = {
    "features": FEATURES.tolist(),
    "n_wikitext": len(wiki_rho), "n_mmlu": len(mmlu_rho), "n_wmdp_bio": len(wmdp_rho),
    "wikitext_vs_mmlu": {
        "rho": {"mean": tvd_rho_retain_mean, "std": tvd_rho_retain_std},
        "rho_raw": {"mean": tvd_raw_retain_mean, "std": tvd_raw_retain_std},
    },
    "wikitext_vs_wmdp_bio": {
        "rho": {"mean": tvd_rho_forget_mean, "std": tvd_rho_forget_std},
        "rho_raw": {"mean": tvd_raw_forget_mean, "std": tvd_raw_forget_std},
    },
}
out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_rho_comparison_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY: TVD(WikiText vs X), rho vs rho_raw")
print("=" * 70)
print(f"{'comparison':>22} {'rho':>16} {'rho_raw':>16}   (lower TVD = more similar)")
print(f"{'WikiText vs MMLU':>22} {tvd_rho_retain_mean:>7.3f}+/-{tvd_rho_retain_std:.3f} {tvd_raw_retain_mean:>7.3f}+/-{tvd_raw_retain_std:.3f}   (want LOW: retain sets should look alike)")
print(f"{'WikiText vs WMDP-Bio':>22} {tvd_rho_forget_mean:>7.3f}+/-{tvd_rho_forget_std:.3f} {tvd_raw_forget_mean:>7.3f}+/-{tvd_raw_forget_std:.3f}   (want HIGH: retain vs forget should look different)")

print("\nSaved:", out_path)
print("DONE")
