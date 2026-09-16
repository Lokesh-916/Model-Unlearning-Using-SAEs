import gc
import json
import os
import pickle
import random
import sys

import numpy as np
import torch
from functools import partial
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import (
    convert_wmdp_data_to_prompt, get_output_probs_abcd, calculate_MCQ_metrics,
)
from evals.unlearning.utils.feature_activation import get_top_features_percentile
from evals.unlearning.utils.var import PRE_WMDP_BIO
from dsg_utils.activation_collection import get_bos_pad_eos_mask

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
LAYER = 3
HOOK_NAME = "blocks.3.hook_resid_post"
MULTIPLIER = 500
RATIO_PERCENTILE = 95
WINDOW_SIZE = 8
P_DYN = 95

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, "gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities")
METRICS_PKL = os.path.join(ARTIFACTS_FOLDER, "gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/metrics",
                            "clamp_feature_activation_multiplier500_nfeatures20_layer3_retainthres95_seed0.pkl")
QUESTION_IDS_FILE = os.path.join(ARTIFACTS_FOLDER, "data/question_ids/all/wmdp-bio_correct.csv")
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

N_SAMPLE = 60
FILLER_WORD_COUNTS = [0, 150, 400, 800]
SEED = 0

device = "cuda"
random.seed(SEED)


def local_rate_vectorized(am_float, window_size):
    """Sliding-window activation rate, for each position t: sum of am_float over
    [max(0,t-half), min(seq_len,t+half+1)) divided by the FULL window_size
    (a fixed denominator), not by however much of the window actually overlaps
    the sequence. Using the shrunken overlap length as the denominator (an
    earlier version of this script did that) is a real bug: near the sequence
    edges the window can shrink to 1-2 tokens, so a single active token there
    spikes the "rate" to 1.0 -- a spurious edge artifact, not a real signal.
    Dividing by the constant window_size instead makes edge positions read as
    LESS active when they're only partially covered, which is the correct
    behavior (matches an implicit zero-padding convention) and exactly matches
    the true windowed rate for every non-edge position anyway."""
    seq_len = am_float.shape[0]
    half = window_size // 2
    cumsum = torch.cat([torch.zeros(1, device=am_float.device, dtype=am_float.dtype), am_float.cumsum(0)])
    idx = torch.arange(seq_len, device=am_float.device)
    lo = (idx - half).clamp(min=0)
    hi = (idx + half + 1).clamp(max=seq_len)
    window_sum = cumsum[hi] - cumsum[lo]
    return window_sum / float(window_size)


def windowed_clamp_hook(resid, hook, sae, features_to_ablate, multiplier, activation_threshold, window_size):
    feature_activations = sae.encode(resid)
    feature_activations[:, 0, :] = 0.0
    reconstruction = sae.decode(feature_activations)
    error = resid - reconstruction

    target_features = feature_activations[:, :, features_to_ablate]
    activation_mask = (target_features > 0).sum(dim=2) > 0  # [batch, seq]
    am_float = activation_mask.float()

    local_active = torch.zeros_like(activation_mask)
    for b in range(am_float.shape[0]):
        rate_b = local_rate_vectorized(am_float[b], window_size)
        local_active[b] = rate_b > activation_threshold

    final_mask = (activation_mask & local_active).unsqueeze(2)
    feature_activations[:, :, features_to_ablate] = torch.where(
        final_mask, torch.full_like(target_features, -multiplier),
        feature_activations[:, :, features_to_ablate],
    )
    modified_reconstruction = sae.decode(feature_activations)
    return modified_reconstruction + error


print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

print("\n[2] Re-deriving main Bio feature set (p_ratio=95, n_feats=20)...")
forget_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
retain_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
sel_ind_out, _ = get_top_features_percentile(
    forget_sparsity, retain_sparsity, ratio_percentile=RATIO_PERCENTILE,
    folder_name=SPARSITY_DIR, n_features_lst=[20],
)
features = sel_ind_out[:20]
print(f"Features: {features.tolist()}")

print("\n[3] Properly recalibrating threshold FOR THE WINDOWED STATISTIC (using cached retain act_ret.pkl)...")
print("    Attempt 3 fix: pool EVERY individual window-rate across all retain sequences and positions,")
print("    then take the 95th percentile of that pooled distribution -- NOT the max-per-sequence approach")
print("    (attempts 1/2), which is a known statistical trap: taking a max over ~1000 overlapping windows")
print("    per sequence drifts toward the extreme high end almost regardless of the real signal.")
with open(os.path.join(SPARSITY_DIR, "act_ret.pkl"), "rb") as f:
    act_ret = pickle.load(f)
pooled_rates = []
for el in tqdm(act_ret):
    # el: [batch, seq, d_sae] numpy array (batch=1 in this cache)
    arr = torch.from_numpy(el[:, :, features] > 0).float()
    for b in range(arr.shape[0]):
        am = (arr[b].sum(dim=1) > 0).float()
        rate = local_rate_vectorized(am, WINDOW_SIZE)
        pooled_rates.append(rate.numpy())
del act_ret
gc.collect()
pooled_rates = np.concatenate(pooled_rates)
print(f"Pooled {pooled_rates.shape[0]} individual window-rate samples across all retain sequences.")
windowed_threshold = float(np.percentile(pooled_rates, P_DYN))
print(f"Windowed threshold (p_dyn={P_DYN} of POOLED per-position local rate): {windowed_threshold:.4f}")
print(f"(for comparison, whole-sequence-mean threshold was ~0.546; attempt 1 gave 1.0000; attempt 2 gave 1.0417)")

hook_fn = partial(
    windowed_clamp_hook, sae=sae, features_to_ablate=features,
    multiplier=MULTIPLIER, activation_threshold=windowed_threshold, window_size=WINDOW_SIZE,
)


def install_windowed_hook():
    model.reset_hooks()
    model.add_hook(sae.cfg.metadata.hook_name, hook_fn)


print("\n[4] Validating utility + forgetting with the PROPERLY CALIBRATED windowed gate...")
install_windowed_hook()
util_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct", split="all")
    util_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_avg = float(np.mean([util_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))
print(f"\nWindowed gate (calibrated): WMDP-bio={util_results['wmdp-bio']*100:.3f}%  MMLU_avg={mmlu_avg*100:.3f}%")
print(f"(whole-sequence baseline:   WMDP-bio=29.368%  MMLU_avg=99.412%)")
print(f"(earlier UNCALIBRATED windowed attempt: WMDP-bio=27.881%  MMLU_avg=91.522%)")

print("\n[5] Re-running the SAME dilution attack against the windowed gate...")
with open(METRICS_PKL, "rb") as f:
    baseline_metrics = pickle.load(f)
is_correct = baseline_metrics["wmdp-bio"]["is_correct"]
question_ids = np.genfromtxt(QUESTION_IDS_FILE, ndmin=1, dtype=int)
forgotten_raw_ids = question_ids[~is_correct.astype(bool)]
sample_ids = random.sample(list(forgotten_raw_ids), min(N_SAMPLE, len(forgotten_raw_ids)))
print(f"Using the same {len(sample_ids)}-question sample as the original dilution attack.")

wmdp_ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
wiki_text_pool = " ".join(x["text"] for x in wiki if len(x["text"]) > 200)
wiki_words = wiki_text_pool.split()


def build_filler(n_words, offset):
    if n_words == 0:
        return ""
    words = wiki_words[offset:offset + n_words]
    return " ".join(words) + "\n\n"


def compute_windowed_rho_max(prompt):
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, stop_at_layer=LAYER + 1, names_filter=HOOK_NAME)
        resid = cache[HOOK_NAME]
        feats = sae.encode(resid)[0]
    mask = get_bos_pad_eos_mask(tokens, model.tokenizer)[0]
    active = ((feats[:, features] > 0).any(dim=1) & mask).float()
    rate = local_rate_vectorized(active, WINDOW_SIZE)
    return rate.max().item(), tokens.shape[1]


defense_results = {}
for n_words in FILLER_WORD_COUNTS:
    print(f"\n{'='*70}\nFiller ~{n_words} words (windowed defense)\n{'='*70}")
    rhos, token_lens, n_correct = [], [], 0
    for offset, raw_id in enumerate(tqdm(sample_ids)):
        ex = wmdp_ds[int(raw_id)]
        filler = build_filler(n_words, offset=offset * 50)
        pre_question = filler + PRE_WMDP_BIO
        prompt = convert_wmdp_data_to_prompt(
            ex["question"], ex["choices"], prompt_format="GEMMA_INST_FORMAT", without_question=False,
            pre_question=pre_question,
        )
        model.reset_hooks()
        rho, tlen = compute_windowed_rho_max(prompt)
        rhos.append(rho)
        token_lens.append(tlen)

        install_windowed_hook()
        probs = get_output_probs_abcd(model, [prompt], batch_size=1, n_batches=1, verbose=False)
        pred = int(probs[0].argmax().item())
        if pred == int(ex["answer"]):
            n_correct += 1

    mean_rho = float(np.mean(rhos))
    mean_tlen = float(np.mean(token_lens))
    attack_success_rate = n_correct / len(sample_ids)
    print(f"  mean_max_local_rho={mean_rho:.4f}  mean_token_len={mean_tlen:.1f}  "
          f"attack_success={attack_success_rate*100:.1f}% ({n_correct}/{len(sample_ids)})")
    defense_results[str(n_words)] = {
        "mean_rho": mean_rho, "mean_token_len": mean_tlen,
        "attack_success_rate": attack_success_rate, "n_correct": n_correct, "n_total": len(sample_ids),
    }

model.reset_hooks()

out = {
    "windowed_threshold": windowed_threshold, "window_size": WINDOW_SIZE, "features": features.tolist(),
    "utility_validation": {"per_dataset": util_results, "mmlu_avg": mmlu_avg},
    "dilution_defense_results": defense_results,
}
out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_windowed_defense_results_v4_window8.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("FINAL SUMMARY: windowed gate as a defense against the dilution attack")
print("=" * 70)
print(f"Windowed threshold: {windowed_threshold:.4f}")
print(f"Utility: WMDP-bio={util_results['wmdp-bio']*100:.3f}%  MMLU_avg={mmlu_avg*100:.3f}%  "
      f"(whole-seq baseline: 29.368% / 99.412%)")
print(f"\n{'filler_words':>14} {'mean_max_local_rho':>20} {'attack_success (windowed)':>28} {'attack_success (whole-seq, from before)':>42}")
whole_seq_success = {"0": 0.0, "150": 0.983, "400": 0.95, "800": 0.967}
for n_words in FILLER_WORD_COUNTS:
    r = defense_results[str(n_words)]
    print(f"{n_words:>14} {r['mean_rho']:>20.4f} {r['attack_success_rate']*100:>27.1f}% "
          f"{whole_seq_success[str(n_words)]*100:>41.1f}%")

print("\nSaved:", out_path)
print("DONE")
