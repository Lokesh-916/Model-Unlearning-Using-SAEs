import json
import os
import pickle
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.feature_activation import (
    get_shuffled_forget_retain_tokens,
    calculate_sparsity,
    get_top_features_percentile,
)
from evals.unlearning.utils.metrics import modify_and_calculate_metrics

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"  # main Bio SAE (Table 12: same as main experiments)
MULTIPLIER = 500
RATIO_PERCENTILE = 95  # p_ratio for bio
N_FEATURES = [20]

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

FRACTIONS = [80, 60, 40, 20, 10, 5]
SCRATCH = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad"

device = "cuda"

print("[1] Loading model + SAE (main Bio ell0=142)...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

results = {}
# 100% reference point already known from the original Bio reproduction
# (n_feats=20, p_ratio=95, full wikitext+bio-forget-corpus data):
results["100"] = {"wmdp-bio": 0.29368, "mmlu_avg": 0.99412}
print("fraction=100% (reused from prior run): WMDP=29.368%  MMLU_avg=99.412%")

for frac in FRACTIONS:
    print(f"\n{'='*70}\nfraction={frac}%\n{'='*70}")
    random.seed(0)
    torch.manual_seed(0)

    out_folder = os.path.join(SCRATCH, f"dsg_dataeff_frac{frac}")
    os.makedirs(out_folder, exist_ok=True)

    print(f"[2] Sampling forget/retain tokens at dataset_fraction={frac}...")
    forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(
        model,
        batch_size=1024,
        seq_len=1024,
        dataset_fraction=frac,
        forget_corpora="bio-forget-corpus",
        retain_corpora="wikitext",
    )
    print("forget_tokens:", forget_tokens.shape, "retain_tokens:", retain_tokens.shape)

    print("[3] Computing feature sparsity...")
    forget_sparsity, retain_sparsity = calculate_sparsity(
        model, sae, forget_tokens, retain_tokens, 1, folder_name=out_folder
    )

    print("[4] Selecting top 20 features (p_ratio=95)...")
    sel_ind_out, percentile_95 = get_top_features_percentile(
        forget_sparsity, retain_sparsity, ratio_percentile=RATIO_PERCENTILE,
        folder_name=out_folder, n_features_lst=N_FEATURES,
    )
    top20 = sel_ind_out[:20]
    threshold = percentile_95["20"]
    print(f"Selected features: {top20.tolist()}")
    print(f"Threshold: {threshold}")

    # Free the (small, but let's be tidy) raw activation pickles once used.
    for fn in ("act_fgt.pkl", "act_ret.pkl"):
        p = os.path.join(out_folder, fn)
        if os.path.exists(p):
            os.remove(p)

    print("[5] Evaluating on full WMDP-Bio + MMLU test sets (cached baseline reused)...")
    metrics = modify_and_calculate_metrics(
        model,
        mcq_batch_size=1,
        artifacts_folder=ARTIFACTS_FOLDER,
        sae=sae,
        dataset_names=DATASET_NAMES,
        intervention_method="clamp_feature_activation",
        features_to_ablate=top20,
        multiplier=MULTIPLIER,
        activation_threshold=threshold,
    )
    row = {d: metrics[d]["mean_correct"] for d in DATASET_NAMES}
    mmlu_avg = float(np.mean([row[d] for d in DATASET_NAMES if d != "wmdp-bio"]))
    results[str(frac)] = {"wmdp-bio": row["wmdp-bio"], "mmlu_avg": mmlu_avg, "features": top20.tolist(), "threshold": threshold}
    print(f"\n>>> fraction={frac}%: WMDP-bio={row['wmdp-bio']*100:.3f}%  MMLU_avg={mmlu_avg*100:.3f}%")

out_path = os.path.join(SCRATCH, "dsg_dataefficiency_bio_results.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY (Bio data efficiency)")
print("=" * 70)
print(f"{'fraction':>10} {'WMDP-bio':>10} {'MMLU avg':>10}")
for frac in ["100"] + [str(f) for f in FRACTIONS]:
    r = results[frac]
    print(f"{frac+'%':>10} {r['wmdp-bio']*100:>9.3f}% {r['mmlu_avg']*100:>9.3f}%")

print("\nSaved:", out_path)
print("DONE")
