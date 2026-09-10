import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.feature_activation import get_top_features_percentile
from evals.unlearning.utils.metrics import modify_and_calculate_metrics

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
RATIO_PERCENTILE = 95  # p_ratio for bio
N_FEATURES = [20]
CLAMP_STRENGTHS = [10, 25, 50, 100, 200, 300, 400, 500]

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, "gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities")
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

device = "cuda"

print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

print("[2] Loading cached Bio forget/retain sparsity (unchanged, main-experiment data)...")
forget_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
retain_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)

print("[3] Selecting top-20 features (p_ratio=95) -- same selection as main Bio result...")
sel_ind_out, percentile_95 = get_top_features_percentile(
    forget_sparsity, retain_sparsity, ratio_percentile=RATIO_PERCENTILE,
    folder_name=SPARSITY_DIR, n_features_lst=N_FEATURES,
)
top20 = sel_ind_out[:20]
threshold = percentile_95["20"]
print(f"Features: {top20.tolist()}")
print(f"Threshold: {threshold}")

results = {}
for c in CLAMP_STRENGTHS:
    print(f"\n{'='*60}\nclamp strength c={c}\n{'='*60}")
    metrics = modify_and_calculate_metrics(
        model,
        mcq_batch_size=1,
        artifacts_folder=ARTIFACTS_FOLDER,
        sae=sae,
        dataset_names=DATASET_NAMES,
        intervention_method="clamp_feature_activation",
        features_to_ablate=top20,
        multiplier=c,
        activation_threshold=threshold,
    )
    row = {d: metrics[d]["mean_correct"] for d in DATASET_NAMES}
    mmlu_avg = float(np.mean([row[d] for d in DATASET_NAMES if d != "wmdp-bio"]))
    results[str(c)] = {"wmdp-bio": row["wmdp-bio"], "mmlu_avg": mmlu_avg}
    print(f"\n>>> c={c}: WMDP-bio={row['wmdp-bio']*100:.3f}%  MMLU_avg={mmlu_avg*100:.3f}%")

out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_clamp_ablation_results.json"
with open(out_path, "w") as f:
    json.dump({"features": top20.tolist(), "threshold": threshold, "results": results}, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY (Bio clamp-strength ablation, n_feats=20)")
print("=" * 60)
print(f"{'c':>6} {'WMDP-bio':>10} {'MMLU avg':>10}")
for c in CLAMP_STRENGTHS:
    r = results[str(c)]
    print(f"{c:>6} {r['wmdp-bio']*100:>9.3f}% {r['mmlu_avg']*100:>9.3f}%")

print("\nSaved:", out_path)
print("DONE")
