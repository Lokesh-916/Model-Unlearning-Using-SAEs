import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import modify_and_calculate_metrics

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_59"  # paper's zero-shot SAE variant (Table 14)
MULTIPLIER = 500

# Top-20 Neuronpedia features for gemma-2-2b, layer 3, gemmascope-res-16k
# (verified to match the paper's Table 16 almost exactly, same query methodology)
BIO_FEATURES = np.array([
    12382, 9722, 343, 373, 11, 15969, 12117, 5877, 968, 622,
    5231, 10546, 12037, 6150, 14747, 5704, 8786, 10933, 140, 13527,
])
CYBER_FEATURES = np.array([
    15331, 2060, 15286, 11015, 364, 4836, 2905, 10931, 11716, 16160,
    6309, 10543, 11513, 1803, 12681, 11520, 11323, 10415, 3943, 4686,
])

CASES = {
    "bio": {
        "artifacts_folder": "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it",
        "dataset_names": ["wmdp-bio", "high_school_us_history", "college_computer_science",
                           "high_school_geography", "human_aging"],
        "wmdp_key": "wmdp-bio",
        "features": BIO_FEATURES,
    },
    "cyber": {
        "artifacts_folder": "artifacts_dynamic_bs1_cyber/unlearning/gemma-2-2b-it",
        "dataset_names": ["wmdp-cyber", "high_school_us_history", "college_biology",
                           "high_school_geography", "human_aging"],
        "wmdp_key": "wmdp-cyber",
        "features": CYBER_FEATURES,
    },
}

TAU_SWEEP = [round(x, 1) for x in np.arange(0.1, 1.0, 0.1)]

parser = argparse.ArgumentParser()
parser.add_argument("--case", choices=["bio", "cyber"], required=True)
args = parser.parse_args()

case = CASES[args.case]
device = "cuda"

print("=" * 70)
print(f"ZERO-SHOT DSG: {args.case}")
print("=" * 70)

print("\n[1] Loading model + SAE (ell0=59)...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)
print("SAE hook:", sae.cfg.metadata.hook_name)

results = {}
for tau in TAU_SWEEP:
    print(f"\n[2] tau={tau} ...")
    metrics = modify_and_calculate_metrics(
        model,
        mcq_batch_size=1,
        artifacts_folder=case["artifacts_folder"],
        sae=sae,
        dataset_names=case["dataset_names"],
        intervention_method="clamp_feature_activation",
        features_to_ablate=case["features"],
        multiplier=MULTIPLIER,
        activation_threshold=tau,
    )
    row = {d: metrics[d]["mean_correct"] for d in case["dataset_names"]}
    mmlu_avg = np.mean([row[d] for d in case["dataset_names"] if d != case["wmdp_key"]])
    row["mmlu_avg"] = float(mmlu_avg)
    results[str(tau)] = row
    print(f"  WMDP={row[case['wmdp_key']]*100:.2f}%  MMLU_avg={mmlu_avg*100:.2f}%")

out_path = f"/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_zeroshot_{args.case}_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 70)
print(f"SUMMARY ({args.case})")
print("=" * 70)
print(f"{'tau':>6} {'WMDP':>10} {'MMLU avg':>10}")
for tau in TAU_SWEEP:
    row = results[str(tau)]
    print(f"{tau:>6} {row[case['wmdp_key']]*100:>9.2f}% {row['mmlu_avg']*100:>9.2f}%")

print("\nSaved:", out_path)
print("DONE")
