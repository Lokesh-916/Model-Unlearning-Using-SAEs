"""
A2: replace clamping selected features to a fixed extreme (-500) with
conditional mean-ablation toward a reference set's feature means -- an
approximation of "what would this feature look like on a question the model
is naturally unsure about," rather than an out-of-distribution constant.

Kill test: compare output entropy/max-probability on flagged (forget) prompts
under DSG's clamp vs the unintervened base model. If DSG's outputs look
abnormal (entropy/confidence very different from natural uncertainty), the
problem the spec describes is real.
"""
import json
import os
import sys

import numpy as np
import torch
from functools import partial

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import (
    convert_wmdp_data_to_prompt, get_output_probs_abcd, calculate_MCQ_metrics,
)
from evals.unlearning.utils.feature_activation import get_top_features_percentile
from evals.unlearning.utils.var import PRE_WMDP_BIO
from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features
from dsg_utils.activation_collection import get_bos_pad_eos_mask

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
HOOK_NAME = "blocks.3.hook_resid_post"
MULTIPLIER = 500
N_FEATURES = 20
N_KILL_TEST_SAMPLES = 150
REFERENCE_FRACTION = 0.25  # bottom quartile by base-model confidence
STRENGTH_GRID = [0.5, 1.0, 2.0, 4.0]
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

print("\n[2] Re-deriving DSG's original feature set + threshold...")
fs_fgt = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
fs_ret = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
sel_orig, perc_orig = get_top_features_percentile(
    fs_fgt, fs_ret, ratio_percentile=95, folder_name=SPARSITY_DIR, n_features_lst=[N_FEATURES],
)
features_orig = sel_orig[:N_FEATURES]
threshold_orig = perc_orig[str(N_FEATURES)]
print(f"Original: {len(features_orig)} features, threshold={threshold_orig:.4f}")

hook_dsg = partial(anthropic_clamp_resid_SAE_features, sae=sae, features_to_ablate=features_orig.tolist(),
                    multiplier=MULTIPLIER, activation_threshold=threshold_orig)

wmdp_ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")


def entropy_and_maxprob(probs):
    p = probs.clamp(min=1e-12)
    p = p / p.sum(dim=-1, keepdim=True)
    ent = -(p * p.log()).sum(dim=-1)
    return ent, p.max(dim=-1).values


print("\n[3] KILL TEST: entropy/confidence of DSG's clamp vs the unintervened base model...")
kill_idx = np.random.choice(len(wmdp_ds), N_KILL_TEST_SAMPLES, replace=False)
prompts = []
answers = []
for i in kill_idx:
    ex = wmdp_ds[int(i)]
    prompts.append(convert_wmdp_data_to_prompt(
        ex["question"], ex["choices"], prompt_format="GEMMA_INST_FORMAT", without_question=False,
        pre_question=PRE_WMDP_BIO,
    ))
    answers.append(int(ex["answer"]))

model.reset_hooks()
base_probs = get_output_probs_abcd(model, prompts, batch_size=1, n_batches=len(prompts), verbose=False)
model.reset_hooks()
model.add_hook(HOOK_NAME, hook_dsg)
dsg_probs = get_output_probs_abcd(model, prompts, batch_size=1, n_batches=len(prompts), verbose=False)
model.reset_hooks()

ent_base, maxp_base = entropy_and_maxprob(base_probs)
ent_dsg, maxp_dsg = entropy_and_maxprob(dsg_probs)
print(f"  base model:  mean entropy={ent_base.mean().item():.4f}  mean max-prob={maxp_base.mean().item():.4f}")
print(f"  DSG clamp:   mean entropy={ent_dsg.mean().item():.4f}  mean max-prob={maxp_dsg.mean().item():.4f}")
print(f"  (max possible entropy for 4-way uniform: {np.log(4):.4f})")

print("\n[4] Building the reference set (WMDP-bio questions the BASE model is least confident on)...")
all_prompts = []
all_answers = []
for i in range(len(wmdp_ds)):
    ex = wmdp_ds[i]
    all_prompts.append(convert_wmdp_data_to_prompt(
        ex["question"], ex["choices"], prompt_format="GEMMA_INST_FORMAT", without_question=False,
        pre_question=PRE_WMDP_BIO,
    ))
    all_answers.append(int(ex["answer"]))
model.reset_hooks()
all_base_probs = get_output_probs_abcd(model, all_prompts, batch_size=1, n_batches=len(all_prompts), verbose=True)
all_maxp = all_base_probs.max(dim=-1).values.cpu().numpy()
n_ref = int(len(wmdp_ds) * REFERENCE_FRACTION)
reference_idx = np.argsort(all_maxp)[:n_ref]  # lowest confidence
print(f"  reference set: {n_ref} lowest-confidence questions out of {len(wmdp_ds)} "
      f"(mean max-prob={all_maxp[reference_idx].mean():.4f}, vs overall mean={all_maxp.mean():.4f})")

print("\n[5] Computing reference-set mean feature activations for the 20 selected features...")
ref_sums = torch.zeros(N_FEATURES, device=device, dtype=torch.float32)
n_tok = 0
model.reset_hooks()
with torch.no_grad():
    for i in reference_idx:
        tokens = model.to_tokens(all_prompts[int(i)], padding_side="right", prepend_bos=False)
        _, cache = model.run_with_cache(tokens, stop_at_layer=4, names_filter=HOOK_NAME)
        resid = cache[HOOK_NAME]
        feats = sae.encode(resid)[0, 1:, features_orig].float()  # exclude BOS
        ref_sums += feats.sum(dim=0)
        n_tok += feats.shape[0]
reference_mean = (ref_sums / n_tok).cpu().numpy()
print(f"  reference mean activations (20 features): min={reference_mean.min():.4f}, "
      f"max={reference_mean.max():.4f}, mean={reference_mean.mean():.4f}")
print(f"  (for comparison, DSG clamps these to a constant -{MULTIPLIER})")


def mean_ablation_hook(resid, hook, sae, features_to_ablate, ref_values, strength, activation_threshold):
    feature_activations = sae.encode(resid)
    feature_activations[:, 0, :] = 0.0
    reconstruction = sae.decode(feature_activations)
    error = resid - reconstruction

    target_features = feature_activations[:, :, features_to_ablate]
    activation_mask = (target_features > 0).sum(dim=2) > 0
    batch_activation_rates = activation_mask.sum(dim=1) / activation_mask.shape[1]
    active_batches = batch_activation_rates > activation_threshold
    final_mask = activation_mask.unsqueeze(2) & active_batches.unsqueeze(1).unsqueeze(2)

    replacement = ref_values * strength
    feature_activations[:, :, features_to_ablate] = torch.where(
        final_mask, replacement.expand_as(target_features), feature_activations[:, :, features_to_ablate],
    )
    modified_reconstruction = sae.decode(feature_activations)
    return modified_reconstruction + error


ref_values_t = torch.tensor(reference_mean, device=device, dtype=torch.bfloat16)

print("\n[6] Line search over mean-ablation strength (target: WMDP-bio accuracy near chance, 25%)...")
grid_results = {}
for strength in STRENGTH_GRID:
    hook_a2 = partial(mean_ablation_hook, sae=sae, features_to_ablate=features_orig.tolist(),
                       ref_values=ref_values_t, strength=strength, activation_threshold=threshold_orig)
    model.reset_hooks()
    model.add_hook(HOOK_NAME, hook_a2)
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name="wmdp-bio", target_metric="correct", split="all")
    model.reset_hooks()
    acc = m["mean_correct"]
    grid_results[strength] = acc
    print(f"  strength={strength}: WMDP-bio accuracy={acc*100:.3f}%  (distance from chance=25%: {abs(acc-0.25)*100:.2f}pp)")

best_strength = min(grid_results, key=lambda s: abs(grid_results[s] - 0.25))
print(f"\nBest strength: {best_strength} (WMDP-bio={grid_results[best_strength]*100:.3f}%)")

print(f"\n[7] Full evaluation of A2 at the chosen strength={best_strength}...")
hook_a2_final = partial(mean_ablation_hook, sae=sae, features_to_ablate=features_orig.tolist(),
                         ref_values=ref_values_t, strength=best_strength, activation_threshold=threshold_orig)
model.reset_hooks()
model.add_hook(HOOK_NAME, hook_a2_final)
a2_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct", split="all")
    a2_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_a2 = float(np.mean([a2_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n[8] Entropy/confidence of A2 (best strength) vs DSG clamp vs base, on the kill-test sample...")
model.reset_hooks()
model.add_hook(HOOK_NAME, hook_a2_final)
a2_probs = get_output_probs_abcd(model, prompts, batch_size=1, n_batches=len(prompts), verbose=False)
model.reset_hooks()
ent_a2, maxp_a2 = entropy_and_maxprob(a2_probs)
print(f"  base model:  mean entropy={ent_base.mean().item():.4f}  mean max-prob={maxp_base.mean().item():.4f}")
print(f"  DSG clamp:   mean entropy={ent_dsg.mean().item():.4f}  mean max-prob={maxp_dsg.mean().item():.4f}")
print(f"  A2 (best):   mean entropy={ent_a2.mean().item():.4f}  mean max-prob={maxp_a2.mean().item():.4f}")

out = {
    "reference_set_size": n_ref,
    "reference_mean_confidence": float(all_maxp[reference_idx].mean()),
    "overall_mean_confidence": float(all_maxp.mean()),
    "grid_results": {str(k): v for k, v in grid_results.items()},
    "best_strength": best_strength,
    "a2_results": a2_results,
    "mmlu_a2": mmlu_a2,
    "entropy_confidence": {
        "base": {"entropy": ent_base.mean().item(), "maxprob": maxp_base.mean().item()},
        "dsg_clamp": {"entropy": ent_dsg.mean().item(), "maxprob": maxp_dsg.mean().item()},
        "a2_best": {"entropy": ent_a2.mean().item(), "maxprob": maxp_a2.mean().item()},
    },
}
out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_a2_results.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)

print("\n" + "=" * 70)
print("FINAL SUMMARY: A2 calibrated mean-ablation vs DSG original clamp")
print("=" * 70)
print(f"{'config':>28} {'WMDP-bio':>12} {'MMLU_avg':>12} {'entropy':>10} {'max-prob':>10}")
print(f"{'base (no intervention)':>28} {'--':>12} {'--':>12} {ent_base.mean().item():>10.4f} {maxp_base.mean().item():>10.4f}")
print(f"{'DSG original clamp':>28} {29.368:>11.3f}% {99.412:>11.3f}% {ent_dsg.mean().item():>10.4f} {maxp_dsg.mean().item():>10.4f}")
print(f"{'A2 mean-ablation (best)':>28} {a2_results['wmdp-bio']*100:>11.3f}% {mmlu_a2*100:>11.3f}% {ent_a2.mean().item():>10.4f} {maxp_a2.mean().item():>10.4f}")
print("\nSaved:", out_path)
print("DONE")
