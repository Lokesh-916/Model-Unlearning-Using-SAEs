import json
import os
import pickle
import random
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from datasets import load_dataset
from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import convert_wmdp_data_to_prompt, get_output_probs_abcd, modify_model
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

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
SPARSITY_DIR = os.path.join(ARTIFACTS_FOLDER, "gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/sparsities")
METRICS_PKL = os.path.join(ARTIFACTS_FOLDER, "gemma-scope-2b-pt-res_layer_3/width_16k/average_l0_142/results/metrics",
                            "clamp_feature_activation_multiplier500_nfeatures20_layer3_retainthres95_seed0.pkl")
QUESTION_IDS_FILE = os.path.join(ARTIFACTS_FOLDER, "data/question_ids/all/wmdp-bio_correct.csv")

N_SAMPLE = 60
FILLER_WORD_COUNTS = [0, 150, 400, 800]  # approximate; actual token counts measured & reported
SEED = 0

device = "cuda"
random.seed(SEED)

print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)

print("\n[2] Re-deriving main Bio feature set + threshold (p_ratio=95, n_feats=20)...")
forget_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_forget.txt"), dtype=float)
retain_sparsity = np.loadtxt(os.path.join(SPARSITY_DIR, "feature_sparsity_retain.txt"), dtype=float)
sel_ind_out, percentile_95 = get_top_features_percentile(
    forget_sparsity, retain_sparsity, ratio_percentile=RATIO_PERCENTILE,
    folder_name=SPARSITY_DIR, n_features_lst=[20],
)
features = sel_ind_out[:20]
threshold = percentile_95["20"]
print(f"Features: {features.tolist()}")
print(f"Threshold: {threshold:.4f}")

print("\n[3] Identifying WMDP-Bio questions DSG currently forgets successfully (is_correct=False)...")
with open(METRICS_PKL, "rb") as f:
    baseline_metrics = pickle.load(f)
is_correct = baseline_metrics["wmdp-bio"]["is_correct"]
question_ids = np.genfromtxt(QUESTION_IDS_FILE, ndmin=1, dtype=int)
assert len(question_ids) == len(is_correct)
forgotten_raw_ids = question_ids[~is_correct.astype(bool)]  # these are baseline-correct, DSG-wrong = successfully forgotten
print(f"Successfully forgotten questions (baseline-correct, DSG-wrong): {len(forgotten_raw_ids)} / {len(question_ids)}")

sample_ids = random.sample(list(forgotten_raw_ids), min(N_SAMPLE, len(forgotten_raw_ids)))
print(f"Sampled {len(sample_ids)} for the dilution attack test.")

print("\n[4] Loading WMDP-Bio dataset + building filler text pool from WikiText...")
wmdp_ds = load_dataset("cais/wmdp", "wmdp-bio", split="test")
wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
wiki_text_pool = " ".join(x["text"] for x in wiki if len(x["text"]) > 200)
wiki_words = wiki_text_pool.split()
print(f"WikiText filler pool: {len(wiki_words)} words available")


def build_filler(n_words, offset):
    if n_words == 0:
        return ""
    words = wiki_words[offset:offset + n_words]
    return " ".join(words) + "\n\n"


def install_dsg_hook():
    model.reset_hooks()
    modify_model(
        model, sae, intervention_method="clamp_feature_activation",
        features_to_ablate=features, multiplier=MULTIPLIER, activation_threshold=threshold,
    )


answer_strings = ["A", "B", "C", "D", " A", " B", " C", " D"]
answer_tokens_ref = model.to_tokens(answer_strings, prepend_bos=False).flatten()


def compute_rho(prompt):
    """Manually compute the same activation-rate statistic the DSG gate uses,
    for reporting/mechanism-verification purposes (independent of the hook,
    which doesn't expose this number directly to our script)."""
    tokens = model.to_tokens(prompt, prepend_bos=True)
    with torch.no_grad():
        _, cache = model.run_with_cache(tokens, stop_at_layer=LAYER + 1, names_filter=HOOK_NAME)
        resid = cache[HOOK_NAME]
        feats = sae.encode(resid)[0]
    mask = get_bos_pad_eos_mask(tokens, model.tokenizer)[0]
    valid_len = mask.sum().item()
    active = (feats[:, features] > 0).any(dim=1) & mask
    rho = active.sum().item() / valid_len if valid_len > 0 else 0.0
    return rho, tokens.shape[1]


results = {}
for n_words in FILLER_WORD_COUNTS:
    print(f"\n{'='*70}\nFiller ~{n_words} words\n{'='*70}")
    rhos = []
    token_lens = []
    n_correct = 0
    for offset, raw_id in enumerate(tqdm(sample_ids)):
        ex = wmdp_ds[int(raw_id)]
        filler = build_filler(n_words, offset=offset * 50)  # vary offset so filler differs per question
        pre_question = filler + PRE_WMDP_BIO
        prompt = convert_wmdp_data_to_prompt(
            ex["question"], ex["choices"], prompt_format="GEMMA_INST_FORMAT", without_question=False,
            pre_question=pre_question,
        )
        model.reset_hooks()  # clean, unintervened pass for the diagnostic rho measurement
        rho, tlen = compute_rho(prompt)
        rhos.append(rho)
        token_lens.append(tlen)

        install_dsg_hook()  # re-install for the actual (intervened) prediction
        probs = get_output_probs_abcd(model, [prompt], batch_size=1, n_batches=1, verbose=False)
        pred = int(probs[0].argmax().item())
        if pred == int(ex["answer"]):
            n_correct += 1

    mean_rho = float(np.mean(rhos))
    mean_tlen = float(np.mean(token_lens))
    attack_success_rate = n_correct / len(sample_ids)
    print(f"  mean_rho={mean_rho:.4f}  mean_token_len={mean_tlen:.1f}  "
          f"answered-correctly-again (attack success)={attack_success_rate*100:.1f}% ({n_correct}/{len(sample_ids)})")
    results[str(n_words)] = {
        "mean_rho": mean_rho, "mean_token_len": mean_tlen,
        "attack_success_rate": attack_success_rate, "n_correct": n_correct, "n_total": len(sample_ids),
    }

model.reset_hooks()

out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_dilution_attack_results.json"
with open(out_path, "w") as f:
    json.dump({"threshold": threshold, "features": features.tolist(), "results": results}, f, indent=2)

print("\n" + "=" * 70)
print("SUMMARY: dilution attack on DSG's rate-based gate")
print("=" * 70)
print(f"Threshold: {threshold:.4f}")
print(f"{'filler_words':>14} {'mean_token_len':>15} {'mean_rho':>10} {'attack_success':>15}")
for n_words in FILLER_WORD_COUNTS:
    r = results[str(n_words)]
    print(f"{n_words:>14} {r['mean_token_len']:>15.1f} {r['mean_rho']:>10.4f} {r['attack_success_rate']*100:>14.1f}%")

print("\nSaved:", out_path)
print("DONE")
