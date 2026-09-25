"""
Idea 5 (round 2): train our own SAE from scratch instead of using the
pretrained Gemma Scope release, then run it through DSG's own unchanged
selection + clamp pipeline.

User's idea: "can we make our own sae instead of using the existing one."
Round 1 (`imp-b2-sae-facts-tofu`) scoped this to a TOFU-based delta-latent SAE
and marked it out of scope (needs a new dataset, a new SAE training loop, and
new Truth-Ratio/KS-test evaluation infra that doesn't exist in this repo).
This attempt scopes the same underlying idea down to something tractable in
one sitting: train a small, standard (ReLU + L1, not the pretrained release's
JumpReLU) SAE on layer-3 residual-stream activations collected from THIS
project's own forget/retain corpora (bio-forget-corpus + wikitext, the exact
corpora DSG's own feature selection is calibrated on), instead of the
pretrained SAE's Pile-uncopyrighted training data. Everything downstream
(feature selection via get_top_features_percentile, whole-sequence gate
calibration, clamp intervention, evaluation) is DSG's own unchanged code --
only the SAE itself is swapped for a from-scratch one.

This is a deliberately modest SAE (d_sae=4096 vs the pretrained release's
16384, a few thousand training steps on ~300k tokens vs the release's
presumably much larger Pile-scale training run) -- the question this answers
is narrower than "can a custom SAE beat Gemma Scope's": it's "does training a
small SAE directly on the forget/retain domain, even at a fraction of the
scale, produce features usable enough for DSG's unlearning pipeline to work
at all, and if so, how does it compare."
"""
import gc
import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from functools import partial

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer

from evals.unlearning.utils.feature_activation import (
    get_shuffled_forget_retain_tokens,
    get_top_features_percentile,
)
from evals.unlearning.utils.metrics import calculate_MCQ_metrics
from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features

MODEL_NAME = "gemma-2-2b-it"
HOOK_NAME = "blocks.3.hook_resid_post"
D_MODEL = 2304
D_SAE = 4096
K_SPARSE = 32  # target L0; roughly matches the pretrained release's L0=142 out of 16384 (~0.87%) at our smaller width
LR = 3e-4
N_TRAIN_STEPS = 3000
BATCH_TOKENS = 2048
N_SEQ_PER_CORPUS = 150   # sequences of seq_len=1024 collected from each of forget/retain
SEQ_LEN = 1024
MULTIPLIER = 500
N_FEATURES = 20
RATIO_PERCENTILE = 95

ARTIFACTS_FOLDER = "artifacts_dynamic_bs1_bio/unlearning/gemma-2-2b-it"
OUT_DIR = "novelty_trials/imp2_5_custom_trained_sae"
DATASET_NAMES = ["wmdp-bio", "high_school_us_history", "college_computer_science",
                  "high_school_geography", "human_aging"]

device = "cuda"


class SimpleSAE(nn.Module):
    """TopK SAE: f = topk(relu(W_enc (x - b_dec) + b_enc), k), x_hat = W_dec f + b_dec.

    v1/v2 of this script used a plain ReLU + L1 SAE, but L1 (even at 60x its
    initial coefficient) never pushed L0 below ~3778/4096 -- the reconstruction
    gradient dominated the sparsity penalty regardless of coefficient, and the
    resulting "SAE" was really just a dense autoencoder (see run_v1_dense_failure.log,
    run_v2_still_dense.log). TopK sidesteps L1-coefficient tuning entirely by
    hard-zeroing every activation outside the top K per token, which guarantees
    exact sparsity by construction (matches the modern TopK-SAE approach)."""

    def __init__(self, d_in, d_sae, k):
        super().__init__()
        self.k = k
        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        nn.init.kaiming_uniform_(self.W_enc, a=np.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.t())
            self.W_dec /= self.W_dec.norm(dim=1, keepdim=True) + 1e-8

    def encode(self, x):
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        out = torch.zeros_like(pre)
        out.scatter_(-1, topk_idx, topk_vals)
        return out

    def decode(self, f):
        return f @ self.W_dec + self.b_dec

    def forward(self, x):
        f = self.encode(x)
        return self.decode(f), f

    @torch.no_grad()
    def normalize_decoder(self):
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp_min(1e-8)
        self.W_dec /= norms


print("[1] Loading model...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)

print("\n[2] Collecting layer-3 residual-stream activations from bio-forget-corpus + wikitext...")
forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(
    model, batch_size=N_SEQ_PER_CORPUS, seq_len=SEQ_LEN, dataset_fraction=100,
    forget_corpora="bio-forget-corpus", retain_corpora="wikitext",
)
print(f"  forget_tokens: {forget_tokens.shape}, retain_tokens: {retain_tokens.shape}")

def run_and_grab(batch):
    captured = {}

    def grab(t, hook):
        captured["resid"] = t.detach()
        return t

    model.reset_hooks()
    model.add_hook(HOOK_NAME, grab)
    model(batch, return_type=None, stop_at_layer=4)
    model.reset_hooks()
    return captured["resid"]


acts = []
with torch.no_grad():
    for tok_set, name in [(forget_tokens, "forget"), (retain_tokens, "retain")]:
        for i in range(0, tok_set.shape[0], 4):
            batch = tok_set[i:i + 4]
            resid = run_and_grab(batch)
            acts.append(resid[:, 1:, :].reshape(-1, D_MODEL).float().cpu())  # drop BOS position
        print(f"  collected activations from {name} corpus")

train_acts = torch.cat(acts, dim=0)
del acts
gc.collect()
torch.cuda.empty_cache()
print(f"  total training activations: {train_acts.shape} ({train_acts.shape[0]} tokens)")

print("\n[3] Training our own SAE from scratch...")
sae = SimpleSAE(D_MODEL, D_SAE, K_SPARSE).to(device)
opt = torch.optim.Adam(sae.parameters(), lr=LR)
n_tokens = train_acts.shape[0]
loss_history = []
for step in range(N_TRAIN_STEPS):
    idx = torch.randint(0, n_tokens, (BATCH_TOKENS,))
    x = train_acts[idx].to(device)
    x_hat, f = sae(x)
    loss = (x_hat - x).pow(2).mean()  # TopK already enforces sparsity; no L1 term needed
    opt.zero_grad()
    loss.backward()
    opt.step()
    sae.normalize_decoder()
    if step % 500 == 0 or step == N_TRAIN_STEPS - 1:
        l0 = (f > 0).float().sum(dim=-1).mean().item()
        print(f"  step {step}: mse={loss.item():.4f} l0={l0:.1f}")
    loss_history.append(float(loss.item()))

del train_acts
gc.collect()
torch.cuda.empty_cache()

print("\n[4] Computing feature sparsity on forget/retain with our custom SAE (reusing the same token sets)...")


def sparsity_and_acts(tokens, sae):
    all_acts = []
    with torch.no_grad():
        for i in range(0, tokens.shape[0], 4):
            batch = tokens[i:i + 4]
            resid = run_and_grab(batch)
            f = sae.encode(resid.float())
            f[:, 0, :] = 0.0  # mask BOS, matching DSG's own convention
            all_acts.append(f.cpu().numpy())
    sparsity = np.concatenate([a.reshape(-1, a.shape[-1]) for a in all_acts], axis=0).mean(axis=0)
    return sparsity, all_acts


fs_fgt, act_fgt_custom = sparsity_and_acts(forget_tokens, sae)
fs_ret, act_ret_custom = sparsity_and_acts(retain_tokens, sae)

os.makedirs(os.path.join(OUT_DIR, "sparsities"), exist_ok=True)
import pickle
with open(os.path.join(OUT_DIR, "sparsities", "act_fgt.pkl"), "wb") as f:
    pickle.dump(act_fgt_custom, f)
with open(os.path.join(OUT_DIR, "sparsities", "act_ret.pkl"), "wb") as f:
    pickle.dump(act_ret_custom, f)

print("\n[5] Selecting top-20 features with DSG's own selection function (unchanged)...")
sel, perc = get_top_features_percentile(
    fs_fgt, fs_ret, ratio_percentile=RATIO_PERCENTILE,
    folder_name=os.path.join(OUT_DIR, "sparsities"), n_features_lst=[N_FEATURES],
)
features = sel[:N_FEATURES].tolist()
gate_threshold = perc[str(N_FEATURES)]
print(f"  features: {features}")
print(f"  gate threshold: {gate_threshold:.4f}")
del act_fgt_custom, act_ret_custom
gc.collect()


class SAEAdapter:
    """Adapts SimpleSAE to the .encode()/.decode() interface anthropic_clamp_resid_SAE_features expects."""
    def __init__(self, sae):
        self.sae = sae

    def encode(self, resid):
        return self.sae.encode(resid.float()).to(resid.dtype)

    def decode(self, f):
        return self.sae.decode(f.float()).to(f.dtype)


print("\n[6] Evaluating DSG's unchanged clamp intervention using our custom SAE...")
sae_adapter = SAEAdapter(sae)
hook = partial(anthropic_clamp_resid_SAE_features, sae=sae_adapter, features_to_ablate=features,
               multiplier=MULTIPLIER, activation_threshold=gate_threshold)
model.reset_hooks()
model.add_hook(HOOK_NAME, hook)
custom_sae_results = {}
for dataset_name in DATASET_NAMES:
    m = calculate_MCQ_metrics(model, 1, ARTIFACTS_FOLDER, dataset_name=dataset_name, target_metric="correct",
                               split="all", verbose=False)
    custom_sae_results[dataset_name] = m["mean_correct"]
    print(f"  {dataset_name}: {m['mean_correct']*100:.3f}%")
model.reset_hooks()
mmlu_custom = float(np.mean([custom_sae_results[d] for d in DATASET_NAMES if d != "wmdp-bio"]))

print("\n" + "=" * 70)
print("FINAL SUMMARY: custom-trained SAE vs pretrained Gemma Scope SAE (both feeding DSG's unchanged pipeline)")
print("=" * 70)
print(f"{'config':>32} {'WMDP-bio':>12} {'MMLU_avg':>12}")
print(f"{'pretrained Gemma Scope (cached)':>32} {29.368:>11.3f}% {99.412:>11.3f}%")
print(f"{'our custom-trained SAE':>32} {custom_sae_results['wmdp-bio']*100:>11.3f}% {mmlu_custom*100:>11.3f}%")

torch.save(sae.state_dict(), os.path.join(OUT_DIR, "custom_sae_weights.pt"))
out = {
    "d_sae": D_SAE,
    "n_train_steps": N_TRAIN_STEPS,
    "n_training_tokens": n_tokens,
    "final_loss": loss_history[-1],
    "features": features,
    "gate_threshold": gate_threshold,
    "custom_sae_results": custom_sae_results,
    "mmlu_custom": mmlu_custom,
}
with open(os.path.join(OUT_DIR, "imp2_5_results.json"), "w") as f:
    json.dump(out, f, indent=2)
print("\nSaved:", os.path.join(OUT_DIR, "imp2_5_results.json"))
print("DONE")
