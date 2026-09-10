import contextlib
import io
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.getcwd(), "dynamic_sae_guardrails"))

from transformer_lens import HookedTransformer
from sae_lens import SAE

from evals.unlearning.utils.metrics import modify_model

MODEL_NAME = "gemma-2-2b-it"
SAE_NAME = "gemma-scope-2b-pt-res"
SAE_BLOCK = "layer_3/width_16k/average_l0_142"
SEQ_LENS = [256, 512, 1024]
N_SAMPLES = 100
N_WARMUP = 10
MULTIPLIER = 500
FEATURES = np.arange(20)  # identity of features doesn't affect timing
THRESHOLD = 0.5

device = "cuda"

print("[1] Loading model + SAE...")
model = HookedTransformer.from_pretrained(MODEL_NAME, device=device, dtype=torch.bfloat16)
sae, _, _ = SAE.from_pretrained(release=SAE_NAME, sae_id=SAE_BLOCK, device=device)


def make_tokens(seq_len):
    # Random valid token ids (content doesn't affect latency, only shape)
    torch.manual_seed(0)
    ids = torch.randint(low=100, high=model.cfg.d_vocab - 1, size=(1, seq_len), device=device)
    ids[:, 0] = model.tokenizer.bos_token_id
    return ids


def time_forward(tokens, n_samples, n_warmup):
    # The DSG hook prints [DSG DEBUG] diagnostics on every call; that console
    # I/O has real wall-clock cost that has nothing to do with the actual
    # SAE/intervention compute this benchmark is meant to measure, so silence
    # stdout for the timed region (matches how the paper's Table 15 would only
    # be measuring compute, not print overhead).
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        for _ in range(n_warmup):
            model(tokens, return_type=None)
        torch.cuda.synchronize()
        times = []
        for _ in range(n_samples):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(tokens, return_type=None)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    return np.array(times)


results = {}
for seq_len in SEQ_LENS:
    print(f"\n{'='*60}\nseq_len={seq_len}\n{'='*60}")
    tokens = make_tokens(seq_len)

    print("[Original model]")
    model.reset_hooks()
    t_orig = time_forward(tokens, N_SAMPLES, N_WARMUP)
    print(f"  mean={t_orig.mean():.4f}s  std={t_orig.std():.4f}s")

    print("[DSG (dynamic clamping)]")
    model.reset_hooks()
    modify_model(
        model, sae,
        intervention_method="clamp_feature_activation",
        features_to_ablate=FEATURES,
        multiplier=MULTIPLIER,
        activation_threshold=THRESHOLD,
    )
    t_dsg = time_forward(tokens, N_SAMPLES, N_WARMUP)
    model.reset_hooks()
    print(f"  mean={t_dsg.mean():.4f}s  std={t_dsg.std():.4f}s")

    overhead_pct = (t_dsg.mean() - t_orig.mean()) / t_orig.mean() * 100
    results[str(seq_len)] = {
        "original_mean": float(t_orig.mean()), "original_std": float(t_orig.std()),
        "dsg_mean": float(t_dsg.mean()), "dsg_std": float(t_dsg.std()),
        "overhead_pct": float(overhead_pct),
    }
    print(f"  overhead: {overhead_pct:+.2f}%")

out_path = "/tmp/claude-1001/-home-amaloch-projects-mechunlearn-project/de45dc44-5b39-4006-bcf0-de6680f6be3e/scratchpad/dsg_latency_results.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"{'seq_len':>8} {'original (s)':>16} {'DSG (s)':>16} {'overhead':>10}")
for seq_len in SEQ_LENS:
    r = results[str(seq_len)]
    print(f"{seq_len:>8} {r['original_mean']:>9.4f}±{r['original_std']:.4f} "
          f"{r['dsg_mean']:>9.4f}±{r['dsg_std']:.4f} {r['overhead_pct']:>+9.2f}%")

print("\nSaved:", out_path)
print("DONE")
