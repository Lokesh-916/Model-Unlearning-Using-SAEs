"""Shared helpers for the later server jobs (gpuws). Imported by cluster/<job>.py.

* DSG_TINY=1 -> every model is a tiny random Gemma-2 (CPU smoke tests); the real gemma tokenizer is
  still used (small, offline). Never use tiny mode for results.
* job_dir(name) -> $DSG_RESULTS/jobs/<name>; summary(name, obj) writes summary.json with the hardware
  label (per-GPU baselines: gpuws numbers are never mixed with lab-PC numbers).
* dsg_guard(model): the DSG-faithful clamp (paper config N=20, retain 95, x500, layer 3) on an HF model,
  features/tau from the bio activation cache (seed 0) - the same features the sanity gate checks.
* Hazardous text never goes to results/: generations and prompts only under $DSG_PRIVATE (hashes elsewhere).
"""
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import torch

from dsgx import paths
from dsgx.util import atomic_write_json, gpu_info, hardware_label, now_iso, package_versions

TINY = os.environ.get("DSG_TINY") == "1"
HF_2B = "google/gemma-2-2b-it"


def device():
    return "cuda" if torch.cuda.is_available() and not TINY else "cpu"


def job_dir(name: str) -> Path:
    d = paths.results_dir() / "jobs" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def private_dir(name: str) -> Path:
    d = paths.private_dir() / "jobs" / name
    d.mkdir(parents=True, exist_ok=True)
    os.chmod(d, 0o700)
    return d


def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def log(name, msg):
    print(f"[{name}] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def summary(name: str, obj: dict):
    out = {"job": name, "time": now_iso(), "hardware": gpu_info(), "hardware_label": hardware_label(),
           "versions": package_versions(), "tiny": TINY, "slurm_job": os.environ.get("SLURM_JOB_ID"), **obj}
    atomic_write_json(job_dir(name) / "summary.json", out)
    return out


def gpu_free_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.mem_get_info()[0] / 2**30


# ----------------------------------------------------------------------------- models
def load_tok(name=HF_2B):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF_2B if TINY else name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def tiny_config(vocab_size=256000):
    from transformers import Gemma2Config

    return Gemma2Config(vocab_size=vocab_size, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
                        num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=1024,
                        sliding_window=64, attn_implementation="eager")


def load_lm(name_or_path=HF_2B, dtype=torch.bfloat16, train=False):
    """HF causal LM on the job device. Tiny mode: a seeded random tiny Gemma-2 (or the saved tiny model
    if name_or_path is a directory written by save_pretrained)."""
    from transformers import AutoModelForCausalLM, Gemma2ForCausalLM

    if TINY:
        p = Path(str(name_or_path))
        if p.is_dir() and (p / "config.json").exists():
            m = AutoModelForCausalLM.from_pretrained(p, dtype=torch.float32)
        else:
            torch.manual_seed(0)
            m = Gemma2ForCausalLM(tiny_config())
        return m.to("cpu")
    m = AutoModelForCausalLM.from_pretrained(name_or_path, dtype=dtype, attn_implementation="eager")
    if train:
        m.gradient_checkpointing_enable()
        m.config.use_cache = False
    return m.to(device())


class TinySAE(torch.nn.Module):
    """JumpReLU-shaped random SAE matching a tiny model's width (smoke tests only)."""

    def __init__(self, d=64, n=256):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.W_enc = torch.nn.Parameter(torch.randn(d, n, generator=g) / d ** 0.5)
        self.W_dec = torch.nn.Parameter(torch.randn(n, d, generator=g) / n ** 0.5)
        self.b_enc = torch.nn.Parameter(torch.zeros(n))
        self.b_dec = torch.nn.Parameter(torch.zeros(d))
        self.threshold = torch.nn.Parameter(torch.full((n,), 0.05))

    def encode(self, x):
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        return torch.relu(pre) * (pre > self.threshold)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec


def load_sae(release="gemma-scope-2b-pt-res", sae_id="layer_3/width_16k/average_l0_142", dtype="bfloat16"):
    if TINY:
        return TinySAE()
    from dsgx.models.loader import get_sae

    return get_sae(release, sae_id, device(), dtype)


def dsg_features(case="bio", n=20, pct=95, sae_release="gemma-scope-2b-pt-res",
                 sae_id="layer_3/width_16k/average_l0_142"):
    """(features, tau) of the DSG paper config from the activation cache (no model needed)."""
    if TINY:
        return list(range(0, 40, 2)), 0.1
    from dsgx.data import activation_cache as ac
    from dsgx.methods import dsg

    cache = ac.ActivationCache(ac.open_cache("gemma-2-2b-it", sae_release, sae_id, f"{case}-forget-corpus",
                                             "wikitext", 0))
    feats = dsg.select_features(cache, n, pct)
    return [int(f) for f in feats], float(dsg.calibrate_tau(cache, feats, pct))


def dsg_guard(model, case="bio", layer=3, multiplier=500):
    """Install the DSG-faithful clamp on an HF model. Returns (hook, handle, info)."""
    from dsgx.train.core import add_dsg_hook_hf

    sae = load_sae()
    feats, tau = dsg_features(case)
    if TINY:
        layer = 1
    hook, handle = add_dsg_hook_hf(model, sae, feats, multiplier, tau, layer)
    return hook, handle, {"features": feats, "tau": tau, "multiplier": multiplier, "layer": layer}


# ----------------------------------------------------------------------------- data
def forget_passages(case="bio", n=None, max_chars=3000, seed=0):
    """Forget-corpus passages (HAZARDOUS text: kept in memory only, never printed or written)."""
    import random

    from dsgx.data.corpora import load_forget_docs

    if TINY:
        return [f"tiny forget passage {i} " * 20 for i in range(n or 16)]
    docs = load_forget_docs(f"{case}-forget-corpus")
    idx = list(range(len(docs)))
    random.Random(seed).shuffle(idx)
    return [docs[i][:max_chars] for i in idx[: n or len(idx)]]


def retain_passages(n=None, max_chars=3000):
    if TINY:
        return [f"tiny retain passage about geography {i} " * 20 for i in range(n or 16)]
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    out, cur = [], ""
    for t in ds["text"]:
        cur += t
        if len(cur) >= max_chars:
            out.append(cur[:max_chars])
            cur = ""
            if n and len(out) >= n:
                break
    return out


def mcq_items(dataset, split="test", n=None):
    from dsgx.data.mcq import load_mcq
    from dsgx.data.splits import get_split

    items = load_mcq(dataset)
    ids = get_split(dataset, split)
    return [items[i] for i in ids[: n or len(ids)]]


def harness_eval(exp_id, label, method, weights=None, datasets=("@forget", "@utility"), split="test", seed=0,
                 attack=None, limit=None, model=None):
    """One harness TEST run (bs=1, both views) -> run dir. Same format as every lab-PC run."""
    from dsgx.run import run

    cfg = {"exp_id": exp_id, "case": "bio", "split": split, "view": "both", "seed": seed, "batch_size": 1,
           "datasets": list(datasets), "dataset_label": label, "method": dict(method),
           "attack": dict(attack or {"name": "none"})}
    if weights or model:
        cfg["model"] = {**(model or {}), **({"weights": str(weights)} if weights else {})}
    if limit:
        cfg["limit"] = limit
    return run(cfg)


def headlines(run_dirs: dict) -> dict:
    """{name: {forget, utility}} (raw view, mean/CI/n) from harness run dirs."""
    out = {}
    for k, d in run_dirs.items():
        m = read_json(Path(d) / "metrics.json")
        if m:
            v = m["views"]["raw"]
            out[k] = {"forget": v.get("forget"), "utility": (v.get("utility") or {}).get("pooled")}
    return out


def require_gpu(min_gb=40):
    if TINY:
        return
    free = gpu_free_gb()
    if free is None or free < min_gb:
        print(f"[precheck] ABORT: {free} GB GPU memory free (< {min_gb})", flush=True)
        sys.exit(75)


def read_json(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default
