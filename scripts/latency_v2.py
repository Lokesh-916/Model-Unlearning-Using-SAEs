#!/usr/bin/env python
"""Latency of DSG, StreamGuard v1, v2 and v2-R against the unguarded model (measurement only; nothing is tuned).

Batch size 1, bfloat16, one GPU that runs nothing else. Two workloads:
  mcq     one forward pass, read the A/B/C/D probabilities at the last position (one softmax over the vocabulary; the
          evaluation harness's scorer dsgx.eval.mcq_eval.answer_probs recomputes the unembedding in chunks twice, about
          200 ms per question whatever the method, which would hide the overheads; it is used only in the checks)
  answer  greedy answer of exactly 96 new tokens with a KV cache (end tokens are ignored so every cell does equal work)
Prompts are harmless: an MMLU DEV question (hazard-adjacent subjects excluded) behind WikiText filler, cut to exactly
64 / 256 / 1024 / 2048 tokens inside the chat template. Because these prompts are benign, the gate decision is FORCED
(not fired / fired) after the gate has been computed; the gate itself always runs. Logging inside the hooks is off.

Paths (nf = gate not fired, f = gate fired on the prompt):
  base      plain model
  dsg       one pass; layer-3 SAE encode + decode at every step; f = the -500 clamp is applied
  v1        one pass; CUSUM score on the same SAE features; f = the clamp and one more SAE decode
  v2        pass 1: prompt through layer 8 + layer-8 SAE + CUSUM (frozen l08_score). Pass 2: nf = the plain model,
            f = the model with the clamp. In `answer` the gate is checked on the prompt only
  v2r       nf: the layer-8 gate runs inside the answering pass and after every generated token. f, mcq: pass 1 as v2,
            pass 2 = refusal vector at layer 10. f, answer: the prompt pass with the gate, then the prompt is fed again
            with the refusal vector (prompt + first 8 generated tokens); the gate is latched
  v2cached  v2, but pass 2 starts at layer 3 from pass 1's activations (layers 0-2 not recomputed).
            NOT USED IN THE PAPER'S RESULTS
TransformerLens copies the transposed unembedding matrix (1.2 GB) on every call; the copy is made once here, for every
method alike (bit-identical logits, checked), so that it does not hide the overheads. The layers are unchanged
TransformerLens code, which is slower than an optimised server: read the overheads, not the absolute times.
Methods are interleaved in rotating order inside each repetition. A run stops if another compute process is on the GPU.

  export PYTHONPATH=src DSG_CACHE=...            # DSG_CACHE needs the bio activation cache (DSG's threshold)
  python scripts/latency_v2.py bench  --json latency_v2.json          # ~10 h on an RTX 2000 Ada; resumes cell by cell
  python scripts/latency_v2.py report --json latency_v2.json --md latency_v2.md
"""
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

LENGTHS = [64, 256, 1024, 2048]
WARMUP, REPS, MAX_NEW = 10, 50, 96
METHODS = ["base", "dsg-nf", "dsg-f", "v1-nf", "v1-f", "v2-nf", "v2-f", "v2r-nf", "v2r-f", "v2cached-nf", "v2cached-f"]
LABEL = {"base": "base model", "dsg": "DSG", "v1": "StreamGuard v1", "v2": "StreamGuard v2", "v2r": "StreamGuard v2-R",
         "v2cached": "v2 cached (not used in the paper's results)"}
# SAEs a method needs; the bytes of the others are subtracted from its peak memory (all are loaded in one process)
NEEDS = {"base": (), "dsg": ("L3",), "v1": ("L3",), "v2": ("L3", "L8"), "v2r": ("L8",), "v2cached": ("L3", "L8")}
# measured gate fire rates, TEST seed 0: (MMLU false alarms, WMDP-Bio). v2 / v2-R: 0.019 and 0.94; DSG / v1 from the
# StreamGuard v2 TEST tables
FIRE = {"dsg": (0.017, 0.878), "v1": (0.046, 0.926), "v2": (0.019, 0.94), "v2r": (0.019, 0.94), "v2cached": (0.019, 0.94)}
WORKLOADS = {"mcq": "multiple choice (one forward pass)", "answer": "written answer (96 generated tokens)"}


# ------------------------------------------------------------------------------------------------ prompts
def fit_ids(prefix, filler, body, suffix, n):
    """prefix + filler (as much as needed) + body + suffix = exactly n ids; None if the body does not fit."""
    pad = n - len(prefix) - len(body) - len(suffix)
    if pad < 0 or pad > len(filler):
        return None
    return list(prefix) + list(filler[:pad]) + list(body) + list(suffix)


def build_prompts(tok, n_prompts, lengths):
    """{workload: {L: [ids]}} from MMLU DEV questions (utility subjects) and WikiText; deterministic."""
    import random

    from datasets import load_dataset

    from dsgx.data.mcq import HAZARD_ADJACENT, load_mcq, pre_question, utility_subjects
    from dsgx.data.splits import get_split

    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    wiki = [t for t in load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")["text"] if len(t) > 200]
    rng = random.Random(0)
    subjects = [s for s in utility_subjects("bio") if s not in HAZARD_ADJACENT["cyber"] and s != "security_studies"]
    items = [load_mcq(s)[i] for s in subjects for i in get_split(s, "dev")]
    rng.shuffle(items)
    head, tail = enc("<bos><start_of_turn>user\n"), enc("<end_of_turn>\n<start_of_turn>model\n")
    out = {w: {L: [] for L in lengths} for w in WORKLOADS}
    for it in items:
        choices = "".join(f"\n{c}. {x}" for c, x in zip("ABCD", it.choices))
        body = {"mcq": enc(pre_question(it.dataset) + it.question + choices), "answer": enc(it.question + "\nAnswer briefly.")}
        filler = []
        while len(filler) < max(lengths):
            filler += enc(wiki[rng.randrange(len(wiki))] + "\n")
        for w in WORKLOADS:
            suf = tail + (enc("Answer: (") if w == "mcq" else [])
            for L in lengths:
                if len(out[w][L]) < n_prompts:
                    ids = fit_ids(head, filler, body[w], suf, L)
                    if ids is not None:
                        out[w][L].append(ids)
        if all(len(v) == n_prompts for d in out.values() for v in d.values()):
            break
    short = {(w, L): len(v) for w, d in out.items() for L, v in d.items() if len(v) < n_prompts}
    if short:
        raise RuntimeError(f"not enough prompts: {short}")
    return out


# ------------------------------------------------------------------------------------------------ runners
def other_gpu_processes():
    r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], capture_output=True, text=True)
    return [int(x) for x in r.stdout.split() if x.strip().isdigit() and int(x) != os.getpid()]


class Runner:
    def __init__(self):
        import torch

        from dsgx.eval.mcq_eval import ANSWER_STRINGS
        from dsgx.methods.registry import make_method
        from dsgx.models.loader import get_bundle
        from experiments.E5 import batchgen as bg
        from experiments.E5.common import load_frozen_v2r
        from experiments.E5.gen import load_gate
        from experiments.V2.common import DSG_FEATURES, METHOD, SG1, THRESHOLD

        torch.set_grad_enabled(False)
        self.torch, self.bg = torch, bg
        bg.end_ids = lambda model: set()               # exactly 96 tokens: end tokens never stop a generation
        b = self.b = get_bundle()
        self.m = b.model
        self.dev = b.device
        u = self.m.unembed
        self.slow_unembed = u.forward
        W = u.W_U.T.contiguous()
        u.forward = lambda x: torch.nn.functional.linear(x, W, u.b_U)
        self.dsg = make_method({"name": "dsg-faithful", "n_features": 20, "retain_pct": 95, "multiplier": 500}, b, 0)
        self.v1 = make_method(SG1, b, 0)
        self.v2 = make_method(METHOD, b, 0)
        for h in (self.dsg.hook, self.v1.hook, self.v2.hook):
            h.record = False
        if list(self.dsg.features) != list(DSG_FEATURES):
            raise RuntimeError("DSG features differ from the frozen list")
        self.gate = load_gate(b)                       # (sae8, feats, w1, w0)
        fz = load_frozen_v2r()
        self.rL = fz["layer"]
        self.rvec = float(fz["selection"]["s"]) * fz["norm"] * torch.tensor(fz["dir"], dtype=torch.float32, device=self.dev)
        self.clamp = (b.sae, DSG_FEATURES, 500.0, b.hook_name)
        self.threshold = THRESHOLD
        self.answer_tokens = self.m.to_tokens(ANSWER_STRINGS, prepend_bos=False).flatten()
        size = lambda sae: sum(p.numel() * p.element_size() for p in sae.parameters())  # noqa: E731
        self.sae_bytes = {"L3": size(b.sae), "L8": size(self.gate[0])}
        self.harness_readout = False                   # checks only: the evaluation scorer instead of the plain readout
        self.gate_fired = None                         # natural decision of the last v2-style pass 1 (reported only)

    # -- shared pieces
    def probs(self, toks, model=None):
        from dsgx.eval.mcq_eval import answer_probs

        m = model or self.m
        if self.harness_readout:
            nti = self.torch.tensor([toks.shape[1] - 1], device=self.dev)
            return answer_probs(m, toks, nti, self.answer_tokens).float().cpu()
        x = m.ln_final(m(toks, return_type=None, stop_at_layer=m.cfg.n_layers)[:, -1:])
        z = m.unembed(x)[0, -1].float()
        cap = m.cfg.output_logits_soft_cap
        z = cap * self.torch.tanh(z / cap) if cap and cap > 0 else z
        return self.torch.softmax(z, -1)[self.answer_tokens].view(2, 4).max(0)[0][None].cpu()

    def pass1(self, toks):
        from experiments.V2.method import l08_score

        s = l08_score(self.m, self.gate[0], 8, *self.gate[1:], toks)
        self.gate_fired = s > self.threshold
        return s

    def pass1_cached(self, toks, kv=None):
        """pass1 that also returns the input of layer 3 (and fills kv's layers 0-8 if a KV cache is given)."""
        from experiments.V2.method import cusum_max

        st, m = {}, self.m
        keep = lambda r, hook: st.__setitem__(hook.name, r)  # noqa: E731
        m.reset_hooks()
        m.run_with_hooks(toks, stop_at_layer=9, past_kv_cache=kv,
                         fwd_hooks=[("blocks.3.hook_resid_pre", lambda r, hook: keep(r, hook)),
                                    ("blocks.8.hook_resid_post", lambda r, hook: keep(r, hook))])
        sae, feats, w1, w0 = self.gate
        x = (sae.encode(st["blocks.8.hook_resid_post"][0])[1:, feats] > 0).float()
        s = cusum_max((x @ w1 + (1 - x) @ w0).tolist())
        self.gate_fired = s > self.threshold
        return s, st["blocks.3.hook_resid_pre"]

    def clamp_hook(self):
        sae, feats, mult, name = self.clamp
        return name, self.bg.BatchClamp(sae, feats, self.torch.tensor([True], device=self.dev), mult)

    # -- workload (a): one forward pass
    def mcq(self, meth, toks):
        m, L = self.m, toks.shape[1]
        name, _, path = meth.partition("-")
        fire = path == "f"
        m.reset_hooks()
        try:
            if name == "base":
                return self.probs(toks)
            if name == "dsg":
                self.dsg.hook.tau = -1.0 if fire else 2.0
                m.add_hook(self.b.hook_name, self.dsg.hook)
                return self.probs(toks)
            if name == "v1":
                self.v1.hook.forced, self.v1.hook.lengths = [fire], [L]
                m.add_hook(self.b.hook_name, self.v1.hook)
                return self.probs(toks)
            if name == "v2cached":
                _, r3 = self.pass1_cached(toks)
                m.reset_hooks()
                if fire:
                    m.add_hook(*self.clamp_hook())
                return self.probs(toks, _FromLayer3(m, r3))
            self.pass1(toks)
            m.reset_hooks()
            if fire and name == "v2":                  # frozen v2 response object (no hook at all when not fired)
                self.v2.hook.forced, self.v2.hook.lengths = [True], [L]
                m.add_hook(self.b.hook_name, self.v2.hook)
            elif fire and name == "v2r":
                m.add_hook(f"blocks.{self.rL}.hook_resid_post", self.bg.BatchSteer(self.rvec[None]))
            return self.probs(toks)
        finally:
            m.reset_hooks()

    # -- workload (b): 96 greedy tokens
    def answer(self, meth, toks, max_new=MAX_NEW):
        m, bg = self.m, self.bg
        name, _, path = meth.partition("-")
        fire = path == "f"
        if name == "base":
            return bg.phase1(m, toks, max_new=max_new)[0]
        if name in ("dsg", "v1"):
            hook = StreamHook(self, self.dsg.hook.sae, self.dsg.features if name == "dsg" else self.v1.gate.features,
                              fire, self.v1.gate if name == "v1" else None)
            return self.greedy(toks, hook, max_new)
        row = {"clamp": True, "vec": None} if name != "v2r" else {"clamp": False, "vec": self.rvec, "limit": 8}
        if name == "v2":
            self.pass1(toks)
            if not fire:
                return bg.phase1(m, toks, max_new=max_new)[0]
            return bg.continue_batch(m, toks, max_new, [row], self.clamp, self.rL)[0][0]
        if name == "v2r":
            det = bg.L08Detector(*self.gate)
            if not fire:                               # phase1 only records fires: the not-fired path is never left
                out = bg.phase1(m, toks, "blocks.8.hook_resid_post", det, max_new)[0]
                self.gate_fired = det.fired()
                return out
            bg.phase1(m, toks, "blocks.8.hook_resid_post", det, 0)
            self.gate_fired = det.fired()
            return bg.continue_batch(m, toks, max_new, [row], self.clamp, self.rL)[0][0]
        from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache as KV

        kv1 = KV.init_cache(m.cfg, self.dev, 1)
        _, r3 = self.pass1_cached(toks, kv1)
        m.reset_hooks()
        try:
            if fire:
                m.add_hook(*self.clamp_hook())
            kv = KV.init_cache(m.cfg, self.dev, 1)
            last = m(r3, start_at_layer=3, past_kv_cache=kv)[0, -1]
            for i in range(3):
                kv.entries[i] = kv1.entries[i]
            kv.previous_attention_mask = kv1.previous_attention_mask
            del kv1, r3
            return self.loop(last, kv, max_new)
        finally:
            m.reset_hooks()

    def loop(self, last, kv, max_new):
        out = []
        for _ in range(max_new):
            nxt = int(last.argmax())
            out.append(nxt)
            last = self.m(self.torch.tensor([[nxt]], device=self.dev), past_kv_cache=kv)[0, -1]
        return out

    def greedy(self, toks, hook, max_new):
        from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache as KV

        m = self.m
        m.reset_hooks()
        m.add_hook(self.b.hook_name, hook)
        try:
            kv = KV.init_cache(m.cfg, self.dev, 1)
            return self.loop(m(toks, past_kv_cache=kv)[0, -1], kv, max_new)
        finally:
            m.reset_hooks()

    def run(self, workload, meth, toks):
        return self.mcq(meth, toks) if workload == "mcq" else self.answer(meth, toks)


class _FromLayer3:
    """Model proxy for answer_probs: the forward starts at layer 3 from a stored residual."""

    def __init__(self, model, resid3):
        self._m, self._r = model, resid3

    def __call__(self, _tokens, **kw):
        return self._m(self._r, start_at_layer=3, **kw)

    def __getattr__(self, k):
        return getattr(self._m, k)


class StreamHook:
    """The streaming gate hook of dsgx.gen.stream (DSG: rho of the fire flags; v1: CUSUM of the per-token LLR, the score
    recomputed on the whole sequence at every token), with the decision forced to `on` after it has been computed."""

    def __init__(self, r, sae, feats, on, cusum_gate=None):
        self.torch, self.sae, self.feats, self.on, self.g = r.torch, sae, [int(f) for f in feats], on, cusum_gate
        self.fires = []

    def __call__(self, resid, hook=None):
        from dsgx.methods.gates import score_cusum, token_llr

        torch, sae, feats = self.torch, self.sae, self.feats
        acts = sae.encode(resid)
        if not self.fires:
            acts[:, 0, :] = 0.0
        err = resid - sae.decode(acts)
        tgt = acts[:, :, feats]
        any_fire = (tgt > 0).any(dim=2)[0]
        if self.g is None:
            self.fires += any_fire.tolist()
            self.score = sum(self.fires) / len(self.fires)
        else:
            self.fires += token_llr(tgt[0] > 0, self.g.w1, self.g.w0).tolist()
            self.score = score_cusum(torch.tensor(self.fires), len(self.fires))[0]
        if not self.on:
            return resid
        mask = any_fire[None, :, None]
        acts[:, :, feats] = torch.where(mask, torch.full_like(tgt, -500.0), tgt)
        return sae.decode(acts) + err


# ------------------------------------------------------------------------------------------------ bench
def checks(r, prompts, n=3):
    """The timed code gives the frozen code's outputs: v2 mcq = the sg2-l08 method object; cached v2 = v2."""
    torch, out = r.torch, {}
    r.harness_readout = True
    x = torch.randn(1, 8, r.m.cfg.d_model, device=r.dev, dtype=r.m.unembed.W_U.dtype)
    out["unembed_bit_identical"] = bool(torch.equal(r.m.unembed(x), r.slow_unembed(x)))
    for L in LENGTHS:
        bad = {"v2_mcq_vs_method": 0, "cached_mcq_argmax": 0, "cached_mcq_maxdiff": 0.0, "cached_answer": 0, "forced_equal_base": 0}
        for ids in prompts["mcq"][L][:n]:
            t = torch.tensor([ids], device=r.dev)
            r.v2.set_lengths([L])
            r.v2.before_forward(t, [L])
            natural = bool(r.v2.hook.forced[0])
            p = r.probs(t)
            r.v2.hook.records, r.v2._scores = [], []
            bad["v2_mcq_vs_method"] += int(not torch.equal(p, r.mcq("v2-f" if natural else "v2-nf", t)))
            for q in ("nf", "f"):
                a, c = r.mcq(f"v2-{q}", t), r.mcq(f"v2cached-{q}", t)
                bad["cached_mcq_argmax"] += int(a.argmax() != c.argmax())
                bad["cached_mcq_maxdiff"] = max(bad["cached_mcq_maxdiff"], float((a - c).abs().max()))
            bad["forced_equal_base"] += int(not torch.equal(r.mcq("base", t), r.mcq("v2-nf", t)))
        for nm in ("mcq", "answer"):
            u, fast = r.m.unembed, r.m.unembed.forward
            t = torch.tensor([prompts[nm][L][0]], device=r.dev)
            a = r.answer("base", t, 8)
            u.forward = r.slow_unembed
            bad["unembed_tokens"] = bad.get("unembed_tokens", 0) + int(a != r.answer("base", t, 8))
            u.forward = fast
        bad["plain_readout_argmax"] = 0
        for ids in prompts["mcq"][L][:n]:
            t = torch.tensor([ids], device=r.dev)
            a = r.mcq("base", t).argmax()
            r.harness_readout = False
            bad["plain_readout_argmax"] += int(a != r.mcq("base", t).argmax())
            r.harness_readout = True
        for ids in prompts["answer"][L][:n]:
            t = torch.tensor([ids], device=r.dev)
            for q in ("nf", "f"):
                bad["cached_answer"] += int(r.answer(f"v2-{q}", t, 24) != r.answer(f"v2cached-{q}", t, 24))
        out[str(L)] = bad
    r.harness_readout = False
    return out


def bench(a):
    import torch

    from dsgx.util import atomic_write_json

    path = Path(a.json)
    if other_gpu_processes():
        raise SystemExit("another compute process is on the GPU; not timing")
    r = Runner()
    prompts = build_prompts(r.m.tokenizer, WARMUP + REPS, LENGTHS)
    rec = json.loads(path.read_text()) if path.exists() else {"cells": {}}
    if a.check or "checks" not in rec:
        rec["checks"] = checks(r, prompts)
        print("checks", json.dumps(rec["checks"]), flush=True)
        if a.check:
            return
    rec["meta"] = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "dtype": "bfloat16", "batch_size": 1,
                   "warmup": a.warmup, "reps": a.reps, "max_new": MAX_NEW, "lengths": LENGTHS, "methods": METHODS,
                   "sae_bytes": r.sae_bytes, "fire_rates": FIRE}
    for w in WORKLOADS:
        for L in a.lengths:
            key = f"{w}|{L}"
            if key in rec["cells"]:
                continue
            toks = [torch.tensor([ids], device=r.dev) for ids in prompts[w][L]]
            times, peak, natural = {m: [] for m in METHODS}, {m: 0 for m in METHODS}, []
            load0, t_cell = os.getloadavg()[0], time.time()
            for i in range(a.warmup + a.reps):
                if other_gpu_processes():
                    raise SystemExit(f"another compute process appeared on the GPU during {key}; cell discarded")
                k = i % len(METHODS)
                for meth in METHODS[k:] + METHODS[:k]:
                    t = toks[i % len(toks)]
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    t0 = time.perf_counter()
                    r.run(w, meth, t)
                    torch.cuda.synchronize()
                    dt = time.perf_counter() - t0
                    if i >= a.warmup:
                        times[meth].append(dt * 1000)
                        peak[meth] = max(peak[meth], torch.cuda.max_memory_allocated())
                    if meth == "v2-nf":
                        natural.append(bool(r.gate_fired))
            cell = {}
            for meth in METHODS:
                unused = sum(v for k2, v in r.sae_bytes.items() if k2 not in NEEDS[meth.partition("-")[0]])
                cell[meth] = {"median_ms": float(np.median(times[meth])), "p95_ms": float(np.percentile(times[meth], 95)),
                              "mean_ms": float(np.mean(times[meth])), "n": len(times[meth]),
                              "peak_gib": (peak[meth] - unused) / 2**30}
            rec["cells"][key] = {"methods": cell, "natural_v2_fires": int(sum(natural)), "n_prompts": len(natural),
                                 "load_before": load0, "load_after": os.getloadavg()[0], "wall_s": time.time() - t_cell}
            atomic_write_json(path, rec)
            print(key, f"{time.time() - t_cell:.0f}s", {m: round(c["median_ms"], 1) for m, c in cell.items()}, flush=True)


# ------------------------------------------------------------------------------------------------ report (CPU)
def expected_ms(nf, f, rate):
    return (1 - rate) * nf + rate * f


def pct(x, base):
    return f"{100 * (x / base - 1):+.1f} %"


def report_md(rec):
    out = []
    for w, title in WORKLOADS.items():
        out += [f"## {title[0].upper() + title[1:]}", "",
                "Each cell: median / 95th percentile in ms, overhead of the median against the base model, peak GPU memory.", "",
                "| method | path | " + " | ".join(f"{L} tokens" for L in rec["meta"]["lengths"]) + " |",
                "|---|---|" + "---|" * len(rec["meta"]["lengths"])]
        cells = {L: rec["cells"][f"{w}|{L}"]["methods"] for L in rec["meta"]["lengths"]}
        for meth in rec["meta"]["methods"]:
            name, _, path = meth.partition("-")
            row = [LABEL[name], {"": "–", "nf": "not fired", "f": "fired"}[path]]
            for L, c in cells.items():
                x = c[meth]
                over = "" if name == "base" else f", {pct(x['median_ms'], c['base']['median_ms'])}"
                row.append(f"{x['median_ms']:.1f} / {x['p95_ms']:.1f}{over}, {x['peak_gib']:.2f} GiB")
            out.append("| " + " | ".join(row) + " |")
        for j, what in ((0, "expected average overhead at the MMLU false-alarm rate"),
                        (1, "expected average overhead at the WMDP-Bio fire rate")):
            for name in FIRE:
                fr = rec["meta"]["fire_rates"][name][j]
                row = [LABEL[name], f"{what} ({fr:g})"]
                for L, c in cells.items():
                    e = expected_ms(c[f"{name}-nf"]["median_ms"], c[f"{name}-f"]["median_ms"], fr)
                    row.append(f"{e:.1f} ms, {pct(e, c['base']['median_ms'])}")
                out.append("| " + " | ".join(row) + " |")
        out.append("")
    return "\n".join(out)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench")
    b.add_argument("--json", required=True)
    b.add_argument("--check", action="store_true", help="only run the output checks")
    b.add_argument("--warmup", type=int, default=WARMUP)
    b.add_argument("--reps", type=int, default=REPS)
    b.add_argument("--lengths", type=int, nargs="+", default=LENGTHS)
    r = sub.add_parser("report")
    r.add_argument("--json", required=True)
    r.add_argument("--md", required=True)
    a = p.parse_args(argv)
    if a.cmd == "bench":
        return bench(a)
    Path(a.md).write_text(report_md(json.loads(Path(a.json).read_text())) + "\n")


if __name__ == "__main__":
    main()
