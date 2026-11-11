"""A2-tofu-full: FULL fine-tune of gemma-2-2b-it on TOFU, then DSG and the best gate on top, TOFU metrics.

1. Fine-tune (all weights, bf16, AdamW8bit, lr 1e-5, 5 epochs, batch 16 = 4 x grad-accum 4, loss on answer
   tokens, chat template) two models, resumable (trainer state every 200 steps):
     full   = forget10 + retain90 (the TOFU "full" set, as in exp/A2)
     retain = retain90 only (the never-learned reference; also used by Q3)
   saved to $DSG_CACHE/models/A2-tofu-full/{full,retain} with train_version.json (2 = fixed Trainer accumulation);
   retain is deleted once the metrics are written (--keep-retain keeps it). Budget (DSG_BUDGET_MIN) and disk
   (free >= 50 GB after 16 GB) are checked before each fine-tune; training stops at a checkpoint, next sbatch resumes.
2. DSG on TOFU (layer-3 Gemma Scope SAE on the fine-tuned model): per-feature fire rate on forget10 vs
   retain90 QA text; top-20 features by (p_forget + 1e-4) / (p_retain + 1e-4) among features firing on
   >= 1% of forget tokens; tau = 95th percentile of rho on retain sequences (DSG's retain-percentile rule).
   Best gate: a window gate (w from COMBINE_SELECTION.json if its detector is window-w*, else 16), threshold
   = 95th percentile of the window score on retain sequences. Both clamp the selected features to -500.
3. TOFU metrics per condition {retain, full, full+dsg, full+best-gate} (Maini et al. 2024): answer
   probability, truth ratio, ROUGE-L recall of greedy answers (64 new tokens) on forget10, retain, real
   authors and world facts; model utility = harmonic mean of the 9 retain/real/world numbers; forget
   quality = KS-test p-value of the forget truth-ratio distribution vs the retain model's.
Writes $DSG_RESULTS/runs/A2-tofu-full/tofu-metrics/{metrics.json,DONE} (exp/A2's tofu-metrics format) and
jobs/tofu-full/summary.json. TOFU is fictitious, but generations are still not written (metrics only).
"""
import argparse
import gc
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from cluster import jobcommon as jc
from dsgx import paths
from dsgx.util import atomic_write_json, now_iso

NAME = "tofu-full"
EPOCHS, LR, BS, ACCUM, MAXLEN = 5, 1e-5, 4, 4, 256
# 2: Trainer zero_grad before step_fn (f40dde8), all `accum` micro-batches reach the optimizer. 1 (jobs 84/96,
# no marker): only the last micro-batch did. Models and partial results without version 2 are never reused.
TRAIN_VERSION = 2
TRAIN_MIN_START, STOP_MARGIN, EVAL_MIN = 15, 6, 30  # minutes of DSG_BUDGET_MIN (slurm/later.sh)
DISK_NEED_GB = 16  # one model being trained (5.2) + trainer state (~10)


class _Stop(Exception):
    pass


def disk_ok(need_gb=DISK_NEED_GB) -> bool:
    """Rule 5 inside the job: free >= 50 GB after `need_gb`, and our total stays < 100 GB."""
    import os

    root = Path(os.environ.get("DSGC", paths.cache_dir()))
    free = shutil.disk_usage(root).free / 1e9
    ours = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9 if not jc.TINY else 0
    ok = free - need_gb >= 50 and ours + need_gb < 100
    jc.log(NAME, f"disk: free {free:.0f} GB, ours {ours:.0f} GB, need {need_gb} GB -> {'ok' if ok else 'REFUSING'}")
    return ok or jc.TINY


def model_ok(path) -> bool:
    """A fine-tuned TOFU model of the current training loop (used by tofu-full, figparity highlight, q2)."""
    return (jc.read_json(Path(path) / "train_version.json", {}) or {}).get("train_version") == TRAIN_VERSION


def tofu(config):
    from datasets import load_dataset

    d = load_dataset("locuslab/TOFU", config, split="train")
    return [dict(r) for r in d]


def chat(tok, q, a=None):
    p = tok.apply_chat_template([{"role": "user", "content": q}], tokenize=False, add_generation_prompt=True)
    return p if a is None else (p, a)


def finetune(tag, rows, a, tok, budget):
    """Saved model dir, or None when the budget ran out (checkpointed; the next chained sbatch resumes)."""
    from dsgx.train.core import Trainer

    out = paths.cache_dir() / "models" / "A2-tofu-full" / tag
    if (out / "config.json").exists():
        if model_ok(out):
            return out
        raise SystemExit(f"{out} was trained by the old loop (no train_version {TRAIN_VERSION}): move it away first")
    if not budget.fits(TRAIN_MIN_START) or not disk_ok():
        return None
    model = jc.load_lm(jc.HF_2B, train=True)
    dev = next(model.parameters()).device
    g = torch.Generator().manual_seed(0)
    steps = max(1, a.epochs * len(rows) // (a.bs * a.accum))
    order = torch.cat([torch.randperm(len(rows), generator=g) for _ in range(a.epochs + 1)]).tolist()

    tr = None

    def step_fn(step):
        if step % 200 == 0 and step > tr.start and not budget.fits(STOP_MARGIN):
            raise _Stop(step)  # the checkpoint of `step` was written at the end of the previous step
        tot = 0.0
        for j in range(a.accum):
            base = (step * a.accum + j) * a.bs
            pairs = [chat(tok, rows[i]["question"], rows[i]["answer"]) for i in order[base: base + a.bs]]
            enc = tok([p + ans + tok.eos_token for p, ans in pairs], return_tensors="pt", padding=True,
                      truncation=True, max_length=a.maxlen, add_special_tokens=False)
            labels = enc["input_ids"].clone()
            labels[enc["attention_mask"] == 0] = -100
            for i, (p, _) in enumerate(pairs):
                labels[i, : len(tok(p, add_special_tokens=False)["input_ids"])] = -100
            loss = model(**{k: v.to(dev) for k, v in enc.items()}, labels=labels.to(dev)).loss / a.accum
            if j < a.accum - 1:
                loss.backward()          # earlier micro-batches: backward now, keep only the value
                tot = tot + loss.detach()
            else:
                tot = loss + tot         # the Trainer backpropagates the last micro-batch
        return {"loss": tot}

    tr = Trainer(model, model.parameters(), jc.job_dir(NAME) / f"ft-{tag}", lr=a.lr, steps=steps,
                 ckpt_every=200, optimizer="adamw" if jc.TINY else "adamw8bit")
    jc.log(NAME, f"{tag}: {len(rows)} rows x {a.epochs} epochs = {steps} steps (start {tr.start})")
    try:
        tr.run(step_fn)
    except _Stop as e:
        jc.log(NAME, f"{tag}: budget low, stopped at step {e.args[0]} (checkpointed); next job resumes")
        del model, tr
        gc.collect()
        torch.cuda.empty_cache()
        return None
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    atomic_write_json(out / "train_version.json", {"train_version": TRAIN_VERSION, "steps": steps, "micro_batch": a.bs,
                                                   "accum": a.accum, "effective_batch": a.bs * a.accum, "time": now_iso()})
    shutil.rmtree(tr.out / "last", ignore_errors=True)
    del model, tr
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- gates on HF hidden states
class Gate:
    """rho (DSG-faithful rate over all tokens) or window (max firing fraction over w tokens) + clamp."""

    def __init__(self, sae, feats, kind, thr, w=16, mult=500.0):
        self.sae, self.kind, self.thr, self.w, self.mult = sae, kind, float(thr), int(w), float(mult)
        self.f = torch.tensor(feats, device=sae.W_dec.device)
        self.enabled = True

    def score(self, fire):  # fire [L] bool
        if self.kind == "rho":
            return float(fire.float().mean())
        f = fire[1:].float()
        if len(f) <= self.w:
            return float(f.mean()) if len(f) else 0.0
        c = torch.cat([torch.zeros(1, device=f.device), f.cumsum(0)])
        return float(((c[self.w:] - c[:-self.w]) / self.w).max())

    def __call__(self, module, inputs, output):
        if not self.enabled:
            return output
        hs = output[0] if isinstance(output, tuple) else output
        x = hs.to(self.sae.W_dec.dtype)
        acts = self.sae.encode(x)
        new = x.clone()
        for b in range(x.shape[0]):
            fire = (acts[b][:, self.f] > 0).any(-1)
            if self.score(fire) > self.thr:
                a2 = acts[b].clone()
                pos = fire.nonzero().flatten()
                a2[pos[:, None], self.f[None, :]] = -self.mult
                new[b] = self.sae.decode(a2) + (x[b] - self.sae.decode(acts[b]))
        new = new.to(hs.dtype)
        return (new, *output[1:]) if isinstance(output, tuple) else new


@torch.no_grad()
def fire_stats(model, tok, sae, layer, texts, feats=None):
    """Per-feature token fire rate, and per-text fire indicators of `feats` (for thresholds)."""
    from dsgx.train.core import decoder_layers

    got = {}
    h = decoder_layers(model)[layer].register_forward_hook(lambda m, i, o: got.__setitem__("x", o[0] if isinstance(o, tuple) else o))
    dev = next(model.parameters()).device
    tot, n, per = None, 0, []
    for t in texts:
        enc = tok(t, return_tensors="pt", add_special_tokens=True).to(dev)
        model(**enc)
        acts = sae.encode(got["x"][0].to(sae.W_dec.dtype))
        acts[0] = 0
        fr = (acts > 0).float().sum(0)
        tot = fr if tot is None else tot + fr
        n += acts.shape[0]
        if feats is not None:
            per.append((acts[:, feats] > 0).any(-1).cpu())
    h.remove()
    return (tot / max(n, 1)).float().cpu().numpy(), per


# ----------------------------------------------------------------------------- TOFU metrics
@torch.no_grad()
def seq_prob(model, tok, q, ans):
    """P(ans | q)^(1/|ans|) (length-normalised), as in TOFU."""
    p = chat(tok, q)
    ids_p = tok(p, add_special_tokens=False)["input_ids"]
    ids = tok(p + ans, return_tensors="pt", add_special_tokens=False)["input_ids"].to(next(model.parameters()).device)
    logits = model(ids).logits[0, :-1].float()
    lp = logits.log_softmax(-1)[torch.arange(ids.shape[1] - 1), ids[0, 1:]]
    lp = lp[len(ids_p) - 1:]
    return float(lp.mean().exp()) if len(lp) else float("nan")


@torch.no_grad()
def rouge_recall(model, tok, q, ans, max_new=64):
    from rouge_score import rouge_scorer

    p = chat(tok, q)
    enc = tok(p, return_tensors="pt", add_special_tokens=False).to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
    return rouge_scorer.RougeScorer(["rougeL"]).score(ans, gen)["rougeL"].recall


METRIC_VERSION = 2  # v1 (job 96): utility used max(0, 1 - mean R) and raw answer prob on real/world sets


def truth_ratio(model, tok, row, probs=False):
    """R = mean P(perturbed) / P(paraphrased), length-normalised probabilities (TOFU). probs=True also returns the
    perturbed probabilities (for the option-normalised answer probability of real_authors / world_facts)."""
    pert = row["perturbed_answer"] if isinstance(row["perturbed_answer"], list) else [row["perturbed_answer"]]
    good = row.get("paraphrased_answer") or row["answer"]
    pps = [seq_prob(model, tok, row["question"], x) for x in pert]
    pg = seq_prob(model, tok, row["question"], good)
    r = float(np.mean(pps) / max(pg, 1e-12))
    return (r, pps) if probs else r


def evaluate(model, tok, sets, n):
    """TOFU metrics as in Maini et al. 2024 and the TOFU code (aggregate_eval_stat.py):
    answer prob = mean P(a|q)^(1/|a|) (retain, forget); on real_authors / world_facts the option-normalised
    P(a) / (P(a) + sum P(perturbed)); truth-ratio score on utility sets = mean_i max(0, 1 - R_i) (per item);
    model utility = harmonic mean of the 9 retain / real / world numbers."""
    out = {}
    for name, rows in sets.items():
        rows = rows[:n]
        ap = [seq_prob(model, tok, r["question"], r["answer"]) for r in rows]
        trp = [truth_ratio(model, tok, r, probs=True) for r in rows]
        tr = [t for t, _ in trp]
        rl = [rouge_recall(model, tok, r["question"], r["answer"]) for r in rows]
        apn = [a / max(a + sum(pp), 1e-12) for a, (_, pp) in zip(ap, trp)]
        out[name] = {"answer_prob": float(np.mean(apn if name in ("real_authors", "world_facts") else ap)),
                     "answer_prob_raw": float(np.mean(ap)), "answer_prob_values": ap,
                     "truth_ratio": float(np.mean(tr)), "truth_ratio_median": float(np.median(tr)),
                     "truth_ratio_score": float(np.mean([max(0.0, 1 - t) for t in tr])),
                     "truth_ratio_forget_score": float(np.mean([min(t, 1 / t) if t > 0 else 0.0 for t in tr])),
                     "rougeL_recall": float(np.mean(rl)), "rougeL_values": rl, "truth_ratio_values": tr, "n": len(rows)}
    util = []
    for s in ("retain", "real_authors", "world_facts"):
        if s in out:
            util += [out[s]["answer_prob"], out[s]["truth_ratio_score"], out[s]["rougeL_recall"]]
    util = [max(u, 1e-6) for u in util]
    out["model_utility"] = float(len(util) / sum(1 / u for u in util)) if util else None
    out["metric_version"] = METRIC_VERSION
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--accum", type=int, default=ACCUM)
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    ap.add_argument("--n-eval", type=int, default=400)
    ap.add_argument("--n-calib", type=int, default=400)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--keep-retain", action="store_true", help="keep the retain model after the metrics are written")
    a = ap.parse_args(argv)
    budget = jc.Budget(a.budget_min)
    d = paths.runs_dir() / "A2-tofu-full" / "tofu-metrics"
    if (d / "DONE").exists() and (jc.read_json(d / "metrics.json", {}) or {}).get("train_version") == TRAIN_VERSION:
        jc.log(NAME, "done (train_version 2), skip")
        return 0
    from scipy.stats import ks_2samp

    jc.require_gpu(40)
    tok = jc.load_tok()
    tok.padding_side = "right"
    f10, r90 = tofu("forget10"), tofu("retain90")
    if jc.TINY:
        f10, r90 = f10[:8], r90[:8]
    paths_ = {"full": finetune("full", f10 + r90, a, tok, budget)}
    paths_["retain"] = paths_["full"] and finetune("retain", r90, a, tok, budget)
    if not all(paths_.values()):
        jc.log(NAME, "partial: fine-tuning not finished; resubmit tofu-full-v3.sbatch to continue")
        return 0
    sets = {"forget": tofu("forget10_perturbed"), "retain": tofu("retain_perturbed"),
            "real_authors": tofu("real_authors_perturbed"), "world_facts": tofu("world_facts_perturbed")}
    n = 2 if jc.TINY else a.n_eval
    # Per-condition checkpoint: each finished condition is saved, so a job stopped by --time resumes at the next.
    part = paths.runs_dir() / "A2-tofu-full" / "tofu-metrics" / "partial"
    part.mkdir(parents=True, exist_ok=True)
    res = {k: jc.read_json(part / f"{k}.json", None) for k in ("retain-model", "full", "full+dsg", "full+best-gate")}
    res = {k: v for k, v in res.items() if v is not None and v.get("n") == n and v.get("metric_version") == METRIC_VERSION
           and v.get("train_version") == TRAIN_VERSION}
    for k in res:
        jc.log(NAME, f"resumed: {k} already evaluated")

    def save(k):
        atomic_write_json(part / f"{k}.json", {**res[k], "n": n, "train_version": TRAIN_VERSION})
        jc.log(NAME, f"eval {k} saved")

    def out_of_budget(k):
        if k in res or budget.fits(EVAL_MIN):
            return False
        jc.log(NAME, f"partial: budget too low for eval {k}; resubmit tofu-full-v3.sbatch to continue")
        return True

    if out_of_budget("retain-model"):
        return 0
    if "retain-model" not in res:
        m = jc.load_lm(paths_["retain"])
        m.eval()
        res["retain-model"] = evaluate(m, tok, sets, n)
        save("retain-model")
        del m
        gc.collect()
        torch.cuda.empty_cache()
    if out_of_budget("full"):
        return 0
    m = jc.load_lm(paths_["full"])
    m.eval()
    if "full" not in res:
        res["full"] = evaluate(m, tok, sets, n)
        save("full")
    # DSG / best gate on the full model
    from dsgx.train.core import decoder_layers

    sae, layer = jc.load_sae(), (1 if jc.TINY else 3)
    calib = 4 if jc.TINY else a.n_calib
    ftxt = [chat(tok, r["question"]) + r["answer"] for r in f10[:calib]]
    rtxt = [chat(tok, r["question"]) + r["answer"] for r in r90[:calib]]
    pf, _ = fire_stats(m, tok, sae, layer, ftxt)
    pr, _ = fire_stats(m, tok, sae, layer, rtxt)
    score = np.where(pf >= 0.01, (pf + 1e-4) / (pr + 1e-4), 0)
    feats = [int(i) for i in np.argsort(-score)[:20]]
    _, per = fire_stats(m, tok, sae, layer, rtxt, feats)
    w = a.window
    sel = jc.read_json(paths.results_dir() / "COMBINE_SELECTION.json", {}) or {}
    det = (sel.get("slots") or {}).get("detector") or ""
    if w is None:
        w = int(det.split("-w")[1]) if det.startswith("window-w") else 16
    rho_g = Gate(sae, feats, "rho", 0, mult=500)
    win_g = Gate(sae, feats, "window", 0, w=w, mult=500)
    rho_g.thr = float(np.percentile([rho_g.score(f) for f in per], 95))
    win_g.thr = float(np.percentile([win_g.score(f) for f in per], 95))
    for tag, g in (("full+dsg", rho_g), ("full+best-gate", win_g)):
        if tag in res:
            continue
        if out_of_budget(tag):
            return 0
        h = decoder_layers(m)[layer].register_forward_hook(g)
        res[tag] = evaluate(m, tok, sets, n)
        res[tag]["gate"] = {"kind": g.kind, "threshold": g.thr, "w": g.w if g.kind == "window" else None,
                            "features": feats, "best_gate_source": det or "default window-w16"}
        h.remove()
        save(tag)
    res = {k: {kk: vv for kk, vv in v.items() if kk != "n"} for k, v in res.items()}
    ref = res["retain-model"]["forget"]["truth_ratio_values"]
    for k, v in res.items():
        v["forget_quality_ks_p"] = float(ks_2samp(v["forget"]["truth_ratio_values"], ref).pvalue) if k != "retain-model" else None
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_json(d / "metrics.json", {"conditions": res, "n_forget": n, "n_retain": n, "metric_version": METRIC_VERSION,
                                           "train_version": TRAIN_VERSION, "models": {k: str(v) for k, v in paths_.items()}, "hardware_label": jc.hardware_label()})
    atomic_write_json(d / "DONE", {"time": now_iso(), "headline": {k: v.get("forget_quality_ks_p") for k, v in res.items()}})
    jc.summary(NAME, {"conditions": {k: {"forget_quality_ks_p": v.get("forget_quality_ks_p"), "model_utility": v.get("model_utility"),
                                         "forget_truth_ratio": v["forget"]["truth_ratio"]} for k, v in res.items()},
                      "models": {k: str(v) for k, v in paths_.items()}})
    if not a.keep_retain:  # only the metrics need it; frees 5.2 GB for the next job (Q2 / FP-highlight use `full`)
        del m
        gc.collect()
        shutil.rmtree(paths_["retain"], ignore_errors=True)
        jc.log(NAME, f"removed {paths_['retain']} (metrics written)")
    jc.log(NAME, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
