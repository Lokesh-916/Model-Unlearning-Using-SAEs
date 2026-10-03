"""A6-full: FULL fine-tune relearning attack (server version of exp/A6's LoRA relearning).

For each target model and each budget k (number of forget passages), fine-tune ALL weights on the k
passages (lr 1e-5, AdamW8bit, batch 4 x 512, STEPS steps) and log forget accuracy (WMDP-Bio TEST, first
N_EVAL items) and utility (N_UTIL MMLU TEST items over the utility subjects) before and at EVAL_AT steps.
Targets (skipped with a note when the weights are not staged):
  dsg-hook    base weights + DSG clamp during relearning and eval (trained through the gate)
  dsg-nohook  base weights, no hook (an open-weights attacker simply removes the hook)
  student     legacy LoRA student (exp/D1 sameref_a0.0, merged)       $DSG_CACHE/models/D1/sameref_a0.0
  d1          D1-full student (job d1-full)                           $DSG_CACHE/models/D1-full/<--d1-tag>
  d2          null-space edit (exp/D2)                                $DSG_CACHE/models/D2/nullspace
  rmu         RMU trained on gpuws (job rmu)                          $DSG_CACHE/models/RMU-cluster/best
  d1-a0.1     D1-full student undo_a0.1 (a0.3/a0.5 collapse MMLU to chance) $DSG_CACHE/models/D1-full/undo_a0.1
  rmu-v2      RMU v2 selected config (job rmu-v2)                     $DSG_CACHE/models/RMU-v2/best
Output per cell: $DSG_RESULTS/runs/A6-full/relearn-<target>-k<k>/{metrics.json,DONE} in exp/A6's format
(condition, k, rank='full', before, curve, n_eval, n_util), so the report's C-H6 rule and figures read it.
A cell with DONE is skipped, so a Slurm requeue continues with the next cell. No weights are saved.
"""
import argparse
import gc
import json
import shutil

import numpy as np
import torch

from cluster import jobcommon as jc
from dsgx import paths
from dsgx.util import atomic_write_json, now_iso

NAME = "a6-full"
KS = [10, 100, 1000]
STEPS, EVAL_AT, LR, BS, MAXLEN = 200, [25, 50, 100, 200], 1e-5, 4, 512
N_EVAL, N_UTIL = 300, 200


def targets(d1_tag):
    c = paths.cache_dir() / "models"
    return {"dsg-hook": (None, True), "dsg-nohook": (None, False),
            "student": (c / "D1" / "sameref_a0.0", False), "d1": (c / "D1-full" / d1_tag, False),
            "d2": (c / "D2" / "nullspace", False), "rmu": (c / "RMU-cluster" / "best", False),
            "d1-a0.1": (c / "D1-full" / "undo_a0.1", False), "rmu-v2": (c / "RMU-v2" / "best", False)}


def eval_sets(n_eval, n_util):
    from dsgx.data.mcq import utility_subjects

    forget = jc.mcq_items("wmdp-bio", "test", n_eval)
    subs = utility_subjects("bio")
    per = max(1, n_util // len(subs) + 1)
    util = [it for s in subs for it in jc.mcq_items(s, "test", per)][:n_util]
    return forget, util


def acc(model, tok, items):
    from dsgx.train.core import mcq_accuracy_hf

    model.eval()
    return float(np.mean(mcq_accuracy_hf(model, tok, items, bs=8))) if items else None


def cell(target, wpath, hook_on, k, a, tok, forget_eval, util_eval, passages):
    from dsgx.train.core import Trainer

    d = paths.runs_dir() / "A6-full" / f"relearn-{target}-k{k}"
    if (d / "DONE").exists():
        jc.log(NAME, f"{d.name}: done, skip")
        return
    d.mkdir(parents=True, exist_ok=True)
    model = jc.load_lm(str(wpath) if wpath else jc.HF_2B, train=True)
    info = None
    if hook_on:
        hook, handle, info = jc.dsg_guard(model)
    before = {"forget_acc": acc(model, tok, forget_eval), "util_acc": acc(model, tok, util_eval)}
    curve = []
    g = torch.Generator().manual_seed(k)
    docs = [passages[i] for i in torch.randperm(len(passages), generator=g)[:k].tolist()]
    dev = next(model.parameters()).device

    def step_fn(step):
        idx = torch.randint(0, len(docs), (a.bs,), generator=g).tolist()
        enc = tok([docs[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=a.maxlen)
        enc = {kk: v.to(dev) for kk, v in enc.items()}
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return {"loss": model(**enc, labels=labels).loss}

    def eval_fn(step):
        if step in a.eval_at:
            r = {"step": step, "forget_acc": acc(model, tok, forget_eval), "util_acc": acc(model, tok, util_eval)}
            curve.append(r)
            return {"forget_acc": r["forget_acc"], "util_acc": r["util_acc"]}
        return {}

    tr = Trainer(model, model.parameters(), jc.job_dir(NAME) / f"{target}-k{k}", lr=a.lr, steps=a.steps,
                 ckpt_every=10 ** 9, eval_fn=eval_fn, eval_every=1, optimizer="adamw" if jc.TINY else "adamw8bit")
    tr.run(step_fn)
    shutil.rmtree(tr.out / "last", ignore_errors=True)  # final trainer state (~10 GB): never kept or fetched
    met = {"condition": target, "k": k, "rank": "full", "steps": a.steps, "lr": a.lr, "weights": str(wpath) if wpath else None,
           "hook": hook_on, "dsg": info, "before": before, "curve": curve, "n_eval": len(forget_eval),
           "n_util": len(util_eval), "time": now_iso(), "hardware_label": jc.hardware_label()}
    atomic_write_json(d / "metrics.json", met)
    atomic_write_json(d / "DONE", {"time": now_iso(), "headline": {"before": before, "after": curve[-1] if curve else None}})
    jc.log(NAME, f"{d.name}: forget {before['forget_acc']:.3f} -> {curve[-1]['forget_acc'] if curve else float('nan'):.3f}")
    del model, tr
    gc.collect()
    torch.cuda.empty_cache()


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=int, nargs="*", default=KS)
    ap.add_argument("--targets", nargs="*", default=None)
    ap.add_argument("--d1-tag", default="undo_a0.3")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--eval-at", type=int, nargs="*", default=EVAL_AT)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    ap.add_argument("--n-eval", type=int, default=N_EVAL)
    ap.add_argument("--n-util", type=int, default=N_UTIL)
    a = ap.parse_args(argv)
    jc.require_gpu(40)
    tok = jc.load_tok()
    tok.padding_side = "right"
    forget_eval, util_eval = eval_sets(a.n_eval, a.n_util)
    passages = jc.forget_passages("bio", n=max(a.ks) if not jc.TINY else 16)
    T = targets(a.d1_tag)
    todo, skipped = [], {}
    for t in a.targets or list(T):
        w, h = T[t]
        if w is not None and not (jc.TINY or (w / "config.json").exists()):
            skipped[t] = f"weights not staged: {w}"
            continue
        todo.append((t, None if jc.TINY else w, h))
    for t, w, h in todo:
        for k in a.ks:
            cell(t, w, h, k, a, tok, forget_eval, util_eval, passages)
    rows = {}
    for p in sorted((paths.runs_dir() / "A6-full").glob("relearn-*/metrics.json")):
        m = json.loads(p.read_text())
        rows[p.parent.name] = {"before": m["before"]["forget_acc"], "after": m["curve"][-1]["forget_acc"] if m["curve"] else None}
    jc.summary(NAME, {"targets": [t for t, _, _ in todo], "skipped": skipped, "ks": a.ks, "steps": a.steps, "cells": rows})
    jc.log(NAME, f"done; skipped {skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
