"""D1 v2: UNDO full-parameter distillation with smaller noise and more steps, DEV selection, TEST, then relearning.

Why v2 (d1-full, job 93, gpuws): noise alpha 0.3 / 0.5 collapsed MMLU to chance (0.238 / 0.232 TEST); alpha 0.1
kept it (0.514 vs base 0.564) at WMDP 0.396, and the retain KL was still falling at step 2000 for every alpha
(alpha 0.1: 0.088 at 500 -> 0.071 at 2000; alpha 0.3: 0.86 -> 0.38). So v2 uses alpha {0.05, 0.1, 0.2} and
4000 steps (2x), the same recipe otherwise (cluster/d1_full.py: shrink-and-perturb init, KL to the DSG-guarded
teacher on forget passages + KL to the unguarded teacher on WikiText-2 + high_school_geography DEV prompts,
AdamW8bit lr 1e-5 with 5% warmup, batch 4 x 512, trainer state every 100 steps).

    python cluster/d1_v2.py train [--budget-min M]   # next unfinished alpha: train (resumable) + DEV harness eval
    python cluster/d1_v2.py test  [--budget-min M]   # DEV selection, then TEST of base / DSG paper / selected student

train  Works on the alphas in order; stops cleanly (after a checkpoint) when the budget is used up, so the next
       chained `d1-v2-train.sbatch` continues. During training a DEV curve (forget and utility accuracy on fixed
       DEV subsets) is logged every 500 steps to train_log.parquet. After training: HF weights to
       $DSG_CACHE/models/D1-v2/undo_a<alpha>, then a harness DEV run (bs 16, both views; base DEV run once).
test   Selection rule (fixed here before any v2 result): among students whose DEV utility drop vs base is
       <= 0.02 (raw full-MMLU utility pooled over the utility subjects except high_school_geography, whose DEV
       prompts are in the retain set), take the lowest DEV forget accuracy (raw); ties -> smaller alpha. If none
       qualifies: the smallest utility drop, flagged bound_met=false. TEST (bs 1, both views, reporting only):
       base, DSG paper config, the selected student. SELECTION.json names the student for a6_full --targets d1v2.
All runs are exp id D1-v2 on gpuws; summary.json / SUMMARY.md also quote d1-full alpha 0.1 (same hardware).
"""
import argparse
import gc
import shutil
import sys

import numpy as np
import torch

from cluster import d1_full
from cluster import jobcommon as jc
from dsgx import paths
from dsgx.util import atomic_write_json, atomic_write_text, now_iso

NAME = "d1-v2"
EXP = "D1-v2"
ALPHAS = [0.05, 0.1, 0.2]
STEPS, BS, MAXLEN, LR, CKPT = 4000, 4, 512, 1e-5, 100
CURVE_EVERY, N_CURVE = 500, 300
UTIL_DROP = 0.02
EXCLUDED_UTIL = ["high_school_geography"]  # its DEV prompts are retain-side training data
SEC_PER_STEP = 2.05                         # measured in d1-full (RTX 6000 Ada, incl. checkpoints)
DEV_EVAL_MIN = 12                            # final save + harness DEV run (bs 16)


class _Stop(Exception):
    pass


def tag(alpha):
    return f"undo_a{alpha}"


def out_root():
    return paths.cache_dir() / "models" / "D1-v2"


def dev_eval(label, weights=None):
    rd = jc.harness_eval(EXP, label, {"name": "base"}, weights=weights, split="dev", batch_size=16, purpose="select")
    f = jc.job_dir(NAME) / "dev_runs.json"
    m = jc.read_json(f, {}) or {}
    m[label] = str(rd)
    atomic_write_json(f, m)
    return rd


def curve_sets():
    from dsgx.data.mcq import utility_subjects

    forget = jc.mcq_items("wmdp-bio", "dev", N_CURVE)
    subs = [s for s in utility_subjects("bio") if s not in EXCLUDED_UTIL]
    per = max(1, N_CURVE // len(subs) + 1)
    util = [it for s in subs for it in jc.mcq_items(s, "dev", per)][:N_CURVE]
    return forget, util


def train_alpha(alpha, a, budget, forget, retain, tok, csets):
    """True when the final weights exist (trained now or before); False when stopped by the budget."""
    from dsgx.train.core import Trainer, mcq_accuracy_hf

    final = out_root() / tag(alpha)
    if (final / "config.json").exists():
        return True
    work = jc.job_dir(NAME) / tag(alpha)
    teacher = jc.load_lm(jc.HF_2B)
    teacher.eval().requires_grad_(False)
    hook, handle, info = jc.dsg_guard(teacher)
    student = jc.load_lm(jc.HF_2B, train=True)
    d1_full.noise_(student, alpha, seed=int(round(alpha * 1000)))
    dev = next(student.parameters()).device

    def batch(src, step, salt):
        g = torch.Generator().manual_seed(step * 2 + salt)  # per-step seed: a resumed run sees the same batches
        idx = torch.randint(0, len(src), (a.bs,), generator=g).tolist()
        enc = tok([src[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=a.maxlen)
        return {k: v.to(dev) for k, v in enc.items()}

    tr = None

    def step_fn(step):
        if step % a.ckpt == 0 and step > tr.start and not budget.fits(DEV_EVAL_MIN + 2):
            raise _Stop(step)  # the checkpoint of `step` was written at the end of the previous step
        fb, rb = batch(forget, step, 0), batch(retain, step, 1)
        with torch.no_grad():
            hook.enabled = True
            tf = teacher(**fb).logits
            hook.enabled = False
            trl = teacher(**rb).logits
        kf = d1_full.kl_tokens(student(**fb).logits, tf, fb["attention_mask"])
        kr = d1_full.kl_tokens(student(**rb).logits, trl, rb["attention_mask"])
        return {"loss": kf + kr, "kl_forget": kf.detach(), "kl_retain": kr.detach()}

    def eval_fn(step):
        if step % CURVE_EVERY and step != a.steps:
            return {}
        return {"dev_forget_acc": float(np.mean(mcq_accuracy_hf(student, tok, csets[0], bs=8))),
                "dev_util_acc": float(np.mean(mcq_accuracy_hf(student, tok, csets[1], bs=8)))}

    tr = Trainer(student, student.parameters(), work, lr=a.lr, steps=a.steps, ckpt_every=a.ckpt,
                 eval_fn=eval_fn, eval_every=1, optimizer="adamw" if jc.TINY else "adamw8bit")
    jc.log(NAME, f"{tag(alpha)}: steps {tr.start}->{a.steps}, budget left {budget.left():.0f} min")
    done = True
    try:
        tr.run(step_fn)
    except _Stop as e:
        jc.log(NAME, f"{tag(alpha)}: budget used, stopped at step {e.args[0]} (checkpointed); next sbatch resumes")
        done = False
    if done:
        final.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(final)
        tok.save_pretrained(final)
        atomic_write_json(final / "d1_v2.json", {"alpha": alpha, "steps": a.steps, "dsg": info,
                                                 "last": tr.log[-1] if tr.log else None, "time": now_iso()})
        shutil.copy2(work / "train_log.parquet", final.parent / f"{tag(alpha)}.train_log.parquet")
        shutil.rmtree(work / "last", ignore_errors=True)  # trainer state (~10 GB)
    handle.remove()
    del teacher, student, tr
    gc.collect()
    torch.cuda.empty_cache()
    return done


def cmd_train(a):
    budget = jc.Budget(a.budget_min)
    jc.require_gpu(40)
    tok = jc.load_tok()
    tok.padding_side = "right"
    forget = jc.forget_passages("bio", n=None if not jc.TINY else 16)
    from dsgx.data.mcq import format_prompt

    retain = jc.retain_passages(n=None if not jc.TINY else 16)
    retain += [format_prompt(it) for it in jc.mcq_items("high_school_geography", "dev")[: (8 if jc.TINY else None)]]
    csets = curve_sets() if not jc.TINY else (jc.mcq_items("wmdp-bio", "dev", 2), jc.mcq_items("human_aging", "dev", 2))
    if not jc.TINY and budget.fits(DEV_EVAL_MIN):
        dev_eval("base")
    for alpha in a.alphas:
        final = out_root() / tag(alpha)
        if not (final / "config.json").exists():
            need = 6 + (a.steps * SEC_PER_STEP / 60 if not (jc.job_dir(NAME) / tag(alpha) / "last").exists() else 15)
            if not budget.fits(min(need, 20)):
                jc.log(NAME, f"{tag(alpha)}: {budget.left():.0f} min left, not starting; next sbatch continues")
                break
            if not train_alpha(alpha, a, budget, forget, retain, tok, csets):
                break
        if not jc.TINY:
            if not budget.fits(DEV_EVAL_MIN - 4):
                break
            dev_eval(f"undo-v2-a{alpha}", weights=final)
    status = {str(al): (out_root() / tag(al) / "config.json").exists() for al in a.alphas}
    jc.log(NAME, f"trained: {status}")
    return 0


def _dev_metrics(label):
    """(forget raw mean, utility raw mean excluding EXCLUDED_UTIL) of the finished DEV run `label`."""
    import pandas as pd

    from pathlib import Path

    rd = (jc.read_json(jc.job_dir(NAME) / "dev_runs.json", {}) or {}).get(label)
    if not rd:
        return None
    rd = Path(rd)
    if not (rd / "DONE").exists():
        return None
    it = pd.read_parquet(rd / "items.parquet", columns=["dataset", "correct"])
    f = it[it["dataset"] == "wmdp-bio"]["correct"].mean()
    u = it[(it["dataset"] != "wmdp-bio") & ~it["dataset"].isin(EXCLUDED_UTIL)]["correct"].mean()
    return {"forget": float(f), "utility": float(u), "n_forget": int((it["dataset"] == "wmdp-bio").sum()),
            "n_util": int(((it["dataset"] != "wmdp-bio") & ~it["dataset"].isin(EXCLUDED_UTIL)).sum()), "run": str(rd)}


def select(alphas):
    base = _dev_metrics("base")
    rows = {al: _dev_metrics(f"undo-v2-a{al}") for al in alphas}
    if base is None or any(v is None for v in rows.values()):
        return None
    for al, r in rows.items():
        r["util_drop"] = base["utility"] - r["utility"]
        r["eligible"] = r["util_drop"] <= UTIL_DROP + 1e-9  # a drop of exactly 0.02 qualifies
    ok = [al for al in alphas if rows[al]["eligible"]]
    if ok:
        pick, met = min(ok, key=lambda al: (rows[al]["forget"], al)), True
    else:
        pick, met = min(alphas, key=lambda al: (rows[al]["util_drop"], al)), False
    return {"rule": f"lowest DEV forget among DEV utility drop <= {UTIL_DROP} (excl. {EXCLUDED_UTIL}); ties -> smaller alpha; "
                    "fallback smallest drop", "base_dev": base, "students_dev": {str(k): v for k, v in rows.items()},
            "selected_alpha": pick, "bound_met": met, "checkpoint": str(out_root() / tag(pick)), "time": now_iso()}


def d1_full_reference():
    """d1-full alpha 0.1 (job 93, same hardware) from its summary, if present on this machine."""
    s = jc.read_json(paths.results_dir() / "jobs" / "d1-full" / "summary.json", {}) or {}
    h = s.get("headline") or {}
    return {k: h.get(k) for k in ("base", "dsg_paper", "undo_a0.1")} if h else None


def cmd_test(a):
    budget = jc.Budget(a.budget_min)
    sel = select(a.alphas)
    if sel is None:
        jc.log(NAME, "ABORT: not every student has a finished DEV run; resubmit d1-v2-train.sbatch (resumes)")
        return 3
    atomic_write_json(jc.job_dir(NAME) / "SELECTION.json", sel)
    jc.log(NAME, f"selected alpha {sel['selected_alpha']} (bound met: {sel['bound_met']})")
    if jc.TINY:
        return 0
    jc.require_gpu(40)
    runs = {}
    for label, method, w in (("base", {"name": "base"}, None),
                             ("dsg-paper", {"name": "dsg-faithful", "n_features": 20, "retain_pct": 95, "multiplier": 500}, None),
                             (f"undo-v2-a{sel['selected_alpha']}", {"name": "base"}, sel["checkpoint"])):
        if not budget.fits(10):
            jc.log(NAME, "budget used before all TEST runs; resubmit d1-v2-test.sbatch (finished runs are skipped)")
            return 3
        runs[label] = str(jc.harness_eval(EXP, label, method, weights=w))
    head = jc.headlines(runs)
    ref = d1_full_reference()
    jc.summary(NAME, {"alphas": a.alphas, "steps": STEPS, "selection": sel, "runs": runs, "headline": head,
                      "d1_full_reference": ref})
    L = ["# D1 v2 (gpuws)", "", f"Selection (DEV, rule fixed in cluster/d1_v2.py): alpha {sel['selected_alpha']}, "
         f"bound met: {sel['bound_met']}.", "", "| student | DEV forget | DEV utility | drop vs base | eligible |", "|---|---|---|---|---|",
         f"| base | {sel['base_dev']['forget']:.3f} | {sel['base_dev']['utility']:.3f} | - | - |"]
    L += [f"| a{k} | {v['forget']:.3f} | {v['utility']:.3f} | {v['util_drop']:+.3f} | {v['eligible']} |" for k, v in sel["students_dev"].items()]
    L += ["", "TEST (raw view; mean [95% CI], n):", "", "| condition | forget | utility |", "|---|---|---|"]
    fmt = lambda m: f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}] n={m['n']}" if m else "-"  # noqa: E731
    L += [f"| {k} | {fmt(v.get('forget'))} | {fmt(v.get('utility'))} |" for k, v in head.items()]
    if ref:
        L += ["", "d1-full (2000 steps) on the same hardware, for reference:", ""]
        L += [f"| d1-full {k} | {fmt((v or {}).get('forget'))} | {fmt((v or {}).get('utility'))} |" for k, v in ref.items()]
    atomic_write_text(jc.job_dir(NAME) / "SUMMARY.md", "\n".join(L) + "\n")
    jc.log(NAME, "done")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["train", "test"])
    ap.add_argument("--alphas", type=float, nargs="*", default=ALPHAS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--ckpt", type=int, default=CKPT)
    ap.add_argument("--budget-min", type=float, default=None)
    a = ap.parse_args(argv)
    return cmd_train(a) if a.cmd == "train" else cmd_test(a)


if __name__ == "__main__":
    sys.exit(main())
