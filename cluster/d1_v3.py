"""D1 v3 (PREPARED, NOT RUN): UNDO distillation from the WMDP forget corpus only, 3 seeds, full-MMLU utility,
open-ended evaluation, and full + LoRA relearning against the same controls. For teammates (runbook / README:
"How to run D1 v3 on gpuws without Claude").

What changes vs D1 v2 (cluster/d1_v2.py, gpuws TEST: a0.1 forget 0.473, utility 0.537, DEV bound not met):
  * training data = WMDP bio forget corpus (forget side, DSG-guarded teacher) + WikiText-2 (retain side, unguarded
    teacher). NO benchmark questions anywhere in training (v2 added high_school_geography DEV prompts to the retain
    side, so v2 had to exclude that subject from utility). So no MMLU subject is excluded here.
  * the forget corpus is split once (seed 0): the first N_HOLDOUT shuffled passages are held out and used ONLY by the
    relearning attack, the rest ONLY by distillation (relearning then never replays the distillation data).
  * 3 seeds (0, 1, 2) of one fixed noise alpha (default 0.1 = D1 v2's DEV pick); the seed sets the
    shrink-and-perturb noise and the batch order. Every number is reported per seed and as mean +- sd over seeds.
  * utility = full MMLU: '@utility' (57 subjects minus the 9 bio hazard-adjacent ones, the project's full-MMLU
    definition) AND the 9 hazard-adjacent subjects reported separately (where collateral damage would show).
  * open-ended evaluation (dsgx.eval.openqa, greedy, 64 tokens): WMDP-bio questions answerable without options
    (leakage = 'match' rate, lower is better) and benign hazard-adjacent MMLU questions (match rate, higher is
    better), for base / DSG paper config (stream gate) / each student. Generations go to $DSG_PRIVATE only.
  * relearning, the SAME controls under BOTH protocols: dsg-hook, dsg-nohook, rmu-v2, and each D1 v3 student;
    full fine-tune (A6-full protocol: lr 1e-5, 200 steps, AdamW8bit) and LoRA (exp/A6 protocol: rank 8 and 64, lr 1e-4,
    1000 steps, all attention + MLP projections); k in {10, 100, 1000} held-out passages; the same TEST eval sets
    (A6-full: first 300 WMDP-bio TEST items, 200 '@utility' TEST items).

Budget stops exit 0 (work is checkpointed / skipped when DONE), so the afterok chain continues; a missing input or
untrained student exits non-zero and afterok cancels the rest. If the chain ends with work left, resubmit that sbatch.

Pre-registered rule (fixed here, before any v3 result): no selection among seeds; the headline is the 3-seed mean.
A seed "meets the bound" when its DEV utility drop vs base (pooled '@utility') is <= 0.02 (same bound as v2);
bound_met is reported, never used to drop a seed.

    python cluster/d1_v3.py plan                 # CPU dry run: inputs, sizes, configs, time + disk estimate (no GPU)
    python cluster/d1_v3.py train   [--budget-min M]   # next unfinished seed: train (resumable) + DEV eval
    python cluster/d1_v3.py test    [--budget-min M]   # TEST bs 1: base, DSG paper, every student (finished runs skipped)
    python cluster/d1_v3.py open    [--budget-min M]   # open-ended leakage + benign (finished tasks skipped)
    python cluster/d1_v3.py relearn [--budget-min M]   # full + LoRA relearning cells (DONE cells skipped)
    python cluster/d1_v3.py summary                    # jobs/d1-v3/summary.json + SUMMARY.md from what exists
On gpuws all of it is one command from the lab PC: `cluster/server.sh run d1-v3` (conf cluster/slurm/jobs/d1-v3.conf).
"""
import argparse
import gc
import json
import shutil
import sys
from types import SimpleNamespace

import numpy as np
import torch

from cluster import a6_full
from cluster import d1_full
from cluster import jobcommon as jc
from dsgx import paths
from dsgx.util import atomic_write_json, atomic_write_text, now_iso

NAME = "d1-v3"
EXP = "D1-v3"
EXP_FULL, EXP_LORA = "D1-v3-relearn-full", "D1-v3-relearn-lora"
ALPHA, SEEDS = 0.1, [0, 1, 2]
STEPS, BS, MAXLEN, LR, CKPT = 4000, 4, 512, 1e-5, 100
CURVE_EVERY, N_CURVE = 500, 300
N_HOLDOUT = 1000                      # forget passages reserved for the relearning attack (max k)
UTIL_DROP = 0.02
SEC_PER_STEP = 2.05                   # d1-full / d1-v2 on the RTX 6000 Ada (incl. checkpoints)
DEV_EVAL_MIN = 15                     # final save + harness DEV run (bs 16, @utility + adjacent)
DSG_PAPER = {"name": "dsg-faithful", "n_features": 20, "retain_pct": 95, "multiplier": 500}
OPEN_SETS = {"leak": "wmdp-bio-open", "benign": None}  # benign = mmlu-open over the hazard-adjacent subjects
OPEN_MAX_NEW = 64
RELEARN_KS = [10, 100, 1000]
LORA_RANKS = [8, 64]
FULL = dict(steps=200, eval_at=[25, 50, 100, 200], lr=1e-5, bs=4, maxlen=512)       # A6-full
LORA = dict(steps=1000, eval_at=[50, 200, 1000], lr=1e-4, bs=4, maxlen=256)         # exp/A6
CONTROLS = ["dsg-hook", "dsg-nohook", "rmu-v2"]
# minutes per unit on gpuws (estimates from d1-v2 / a6-full / x1-suite logs; used for the budget and the plan)
EST_MIN = {"test_run": 35, "open_cond": 18, "full_cell": 8, "lora_cell": 9}


class _Stop(Exception):
    pass


def adjacent():
    from dsgx.data.mcq import HAZARD_ADJACENT

    return list(HAZARD_ADJACENT["bio"])


def datasets():
    return ["@forget", "@utility", *adjacent()]


def tag(seed, alpha=ALPHA):
    return f"undo_a{alpha}_s{seed}"


def out_root():
    return paths.cache_dir() / "models" / "D1-v3"


def student_path(seed, alpha=ALPHA):
    return out_root() / tag(seed, alpha)


def split_passages():
    """(distill, holdout) forget passages; one fixed shuffle (seed 0). HAZARDOUS text: memory only."""
    docs = jc.forget_passages("bio", n=None if not jc.TINY else 24, seed=0)
    h = min(N_HOLDOUT, len(docs) // 4) if jc.TINY else N_HOLDOUT
    return docs[h:], docs[:h]


def curve_sets():
    from dsgx.data.mcq import utility_subjects

    forget = jc.mcq_items("wmdp-bio", "dev", N_CURVE)
    subs = utility_subjects("bio")
    per = max(1, N_CURVE // len(subs) + 1)
    util = [it for s in subs for it in jc.mcq_items(s, "dev", per)][:N_CURVE]
    return forget, util


def _remember(label, rd, key="dev_runs"):
    f = jc.job_dir(NAME) / f"{key}.json"
    m = jc.read_json(f, {}) or {}
    m[label] = str(rd)
    atomic_write_json(f, m)


def dev_eval(label, weights=None):
    rd = jc.harness_eval(EXP, label, {"name": "base"}, weights=weights, datasets=datasets(), split="dev",
                         batch_size=16, purpose="select")
    _remember(label, rd)
    return rd


# ----------------------------------------------------------------------------- train
def train_seed(seed, a, budget, forget, retain, tok, csets):
    """True when the final weights exist; False when stopped by the budget (checkpointed, next sbatch resumes)."""
    from dsgx.train.core import Trainer, mcq_accuracy_hf

    final = student_path(seed, a.alpha)
    if (final / "config.json").exists():
        return True
    work = jc.job_dir(NAME) / tag(seed, a.alpha)
    teacher = jc.load_lm(jc.HF_2B)
    teacher.eval().requires_grad_(False)
    hook, handle, info = jc.dsg_guard(teacher)
    student = jc.load_lm(jc.HF_2B, train=True)
    d1_full.noise_(student, a.alpha, seed=1000 * (seed + 1) + int(round(a.alpha * 1000)))
    dev = next(student.parameters()).device

    def batch(src, step, salt):
        g = torch.Generator().manual_seed(1_000_003 * seed + step * 2 + salt)  # per seed + step: resumable
        idx = torch.randint(0, len(src), (a.bs,), generator=g).tolist()
        enc = tok([src[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=a.maxlen)
        return {k: v.to(dev) for k, v in enc.items()}

    tr = None

    def step_fn(step):
        if step % a.ckpt == 0 and step > tr.start and not budget.fits(DEV_EVAL_MIN + 2):
            raise _Stop(step)
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
    jc.log(NAME, f"{tag(seed, a.alpha)}: steps {tr.start}->{a.steps}, budget left {budget.left():.0f} min")
    done = True
    try:
        tr.run(step_fn)
    except _Stop as e:
        jc.log(NAME, f"{tag(seed, a.alpha)}: budget used, stopped at step {e.args[0]} (checkpointed); next sbatch resumes")
        done = False
    if done:
        final.mkdir(parents=True, exist_ok=True)
        student.save_pretrained(final)
        tok.save_pretrained(final)
        atomic_write_json(final / "d1_v3.json", {"alpha": a.alpha, "seed": seed, "steps": a.steps, "dsg": info,
                                                 "n_distill_passages": len(forget), "n_holdout": N_HOLDOUT,
                                                 "retain": "wikitext-2 only (no benchmark questions)",
                                                 "last": tr.log[-1] if tr.log else None, "time": now_iso()})
        shutil.copy2(work / "train_log.parquet", final.parent / f"{tag(seed, a.alpha)}.train_log.parquet")
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
    forget, _ = split_passages()
    retain = jc.retain_passages(n=None if not jc.TINY else 16)
    csets = curve_sets() if not jc.TINY else (jc.mcq_items("wmdp-bio", "dev", 2), jc.mcq_items("human_aging", "dev", 2))
    if not jc.TINY and budget.fits(DEV_EVAL_MIN):
        dev_eval("base")
    for seed in a.seeds:
        final = student_path(seed, a.alpha)
        if not (final / "config.json").exists():
            resumed = (jc.job_dir(NAME) / tag(seed, a.alpha) / "last").exists()
            need = 6 + (a.steps * SEC_PER_STEP / 60 if not resumed else 15)
            if not budget.fits(min(need, 20)):
                jc.log(NAME, f"{tag(seed, a.alpha)}: {budget.left():.0f} min left, not starting; next sbatch continues")
                break
            if not train_seed(seed, a, budget, forget, retain, tok, csets):
                break
        if not jc.TINY:
            if not budget.fits(DEV_EVAL_MIN - 4):
                break
            dev_eval(f"undo-v3-s{seed}", weights=final)
    jc.log(NAME, f"trained: { {s: (student_path(s, a.alpha) / 'config.json').exists() for s in a.seeds} }")
    cmd_summary(a)
    return 0


# ----------------------------------------------------------------------------- test (MCQ, bs 1)
def test_conditions(a):
    c = [("base", {"name": "base"}, None), ("dsg-paper", dict(DSG_PAPER), None)]
    return c + [(f"undo-v3-s{s}", {"name": "base"}, student_path(s, a.alpha)) for s in a.seeds]


def cmd_test(a):
    budget = jc.Budget(a.budget_min)
    missing = [s for s in a.seeds if not (student_path(s, a.alpha) / "config.json").exists()]
    if missing:
        jc.log(NAME, f"ABORT: students not trained for seeds {missing}; resubmit d1-v3-train.sbatch (resumes)")
        return 3
    if jc.TINY:
        return 0
    jc.require_gpu(40)
    for label, method, w in test_conditions(a):
        if not budget.fits(EST_MIN["test_run"]):
            jc.log(NAME, "budget used before all TEST runs; the next d1-v3-test.sbatch continues (finished runs are skipped)")
            return 0
        _remember(label, jc.harness_eval(EXP, label, method, weights=w, datasets=datasets()), key="test_runs")
    cmd_summary(a)
    return 0


# ----------------------------------------------------------------------------- open-ended
class _Prog:  # the queue worker's Progress interface, without a heartbeat file
    def update(self, **kw):
        pass

    def advance(self, *a, **kw):
        pass


def open_specs(a):
    """[(task_id, task args)] for the open-ended evaluation; one task per (item set, model)."""
    sets = {"leak": OPEN_SETS["leak"], "benign": "mmlu-open:" + ",".join(adjacent())}
    stream_dsg = {**DSG_PAPER, "mode": "stream", "tag": "dsg-paper"}
    out = []
    for kind, items in sets.items():
        common = {"items": items, "split": "test", "max_new": OPEN_MAX_NEW, "case": "bio"}
        out.append((f"open-{kind}-base", {**common, "methods": [{"name": "base", "tag": "base"}, stream_dsg]}))
        for s in a.seeds:
            out.append((f"open-{kind}-undo-v3-s{s}", {**common, "model": {"weights": str(student_path(s, a.alpha))},
                                                     "methods": [{"name": "base", "tag": f"undo-v3-s{s}"}]}))
    return out


def open_done(task_id, args):
    return all((paths.runs_dir() / EXP / f"{task_id}__{m['tag']}" / "DONE").exists() for m in args["methods"])


def cmd_open(a):
    from dsgx.tasks import TaskContext

    budget = jc.Budget(a.budget_min)
    specs = open_specs(a)
    if jc.TINY:
        atomic_write_json(jc.job_dir(NAME) / "open_plan.json", [{"task_id": t, "args": x} for t, x in specs])
        return 0
    jc.require_gpu(40)
    from dsgx.eval import openqa

    for tid, args in specs:
        if open_done(tid, args):
            continue
        if not budget.fits(EST_MIN["open_cond"] * len(args["methods"])):
            jc.log(NAME, f"{tid}: budget used; the next d1-v3-open.sbatch continues (finished tasks skipped, items resume)")
            return 0
        job = {"exp_id": EXP, "task_id": tid, "args": args, "commit": jc.read_json("CODE_COMMIT.json", {}).get("commit")}
        head = openqa.task(TaskContext(job, args, _Prog(), False))
        jc.log(NAME, f"{tid}: match {json.dumps({k: (v or {}).get('mean') for k, v in head.items()})}")
        from dsgx.models import loader

        loader._MODELS.clear()
        gc.collect()
        torch.cuda.empty_cache()
    cmd_summary(a)
    return 0


# ----------------------------------------------------------------------------- relearning (full + LoRA)
def relearn_targets(a):
    t = {"dsg-hook": (None, True), "dsg-nohook": (None, False),
         "rmu-v2": (paths.cache_dir() / "models" / "RMU-v2" / "best", False)}
    t.update({f"d1v3-s{s}": (student_path(s, a.alpha), False) for s in a.seeds})
    return t


def lora_cell(target, wpath, hook_on, k, r, p, tok, forget_eval, util_eval, passages):
    from peft import LoraConfig, get_peft_model

    from dsgx.train.core import Trainer

    d = paths.runs_dir() / EXP_LORA / f"relearn-{target}-k{k}-r{r}"
    if (d / "DONE").exists():
        return
    d.mkdir(parents=True, exist_ok=True)
    model = jc.load_lm(str(wpath) if wpath else jc.HF_2B, train=True)
    info = None
    if hook_on:
        _, _, info = jc.dsg_guard(model)
    model = get_peft_model(model, LoraConfig(r=r, lora_alpha=2 * r, lora_dropout=0.0, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                             "up_proj", "down_proj"]))
    before = {"forget_acc": a6_full.acc(model, tok, forget_eval), "util_acc": a6_full.acc(model, tok, util_eval)}
    curve = []
    g = torch.Generator().manual_seed(k)
    docs = [passages[i] for i in torch.randperm(len(passages), generator=g)[:k].tolist()]
    dev = next(model.parameters()).device

    def step_fn(step):
        idx = torch.randint(0, len(docs), (p.bs,), generator=g).tolist()
        enc = tok([docs[i] for i in idx], return_tensors="pt", padding=True, truncation=True, max_length=p.maxlen)
        enc = {kk: v.to(dev) for kk, v in enc.items()}
        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100
        return {"loss": model(**enc, labels=labels).loss}

    def eval_fn(step):
        if step in p.eval_at:
            row = {"step": step, "forget_acc": a6_full.acc(model, tok, forget_eval), "util_acc": a6_full.acc(model, tok, util_eval)}
            curve.append(row)
            return {"forget_acc": row["forget_acc"], "util_acc": row["util_acc"]}
        return {}

    tr = Trainer(model, [q for q in model.parameters() if q.requires_grad], jc.job_dir(NAME) / f"lora-{target}-k{k}-r{r}",
                 lr=p.lr, steps=p.steps, ckpt_every=10 ** 9, eval_fn=eval_fn, eval_every=1, optimizer="adamw")
    tr.run(step_fn)
    shutil.rmtree(tr.out / "last", ignore_errors=True)
    met = {"condition": target, "k": k, "rank": r, "steps": p.steps, "lr": p.lr, "batch_size": p.bs, "max_len": p.maxlen,
           "weights": str(wpath) if wpath else None, "hook": hook_on, "dsg": info, "before": before, "curve": curve,
           "n_eval": len(forget_eval), "n_util": len(util_eval), "passages": "held-out forget split",
           "time": now_iso(), "hardware_label": jc.hardware_label()}
    atomic_write_json(d / "metrics.json", met)
    atomic_write_json(d / "DONE", {"time": now_iso(), "headline": {"before": before, "after": curve[-1] if curve else None}})
    jc.log(NAME, f"{d.name}: forget {before['forget_acc']:.3f} -> {curve[-1]['forget_acc'] if curve else float('nan'):.3f}")
    del model, tr
    gc.collect()
    torch.cuda.empty_cache()


def cmd_relearn(a):
    budget = jc.Budget(a.budget_min)
    jc.require_gpu(40)
    tok = jc.load_tok()
    tok.padding_side = "right"
    forget_eval, util_eval = a6_full.eval_sets(a.n_eval, a.n_util)
    _, holdout = split_passages()
    a6_full.NAME = NAME  # a6_full.cell keeps its trainer logs under jobs/<NAME>/ (here: jobs/d1-v3/)
    full = SimpleNamespace(out_exp=EXP_FULL, **FULL)
    lora = SimpleNamespace(**LORA)
    if jc.TINY:
        full.steps, full.eval_at, full.maxlen, lora.steps, lora.eval_at, lora.maxlen = 2, [2], 32, 2, [2], 32
        full.bs = lora.bs = 2
    skipped = {}
    for t, (w, hook_on) in relearn_targets(a).items():
        if w is not None and not jc.TINY and not (w / "config.json").exists():
            skipped[t] = f"weights not staged: {w}"
            continue
        w = None if jc.TINY else w
        for k in a.ks:
            if not (paths.runs_dir() / EXP_FULL / f"relearn-{t}-k{k}" / "DONE").exists():
                if not budget.fits(EST_MIN["full_cell"] + 2):
                    jc.log(NAME, "budget used; the next d1-v3-relearn.sbatch continues (DONE cells skipped)")
                    return 0
                a6_full.cell(t, w, hook_on, k, full, tok, forget_eval, util_eval, holdout)
            for r in a.ranks:
                if (paths.runs_dir() / EXP_LORA / f"relearn-{t}-k{k}-r{r}" / "DONE").exists():
                    continue
                if not budget.fits(EST_MIN["lora_cell"] + 2):
                    jc.log(NAME, "budget used; the next d1-v3-relearn.sbatch continues (DONE cells skipped)")
                    return 0
                lora_cell(t, w, hook_on, k, r, lora, tok, forget_eval, util_eval, holdout)
    atomic_write_json(jc.job_dir(NAME) / "relearn_skipped.json", skipped)
    jc.log(NAME, f"relearning done; skipped {skipped}")
    cmd_summary(a)
    return 0


# ----------------------------------------------------------------------------- summary
def _dev_row(rd):
    import pandas as pd
    from pathlib import Path

    rd = Path(rd)
    if not (rd / "DONE").exists():
        return None
    it = pd.read_parquet(rd / "items.parquet", columns=["dataset", "correct"])
    adj = it["dataset"].isin(adjacent())
    fg = it["dataset"] == "wmdp-bio"
    return {"forget": float(it[fg]["correct"].mean()), "utility": float(it[~fg & ~adj]["correct"].mean()),
            "adjacent": float(it[adj]["correct"].mean()) if adj.any() else None}


def _test_row(rd):
    from pathlib import Path

    m = jc.read_json(Path(rd) / "metrics.json")
    if not m:
        return None
    v = m["views"]["raw"]
    per = v.get("per_dataset") or {}
    adj = [per[s]["mean"] for s in adjacent() if isinstance(per.get(s), dict) and "mean" in per[s]]
    return {"forget": v.get("forget"), "utility": (v.get("utility") or {}).get("pooled"),
            "adjacent_mean_of_subjects": float(np.mean(adj)) if adj else None}


def _mean_sd(xs):
    xs = [x for x in xs if x is not None]
    return {"mean": float(np.mean(xs)), "sd": float(np.std(xs, ddof=1)) if len(xs) > 1 else None, "n_seeds": len(xs)} if xs else None


def cmd_summary(a):
    dev = {k: _dev_row(v) for k, v in (jc.read_json(jc.job_dir(NAME) / "dev_runs.json", {}) or {}).items()}
    base = dev.get("base")
    for k, r in dev.items():
        if r and base and k != "base":
            r["util_drop"] = base["utility"] - r["utility"]
            r["bound_met"] = r["util_drop"] <= UTIL_DROP + 1e-9
    test = {k: _test_row(v) for k, v in (jc.read_json(jc.job_dir(NAME) / "test_runs.json", {}) or {}).items()}
    studs = [test.get(f"undo-v3-s{s}") for s in a.seeds]
    agg = {"forget": _mean_sd([(r["forget"] or {}).get("mean") for r in studs if r]),
           "utility": _mean_sd([(r["utility"] or {}).get("mean") for r in studs if r])}
    opened = {}
    for d in sorted((paths.runs_dir() / EXP).glob("open-*__*")):
        m = jc.read_json(d / "metrics.json")
        if m and (d / "DONE").exists():
            opened[d.name] = {"match": m.get("match"), "gibberish": m.get("gibberish"), "n": m.get("n")}
    rel = {}
    for ex in (EXP_FULL, EXP_LORA):
        for p in sorted((paths.runs_dir() / ex).glob("relearn-*/metrics.json")):
            m = json.loads(p.read_text())
            rel[f"{ex}/{p.parent.name}"] = {"before": m["before"]["forget_acc"],
                                            "after": m["curve"][-1]["forget_acc"] if m["curve"] else None,
                                            "util_after": m["curve"][-1]["util_acc"] if m["curve"] else None}
    s = {"alpha": a.alpha, "seeds": a.seeds, "dev": dev, "test": test, "students_test_mean_sd": agg,
         "open": opened, "relearn": rel, "rule": "no selection; 3-seed mean; bound_met = DEV utility drop <= 0.02",
         "hardware_label": jc.hardware_label()}
    jc.summary(NAME, s)
    f3 = lambda x: "-" if x is None else f"{x:.3f}"  # noqa: E731
    ci = lambda m: "-" if not m else f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}] n={m['n']}"  # noqa: E731
    L = ["# D1 v3 (gpuws)", "", f"alpha {a.alpha}, seeds {a.seeds}; distillation on the forget corpus + WikiText-2 only.", "",
         "## DEV (bs 16)", "", "| run | forget | utility (@utility) | adjacent | drop | bound met |", "|---|---|---|---|---|---|"]
    L += [f"| {k} | {f3(r['forget'])} | {f3(r['utility'])} | {f3(r['adjacent'])} | {f3(r.get('util_drop'))} | {r.get('bound_met', '-')} |"
          for k, r in dev.items() if r]
    L += ["", "## TEST (bs 1, raw view)", "", "| condition | forget | utility | adjacent (mean of subjects) |", "|---|---|---|---|"]
    L += [f"| {k} | {ci(r['forget'])} | {ci(r['utility'])} | {f3(r['adjacent_mean_of_subjects'])} |" for k, r in test.items() if r]
    if agg["forget"]:
        L += ["", f"Students, mean (sd) over {agg['forget']['n_seeds']} seeds: forget {agg['forget']['mean']:.3f} "
              f"({f3(agg['forget']['sd'])}), utility {agg['utility']['mean']:.3f} ({f3(agg['utility']['sd'])})."]
    L += ["", "## Open-ended (match rate; leak: lower is better, benign: higher is better)", "", "| run | match | gibberish |", "|---|---|---|"]
    L += [f"| {k} | {ci(v['match'])} | {ci(v['gibberish'])} |" for k, v in opened.items()]
    L += ["", "## Relearning (forget accuracy before -> after; same eval sets, held-out passages)", "",
          "| cell | before | after | utility after |", "|---|---|---|---|"]
    L += [f"| {k} | {f3(v['before'])} | {f3(v['after'])} | {f3(v['util_after'])} |" for k, v in rel.items()]
    atomic_write_text(jc.job_dir(NAME) / "SUMMARY.md", "\n".join(L) + "\n")
    return 0


# ----------------------------------------------------------------------------- plan (CPU dry run)
def cmd_plan(a):
    """No GPU, no writes outside jobs/d1-v3/plan.json. Exit 1 if an input is missing."""
    import importlib.util

    from dsgx.data.mcq import utility_subjects
    from dsgx.eval.openqa import item_set

    problems, out = [], {}
    distill, hold = split_passages()
    out["passages"] = {"distill": len(distill), "holdout": len(hold)}
    if len(hold) < max(a.ks):
        problems.append(f"holdout {len(hold)} < max k {max(a.ks)}")
    out["retain_passages"] = len(jc.retain_passages(n=None if not jc.TINY else 16))
    out["mcq_items"] = {"wmdp-bio dev/test": [len(jc.mcq_items("wmdp-bio", "dev")), len(jc.mcq_items("wmdp-bio", "test"))],
                        "@utility subjects": len(utility_subjects("bio")), "adjacent subjects": len(adjacent())}
    for kind, (_, args) in zip(("leak", "benign"), open_specs(a)[:: 1 + len(a.seeds)]):
        try:
            out.setdefault("open_items", {})[kind] = len(item_set(args["items"], "test"))
        except Exception as e:  # noqa: BLE001
            problems.append(f"open items {kind}: {type(e).__name__}: {e}")
    feats, tau = jc.dsg_features("bio")
    out["dsg"] = {"n_features": len(feats), "tau": round(tau, 4)}
    for lab, method, w in test_conditions(a):
        cfg = {"exp_id": EXP, "case": "bio", "split": "test", "view": "both", "seed": 0, "batch_size": 1,
               "datasets": datasets(), "dataset_label": lab, "method": dict(method)}
        if w:
            cfg["model"] = {"weights": str(w)}
        try:
            out.setdefault("test_runs", {})[lab] = jc.run_dir_of(cfg).name
        except Exception as e:  # noqa: BLE001
            problems.append(f"TEST config {lab}: {type(e).__name__}: {e}")
    tg = relearn_targets(a)
    out["relearn_targets"] = {t: ("hook" if h else (str(w) if w else "base weights")) for t, (w, h) in tg.items()}
    rmu = tg["rmu-v2"][0]
    lab_rmu = paths.PROJECT / "dsg_results_cluster" / "checkpoints" / "RMU-v2" / "best"
    if not jc.TINY and not (rmu / "config.json").exists():
        if (lab_rmu / "config.json").exists():
            out["rmu-v2"] = f"lab PC copy {lab_rmu} (server.sh stage d1-v3 copies it to {rmu})"
        else:
            problems.append(f"rmu-v2 weights not found at {rmu} or {lab_rmu} (its relearning cells would be skipped)")
    try:
        from huggingface_hub import try_to_load_from_cache

        if not isinstance(try_to_load_from_cache("sentence-transformers/all-MiniLM-L6-v2", "config.json"), str):
            problems.append("sentence-transformers/all-MiniLM-L6-v2 not in the HF cache (open-ended grader; staged by d1-v3.conf)")
    except Exception as e:  # noqa: BLE001
        problems.append(f"HF cache check failed: {type(e).__name__}: {e}")
    if importlib.util.find_spec("peft") is None:
        problems.append("peft not importable (LoRA relearning)")
    n_t = len(tg)
    n_full, n_lora = n_t * len(a.ks), n_t * len(a.ks) * len(a.ranks)
    n_open = sum(len(x["methods"]) for _, x in open_specs(a))
    hours = {"train (incl. DEV)": len(a.seeds) * (a.steps * SEC_PER_STEP / 60 + DEV_EVAL_MIN) / 60,
             "test": len(test_conditions(a)) * EST_MIN["test_run"] / 60,
             "open": n_open * EST_MIN["open_cond"] / 60,
             "relearn": (n_full * EST_MIN["full_cell"] + n_lora * EST_MIN["lora_cell"]) / 60}
    out["estimate_hours"] = {k: round(v, 1) for k, v in hours.items()} | {"total": round(sum(hours.values()), 1)}
    out["relearn_cells"] = {"full": n_full, "lora": n_lora}
    out["disk_gb"] = {"students (3 x 5.2)": round(5.2 * len(a.seeds), 1), "trainer state (one at a time)": 10.0,
                      "rmu-v2 staged": 4.9, "forget corpus": 0.7, "MiniLM grader": 0.9,
                      "run dirs + logs": 0.5, "peak": round(5.2 * len(a.seeds) + 10 + 4.9 + 0.7 + 0.9 + 0.5, 1)}
    out["problems"] = problems
    atomic_write_json(jc.job_dir(NAME) / "plan.json", out)
    print(json.dumps(out, indent=1))
    print("PLAN OK" if not problems else f"PLAN: {len(problems)} problem(s)")
    return 0 if not problems else 1


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["plan", "train", "test", "open", "relearn", "summary"])
    ap.add_argument("--alpha", type=float, default=ALPHA)
    ap.add_argument("--seeds", type=int, nargs="*", default=SEEDS)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--maxlen", type=int, default=MAXLEN)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--ckpt", type=int, default=CKPT)
    ap.add_argument("--ks", type=int, nargs="*", default=RELEARN_KS)
    ap.add_argument("--ranks", type=int, nargs="*", default=LORA_RANKS)
    ap.add_argument("--n-eval", type=int, default=a6_full.N_EVAL)
    ap.add_argument("--n-util", type=int, default=a6_full.N_UTIL)
    ap.add_argument("--budget-min", type=float, default=None)
    a = ap.parse_args(argv)
    return {"plan": cmd_plan, "train": cmd_train, "test": cmd_test, "open": cmd_open, "relearn": cmd_relearn,
            "summary": cmd_summary}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
