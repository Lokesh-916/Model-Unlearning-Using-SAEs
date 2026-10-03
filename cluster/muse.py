"""A5-muse (BM1): MUSE (Shi et al. 2024) on gemma-2-2b-it targets built on gpuws, then DSG and our best gate,
scored with the OFFICIAL muse_bench metric code (VerbMem, KnowMem, PrivLeak).

Inputs (fetched on the lab PC with `cluster/fetch_models.sh muse`, staged by `server.sh stage muse`):
  * datasets muse-bench/MUSE-News and muse-bench/MUSE-Books (configs raw, verbmem, knowmem, privleak);
  * the official code github.com/swj0419/muse_bench (wheels/muse/muse_bench-main.tar.gz, sha256 in SHA256SUMS);
    unpacked once to $DSGC/data/muse/muse_bench-main; its metrics/{verbmem,knowmem,privleak}.py are imported
    and called unchanged. A local re-implementation of the same definitions is used only if that import fails
    (recorded as "implementation" in metrics.json).

Per corpus (news, then books), resumable in stages, so a chain of <= 3 h sbatch jobs finishes it
(DSG_BUDGET_MIN from slurm/later.sh; every stage checks the budget before it starts; training stops at a
checkpoint when the budget runs low and the next chained job resumes it):
  1. retrain = full fine-tune of gemma-2-2b-it on raw retain1 (the "never saw forget" reference), then its
     metrics, then its weights are deleted (only one model on disk at a time);
  2. target  = full fine-tune on raw forget + retain1 (MUSE's target recipe on our base model);
     1024-token chunks, batch 8 (2 x accumulation 4), lr 1e-5, 3 epochs, AdamW 8-bit, checkpoint every 100 steps;
  3. conditions on the target, each saved on its own (partial/<cond>.json):
       target            no unlearning (MUSE's "target" = our base for this benchmark)
       target+dsg        DSG on the target: layer-3 SAE, top-20 features by forget/retain fire-rate ratio on
                         256-token raw chunks, tau = 95th percentile of rho on retain chunks, clamp x500
       target+best-gate  our best gate: window gate (w from COMBINE_SELECTION.json if its detector slot is
                         window-w*, else w = 16; same rule as tofu-full), threshold = 95th pct on retain chunks
     Gates decide on the prompt (first forward pass) and keep that decision for every generated token
     (prompt-only gating); with a KV cache a per-token rho would otherwise be computed on one token.
  4. metrics (official definitions): VerbMem = mean ROUGE-L F1 of 128 greedy tokens vs the first 128 gt tokens;
     KnowMem forget / retain = mean ROUGE-L F1 of greedy answers (32 tokens) with the split's own ICL demos;
     PrivLeak = (AUC_model - AUC_retrain) / AUC_retrain x 100 with AUC = forget-vs-holdout Min-40% AUC and
     AUC_retrain from OUR retrain model (the constants in muse_bench are for Llama-2-7B, not this model).
     privleak.eval() of muse_bench main crashes (it calls .keys() on a list), so its eval_data() and sweep()
     are called directly, as eval() intends.
     MUSE Books has no KnowMem files in muse_bench (DEFAULT_DATA None), but the HF dataset has knowmem
     splits; they are scored the same way.
Writes runs/A5-muse/muse-<corpus>/{partial/*.json, metrics.json, DONE} and jobs/muse/summary.json.
Only aggregate metrics are written; MUSE text is copyrighted (not hazardous) and is never printed or saved.

    python cluster/muse.py [--budget-min M] [--corpora news books] [--keep]
    python cluster/muse.py --inspect          # configs / splits / columns (no text)
"""
import argparse
import gc
import os
import shutil
import sys
import tarfile
from pathlib import Path

import numpy as np
import torch

from cluster import jobcommon as jc
from dsgx import paths
from dsgx.util import atomic_write_json, now_iso

NAME = "muse"
EXP = "A5-muse"
REPO = {"news": "muse-bench/MUSE-News", "books": "muse-bench/MUSE-Books"}
EPOCHS, LR, BS, ACCUM, CHUNK, CKPT = 3, 1e-5, 2, 4, 1024, 100
AUC_KEY = "forget_holdout_Min-40%"
CONDITIONS = ("target", "target+dsg", "target+best-gate")
EVAL_MIN = 25          # one condition's metrics (300 generations + 600 scoring passes), with margin
TRAIN_MIN_START = 20   # do not start (or resume) a fine-tune with less budget than this
STOP_MARGIN = 8        # stop training at a checkpoint when less than this is left
DISK_NEED_GB = 16      # one model (5.2) + trainer state (~10) at a time


class _Stop(Exception):
    pass


# ----------------------------------------------------------------------------- data
def ds(corpus, config, split):
    from datasets import load_dataset

    return load_dataset(REPO[corpus], config, split=split)


def inspect(corpus):
    from datasets import get_dataset_config_names, load_dataset

    for c in get_dataset_config_names(REPO[corpus]):
        d = load_dataset(REPO[corpus], c)
        print(corpus, c, {s: (len(d[s]), d[s].column_names) for s in d})


def raw_texts(corpus, split):
    return [r["text"] for r in ds(corpus, "raw", split)]


def chunks(tok, texts, n=CHUNK):
    ids = []
    for t in texts:
        ids += tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return [ids[i:i + n] for i in range(0, len(ids) - n + 1, n)]


def load_eval(corpus, n):
    if jc.TINY:
        vm = [{"prompt": f"Once upon a time {i}", "gt": f"there was a test number {i}"} for i in range(3)]
        qa = [{"question": f"Who is {i}?", "answer": f"person {i}"} for i in range(3)]
        return {"verbmem": vm, "kf": qa, "kf_icl": qa[:1], "kr": qa, "kr_icl": qa[:1],
                "pf": [f"a b c {i} " * 30 for i in range(4)], "pr": [f"g h {i} " * 30 for i in range(4)],
                "ph": [f"d e f {i} " * 30 for i in range(4)]}

    def take(cfg, split, k=None):
        rows = [dict(r) for r in ds(corpus, cfg, split)]
        return rows[:k] if k else rows
    return {"verbmem": take("verbmem", "forget", n), "kf": take("knowmem", "forget_qa", n),
            "kf_icl": take("knowmem", "forget_qa_icl"), "kr": take("knowmem", "retain_qa", n),
            "kr_icl": take("knowmem", "retain_qa_icl"),
            "pf": [r["text"] for r in take("privleak", "forget", n)], "pr": [r["text"] for r in take("privleak", "retain", n)],
            "ph": [r["text"] for r in take("privleak", "holdout", n)]}


# ----------------------------------------------------------------------------- official metric code
def official_dir(arg=None) -> Path | None:
    """Unpack the staged muse_bench tarball once (inside ~/dsg_cluster / the lab project) and return its root."""
    cands = [Path(arg)] if arg else []
    root = Path(os.environ.get("DSGC", Path.home() / "dsg_cluster")) / "data" / "muse"
    lab = Path.home() / "projects/mechunlearn-project/wheels/muse"
    cands += [root / "muse_bench-main", lab / "muse_bench-main"]
    for c in cands:
        if (c / "metrics" / "verbmem.py").exists():
            return c
    for tb in (root / "muse_bench-main.tar.gz", lab / "muse_bench-main.tar.gz"):
        if tb.exists():
            dest = tb.parent
            with tarfile.open(tb) as t:
                t.extractall(dest, members=[m for m in t.getmembers() if m.name.startswith("muse_bench-main/metrics/")],
                             filter="data")
            if (dest / "muse_bench-main/metrics/verbmem.py").exists():
                return dest / "muse_bench-main"
    return None


class Metrics:
    """VerbMem / KnowMem / PrivLeak via the official functions (or the re-implementation as a fallback)."""

    def __init__(self, odir: Path | None):
        self.impl = "reimplementation of muse_bench definitions"
        self.V = self.K = self.P = None
        if odir is not None:
            try:
                sys.path.insert(0, str(odir))
                import metrics.knowmem as K  # noqa: E402  (muse_bench/metrics, relative import of .logger)
                import metrics.privleak as P
                import metrics.verbmem as V

                self.V, self.K, self.P = V, K, P
                self.impl = (f"official muse_bench ({odir.name}/metrics): verbmem.eval, knowmem.eval, "
                             "privleak.eval_data + sweep (privleak.eval itself crashes on main)")
            except Exception as e:  # missing dependency -> fallback, recorded
                jc.log(NAME, f"official metric import failed ({type(e).__name__}: {e}); using the re-implementation")

    @staticmethod
    def _agg(agg):
        keep = ("mean_rougeL", "rougeL_ci_lo", "rougeL_ci_hi", "mean_rougeL_recall")
        return {k: (float(agg[k]) if agg.get(k) is not None and np.isfinite(agg[k]) else None) for k in keep if k in agg}

    def verbmem(self, model, tok, rows):
        prompts, gts = [r["prompt"] for r in rows], [r["gt"] for r in rows]
        if self.V:
            agg, _ = self.V.eval(model=model, tokenizer=tok, prompts=prompts, gts=gts, max_new_tokens=128)
            return self._agg(agg)
        return _re_verbmem(model, tok, prompts, gts)

    def knowmem(self, model, tok, rows, icl):
        q, a = [r["question"] for r in rows], [r["answer"] for r in rows]
        iq, ia = [r["question"] for r in icl], [r["answer"] for r in icl]
        if self.K:
            agg, _ = self.K.eval(model=model, tokenizer=tok, questions=q, answers=a, icl_qs=iq, icl_as=ia, max_new_tokens=32)
            return self._agg(agg)
        return _re_knowmem(model, tok, q, a, iq, ia)

    def privleak_auc(self, model, tok, forget, retain, holdout):
        if self.P:
            # privleak.eval() of muse_bench main crashes (`log['forget'].keys()` on a list), so we call its own
            # eval_data() and sweep() exactly as eval() intends for the key forget_holdout_Min-40%:
            # forget = label 0 ("nonmember" slot), holdout = label 1, ROC on -score. `retain` is not needed.
            key = AUC_KEY.split("_", 2)[2]
            f = [d[key] for d in self.P.eval_data(forget, model, tok)]
            h = [d[key] for d in self.P.eval_data(holdout, model, tok)]
            _, _, auc, _ = self.P.sweep(np.array(f + h), np.array([0] * len(f) + [1] * len(h)))
            return float(auc)
        return _re_auc(model, tok, forget, holdout)


@torch.no_grad()
def _gen(model, tok, prompt, n):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=True).input_ids.to(next(model.parameters()).device)
    out = model.generate(ids, max_new_tokens=n, do_sample=False, pad_token_id=tok.pad_token_id)
    return tok.batch_decode(out[:, ids.shape[1]:], skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]


def _rouge(pairs):
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    f = [sc.score(g, o)["rougeL"].fmeasure for g, o in pairs]
    lo, hi = np.percentile([np.mean(np.random.default_rng(i).choice(f, len(f))) for i in range(1000)], [2.5, 97.5])
    return {"mean_rougeL": float(np.mean(f)), "rougeL_ci_lo": float(lo), "rougeL_ci_hi": float(hi)}


def _re_verbmem(model, tok, prompts, gts):
    pairs = []
    for p, g in zip(prompts, gts):
        gs = tok.batch_decode(tok(g, return_tensors="pt", add_special_tokens=True).input_ids[:, :128],
                              skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        pairs.append((gs, _gen(model, tok, p, 128)))
    return _rouge(pairs)


def _re_knowmem(model, tok, q, a, iq, ia):
    demo = "".join(f"Question: {x}\nAnswer: {y}\n\n" for x, y in zip(iq, ia))
    pairs = []
    for x, y in zip(q, a):
        o = _gen(model, tok, demo + f"Question: {x}\nAnswer: ", 32)
        for w in ("\n\n", "\nQuestion", "Question:"):
            o = o.split(w)[0]
        pairs.append((y, o))
    return _rouge(pairs)


@torch.no_grad()
def _re_auc(model, tok, forget, holdout):
    from sklearn.metrics import auc, roc_curve

    def mink(t):
        ids = torch.tensor(tok.encode(t)).unsqueeze(0).to(next(model.parameters()).device)
        lp = model(ids).logits[0, :-1].float().log_softmax(-1)
        tl = lp[torch.arange(lp.shape[0]), ids[0, 1:]].cpu().numpy()
        k = int(len(tl) * 0.4)
        return float(-np.mean(np.sort(tl)[:k])) if k else float("nan")
    s = np.array([mink(t) for t in forget] + [mink(t) for t in holdout])
    y = np.array([0] * len(forget) + [1] * len(holdout))
    fpr, tpr, _ = roc_curve(y, -s)
    return float(auc(fpr, tpr))


# ----------------------------------------------------------------------------- gates
class PromptGate:
    """Wraps tofu_full.Gate: decide on the prompt (first forward with > 1 token), keep the decision for the
    generated tokens (single-token forwards with a KV cache) and clamp the selected features where they fire."""

    def __init__(self, gate):
        self.g, self.on = gate, None
        self.n_prompts = self.n_fired = 0

    def __call__(self, module, inputs, output):
        g = self.g
        hs = output[0] if isinstance(output, tuple) else output
        x = hs.to(g.sae.W_dec.dtype)
        acts = g.sae.encode(x)
        if x.shape[1] > 1 or self.on is None:
            self.on = [g.score((acts[b][:, g.f] > 0).any(-1)) > g.thr for b in range(x.shape[0])]
            self.n_prompts += x.shape[0]
            self.n_fired += sum(self.on)
        new = x.clone()
        for b in range(x.shape[0]):
            if not self.on[min(b, len(self.on) - 1)]:
                continue
            fire = (acts[b][:, g.f] > 0).any(-1)
            if fire.any():
                a2 = acts[b].clone()
                pos = fire.nonzero().flatten()
                a2[pos[:, None], g.f[None, :]] = -g.mult
                new[b] = g.sae.decode(a2) + (x[b] - g.sae.decode(acts[b]))
        new = new.to(hs.dtype)
        return (new, *output[1:]) if isinstance(output, tuple) else new


def calibrate_gates(m, tok, fr, r1, a):
    """DSG gate (rho) and best gate (window) on the target model; thresholds from retain chunks only."""
    from cluster.tofu_full import Gate, fire_stats

    sae, layer = jc.load_sae(), (1 if jc.TINY else 3)
    fch = [tok.decode(c) for c in chunks(tok, fr, 256)[: a.n_calib]]
    rch = [tok.decode(c) for c in chunks(tok, r1, 256)[: a.n_calib]]
    pf, _ = fire_stats(m, tok, sae, layer, fch)
    pr, _ = fire_stats(m, tok, sae, layer, rch)
    feats = [int(i) for i in np.argsort(-np.where(pf >= 0.01, (pf + 1e-4) / (pr + 1e-4), 0))[:20]]
    _, per = fire_stats(m, tok, sae, layer, rch, feats)
    rho_g = Gate(sae, feats, "rho", 0, mult=500)
    rho_g.thr = float(np.percentile([rho_g.score(f) for f in per], 95))
    sel = jc.read_json(paths.results_dir() / "COMBINE_SELECTION.json", {}) or {}
    det = (sel.get("slots") or {}).get("detector") or ""  # dsgx.combine.select: slot -> candidate name or None
    w = a.window or (int(det.split("-w")[1]) if det.startswith("window-w") else 16)
    win_g = Gate(sae, feats, "window", 0, w=w, mult=500)
    win_g.thr = float(np.percentile([win_g.score(f) for f in per], 95))
    info = {"features": feats, "layer": layer, "n_calib": len(rch), "dsg_tau": rho_g.thr, "window_w": w,
            "window_thr": win_g.thr, "best_gate_source": det or "default window-w16"}
    return {"target+dsg": rho_g, "target+best-gate": win_g}, layer, info


# ----------------------------------------------------------------------------- fine-tuning
def model_dir(corpus, tag):
    return paths.cache_dir() / "models" / EXP / f"{corpus}-{tag}"


def finetune(corpus, tag, texts, a, tok, budget):
    """Resumable full fine-tune; returns the saved model dir, or None if the budget ran out (checkpointed)."""
    from dsgx.train.core import Trainer

    out = model_dir(corpus, tag)
    if (out / "config.json").exists():
        return out
    if not budget.fits(TRAIN_MIN_START):
        return None
    model = jc.load_lm(jc.HF_2B, train=True)
    dev = next(model.parameters()).device
    ch = chunks(tok, texts, a.chunk)
    g = torch.Generator().manual_seed(0)
    order = torch.cat([torch.randperm(len(ch), generator=g) for _ in range(a.epochs + 1)]).tolist()
    steps = max(1, a.epochs * len(ch) // (a.bs * a.accum))
    tr = None

    def step_fn(step):
        if step % a.ckpt == 0 and step > tr.start and not budget.fits(STOP_MARGIN):
            raise _Stop(step)  # the checkpoint of `step` was written at the end of the previous step
        tot = 0.0
        for j in range(a.accum):
            b = (step * a.accum + j) * a.bs
            x = torch.tensor([ch[i] for i in order[b:b + a.bs]], device=dev)
            loss = model(input_ids=x, labels=x).loss / a.accum
            if j < a.accum - 1:
                loss.backward()
                tot = tot + loss.detach()
            else:
                tot = loss + tot
        return {"loss": tot}

    tr = Trainer(model, model.parameters(), jc.job_dir(NAME) / f"ft-{corpus}-{tag}", lr=a.lr, steps=steps,
                 ckpt_every=a.ckpt, optimizer="adamw" if jc.TINY else "adamw8bit")
    jc.log(NAME, f"{corpus}-{tag}: {len(ch)} chunks x {a.epochs} epochs = {steps} steps (start {tr.start})")
    try:
        tr.run(step_fn)
    except _Stop as e:
        jc.log(NAME, f"{corpus}-{tag}: budget low, stopped at step {e.args[0]} (checkpointed); next job resumes")
        del model, tr
        gc.collect()
        torch.cuda.empty_cache()
        return None
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    shutil.rmtree(tr.out / "last", ignore_errors=True)
    del model, tr
    gc.collect()
    torch.cuda.empty_cache()
    return out


# ----------------------------------------------------------------------------- evaluation
def evaluate(mt: Metrics, model, tok, data):
    return {"verbmem_forget": mt.verbmem(model, tok, data["verbmem"]),
            "knowmem_forget": mt.knowmem(model, tok, data["kf"], data["kf_icl"]),
            "knowmem_retain": mt.knowmem(model, tok, data["kr"], data["kr_icl"]),
            "privleak_auc": mt.privleak_auc(model, tok, data["pf"], data["pr"], data["ph"])}


def disk_ok(need_gb=DISK_NEED_GB) -> bool:
    """Rule 5 inside the job: free >= 50 GB after `need_gb`, and our total stays < 100 GB."""
    root = Path(os.environ.get("DSGC", paths.cache_dir()))
    free = shutil.disk_usage(root).free / 1e9
    ours = sum(f.stat().st_size for f in root.rglob("*") if f.is_file() and not f.is_symlink()) / 1e9 if not jc.TINY else 0
    ok = free - need_gb >= 50 and ours + need_gb < 100
    jc.log(NAME, f"disk: free {free:.0f} GB, ours {ours:.0f} GB, need {need_gb} GB -> {'ok' if ok else 'REFUSING'}")
    return ok or jc.TINY


def run_corpus(corpus, a, tok, budget, mt) -> bool:
    """True when the corpus is finished (DONE written)."""
    d = paths.runs_dir() / EXP / f"muse-{corpus}"
    part = d / "partial"
    if (d / "DONE").exists():
        jc.log(NAME, f"{corpus}: done, skip")
        return True
    part.mkdir(parents=True, exist_ok=True)
    if jc.TINY:
        fr, r1 = [f"tiny forget {i} " * 300 for i in range(4)], [f"tiny retain {i} " * 300 for i in range(4)]
    else:
        fr, r1 = raw_texts(corpus, "forget"), raw_texts(corpus, "retain1")
    data = None
    # 1. retrain reference
    if not (part / "retrain.json").exists():
        if not model_dir(corpus, "retrain").exists() and not disk_ok():
            return False
        m_dir = finetune(corpus, "retrain", r1, a, tok, budget)
        if m_dir is None or not budget.fits(EVAL_MIN):
            return False
        data = data or load_eval(corpus, a.n)
        m = jc.load_lm(m_dir).eval()
        atomic_write_json(part / "retrain.json", {"metrics": evaluate(mt, m, tok, data), "time": now_iso()})
        del m
        gc.collect()
        torch.cuda.empty_cache()
        if not a.keep:
            shutil.rmtree(m_dir, ignore_errors=True)
    # 2. target
    todo = [c for c in CONDITIONS if not (part / f"{c}.json").exists()]
    if todo:
        if not model_dir(corpus, "target").exists() and not disk_ok():
            return False
        m_dir = finetune(corpus, "target", fr + r1, a, tok, budget)
        if m_dir is None:
            return False
        m = None
        for c in todo:
            if not budget.fits(EVAL_MIN):
                return False
            data = data or load_eval(corpus, a.n)
            m = m or jc.load_lm(m_dir).eval()
            if c == "target":
                res = {"metrics": evaluate(mt, m, tok, data)}
            else:
                from dsgx.train.core import decoder_layers

                gates, layer, info = calibrate_gates(m, tok, fr, r1, a)
                pg = PromptGate(gates[c])
                h = decoder_layers(m)[layer].register_forward_hook(pg)
                try:
                    res = {"metrics": evaluate(mt, m, tok, data)}
                finally:
                    h.remove()
                res["gate"] = {**info, "kind": gates[c].kind, "threshold": gates[c].thr,
                               "fire_rate_eval_prompts": pg.n_fired / max(1, pg.n_prompts), "n_eval_prompts": pg.n_prompts}
            res["time"] = now_iso()
            atomic_write_json(part / f"{c}.json", res)
            jc.log(NAME, f"{corpus}: {c} done")
        del m
        gc.collect()
        torch.cuda.empty_cache()
    # 3. combine
    res = {k: jc.read_json(part / f"{k}.json")["metrics"] for k in ("retrain", *CONDITIONS)}
    gate = {k: jc.read_json(part / f"{k}.json").get("gate") for k in CONDITIONS[1:]}
    base = res["retrain"]["privleak_auc"]
    for v in res.values():
        v["privleak"] = (v["privleak_auc"] - base) / base * 100 if base else None
    atomic_write_json(d / "metrics.json", {"corpus": corpus, "conditions": res, "gates": gate, "n": a.n, "epochs": a.epochs,
                                           "implementation": mt.impl, "privleak_auc_key": AUC_KEY,
                                           "privleak_reference": "our retrain model (same recipe, retain1 only)",
                                           "gating": "prompt-only (decision on the prompt, kept for generated tokens)",
                                           "hardware": os.environ.get("DSG_HARDWARE", "unknown")})
    atomic_write_json(d / "DONE", {"time": now_iso(), "headline": {k: v["privleak"] for k, v in res.items()}})
    if not a.keep:
        shutil.rmtree(model_dir(corpus, "target"), ignore_errors=True)
    return True


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=["news", "books"])
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--accum", type=int, default=ACCUM)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--ckpt", type=int, default=CKPT)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-calib", type=int, default=400)
    ap.add_argument("--window", type=int, default=None)
    ap.add_argument("--official", default=None, help="muse_bench root (default: unpack the staged tarball)")
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    if a.inspect:
        for c in a.corpora:
            inspect(c)
        return 0
    jc.require_gpu(40)
    budget = jc.Budget(a.budget_min)
    tok = jc.load_tok()
    tok.padding_side = "right"
    if jc.TINY:
        a.chunk, a.n_calib, a.ckpt = 64, 2, 2
    mt = Metrics(official_dir(a.official))
    jc.log(NAME, f"metrics: {mt.impl}; budget {budget.minutes:.0f} min")
    done = [c for c in a.corpora if run_corpus(c, a, tok, budget, mt)]
    rows = {p.parent.name: jc.read_json(p) for p in sorted((paths.runs_dir() / EXP).glob("*/metrics.json"))}
    jc.summary(NAME, {"corpora": a.corpora, "finished": done, "implementation": mt.impl,
                      "results": {k: {c: {m: (v.get(m) if not isinstance(v.get(m), dict) else v[m].get("mean_rougeL"))
                                          for m in ("verbmem_forget", "knowmem_forget", "knowmem_retain", "privleak")}
                                      for c, v in r["conditions"].items()} for k, r in rows.items()}})
    jc.log(NAME, "all corpora done" if len(done) == len(a.corpora) else f"partial: done {done}; resubmit muse.sbatch to continue")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
