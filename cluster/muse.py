"""A5-muse: MUSE (Shi et al. 2024) with gemma-2-2b-it targets built on gpuws, then DSG, then VerbMem /
KnowMem / PrivLeak.

NOT ON THE LAB PC YET (fetch first with cluster/fetch_muse.sh, network, ~0.2 GB):
  * datasets muse-bench/MUSE-News and muse-bench/MUSE-Books (configs raw, verbmem, knowmem, privleak)
  * the official metric code github.com/swj0419/muse_bench (tarball in wheels/muse/). The functions below
    follow its definitions; after fetching, `--inspect` prints configs/splits/columns (no text) and
    `--official DIR` switches VerbMem/KnowMem to the official implementation if it imports.

Per corpus (news | books), one at a time (disk):
 1. target  = full fine-tune of gemma-2-2b-it on raw forget + retain1 (MUSE's target recipe, our base model);
    retrain = fine-tune on raw retain1 only (the reference "never saw forget" model).
    1024-token chunks, bs 8 (2 x accum 4), lr 1e-5, EPOCHS epochs, AdamW8bit, resumable.
 2. DSG on the target: layer-3 SAE, top-20 features by forget/retain fire-rate ratio on raw text, tau =
    95th pct of rho on retain chunks (same rule as tofu-full); clamp x500.
 3. Metrics for target, target+DSG, retrain:
    VerbMem  ROUGE-L F1 of the greedy continuation (len(gt) tokens, max 128) of verbmem prompts (forget);
    KnowMem  ROUGE-L recall of greedy answers (max 32 tokens) to knowmem QA, with the ICL demos, on
             forget_qa and retain_qa;
    PrivLeak Min-K% (k = 40%) AUC of forget (member) vs holdout (non-member) texts, and
             PrivLeak = (AUC_model - AUC_retrain) / AUC_retrain * 100 (0 = indistinguishable from retrain).
 Writes runs/A5-muse/muse-<corpus>/{metrics.json,DONE}; deletes the two models unless --keep.
News/Books text is copyrighted, not hazardous; still only metrics are written.
"""
import argparse
import gc
import shutil

import numpy as np
import torch

from cluster import jobcommon as jc
from dsgx import paths
from dsgx.util import atomic_write_json, now_iso

NAME = "muse"
REPO = {"news": "muse-bench/MUSE-News", "books": "muse-bench/MUSE-Books"}
EPOCHS, LR, BS, ACCUM, CHUNK = 3, 1e-5, 2, 4, 1024


def ds(corpus, config, split):
    from datasets import load_dataset

    return load_dataset(REPO[corpus], config, split=split)


def col(row, *names):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    raise KeyError(f"none of {names} in {list(row)}")


def inspect(corpus):
    from datasets import get_dataset_config_names, load_dataset

    for c in get_dataset_config_names(REPO[corpus]):
        d = load_dataset(REPO[corpus], c)
        print(corpus, c, {s: (len(d[s]), d[s].column_names) for s in d})


def raw_texts(corpus, split):
    d = ds(corpus, "raw", split)
    return [col(r, "text") for r in d]


def chunks(tok, texts, n=CHUNK):
    ids = []
    for t in texts:
        ids += tok(t, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    return [ids[i:i + n] for i in range(0, len(ids) - n + 1, n)]


def finetune(corpus, tag, texts, a, tok):
    from dsgx.train.core import Trainer

    out = paths.cache_dir() / "models" / "A5-muse" / f"{corpus}-{tag}"
    if (out / "config.json").exists():
        return out
    model = jc.load_lm(jc.HF_2B, train=True)
    dev = next(model.parameters()).device
    ch = chunks(tok, texts, a.chunk)
    g = torch.Generator().manual_seed(0)
    order = torch.cat([torch.randperm(len(ch), generator=g) for _ in range(a.epochs + 1)]).tolist()
    steps = max(1, a.epochs * len(ch) // (a.bs * a.accum))

    def step_fn(step):
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
                 ckpt_every=200, optimizer="adamw" if jc.TINY else "adamw8bit")
    tr.run(step_fn)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tok.save_pretrained(out)
    shutil.rmtree(tr.out / "last", ignore_errors=True)
    del model, tr
    gc.collect()
    torch.cuda.empty_cache()
    return out


@torch.no_grad()
def greedy(model, tok, prompt, max_new):
    enc = tok(prompt, return_tensors="pt", add_special_tokens=True, truncation=True, max_length=2048).to(next(model.parameters()).device)
    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False)
    return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)


def verbmem(model, tok, rows):
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"])
    out = []
    for r in rows:
        gt = col(r, "gt", "continuation", "answer")
        n = min(128, len(tok(gt, add_special_tokens=False)["input_ids"]))
        out.append(sc.score(gt, greedy(model, tok, col(r, "prompt", "text"), n))["rougeL"].fmeasure)
    return float(np.mean(out)) if out else None


def knowmem(model, tok, rows, icl):
    from rouge_score import rouge_scorer

    sc = rouge_scorer.RougeScorer(["rougeL"])
    demo = "".join(f"Question: {col(d, 'question')}\nAnswer: {col(d, 'answer')}\n\n" for d in icl)
    out = []
    for r in rows:
        ans = greedy(model, tok, demo + f"Question: {col(r, 'question')}\nAnswer:", 32).split("\n")[0]
        out.append(sc.score(col(r, "answer"), ans)["rougeL"].recall)
    return float(np.mean(out)) if out else None


@torch.no_grad()
def mink(model, tok, text, k=0.4):
    enc = tok(text, return_tensors="pt", truncation=True, max_length=512).to(next(model.parameters()).device)
    lp = model(**enc).logits[0, :-1].float().log_softmax(-1)
    tl = lp[torch.arange(lp.shape[0]), enc["input_ids"][0, 1:]]
    n = max(1, int(k * len(tl)))
    return float(tl.sort().values[:n].mean())


def auc(pos, neg):
    from dsgx.analysis.gate_quality import auroc

    return auroc(np.asarray(pos), np.asarray(neg))


def privleak_auc(model, tok, forget, holdout):
    # members should have HIGHER min-k log-prob; AUC of member vs non-member scores
    return auc([mink(model, tok, t) for t in forget], [mink(model, tok, t) for t in holdout])


def evaluate(model, tok, data):
    return {"verbmem_forget": verbmem(model, tok, data["verbmem"]),
            "knowmem_forget": knowmem(model, tok, data["kf"], data["icl"]),
            "knowmem_retain": knowmem(model, tok, data["kr"], data["icl"]),
            "privleak_auc": privleak_auc(model, tok, data["pf"], data["ph"])}


def load_eval(corpus, n):
    def take(d, k):
        return [dict(r) for r in d][:k]

    if jc.TINY:
        fake = [{"prompt": "Once upon a time", "gt": "there was a test", "question": "Who?", "answer": "a test", "text": "tiny text " * 30}] * 2
        return {"verbmem": fake, "kf": fake, "kr": fake, "icl": fake[:1], "pf": ["a b c " * 40] * 2, "ph": ["d e f " * 40] * 2}
    kn = ds(corpus, "knowmem", "forget_qa")
    return {"verbmem": take(ds(corpus, "verbmem", "forget"), n), "kf": take(kn, n),
            "kr": take(ds(corpus, "knowmem", "retain_qa"), n), "icl": take(ds(corpus, "knowmem", "forget_qa_icl"), 3),
            "pf": [col(r, "text") for r in take(ds(corpus, "privleak", "forget"), n)],
            "ph": [col(r, "text") for r in take(ds(corpus, "privleak", "holdout"), n)]}


def run_corpus(corpus, a, tok):
    from cluster.tofu_full import Gate, fire_stats
    from dsgx.train.core import decoder_layers

    d = paths.runs_dir() / "A5-muse" / f"muse-{corpus}"
    if (d / "DONE").exists():
        jc.log(NAME, f"{corpus}: done, skip")
        return
    if jc.TINY:
        fr, r1 = [f"tiny forget {i} " * 300 for i in range(4)], [f"tiny retain {i} " * 300 for i in range(4)]
    else:
        fr, r1 = raw_texts(corpus, "forget"), raw_texts(corpus, "retain1")
    m_target = finetune(corpus, "target", fr + r1, a, tok)
    m_retrain = finetune(corpus, "retrain", r1, a, tok)
    data = load_eval(corpus, a.n)
    res = {}
    m = jc.load_lm(m_retrain)
    m.eval()
    res["retrain"] = evaluate(m, tok, data)
    del m
    m = jc.load_lm(m_target)
    m.eval()
    res["target"] = evaluate(m, tok, data)
    sae, layer = jc.load_sae(), (1 if jc.TINY else 3)
    fch = [tok.decode(c) for c in chunks(tok, fr, 256)[: a.n_calib]]
    rch = [tok.decode(c) for c in chunks(tok, r1, 256)[: a.n_calib]]
    pf, _ = fire_stats(m, tok, sae, layer, fch)
    pr, _ = fire_stats(m, tok, sae, layer, rch)
    feats = [int(i) for i in np.argsort(-np.where(pf >= 0.01, (pf + 1e-4) / (pr + 1e-4), 0))[:20]]
    _, per = fire_stats(m, tok, sae, layer, rch, feats)
    g = Gate(sae, feats, "rho", 0, mult=500)
    g.thr = float(np.percentile([g.score(f) for f in per], 95))
    h = decoder_layers(m)[layer].register_forward_hook(g)
    res["target+dsg"] = evaluate(m, tok, data)
    res["target+dsg"]["gate"] = {"features": feats, "tau": g.thr}
    h.remove()
    base = res["retrain"]["privleak_auc"]
    for k, v in res.items():
        v["privleak"] = (v["privleak_auc"] - base) / base * 100 if base else None
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_json(d / "metrics.json", {"corpus": corpus, "conditions": res, "n": a.n, "epochs": a.epochs,
                                           "implementation": "reimplementation of muse_bench definitions"})
    atomic_write_json(d / "DONE", {"time": now_iso(), "headline": {k: v["privleak"] for k, v in res.items()}})
    del m
    gc.collect()
    if not a.keep:
        for p in (m_target, m_retrain):
            shutil.rmtree(p, ignore_errors=True)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpora", nargs="*", default=["news", "books"])
    ap.add_argument("--inspect", action="store_true")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--bs", type=int, default=BS)
    ap.add_argument("--accum", type=int, default=ACCUM)
    ap.add_argument("--chunk", type=int, default=CHUNK)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--n-calib", type=int, default=400)
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args(argv)
    if a.inspect:
        for c in a.corpora:
            inspect(c)
        return 0
    jc.require_gpu(40)
    tok = jc.load_tok()
    tok.padding_side = "right"
    if jc.TINY:
        a.chunk, a.n_calib = 64, 2
    for c in a.corpora:
        run_corpus(c, a, tok)
    rows = {p.parent.name: jc.read_json(p)["conditions"] for p in sorted((paths.runs_dir() / "A5-muse").glob("*/metrics.json"))}
    jc.summary(NAME, {"corpora": a.corpora, "results": {k: {c: {m: v.get(m) for m in ("verbmem_forget", "knowmem_forget", "knowmem_retain", "privleak")}
                                                           for c, v in r.items()} for k, r in rows.items()}})
    jc.log(NAME, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
