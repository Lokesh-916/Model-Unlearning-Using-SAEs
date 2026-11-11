"""E2e translate-forget task (GPU): Hindi versions of a forget subset for R7. Training input only.

Docs 0..N_HINDI_DOCS-1 of R6's forget docs; each doc's 511-token excerpt (exactly the text R6 trains on) is split at
sentence ends into chunks of <= 60 words, translated en -> hi with B3's NLLB-200-distilled-600M pipeline
(dsgx.data.translate.Translator), back-translated hi -> en, and kept if the doc-level chrF(back-translation, excerpt)
>= HINDI_MIN_CHRF (B3's back-translation filter at its eval threshold). Records go to $DSG_PRIVATE/E2e/forget_hi.jsonl
only ({doc_index, excerpt_sha256, hi, chrf, n_chunks}); appended per batch, so a restart resumes. Logs and metrics hold
counts and chrF summaries only, never text.
"""
import hashlib
import json
import re

import numpy as np

from dsgx import paths
from dsgx.util import now_iso

from experiments.E2e.common import HINDI_MIN_CHRF, N_HINDI_DOCS

MAX_WORDS = 60


def out_file(smoke=False):
    p = paths.private_dir() / "E2e" / ("forget_hi-smoke.jsonl" if smoke else "forget_hi.jsonl")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load(min_chrf=HINDI_MIN_CHRF, limit=None, smoke=False):
    """{doc_index: hindi text} for kept records."""
    out = {}
    p = out_file(smoke)
    if p.exists():
        for line in p.open():
            r = json.loads(line)
            if r["chrf"] >= min_chrf and (limit is None or r["doc_index"] < limit):
                out[r["doc_index"]] = r["hi"]
    return out


def chunks(text):
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]
    out, cur = [], []
    for s in sents:
        w = s.split()
        while len(w) > MAX_WORDS:
            if cur:
                out.append(" ".join(cur))
                cur = []
            out.append(" ".join(w[:MAX_WORDS]))
            w = w[MAX_WORDS:]
        if len(cur) + len(w) > MAX_WORDS and cur:
            out.append(" ".join(cur))
            cur = []
        cur += w
    if cur:
        out.append(" ".join(cur))
    return out


def task(ctx):
    from transformers import AutoTokenizer

    from dsgx.data.corpora import load_forget_docs
    from dsgx.data.translate import LANGS, SRC, Translator, chrf

    from experiments.E2d.train import HF
    from experiments.E2e.augment import excerpt_ids

    a = ctx.args
    n = int(a.get("limit") or N_HINDI_DOCS)
    tok = AutoTokenizer.from_pretrained(HF)
    docs = load_forget_docs("bio-forget-corpus", min_len=50)[:n]
    p = out_file(ctx.smoke)
    have = set()
    if p.exists():
        have = {json.loads(line)["doc_index"] for line in p.open()}
    todo = [i for i in range(n) if i not in have]
    ctx.write_config(n_docs=n, already=len(have), max_words=MAX_WORDS, min_chrf=HINDI_MIN_CHRF)
    ctx.progress.update(items_total=n, items_done=len(have), force=True)
    tr = Translator(batch_size=int(a.get("batch_size", 16)))
    tgt = LANGS["hi"]
    with p.open("a") as f:
        for s in range(0, len(todo), 16):
            idx = todo[s:s + 16]
            ex = [tok.decode(excerpt_ids(tok, docs[i])) for i in idx]
            ch = [chunks(t) for t in ex]
            flat = [c for cs in ch for c in cs]
            hi = tr(flat, SRC, tgt)
            back = tr(hi, tgt, SRC)
            k = 0
            for i, t, cs in zip(idx, ex, ch):
                h, b = hi[k:k + len(cs)], back[k:k + len(cs)]
                k += len(cs)
                rec = {"doc_index": i, "excerpt_sha256": hashlib.sha256(t.encode()).hexdigest(), "hi": " ".join(h),
                       "chrf": chrf(" ".join(b), t), "n_chunks": len(cs)}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            ctx.progress.advance(len(idx))
    recs = [json.loads(line) for line in p.open()]
    recs = [r for r in recs if r["doc_index"] < n]
    c = np.array([r["chrf"] for r in recs])
    kept = int((c >= HINDI_MIN_CHRF).sum())
    m = {"n_docs": n, "records": len(recs), "kept": kept, "min_chrf": HINDI_MIN_CHRF,
         "chrf_quantiles": {q: float(np.quantile(c, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(c) else None,
         "chunks_total": int(sum(r["n_chunks"] for r in recs)), "time": now_iso()}
    print(f"[E2e-translate-forget] {len(recs)} docs translated, {kept} kept at chrF >= {HINDI_MIN_CHRF}", flush=True)
    ctx.write_metrics(m)
    ctx.finish({"view": "translation-cache", "forget": None, "kept": kept})
    return m
