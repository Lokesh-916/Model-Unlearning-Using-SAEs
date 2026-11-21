"""P3 docs task (GPU): per-DOCUMENT layer-8 16k SAE token-fire counts on E4a's forget and benign-biology documents.

Same corpora, truncation, tokenisation and right-padded batches as E4a's docs task (experiments.E4a.docs, unchanged),
model run to layer 8 only (as P4 docs65). E4a cached only per-chunk sums, which cannot be sub-sampled by document.
Per chunk file $DSG_CACHE/P3/<exp>/docs/<corpus>/chunk_<k>.npz: counts [n, 16384] uint8 (<= 255 non-BOS tokens per
document), ntok [n]. STOP check per chunk: the summed counts and token counts equal E4a's cached L08 chunk exactly.
Item-level resume: a chunk with a matching fingerprint is reused. No text stored or printed.
"""
import io
import time

import numpy as np
import torch

from dsgx.util import atomic_write_bytes, atomic_write_json

from experiments.E4a.common import CHUNK_DOCS, DOC_BATCH, DOC_CHARS, DOC_TOKENS, SAES, load_corpus, sha
from experiments.E4a.common import work_dir as e4a_dir
from experiments.E4a.docs import corpus_fp, tokenize
from experiments.P3.common import E4A_EXP, LAYER, RELEASE, SAE_ID, SUB_CORPORA, work_dir

L08_IDX = [i for i, (t, _, _) in enumerate(SAES) if t == "L08"][0]


@torch.no_grad()
def encode(model, sae, toks):
    hook = f"blocks.{LAYER}.hook_resid_post"
    counts = np.zeros((len(toks), sae.W_dec.shape[0]), np.uint8)
    ntok = np.zeros(len(toks), np.int32)
    pad = model.tokenizer.pad_token_id or 0
    for s in range(0, len(toks), DOC_BATCH):
        batch = toks[s:s + DOC_BATCH]
        L = max(len(x) for x in batch)
        x = torch.full((len(batch), L), pad, dtype=torch.long)
        for j, ids in enumerate(batch):
            x[j, :len(ids)] = torch.tensor(ids)
        _, cache = model.run_with_cache(x.to(model.cfg.device), stop_at_layer=LAYER + 1, names_filter=lambda nm: nm == hook)
        a = sae.encode(cache[hook])
        for j, ids in enumerate(batch):
            nr = len(ids)
            ntok[s + j] = nr - 1
            c = (a[j, 1:nr] > 0).sum(0).cpu().numpy()
            assert c.max() <= 255
            counts[s + j] = c.astype(np.uint8)
        del cache, a
    return {"counts": counts, "ntok": ntok}


def load_docs(exp_id, corpus):
    cs, nt = [], []
    for f in sorted(work_dir(exp_id, "docs", corpus).glob("chunk_*.npz")):
        with np.load(f) as p:
            cs.append(p["counts"])
            nt.append(p["ntok"])
    return np.concatenate(cs), np.concatenate(nt)


def task(ctx):
    from dsgx.models.loader import get_sae

    limit = ctx.args.get("limit")
    b = ctx.bundle()
    m = b.model
    sae = get_sae(RELEASE, SAE_ID, b.dtype, b.device)
    assert sae.cfg.metadata.hook_name == f"blocks.{LAYER}.hook_resid_post"
    corpora = {c: load_corpus(c) for c in SUB_CORPORA}
    if limit:
        corpora = {c: v[: int(limit)] for c, v in corpora.items()}
    fps = {c: corpus_fp(v) for c, v in corpora.items()}
    ctx.write_config(corpora=fps, doc_tokens=DOC_TOKENS, sae=SAE_ID)
    ctx.progress.update(items_total=sum(len(v) for v in corpora.values()), items_done=0, force=True)
    t0, new, check = time.time(), 0, {}
    for c, docs in corpora.items():
        d = work_dir(ctx.exp_id, "docs", c)
        n_chunks = (len(docs) + CHUNK_DOCS - 1) // CHUNK_DOCS
        n_eq = 0
        for k in range(n_chunks):
            s, e = k * CHUNK_DOCS, min((k + 1) * CHUNK_DOCS, len(docs))
            f = d / f"chunk_{k:05d}.npz"
            fp = sha([fps[c]["sha256"], s, e, DOC_TOKENS, DOC_CHARS, SAE_ID, "per-doc"])
            out = None
            if f.exists():
                try:
                    with np.load(f) as p:
                        if str(p["fp"]) == fp:
                            out = {"counts": p["counts"], "ntok": p["ntok"]}
                except Exception:
                    out = None
            if out is None:
                out = encode(m, sae, tokenize(m, docs[s:e]))
                buf = io.BytesIO()
                np.savez_compressed(buf, fp=np.array(fp), **out)
                atomic_write_bytes(f, buf.getvalue())
                new += e - s
            if not limit:
                with np.load(e4a_dir(E4A_EXP, "docs", c) / f"chunk_{k:05d}.npz") as p:
                    ref, rn = p["fire"][L08_IDX], p["ntok"]
                tot = out["counts"].sum(0, dtype=np.int64)
                if not (np.array_equal(tot, ref) and np.array_equal(out["ntok"], rn)):
                    raise RuntimeError(f"STOP: {c} chunk {k}: per-document L08 counts do not sum to E4a's chunk counts "
                                       f"(L1 {int(np.abs(tot - ref).sum())})")
                n_eq += 1
            ctx.progress.advance(e - s)
            if k % 10 == 0:
                print(f"[P3 docs] {c} chunk {k + 1}/{n_chunks}, {new / max(1e-9, time.time() - t0):.1f} new docs/s", flush=True)
        check[c] = {"n_docs": len(docs), "n_chunks": n_chunks, "chunks_equal_e4a": n_eq}
    m_out = {"check_vs_e4a": check, "docs_per_s_new": new / max(1e-9, time.time() - t0),
             "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
    atomic_write_json(work_dir(ctx.exp_id) / "docs_meta.json", m_out)
    ctx.write_metrics(m_out)
    ctx.finish({"view": "p3-docs", "forget": None})
    return m_out
