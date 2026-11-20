"""P4 docs65 task (GPU): layer-8 token-fire counts of the Gemma Scope 65k canonical SAE on E4a's documents.

Same corpora, truncation, tokenisation and right-padded batches as E4a's docs task (experiments.E4a.docs, unchanged),
model run to layer 8 only. Also counts the 16k L08 SAE in the same pass as a check: the summed counts must equal
E4a's cached L08 counts exactly (else the task fails before anything is fitted).
Per chunk file $DSG_CACHE/P4/<exp>/docs65/<corpus>/chunk_<k>.npz: fire65 [65536] int64, fire16 [16384] int64,
ntok [n] (item-level resume: a chunk with a matching fingerprint is reused). No text stored or printed.
"""
import io
import time

import numpy as np
import torch

from dsgx.util import atomic_write_bytes, atomic_write_json

from experiments.E4a.common import CHUNK_DOCS, CORPORA, DOC_BATCH, DOC_CHARS, DOC_TOKENS, SAES, load_corpus, sha
from experiments.E4a.common import work_dir as e4a_dir
from experiments.E4a.docs import corpus_fp, tokenize
from experiments.P4.common import E4A_EXP, RELEASE, SAE16, SAE65, work_dir

L08_IDX = [i for i, (t, _, _) in enumerate(SAES) if t == "L08"][0]


@torch.no_grad()
def encode(model, sae16, sae65, toks):
    hook = "blocks.8.hook_resid_post"
    f16 = np.zeros(sae16.W_dec.shape[0], np.int64)
    f65 = np.zeros(sae65.W_dec.shape[0], np.int64)
    ntok = np.zeros(len(toks), np.int32)
    pad = model.tokenizer.pad_token_id or 0
    for s in range(0, len(toks), DOC_BATCH):
        batch = toks[s:s + DOC_BATCH]
        L = max(len(x) for x in batch)
        x = torch.full((len(batch), L), pad, dtype=torch.long)
        for j, ids in enumerate(batch):
            x[j, :len(ids)] = torch.tensor(ids)
        _, cache = model.run_with_cache(x.to(model.cfg.device), stop_at_layer=9, names_filter=lambda nm: nm == hook)
        r = cache[hook]
        a16 = sae16.encode(r)
        a65 = sae65.encode(r)
        for j, ids in enumerate(batch):
            nr = len(ids)
            ntok[s + j] = nr - 1
            f16 += (a16[j, 1:nr] > 0).sum(0).cpu().numpy().astype(np.int64)
            f65 += (a65[j, 1:nr] > 0).sum(0).cpu().numpy().astype(np.int64)
        del cache, a16, a65
    return {"fire16": f16, "fire65": f65, "ntok": ntok}


def e4a_l08_counts(corpus):
    files = sorted(e4a_dir(E4A_EXP, "docs", corpus).glob("chunk_*.npz"))
    tot, nt = None, 0
    for f in files:
        with np.load(f) as p:
            fr = p["fire"][L08_IDX]
            tot = fr.copy() if tot is None else tot + fr
            nt += int(p["ntok"].sum())
    return tot, nt


def load_counts(exp_id, corpus):
    files = sorted(work_dir(exp_id, "docs65", corpus).glob("chunk_*.npz"))
    f16 = f65 = None
    nt = 0
    for f in files:
        with np.load(f) as p:
            f16 = p["fire16"].copy() if f16 is None else f16 + p["fire16"]
            f65 = p["fire65"].copy() if f65 is None else f65 + p["fire65"]
            nt += int(p["ntok"].sum())
    return {"fire16": f16, "fire65": f65, "ntok": nt}


def task(ctx):
    from dsgx.models.loader import get_sae

    limit = ctx.args.get("limit")
    b = ctx.bundle()
    m = b.model
    sae16 = get_sae(RELEASE, SAE16["L08"][1], b.dtype, b.device)
    sae65 = get_sae(RELEASE, SAE65[2], b.dtype, b.device)
    for s in (sae16, sae65):
        assert s.cfg.metadata.hook_name == "blocks.8.hook_resid_post", s.cfg.metadata.hook_name
    corpora = {c: load_corpus(c) for c in CORPORA}
    if limit:
        corpora = {c: v[: int(limit)] for c, v in corpora.items()}
    fps = {c: corpus_fp(v) for c, v in corpora.items()}
    ctx.write_config(corpora=fps, doc_tokens=DOC_TOKENS, sae16=SAE16["L08"][1], sae65=SAE65[2])
    ctx.progress.update(items_total=sum(len(v) for v in corpora.values()), items_done=0, force=True)
    t0, new = time.time(), 0
    for c, docs in corpora.items():
        d = work_dir(ctx.exp_id, "docs65", c)
        n_chunks = (len(docs) + CHUNK_DOCS - 1) // CHUNK_DOCS
        for k in range(n_chunks):
            s, e = k * CHUNK_DOCS, min((k + 1) * CHUNK_DOCS, len(docs))
            f = d / f"chunk_{k:05d}.npz"
            fp = sha([fps[c]["sha256"], s, e, DOC_TOKENS, DOC_CHARS, SAE16["L08"][1], SAE65[2]])
            if f.exists():
                try:
                    if str(np.load(f)["fp"]) == fp:
                        ctx.progress.advance(e - s)
                        continue
                except Exception:
                    pass
            out = encode(m, sae16, sae65, tokenize(m, docs[s:e]))
            buf = io.BytesIO()
            np.savez(buf, fp=np.array(fp), **out)
            atomic_write_bytes(f, buf.getvalue())
            ctx.progress.advance(e - s)
            new += e - s
            if k % 10 == 0:
                print(f"[P4 docs65] {c} chunk {k + 1}/{n_chunks}, {new / (time.time() - t0):.1f} docs/s", flush=True)
    check = {}
    for c in corpora:
        mine = load_counts(ctx.exp_id, c)
        if limit:
            check[c] = {"skipped": "smoke (limit): E4a counts cover the full corpora"}
            continue
        ref, nt = e4a_l08_counts(c)
        diff = int(np.abs(mine["fire16"] - ref).sum())
        check[c] = {"ntok": mine["ntok"], "ntok_e4a": nt, "l1_diff_l08_16k": diff}
        if diff or nt != mine["ntok"]:
            raise RuntimeError(f"STOP: {c}: L08 16k fire counts differ from E4a (L1 {diff}, ntok {mine['ntok']} vs {nt})")
    m_out = {"check_vs_e4a": check, "docs_per_s_new": new / max(1e-9, time.time() - t0),
             "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else None}
    atomic_write_json(work_dir(ctx.exp_id) / "docs65_meta.json", m_out)
    ctx.write_metrics(m_out)
    ctx.finish({"view": "p4-docs65", "forget": None})
    return m_out
