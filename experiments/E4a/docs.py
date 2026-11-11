"""E4a docs task (GPU): residual summaries + SAE fire counts per document, layers 3-14.

Per corpus (forget / benign / wiki), documents in order, CHUNK_DOCS per resumable chunk file
$DSG_CACHE/E4a/<exp>/docs/<corpus>/chunk_<k>.npz (atomic; a chunk whose fingerprint matches is reused):
  mean  [n, 12, d] fp16   mean residual (resid_post) over non-BOS positions, layers 3..14
  tok   [n, T, 12, d] fp16 residuals at TOK_PER_DOC sampled non-BOS positions (seeded per document; -1 = none)
  ntok  [n]               non-BOS tokens
  fire  [13, F] int64     per-SAE token-fire counts summed over the chunk's non-BOS tokens
No text is stored, logged or printed: corpora are fingerprinted by count + sha256 only.
"""
import hashlib
import io
import time

import numpy as np
import torch

from dsgx.util import atomic_write_bytes, atomic_write_json

from experiments.E4a.common import (CHUNK_DOCS, CORPORA, DOC_BATCH, DOC_CHARS, DOC_TOKENS, LAYERS, RELEASE, SAES,
                                    TOK_PER_DOC, load_corpus, sha, work_dir)


def corpus_fp(docs) -> dict:
    h = hashlib.sha256()
    for t in docs:
        h.update(t.encode())
        h.update(b"\n\x00")
    return {"n": len(docs), "sha256": h.hexdigest()}


def tokenize(model, docs):
    tok = model.tokenizer
    out = []
    for t in docs:
        ids = tok(t[:DOC_CHARS], add_special_tokens=False)["input_ids"][: DOC_TOKENS - 1]
        out.append([tok.bos_token_id] + ids)
    return out


def _sample_pos(n_real: int, key: int) -> np.ndarray:
    rng = np.random.default_rng(1_000_003 * 7 + key)
    pos = np.arange(1, n_real)
    pick = np.sort(rng.choice(pos, size=min(TOK_PER_DOC, len(pos)), replace=False)) if len(pos) else np.array([], int)
    return np.concatenate([pick, -np.ones(TOK_PER_DOC - len(pick), int)]).astype(np.int16)


@torch.no_grad()
def encode_chunk(model, saes, toks, base_index: int):
    names = [f"blocks.{l}.hook_resid_post" for l in LAYERS]
    d = model.cfg.d_model
    n = len(toks)
    mean = np.zeros((n, len(LAYERS), d), np.float16)
    tok = np.zeros((n, TOK_PER_DOC, len(LAYERS), d), np.float16)
    tpos = np.zeros((n, TOK_PER_DOC), np.int16)
    ntok = np.zeros(n, np.int32)
    fire = np.zeros((len(saes), saes[0][2].W_dec.shape[0]), np.int64)
    pad = model.tokenizer.pad_token_id or 0
    for s in range(0, n, DOC_BATCH):
        batch = toks[s:s + DOC_BATCH]
        L = max(len(x) for x in batch)
        x = torch.full((len(batch), L), pad, dtype=torch.long)
        for j, ids in enumerate(batch):
            x[j, :len(ids)] = torch.tensor(ids)
        _, cache = model.run_with_cache(x.to(model.cfg.device), stop_at_layer=max(LAYERS) + 1,
                                        names_filter=lambda nm: nm in names)
        lens = [len(ids) for ids in batch]
        for j, nr in enumerate(lens):
            i = s + j
            ntok[i] = nr - 1
            tpos[i] = _sample_pos(nr, base_index + i)
            valid = tpos[i][tpos[i] >= 0].astype(int)
            for li, nm in enumerate(names):
                r = cache[nm][j, 1:nr].float()
                mean[i, li] = r.mean(0).cpu().numpy() if nr > 1 else 0
                if len(valid):
                    tok[i, :len(valid), li] = cache[nm][j, valid].float().cpu().numpy()
        for k, (_, layer, sae) in enumerate(saes):
            r = cache[f"blocks.{layer}.hook_resid_post"]
            a = sae.encode(r)
            for j, nr in enumerate(lens):
                fire[k] += (a[j, 1:nr] > 0).sum(0).cpu().numpy().astype(np.int64)
        del cache
    return {"mean": mean, "tok": tok, "tpos": tpos, "ntok": ntok, "fire": fire}


def load_all(exp_id: str, corpus: str) -> dict:
    """Concatenate a corpus' chunks (mean, tok, tpos, ntok; fire summed)."""
    files = sorted(work_dir(exp_id, "docs", corpus).glob("chunk_*.npz"))
    out, fire = {k: [] for k in ("mean", "tok", "tpos", "ntok")}, None
    for f in files:
        with np.load(f) as p:
            for k in out:
                out[k].append(p[k])
            fire = p["fire"].copy() if fire is None else fire + p["fire"]
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["fire"] = fire
    return out


def task(ctx):
    from dsgx.models.loader import get_sae

    a = ctx.args
    limit = a.get("limit")
    b = ctx.bundle()
    m = b.model
    saes = [(tag, layer, get_sae(RELEASE, sid, b.dtype, b.device)) for tag, layer, sid in SAES]
    for tag, layer, sae in saes:
        assert sae.cfg.metadata.hook_name == f"blocks.{layer}.hook_resid_post", (tag, sae.cfg.metadata.hook_name)
    corpora = {c: load_corpus(c) for c in CORPORA}
    if limit:
        corpora = {c: v[: int(limit)] for c, v in corpora.items()}
    fps = {c: corpus_fp(v) for c, v in corpora.items()}
    ctx.write_config(corpora=fps, doc_tokens=DOC_TOKENS, tok_per_doc=TOK_PER_DOC, layers=LAYERS,
                     saes=[(t, l, s) for t, l, s in SAES])
    total = sum(len(v) for v in corpora.values())
    ctx.progress.update(items_total=total, items_done=0, force=True)
    t0, done_new = time.time(), 0
    meta = {}
    for c, docs in corpora.items():
        d = work_dir(ctx.exp_id, "docs", c)
        n_chunks = (len(docs) + CHUNK_DOCS - 1) // CHUNK_DOCS
        reused = 0
        for k in range(n_chunks):
            s, e = k * CHUNK_DOCS, min((k + 1) * CHUNK_DOCS, len(docs))
            f = d / f"chunk_{k:05d}.npz"
            fp = sha([fps[c]["sha256"], s, e, DOC_TOKENS, DOC_CHARS, TOK_PER_DOC, SAES])
            if f.exists():
                try:
                    if str(np.load(f)["fp"]) == fp:
                        reused += 1
                        ctx.progress.advance(e - s)
                        continue
                except Exception:
                    pass
            out = encode_chunk(m, saes, tokenize(m, docs[s:e]), s)
            buf = io.BytesIO()
            np.savez(buf, fp=np.array(fp), **out)
            atomic_write_bytes(f, buf.getvalue())
            ctx.progress.advance(e - s)
            done_new += e - s
            if done_new and k % 10 == 0:
                rate = done_new / (time.time() - t0)
                print(f"[E4a docs] {c} chunk {k + 1}/{n_chunks}, {rate:.1f} docs/s", flush=True)
        meta[c] = {**fps[c], "chunks": n_chunks, "chunks_reused": reused}
    for f in work_dir(ctx.exp_id, "docs").glob("*/chunk_*.npz"):
        k = int(f.stem.split("_")[1])
        c = f.parent.name
        if k >= (len(corpora[c]) + CHUNK_DOCS - 1) // CHUNK_DOCS:
            f.unlink()  # stale chunk from a larger earlier run (e.g. different limit)
    ntoks = {c: int(load_all(ctx.exp_id, c)["ntok"].sum()) for c in corpora}
    m_out = {"corpora": meta, "ntok": ntoks, "docs_per_s_new": done_new / max(1e-9, time.time() - t0),
             "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
    atomic_write_json(work_dir(ctx.exp_id) / "docs_meta.json", m_out)
    ctx.write_metrics(m_out)
    ctx.finish({"view": "e4a-docs", "forget": None, "n_docs": total})
    return m_out
