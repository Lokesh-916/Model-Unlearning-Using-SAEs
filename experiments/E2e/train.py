"""E2e train task (GPU): R6 and R7 as fp64 deltas of mlp.down_proj, layers 1-3 (E2d delta format, loadable by
experiments.E2d.response with response_exp E2e).

Recipe = R5 (experiments/E2d/train.py): steering 2 x c14 along c14's control direction, alpha 300, lr 5e-5, layer 3,
update layers 1-3, batch 48, retain = wikitext + filtered benign-bio (R5's stream, same leakage scan), truncated
models (bit-identical layer-3 output checked), micro-batches padded to the batch length.
Changes (DEVIATIONS 2026-11-11): forget stream = experiments.E2e.augment.build_stream (R6: 1440 docs x 5 forms;
R7: + Hindi forget subset); the forget loss is a per-sequence masked mean of the squared distance to u (mask = forget
tokens; orig / Hindi forms: every real token), averaged over the batch; steps = ceil(sequences / 48) (R6: 150).
The forget corpus and its translations are read only as training input: counts and hashes only.
Step-level resume: trainer state every 50 steps under $DSG_CACHE/models/E2e/<R>/last.
"""
import hashlib
import json
import math
import shutil
import time

import numpy as np
import torch

from dsgx.util import atomic_write_json, gpu_info, now_iso, sha256_file

from experiments.E2d.common import c14_control_vec, response_dir
from experiments.E2d.train import (CKPT_EVERY, HF, LayerTap, _load, benign_bio_retain, check_truncation, key,
                                   make_batches, save_delta)
from experiments.E2e.common import LADDER_E2E


def _pad_left(rows, pad_id):
    T = max(len(r[0]) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(rows), T), dtype=torch.long)
    msk = torch.zeros((len(rows), T), dtype=torch.float32)
    for b, (i, m, _) in enumerate(rows):
        n = len(i)
        ids[b, T - n:] = torch.tensor(i)
        att[b, T - n:] = 1
        msk[b, T - n:] = torch.tensor(m, dtype=torch.float32)
    return ids, att, msk


def train(rid, exp, hp, fseqs, retain_docs, log, progress=None, micro=12):
    from transformers import AutoTokenizer

    d = response_dir(exp, rid)
    dev = "cuda"
    torch.set_grad_enabled(True)
    torch.manual_seed(hp["seed"])
    torch.cuda.manual_seed(hp["seed"])
    tok = AutoTokenizer.from_pretrained(HF)
    tok.padding_side = "left"
    L = hp["layer_id"]
    torch.cuda.reset_peak_memory_stats()
    frozen, updated = _load(L, dev), _load(L, dev)
    frozen.eval()
    params = []
    for i in hp["layer_ids"]:
        p = updated.model.layers[i].mlp.down_proj.weight
        p.requires_grad_(True)
        params.append(p)
    opt = torch.optim.AdamW(params, lr=hp["lr"])
    u0 = torch.tensor(np.load(c14_control_vec()), dtype=torch.float32)
    u = (u0 / u0.norm() * hp["steering_coeff"]).to(dev, updated.dtype).reshape(1, 1, -1)
    ftap, utap = LayerTap(frozen.model.layers[L]), LayerTap(updated.model.layers[L])
    bs = hp["batch_size"]
    n = hp["max_num_batches"]
    fb = make_batches(fseqs, bs, n, cycle=True)
    rb = make_batches(retain_docs, bs, n, cycle=True)
    sp = d / "last" / "state.pt"
    start, tlog = 0, []
    if sp.exists():
        s = torch.load(sp, weights_only=False)
        with torch.no_grad():
            for p, v in zip(params, s["params"]):
                p.copy_(v.to(p.device, p.dtype))
        opt.load_state_dict(s["opt"])
        start, tlog, micro = s["step"], s["log"], s.get("micro", micro)
        log(f"{rid}: resumed at step {start}")
    t0 = time.time() - (tlog[-1]["time"] if tlog else 0.0)
    updated.train()
    step = start
    while step < n:
        fids, fatt, fmsk = _pad_left(fb[step], tok.pad_token_id)
        renc = tok(rb[step], return_tensors="pt", padding=True, truncation=True, max_length=hp["max_len"])
        try:
            opt.zero_grad(set_to_none=True)
            ul = rl = 0.0
            hf_norm = []
            for s0 in range(0, bs, micro):
                w = min(micro, bs - s0) / bs
                ids, att = fids[s0:s0 + micro].to(dev), fatt[s0:s0 + micro].to(dev)
                m = fmsk[s0:s0 + micro].to(dev)
                h_f = utap.run(updated, ids, att)
                err = ((h_f.float() - u.float()) ** 2).mean(-1)               # [B, T]
                per = (err * m).sum(1) / m.sum(1)                              # per-sequence masked mean
                lu = per.mean() * w
                lu.backward()
                ul += float(lu)
                hf_norm.append(float((h_f.detach().float().norm(dim=-1) * m).sum() / m.sum()))
                del h_f, err, per, lu
                ids, msk = renc["input_ids"][s0:s0 + micro].to(dev), renc["attention_mask"][s0:s0 + micro].to(dev)
                h_r = utap.run(updated, ids, msk)
                with torch.no_grad():
                    h_r0 = ftap.run(frozen, ids, msk)
                lr_ = torch.nn.functional.mse_loss(h_r, h_r0) * hp["alpha"] * w
                lr_.backward()
                rl += float(lr_)
                del h_r, h_r0, lr_
        except torch.cuda.OutOfMemoryError:
            opt.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            if micro == 1:
                raise
            micro = max(1, micro // 2)
            log(f"{rid}: OOM at step {step + 1}, micro-batch -> {micro}")
            continue
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params)))
        opt.step()
        step += 1
        tlog.append({"step": step, "loss": ul + rl, "unlearn_loss": ul, "retain_loss": rl, "grad_norm": gn,
                     "time": time.time() - t0, "micro": micro, "h_forget_norm": float(np.mean(hf_norm)),
                     "forget_len_max": int(fids.shape[1]), "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9})
        if progress:
            progress(step)
        if step % 10 == 0:
            log(f"{rid} step {step}/{n}: unlearn {ul:.4g} retain {rl:.4g} gn {gn:.3g} micro {micro} "
                f"({tlog[-1]['time'] / step:.1f} s/step)")
        if step % CKPT_EVERY == 0 or step == n:
            (d / "last").mkdir(exist_ok=True)
            tmp = d / "last" / "state.pt.tmp"
            torch.save({"step": step, "params": [p.detach().cpu() for p in params], "opt": opt.state_dict(),
                        "log": tlog, "micro": micro}, tmp)
            tmp.replace(sp)
    peak = {"peak_gpu_allocated_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3),
            "peak_gpu_reserved_gb": round(torch.cuda.max_memory_reserved() / 1e9, 3)}
    new = {key(i): updated.model.layers[i].mlp.down_proj.weight.detach().to(torch.bfloat16).cpu()
           for i in hp["layer_ids"]}
    ftap.h.remove()
    utap.h.remove()
    del frozen, updated, opt, params
    torch.cuda.empty_cache()
    return {"new": new, "log": tlog, "peak": peak, "micro_final": micro}


def _stream_sha(seqs):
    h = hashlib.sha256()
    for ids, m, f in seqs:
        h.update(f.encode())
        h.update(np.asarray(ids, dtype=np.int64).tobytes())
        h.update(np.asarray(m, dtype=np.int8).tobytes())
    return h.hexdigest()


def task(ctx):
    import pandas as pd
    from transformers import AutoTokenizer

    from dsgx import paths
    from dsgx.data.corpora import load_forget_docs

    from experiments.E2e.augment import build_stream
    from experiments.E2e.translate_forget import load as load_hindi

    def log(msg):
        print(f"[E2e-train] {msg}", flush=True)

    a = ctx.args
    exp = ctx.exp_id  # E2e or E2e-smoke: deltas under $DSG_CACHE/models/<exp>/<R>
    rids = a.get("responses", list(LADDER_E2E))
    tok = AutoTokenizer.from_pretrained(HF)
    docs = retain = leak = None
    streams = {}
    for rid in rids:
        hp = dict(LADDER_E2E[rid])
        if a.get("n_aug_docs"):  # smoke only
            hp["n_aug_docs"] = int(a["n_aug_docs"])
            if "hindi_docs" in hp:
                hp["hindi_docs"] = min(hp["hindi_docs"], int(a["n_aug_docs"]))
        if docs is None:
            docs = load_forget_docs("bio-forget-corpus", min_len=hp["min_len"])
        hindi = load_hindi(hp["hindi_min_chrf"], hp["hindi_docs"], ctx.smoke) if "hindi_docs" in hp else None
        if "hindi_docs" in hp and not hindi:
            raise RuntimeError(f"{rid}: no Hindi forget records (run translate-forget first)")
        seqs, counts = build_stream(tok, docs, hp["n_aug_docs"], hindi, seed=hp["seed"])
        hp["max_num_batches"] = int(a.get("max_num_batches") or math.ceil(len(seqs) / hp["batch_size"]))
        streams[rid] = (hp, seqs, counts)
    total = sum(streams[r][0]["max_num_batches"] for r in rids)
    ctx.progress.update(items_total=total, items_done=0, force=True)
    done = 0
    out = {}
    for rid in rids:
        hp, seqs, counts = streams[rid]
        rd = ctx.run_dir(rid)
        if (rd / "DONE").exists() and (response_dir(exp, rid) / "delta.safetensors").exists():
            out[rid] = json.loads((rd / "metrics.json").read_text())
            done += hp["max_num_batches"]
            continue
        ctx.write_config(name=rid, response=rid, hp=hp)
        if retain is None:
            from dsgx.data.corpora import load_retain_docs

            trunc = check_truncation(load_retain_docs("wikitext", min_len=hp["min_len"]))
            atomic_write_json(ctx.run_dir() / "truncation_check.json", trunc)
            log(f"truncation check: {trunc}")
            if not trunc["equal"]:
                raise RuntimeError(f"truncated layer-3 output differs from the full model: {trunc}")
            retain, leak = benign_bio_retain(exp, log)  # R5's retain stream + leakage scan (written under <exp>/R5)
            src = response_dir(exp, "R5") / "retain_leakage.json"
            shutil.move(str(src), str(response_dir(exp, "_shared") / "retain_leakage.json"))
            shutil.rmtree(response_dir(exp, "R5"), ignore_errors=True)
        lens = np.array([len(s[0]) for s in seqs])
        msum = np.array([sum(s[1]) for s in seqs])
        log(f"{rid}: {len(seqs)} forget sequences {counts}, {hp['max_num_batches']} steps, len median "
            f"{int(np.median(lens))} max {int(lens.max())}, forget tokens median {int(np.median(msum))}")
        t0 = time.time()
        base_done = done
        r = train(rid, exp, hp, seqs, retain, log, progress=lambda s: ctx.progress.update(items_done=base_done + s))
        pd.DataFrame(r["log"]).to_parquet(rd / "train_log.parquet", index=False)
        meta = save_delta(rid, exp, r["new"], {
            "hp": hp, "source": "trained on the lab PC", "hardware_trained": "labpc", "gpu": gpu_info(),
            "train_seconds_last_attempt": round(time.time() - t0, 1), **r["peak"], "micro_batch_final": r["micro_final"],
            "final": {k: r["log"][-1][k] for k in ("unlearn_loss", "retain_loss", "grad_norm")},
            "data": {"forget_corpus_sha256": sha256_file(paths.forget_corpus_jsonl()), "n_forget_docs": len(docs),
                     "forget_docs_used": hp["n_aug_docs"], "forget_sequences": len(seqs), "forms": counts,
                     "forget_stream_sha256": _stream_sha(seqs), "seq_len_median": int(np.median(lens)),
                     "seq_len_max": int(lens.max()), "forget_tokens_median": int(np.median(msum)),
                     "n_retain_docs": len(retain), "retain": hp["retain"], "retain_leakage": leak},
            "u": "c14 control_vec.npy direction rescaled to steering_coeff", "time_done": now_iso()})
        shutil.rmtree(response_dir(exp, rid) / "last", ignore_errors=True)
        done += hp["max_num_batches"]
        m = dict(meta)
        ctx.write_metrics(m, name=rid)
        ctx.finish({"view": "e2d-response", "response": rid, "peak_gpu_gb": meta.get("peak_gpu_allocated_gb")}, name=rid)
        out[rid] = m
    ctx.write_metrics({"responses": {r: {k: out[r].get(k) for k in ("delta_file_sha256", "delta_bytes",
                                                                     "peak_gpu_allocated_gb", "peak_gpu_reserved_gb")}
                                     for r in out}})
    ctx.finish({"view": "e2e-train", "responses": list(out)})
    return {r: out[r].get("delta_file_sha256") for r in out}
