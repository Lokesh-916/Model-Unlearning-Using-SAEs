"""E2d train task (GPU): the response ladder R1-R5 as fp64 weight deltas of mlp.down_proj in layers 1-3.

R1 = RMU v2 c14 as trained on gpuws: its delta is read from the lab copy of the weights (every other tensor is
checked bit-identical to the base checkpoint). R2-R5 are trained here with the c14 recipe (port of
cluster/rmu_v2_train.py, itself the official WMDP rmu/unlearn.py port): frozen + updated copy, MSE of the layer-3
output to the fixed control vector u on forget docs, alpha * MSE to the frozen layer-3 output on retain docs,
gradients only on mlp.down_proj of layers 1-3, AdamW lr 5e-5, left padding, truncation 512, docs > 50 chars,
consecutive forget batches (never repeated), retain batches cycled in order. u = c14's saved control direction
rescaled to the response's steering coefficient.

Lab-PC memory: both copies are truncated after layer 3 (the loss only reads layer 3's output and the forward is
stopped there, as in the original LayerTap, so this changes nothing numerically), and a 48-doc step is split into
micro-batches padded to the full batch's length with each micro loss weighted by its share of the batch, so the
gradient equals the full-batch gradient. Peak GPU memory is recorded per response.

The forget corpus is read only as training input: never printed, logged or stored (counts and hashes only).
Item/step-level resume: trainer state every 50 steps under $DSG_CACHE/models/<exp>/<R>/last.
"""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from dsgx.util import atomic_write_json, gpu_info, now_iso, sha256_file

from experiments.E2d.common import LADDER, c14_control_vec, c14_weights, response_dir

HF = "google/gemma-2-2b-it"
CKPT_EVERY = 50


class _Stop(Exception):
    pass


class LayerTap:
    """Forward hook on a decoder layer: stores its output and stops the forward (as cluster/rmu_train.py)."""

    def __init__(self, layer):
        self.out = None
        self.h = layer.register_forward_hook(self)

    def __call__(self, module, inputs, output):
        self.out = output[0] if isinstance(output, tuple) else output
        raise _Stop

    def run(self, model, ids, mask):
        try:
            model(input_ids=ids, attention_mask=mask, use_cache=False)
        except _Stop:
            pass
        return self.out


def key(i: int) -> str:
    return f"model.layers.{i}.mlp.down_proj.weight"


def snapshot() -> Path:
    from huggingface_hub import try_to_load_from_cache

    p = try_to_load_from_cache(HF, "model.safetensors.index.json")
    if not isinstance(p, str):
        raise FileNotFoundError(f"{HF} checkpoint index not in the local HF cache")
    return Path(p).parent


def base_tensors(keys) -> dict:
    """bf16 base tensors for the given HF keys, read from the local checkpoint shards."""
    from safetensors import safe_open

    snap = snapshot()
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    out = {}
    for k in keys:
        with safe_open(str(snap / idx[k]), "pt") as f:
            out[k] = f.get_tensor(k)
    return out


def _sha_tensors(d: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(d):
        h.update(k.encode())
        h.update(d[k].contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def save_delta(rid: str, exp_id: str, new: dict, extra: dict) -> dict:
    """new: {key: bf16 tensor (CPU)}. Writes delta.safetensors (fp64) + meta.json; checks exact reconstruction."""
    from safetensors.torch import save_file

    base = base_tensors(list(new))
    delta = {k: (new[k].double() - base[k].double()).contiguous() for k in new}  # exact for any two bf16 values
    exact = all(torch.equal((base[k].double() + delta[k]).to(torch.bfloat16), new[k].to(torch.bfloat16)) for k in new)
    if not exact:
        raise RuntimeError(f"{rid}: bf16(base + delta) != trained weights")
    d = response_dir(exp_id, rid)
    tmp = d / "delta.safetensors.tmp"
    save_file(delta, str(tmp))
    tmp.replace(d / "delta.safetensors")
    per = {k: {"fro": float(delta[k].norm()), "rel_fro": float(delta[k].norm() / base[k].double().norm()),
               "max_abs": float(delta[k].abs().max()), "frac_changed": float((delta[k] != 0).float().mean())}
           for k in sorted(delta)}
    meta = {"rid": rid, "keys": sorted(delta), "dtype": "float64", "exact_bf16_reconstruction": exact,
            "base_sha256": _sha_tensors(base), "new_sha256": _sha_tensors({k: new[k].to(torch.bfloat16) for k in new}),
            "delta_file_sha256": sha256_file(d / "delta.safetensors"),
            "delta_bytes": (d / "delta.safetensors").stat().st_size, "delta_stats": per, "time": now_iso(), **extra}
    atomic_write_json(d / "meta.json", meta)
    return meta


def extract_r1(exp_id: str, log) -> dict:
    """R1: delta of the lab copy of RMU v2 c14; every other tensor must equal the base checkpoint."""
    from safetensors import safe_open

    w = c14_weights() / "model.safetensors"
    sel = json.loads((c14_weights() / "SELECTED.json").read_text())
    snap = snapshot()
    idx = json.loads((snap / "model.safetensors.index.json").read_text())["weight_map"]
    changed, same, missing = [], 0, []
    handles = {}
    with safe_open(str(w), "pt") as f:
        for k in f.keys():
            if k not in idx:
                missing.append(k)
                continue
            if idx[k] not in handles:
                handles[idx[k]] = safe_open(str(snap / idx[k]), "pt")
            if torch.equal(f.get_tensor(k), handles[idx[k]].get_tensor(k)):
                same += 1
            else:
                changed.append(k)
        expect = sorted(key(i) for i in sel["hp"]["layer_ids"])
        if sorted(changed) != expect:
            raise RuntimeError(f"R1: changed tensors {sorted(changed)} != expected {expect}")
        new = {k: f.get_tensor(k) for k in changed}
    log(f"R1: {len(changed)} tensors differ from base ({', '.join(sorted(changed))}), {same} identical, "
        f"{len(missing)} not in the base index")
    return save_delta("R1", exp_id, new, {
        "hp": {**LADDER["R1"], **{k: sel["hp"][k] for k in ("steering_coeff", "alpha", "layer_id", "layer_ids",
                                                             "batch_size", "max_num_batches", "lr")}},
        "source": "reuse: lab copy of gpuws RMU v2 c14 weights", "hardware_trained": "gpuws",
        "source_weights_sha256": sha256_file(w), "selected_dev": sel.get("dev"),
        "n_identical_tensors": same, "n_unindexed": len(missing), "peak_gpu_gb_train": sel["dev"].get("peak_vram_gb_train"),
        "peak_gpu_note": "gpuws, full (untruncated) models, bs 48 in one pass (SELECTED.json)"})


# ------------------------------------------------------------------------------------------ data
def make_batches(docs, bs, n, cycle):
    if not cycle:
        assert bs * n <= len(docs), f"need {bs * n} docs, have {len(docs)}"
        return [docs[i * bs:(i + 1) * bs] for i in range(n)]
    return [[docs[(i * bs + j) % len(docs)] for j in range(bs)] for i in range(n)]


def benign_bio_retain(exp_id: str, log) -> tuple[list[str], dict]:
    """B1's benign_bio filler pool minus every doc sharing any word 8-gram with any WMDP-Bio / MMLU item;
    then a leakage-checker style containment scan (fail >= 0.8, warn >= 0.5) of the final retain stream."""
    from dsgx.attacks.transforms import _filler_texts
    from dsgx.checks.leakage import _ngrams, _words, positive_control
    from dsgx.data.corpora import load_retain_docs
    from dsgx.data.mcq import MMLU_SUBJECTS, load_mcq

    pool = list(_filler_texts("benign_bio"))
    eval_ds = ["wmdp-bio"] + list(MMLU_SUBJECTS)
    items = [it for d in eval_ds for it in load_mcq(d)]
    item_grams = [_ngrams(_words(it.question + " " + " ".join(it.choices))) for it in items]
    allg = set().union(*item_grams)
    keep, dropped = [], 0
    for t in pool:
        if _ngrams(_words(t)) & allg:
            dropped += 1
        else:
            keep.append(t)
    wiki = load_retain_docs("wikitext", min_len=50)
    stream_grams = set()
    for t in keep + wiki:
        stream_grams |= _ngrams(_words(t))
    fails, warns = 0, 0
    for g in item_grams:
        if len(g) < 3:
            continue
        frac = len(g & stream_grams) / len(g)
        fails += frac >= 0.8
        warns += 0.5 <= frac < 0.8
    pc = positive_control(stream_grams, keep[0] if keep else "")
    rep = {"pool": len(pool), "dropped_any_8gram_overlap": dropped, "kept": len(keep),
           "kept_sha256": hashlib.sha256("\n\x00".join(keep).encode()).hexdigest(),
           "eval_items_scanned": len(items), "eval_datasets": len(eval_ds),
           "stream": "kept benign-bio + wikitext-2 test (> 50 chars)", "n_fail": int(fails), "n_warn": int(warns),
           "positive_control": pc, "pass": bool(fails == 0 and pc.get("pass")), "time": now_iso()}
    atomic_write_json(response_dir(exp_id, "R5") / "retain_leakage.json", rep)
    log(f"R5 retain: pool {len(pool)}, dropped {dropped} (8-gram overlap), kept {len(keep)}; scan fail {fails} "
        f"warn {warns}; positive control {pc.get('detected')}/{pc.get('n')}")
    if not rep["pass"]:
        raise RuntimeError(f"R5 retain leakage check failed: {rep}")
    n = max(len(wiki), len(keep))
    docs = [x for i in range(n) for x in (wiki[i % len(wiki)], keep[i % len(keep)])]
    return docs, rep


# ------------------------------------------------------------------------------------------ training
def _load(L: int, device):
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, attn_implementation="eager")
    m.model.layers = m.model.layers[:L + 1]
    m.config.num_hidden_layers = L + 1
    if getattr(m.config, "layer_types", None):
        m.config.layer_types = m.config.layer_types[:L + 1]
    for p in m.parameters():
        p.requires_grad_(False)
    return m.to(device)


def check_truncation(docs, L: int = 3, n: int = 4) -> dict:
    """Layer-L output of the truncated model == the full model's (stopped forward), on n benign retain docs."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(HF)
    tok.padding_side = "left"
    enc = tok(docs[:n], return_tensors="pt", padding=True, truncation=True, max_length=128)
    full = AutoModelForCausalLM.from_pretrained(HF, dtype=torch.bfloat16, attn_implementation="eager").to("cuda")
    tap = LayerTap(full.model.layers[L])
    with torch.no_grad():
        a = tap.run(full, enc["input_ids"].cuda(), enc["attention_mask"].cuda()).clone()
    tap.h.remove()
    del full
    torch.cuda.empty_cache()
    tr = _load(L, "cuda")
    tap = LayerTap(tr.model.layers[L])
    with torch.no_grad():
        b = tap.run(tr, enc["input_ids"].cuda(), enc["attention_mask"].cuda()).clone()
    tap.h.remove()
    del tr
    torch.cuda.empty_cache()
    return {"equal": bool(torch.equal(a, b)), "max_abs_diff": float((a.float() - b.float()).abs().max()), "n_docs": n}


def train(rid: str, exp_id: str, hp: dict, forget_docs, retain_docs, log, progress=None, micro: int = 12) -> dict:
    from transformers import AutoTokenizer

    d = response_dir(exp_id, rid)
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
    n, bs = hp["max_num_batches"], hp["batch_size"]
    fb = make_batches(forget_docs, bs, n, cycle=False)
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
        fenc = tok(fb[step], return_tensors="pt", padding=True, truncation=True, max_length=hp["max_len"])
        renc = tok(rb[step], return_tensors="pt", padding=True, truncation=True, max_length=hp["max_len"])
        try:
            opt.zero_grad(set_to_none=True)
            ul = rl = 0.0
            hf_norm = []
            for s0 in range(0, bs, micro):
                w = min(micro, bs - s0) / bs
                ids, msk = fenc["input_ids"][s0:s0 + micro].to(dev), fenc["attention_mask"][s0:s0 + micro].to(dev)
                h_f = utap.run(updated, ids, msk)
                lu = torch.nn.functional.mse_loss(h_f, u.expand_as(h_f)) * w
                lu.backward()
                ul += float(lu)
                hf_norm.append(float(h_f.detach().float().norm(dim=-1).mean()))
                del h_f, lu
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
        row = {"step": step, "loss": ul + rl, "unlearn_loss": ul, "retain_loss": rl, "grad_norm": gn,
               "time": time.time() - t0, "micro": micro, "h_forget_norm": float(np.mean(hf_norm)),
               "peak_alloc_gb": torch.cuda.max_memory_allocated() / 1e9}
        tlog.append(row)
        if progress:
            progress(step)
        if step % 10 == 0:
            log(f"{rid} step {step}/{n}: unlearn {ul:.4g} retain {rl:.4g} gn {gn:.3g} micro {micro}")
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


def task(ctx):
    import pandas as pd

    from dsgx.data.corpora import load_forget_docs, load_retain_docs
    from dsgx import paths

    def log(msg):
        print(f"[E2d-train] {msg}", flush=True)

    exp = ctx.exp_id
    rids = ctx.args.get("responses", list(LADDER))
    steps_override = ctx.args.get("max_num_batches")  # smoke only
    total = sum(int(steps_override or LADDER[r]["max_num_batches"]) for r in rids if r != "R1")
    ctx.progress.update(items_total=total, items_done=0, force=True)
    done_steps = 0
    out = {}
    forget_docs = retain_wiki = None
    for rid in rids:
        rd = ctx.run_dir(rid)
        if (rd / "DONE").exists() and (response_dir(exp, rid) / "delta.safetensors").exists():
            out[rid] = json.loads((rd / "metrics.json").read_text())
            done_steps += 0 if rid == "R1" else int(steps_override or LADDER[rid]["max_num_batches"])
            continue
        ctx.write_config(name=rid, response=rid, hp=LADDER[rid])
        if rid == "R1":
            meta = extract_r1(exp, log)
        else:
            hp = dict(LADDER[rid])
            if steps_override:
                hp["max_num_batches"] = int(steps_override)
            if forget_docs is None:
                forget_docs = load_forget_docs("bio-forget-corpus", min_len=hp["min_len"])
                retain_wiki = load_retain_docs("wikitext", min_len=hp["min_len"])
                trunc = check_truncation(retain_wiki)
                atomic_write_json(ctx.run_dir() / "truncation_check.json", trunc)
                log(f"truncation check: {trunc}")
                if not trunc["equal"]:
                    raise RuntimeError(f"truncated layer-3 output differs from the full model: {trunc}")
            leak = None
            if hp["retain"] == "wikitext":
                retain = retain_wiki
            else:
                retain, leak = benign_bio_retain(exp, log)
            t0 = time.time()
            base_done = done_steps
            r = train(rid, exp, hp, forget_docs, retain, log,
                      progress=lambda s: ctx.progress.update(items_done=base_done + s))
            pd.DataFrame(r["log"]).to_parquet(rd / "train_log.parquet", index=False)
            meta = save_delta(rid, exp, r["new"], {
                "hp": hp, "source": "trained on the lab PC", "hardware_trained": "labpc", "gpu": gpu_info(),
                "train_seconds_last_attempt": round(time.time() - t0, 1), **r["peak"],
                "micro_batch_final": r["micro_final"], "final": {k: r["log"][-1][k] for k in
                                                                ("unlearn_loss", "retain_loss", "grad_norm")},
                "data": {"forget_corpus_sha256": sha256_file(paths.forget_corpus_jsonl()),
                         "n_forget_docs": len(forget_docs), "forget_docs_used": hp["batch_size"] * hp["max_num_batches"],
                         "n_retain_docs": len(retain), "retain": hp["retain"], "retain_leakage": leak},
                "u": "c14 control_vec.npy direction rescaled to steering_coeff"})
            import shutil

            shutil.rmtree(response_dir(exp, rid) / "last", ignore_errors=True)
            done_steps += hp["max_num_batches"]
        m = {k: v for k, v in meta.items() if k not in ("delta_stats",)}
        m["delta_stats"] = meta["delta_stats"]
        ctx.write_metrics(m, name=rid)
        ctx.finish({"view": "e2d-response", "response": rid,
                    "peak_gpu_gb": meta.get("peak_gpu_allocated_gb") or meta.get("peak_gpu_gb_train")}, name=rid)
        out[rid] = m
    ctx.write_metrics({"responses": {r: {k: out[r].get(k) for k in ("delta_file_sha256", "delta_bytes",
                                                                     "peak_gpu_allocated_gb", "peak_gpu_reserved_gb",
                                                                     "peak_gpu_gb_train")} for r in out}})
    ctx.finish({"view": "e2d-train", "responses": list(out)})
    return {r: out[r].get("delta_file_sha256") for r in out}
