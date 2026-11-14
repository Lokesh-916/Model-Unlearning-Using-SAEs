"""RMU on gemma-2-2b-it (WMDP Bio): official recipe + a small dev grid. Resumable.

Port of the official WMDP RMU loop (Li et al. 2024, github.com/centerforaisafety/wmdp,
rmu/unlearn.py), unchanged except for the model:
  * frozen copy + updated copy of the model (bf16), activations = output of decoder layer
    `layer_id`; only mlp.down_proj (param index 6) of layers `layer_ids` is trained;
  * control vector u = U[0,1)^d / ||.|| * steering_coeff, fixed per run (seed 42);
  * loss = MSE(h_upd(forget), u) + alpha * MSE(h_upd(retain), h_frozen(retain)), all positions;
  * AdamW lr 5e-5, no schedule, no clipping; batch 4, truncation 512 tokens, docs > 50 chars;
    150 batches; forget = bio-forget-corpus (file order), retain = wikitext-2-raw-v1 test.
  * Official Zephyr-7B (32 layers) values: layer 7, update 5,6,7. Gemma-2-2b (26 layers): same.
Only the forward up to layer `layer_id` is computed (the loss does not depend on later layers).

Steering is scale dependent, so the grid is expressed in units of r = median per-token norm of
the frozen layer-7 output on retain text (BOS excluded), measured at the start and logged.
Grid (6 configs): steering in {1, 2, 4} x r, alpha in {300, 1200}.

Selection (DEV only): each config is evaluated with the harness on the dev split (WMDP-Bio dev +
full-MMLU utility dev, bs=16 as allowed for dev sweeps). Among configs whose pooled utility drop
vs base is <= 0.02, pick the lowest WMDP-Bio dev accuracy; if none qualifies, the smallest drop.
Kept on disk: `best` (full HF weights of the current best config) and `last` (trainer state of
the config being trained, checkpointed every 50 steps). Everything else is deleted.

Hazardous text is never printed or logged: only counts, hashes and metrics.
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from dsgx import paths
from dsgx.logging.run_logger import RunLogger
from dsgx.util import atomic_write_json, now_iso, sha256_file

OFFICIAL = {"layer_id": 7, "layer_ids": [5, 6, 7], "param_name": "mlp.down_proj.weight", "lr": 5e-5,
            "batch_size": 4, "max_len": 512, "min_len": 50, "max_num_batches": 150, "seed": 42}
STEER_MULTS = [1.0, 2.0, 4.0]
ALPHAS = [300.0, 1200.0]
MAX_UTIL_DROP = 0.02
CKPT_EVERY = 50
EVAL_EVERY = 50
HF_NAME = "google/gemma-2-2b-it"

MODELS = paths.cache_dir() / "models" / "RMU-cluster"
TMP = paths.cache_dir() / "models" / "RMU-cluster-tmp"
JOBDIR = paths.results_dir() / "jobs" / "rmu"
STATE = JOBDIR / "grid_state.json"


class _Stop(Exception):
    pass


class LayerTap:
    """Forward hook on decoder layer `layer_id`: stores its output; optionally stops the forward."""

    def __init__(self, layer):
        self.out, self.stop = None, False
        self.h = layer.register_forward_hook(self)

    def __call__(self, module, inputs, output):
        self.out = output[0] if isinstance(output, tuple) else output
        if self.stop:
            raise _Stop

    def run(self, model, enc):
        self.stop = True
        try:
            model(**enc)
        except _Stop:
            pass
        finally:
            self.stop = False
        return self.out


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"configs": {}, "best": None, "base_dev": None, "r": None, "started": now_iso()}


def save_state(st):
    atomic_write_json(STATE, st)


def dev_eval(weights, tag, hp=None):
    """Harness dev eval (WMDP-Bio dev + full-MMLU utility dev); returns (run_dir, summary)."""
    from dsgx.models.loader import clear
    from dsgx.run import run

    cfg = {"exp_id": "RMU-cluster-dev", "case": "bio", "split": "dev", "view": "both", "seed": 0,
           "batch_size": 16, "purpose": "tune", "datasets": ["@forget", "@utility"],
           "dataset_label": f"{tag}-wmdp+mmlu48", "method": {"name": "base"},
           "model": {"weights": str(weights)} if weights else {}, "rmu": hp}
    rd = run(cfg)
    clear()
    m = json.loads((rd / "metrics.json").read_text())["views"]["raw"]
    return rd, {"wmdp": m["forget"], "util_pooled": m["utility"]["pooled"],
                "util_unweighted": m["utility"]["unweighted"], "run_dir": str(rd)}


def sel_key(s, base):
    drop = base["util_pooled"]["mean"] - s["util_pooled"]["mean"]
    return (0, s["wmdp"]["mean"]) if drop <= MAX_UTIL_DROP else (1, drop)


def batches(docs, bs, n):
    return [docs[i * bs:(i + 1) * bs] for i in range(n)]


def mini_eval_sets():
    from dsgx.data.mcq import load_mcq, utility_subjects
    from dsgx.data.splits import get_split

    wm = load_mcq("wmdp-bio")
    forget = [wm[i] for i in get_split("wmdp-bio", "dev")[:200]]
    util = []
    for s in utility_subjects("bio"):
        items = load_mcq(s)
        util += [items[i] for i in get_split(s, "dev")[:5]]
    return forget, util


def measure_r(model, tok, tap, retain_docs, n_batches=10):
    norms = []
    with torch.no_grad():
        for b in batches(retain_docs, 4, n_batches):
            enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to(model.device)
            h = tap.run(model, enc).float()
            mask = enc["attention_mask"].bool() & (enc["input_ids"] != tok.bos_token_id)
            norms.append(h.norm(dim=-1)[mask].cpu())
    return float(torch.cat(norms).median())


def train_config(k, hp, forget_b, retain_b, mini, st):
    from dsgx.train.core import decoder_layers, load_hf, mcq_accuracy_hf

    torch.set_grad_enabled(True)  # the harness eval (get_bundle) turns grad off globally
    torch.manual_seed(OFFICIAL["seed"])
    torch.cuda.manual_seed(OFFICIAL["seed"])
    frozen, tok = load_hf(HF_NAME)
    tok.padding_side = "left"  # fixed (mcq_probs_hf also uses left padding)
    updated, _ = load_hf(HF_NAME)
    frozen.eval()
    for p in frozen.parameters():
        p.requires_grad_(False)
    params = []
    for n, p in updated.named_parameters():
        p.requires_grad_(False)
    for lid in hp["layer_ids"]:
        p = decoder_layers(updated)[lid].mlp.down_proj.weight
        p.requires_grad_(True)
        params.append(p)
    opt = torch.optim.AdamW(params, lr=hp["lr"])
    d = updated.config.hidden_size
    u = torch.rand(1, 1, d, dtype=updated.dtype, device=updated.device)
    u = u / torch.norm(u) * hp["steering_coeff"]
    ftap = LayerTap(decoder_layers(frozen)[hp["layer_id"]])
    utap = LayerTap(decoder_layers(updated)[hp["layer_id"]])

    last = MODELS / "last"
    start, log = 0, []
    sp = last / "state.pt"
    if sp.exists():
        s = torch.load(sp, weights_only=False)
        if s.get("cfg") == k:
            with torch.no_grad():
                for p, v in zip(params, s["params"]):
                    p.copy_(v.to(p.device, p.dtype))
            opt.load_state_dict(s["opt"])
            u = s["u"].to(updated.device, updated.dtype)
            start, log = s["step"], s["log"]
            print(f"[rmu] cfg {k}: resumed at step {start}", flush=True)
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time() - (log[-1]["time"] if log else 0.0)
    updated.train()
    for step in range(start, hp["max_num_batches"]):
        fenc = tok(forget_b[step], return_tensors="pt", padding=True, truncation=True,
                   max_length=hp["max_len"]).to(updated.device)
        renc = tok(retain_b[step], return_tensors="pt", padding=True, truncation=True,
                   max_length=hp["max_len"]).to(updated.device)
        h_f = utap.run(updated, fenc)
        unlearn_loss = torch.nn.functional.mse_loss(h_f, u.expand_as(h_f))
        h_r = utap.run(updated, renc)
        with torch.no_grad():
            h_r0 = ftap.run(frozen, renc)
        retain_loss = torch.nn.functional.mse_loss(h_r, h_r0) * hp["alpha"]
        loss = unlearn_loss + retain_loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = float(torch.sqrt(sum((p.grad.float() ** 2).sum() for p in params)))
        opt.step()
        row = {"step": step + 1, "loss": float(loss), "unlearn_loss": float(unlearn_loss),
               "retain_loss": float(retain_loss), "lr": hp["lr"], "grad_norm": gn,
               "time": time.time() - t0, "vram_gb": torch.cuda.max_memory_allocated() / 1e9,
               "h_forget_norm": float(h_f.float().norm(dim=-1).mean())}
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == hp["max_num_batches"]:
            updated.eval()
            with torch.no_grad():
                row["mini_forget_acc"] = float(mcq_accuracy_hf(updated, tok, mini[0], bs=16).mean())
                row["mini_mmlu_acc"] = float(mcq_accuracy_hf(updated, tok, mini[1], bs=16).mean())
            updated.train()
        log.append(row)
        if (step + 1) % 10 == 0:
            print(f"[rmu] cfg {k} step {step + 1}: unlearn {row['unlearn_loss']:.4g} retain {row['retain_loss']:.4g}"
                  + (f" mini_forget {row['mini_forget_acc']:.3f} mini_mmlu {row['mini_mmlu_acc']:.3f}"
                     if "mini_forget_acc" in row else ""), flush=True)
        if (step + 1) % CKPT_EVERY == 0 or step + 1 == hp["max_num_batches"]:
            last.mkdir(parents=True, exist_ok=True)
            tmp = last / "state.pt.tmp"
            torch.save({"cfg": k, "step": step + 1, "params": [p.detach().cpu() for p in params],
                        "opt": opt.state_dict(), "u": u.cpu(), "log": log}, tmp)
            tmp.replace(sp)
    updated.eval()
    out = TMP / f"cfg{k}"
    shutil.rmtree(out, ignore_errors=True)
    updated.save_pretrained(out, safe_serialization=True)
    u_np = u.float().cpu().numpy()
    del frozen, updated, opt, params, ftap, utap
    torch.cuda.empty_cache()
    return out, log, u_np


def main():
    import pandas as pd

    from dsgx.data.corpora import load_forget_docs, load_retain_docs
    from dsgx.train.core import decoder_layers, load_hf

    JOBDIR.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    st = load_state()
    t_job = time.time()

    forget_docs = load_forget_docs("bio-forget-corpus", min_len=OFFICIAL["min_len"])
    retain_docs = load_retain_docs("wikitext", min_len=OFFICIAL["min_len"])
    nb = OFFICIAL["max_num_batches"]
    forget_b = batches(forget_docs, OFFICIAL["batch_size"], nb)
    retain_b = batches(retain_docs, OFFICIAL["batch_size"], nb)
    assert len(forget_b[-1]) == 4 and len(retain_b[-1]) == 4
    data_info = {"forget_corpus_sha256": sha256_file(paths.forget_corpus_jsonl()), "n_forget_docs": len(forget_docs),
                 "n_retain_docs": len(retain_docs), "docs_used_per_corpus": nb * OFFICIAL["batch_size"]}
    print("[rmu] data:", json.dumps(data_info), flush=True)

    if st["base_dev"] is None:
        _, st["base_dev"] = dev_eval(None, "base")
        save_state(st)
    base = st["base_dev"]
    print(f"[rmu] base dev: wmdp {base['wmdp']['mean']:.4f} util {base['util_pooled']['mean']:.4f}", flush=True)

    if st["r"] is None:
        m, tok = load_hf(HF_NAME)
        tok.padding_side = "left"
        tap = LayerTap(decoder_layers(m)[OFFICIAL["layer_id"]])
        st["r"] = measure_r(m, tok, tap, retain_docs)
        del m, tap
        torch.cuda.empty_cache()
        save_state(st)
    r = st["r"]
    print(f"[rmu] median token norm at layer {OFFICIAL['layer_id']}: r = {r:.2f}", flush=True)

    grid = [(sm, a) for sm in STEER_MULTS for a in ALPHAS]
    mini = mini_eval_sets()
    for k, (sm, a) in enumerate(grid):
        key = str(k)
        if st["configs"].get(key, {}).get("status") == "done":
            continue
        hp = {**OFFICIAL, "steering_mult": sm, "steering_coeff": round(sm * r, 2), "alpha": a, "r": round(r, 4)}
        print(f"[rmu] cfg {k}: steering {hp['steering_coeff']} ({sm} x r), alpha {a}", flush=True)
        st["configs"][key] = {"status": "training", "hp": hp}
        save_state(st)
        t0 = time.time()
        wdir, log, u = train_config(k, hp, forget_b, retain_b, mini, st)
        t_train = time.time() - t0
        rd, summ = dev_eval(wdir, f"rmu-c{k}", hp)
        # training run dir (MASTER_PLAN 7): config, train_log, metrics, control vector
        tcfg = {"exp_id": "RMU-cluster-train", "case": "bio", "split": "dev", "seed": OFFICIAL["seed"],
                "datasets": ["bio-forget-corpus", "wikitext"], "dataset_label": f"c{k}",
                "method": {"name": "rmu", **hp}}
        lg = RunLogger(tcfg)
        lg.write_config({"data": data_info, "official_reference": "centerforaisafety/wmdp rmu/unlearn.py",
                         "slurm_job": os.environ.get("SLURM_JOB_ID")})
        pd.DataFrame(log).to_parquet(lg.dir / "train_log.parquet", index=False)
        np.save(lg.dir / "control_vec.npy", u)
        summ.update({"train_seconds": round(t_train, 1), "peak_vram_gb_train": max(x["vram_gb"] for x in log),
                     "final": {k2: log[-1][k2] for k2 in ("unlearn_loss", "retain_loss", "mini_forget_acc", "mini_mmlu_acc")}})
        lg.write_metrics({"dev": summ, "base_dev": base, "util_drop": base["util_pooled"]["mean"] - summ["util_pooled"]["mean"],
                          "selection_key": list(sel_key(summ, base))})
        lg.mark_done({"forget": summ["wmdp"], "utility_pooled": summ["util_pooled"], "view": "raw (dev)"})

        best = st["best"]
        if best is None or sel_key(summ, base) < sel_key(st["configs"][str(best)]["dev"], base):
            shutil.rmtree(MODELS / "best", ignore_errors=True)
            shutil.move(str(wdir), str(MODELS / "best"))
            st["best"] = k
            print(f"[rmu] cfg {k} is the new best", flush=True)
        else:
            shutil.rmtree(wdir, ignore_errors=True)
        st["configs"][key] = {"status": "done", "hp": hp, "dev": summ, "train_run_dir": str(lg.dir)}
        save_state(st)
        print(f"[rmu] cfg {k} dev: wmdp {summ['wmdp']['mean']:.4f} util {summ['util_pooled']['mean']:.4f} "
              f"(drop {base['util_pooled']['mean'] - summ['util_pooled']['mean']:+.4f})", flush=True)

    b = st["best"]
    st["selected"] = {"cfg": b, "hp": st["configs"][str(b)]["hp"], "dev": st["configs"][str(b)]["dev"],
                      "rule": f"lowest WMDP-Bio dev acc among configs with pooled utility drop <= {MAX_UTIL_DROP}; else smallest drop",
                      "weights": str(MODELS / "best")}
    st["finished"] = now_iso()
    st["job_seconds_last_attempt"] = round(time.time() - t_job, 1)
    save_state(st)
    (MODELS / "best" / "SELECTED.json").write_text(json.dumps(st["selected"], indent=1, default=str))
    print("[rmu] selected:", json.dumps({"cfg": b, "hp": st["selected"]["hp"]}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
