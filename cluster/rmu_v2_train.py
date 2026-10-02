"""RMU v2 on gemma-2-2b-it (WMDP Bio): wider DEV grid, largest safe batch size. Resumable.

Why v2: the v1 grid (cluster/rmu_train.py, 6 configs, steering {1,2,4} x r, alpha {300,1200}, layer 7)
selected its strongest steering (4 x r, alpha 300) at the edge of the grid; its TEST WMDP-Bio was 0.556
vs base 0.644 and the third-party RMU's 0.498 (dsg_results_cluster/jobs/rmu/SUMMARY.md). v1 utility drops
were all <= 0.0064, far below the 0.02 budget, so there is room for stronger forgetting.

Recipe: identical to v1 (official WMDP rmu/unlearn.py port: frozen + updated copy, MSE to a fixed random
control vector u on forget text, alpha * MSE to the frozen activations on retain text, only mlp.down_proj
of the update layers trained, AdamW lr 5e-5, truncation 512, docs > 50 chars, seed 42) except:
  * batch size B = the largest of BS_CANDIDATES whose measured peak GPU memory (one full training step on
    512-token batches, the worst case) stays within the budget min(0.70 x total, free at start - 4 GiB)
    (~33 GiB of 48). B is probed once and stored in grid_state.json, so a restart keeps it.
    Official RMU uses B=4; a larger B sees B/4 x more documents per step (deviation, logged).
  * forget batches are consecutive corpus documents (file order, as v1); the retain corpus (wikitext-2
    test, 1962 docs > 50 chars) is smaller than B x steps, so retain batches cycle through it in order.
  * steering is in units of r_L = median per-token norm of the frozen layer-L output on retain text
    (BOS excluded), measured separately for L = 3 and L = 7.
  * dev eval batch size = the largest of EVAL_BS_CANDIDATES that fits the same budget on the longest dev
    prompt (decision 1 allows batched dev sweeps; TEST reporting stays at bs=1 in rmu_v2_eval.py).

Grid: the full factorial steering {4,8,12,20} x r, alpha {100,300,1200}, layer {3 (update 1,2,3),
7 (update 5,6,7)}, steps {150,300} has 48 configs; GRID below keeps 16 (rationale in GRID_RATIONALE,
written to grid_state.json and GRID.md).

Selection (DEV only, same rule as v1): among configs whose pooled full-MMLU utility drop vs base (dev,
same batch size) is <= 0.02, the lowest WMDP-Bio dev accuracy; if none qualifies, the smallest drop.
Kept on disk: models/RMU-v2/best (HF weights of the current best) and models/RMU-v2/last (trainer state
of the config in training, every 50 steps). Everything else is deleted.

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

from cluster import jobcommon as jc
from cluster.rmu_train import LayerTap, MAX_UTIL_DROP, mini_eval_sets, sel_key
from dsgx import paths
from dsgx.util import atomic_write_json, now_iso, sha256_file

RECIPE = {"param_name": "mlp.down_proj.weight", "lr": 5e-5, "max_len": 512, "min_len": 50, "seed": 42}
UPDATE_LAYERS = {3: [1, 2, 3], 7: [5, 6, 7]}
# (steering multiple of r_L, alpha, layer, steps)
GRID = [
    (4, 300, 7, 150),     # v1's selected config (anchor; now at batch size B)
    (4, 100, 7, 150),
    (8, 100, 7, 150), (8, 300, 7, 150),
    (12, 100, 7, 150), (12, 300, 7, 150), (12, 1200, 7, 150),
    (20, 100, 7, 150), (20, 300, 7, 150), (20, 1200, 7, 150),
    (8, 300, 7, 300), (20, 300, 7, 300), (20, 1200, 7, 300),
    (8, 300, 3, 150), (20, 300, 3, 150), (20, 1200, 3, 150),
]
GRID_RATIONALE = [
    "Full factorial 4 steering x 3 alpha x 2 layers x 2 step counts = 48 configs; kept 16 (training ~2-12 min "
    "+ dev eval ~2 min each).",
    "Layer 7 / update 5,6,7 is the official RMU setting and v1's only layer: 13 of 16 configs.",
    "v1: alpha 1200 almost cancelled forgetting at <= 4 x r (dev WMDP drop 0.01-0.02 vs 0.03-0.08 at alpha 300), "
    "so alpha 1200 is only paired with 12 x and 20 x r, where stronger retain pressure may be needed; "
    "4x/1200 and 8x/1200 are dropped as dominated.",
    "v1 utility drops were all <= 0.0064 (budget 0.02): the budget goes to stronger steering and lower alpha "
    "(alpha 100 at every steering level on layer 7).",
    "4 x r / alpha 300 / 150 steps repeats v1's selected config as an anchor at the new batch size.",
    "300 steps only where longer training plausibly changes the outcome: 8x/300, 20x/300, 20x/1200.",
    "Layer 3 / update 1,2,3 (the layer of DSG's SAE and of the third-party RMU): 3 configs at 150 steps, "
    "8x/300, 20x/300, 20x/1200 (the strongest settings; r_3 is measured separately).",
]
BS_CANDIDATES = [64, 48, 32, 24, 16, 8, 4]
EVAL_BS_CANDIDATES = [128, 96, 64, 48, 32, 16]
MEM_FRACTION, MEM_RESERVE_GIB = 0.70, 4.0
MINI_BS = 16
CKPT_EVERY = 50
EVAL_EVERY = 50
HF_NAME = jc.HF_2B

MODELS = paths.cache_dir() / "models" / "RMU-v2"
TMP = paths.cache_dir() / "models" / "RMU-v2-tmp"
JOBDIR = paths.results_dir() / "jobs" / "rmu-v2"
STATE = JOBDIR / "grid_state.json"


def grid_hps(r_by_layer: dict, bs: int) -> list[dict]:
    """The 16 resolved hyper-parameter dicts (keys stable across restarts)."""
    out = []
    for sm, a, layer, steps in GRID:
        r = r_by_layer[str(layer)]
        out.append({**RECIPE, "layer_id": layer, "layer_ids": UPDATE_LAYERS[layer], "batch_size": bs,
                    "max_num_batches": steps, "steering_mult": float(sm), "steering_coeff": round(sm * r, 2),
                    "alpha": float(a), "r": round(r, 4)})
    return out


def make_batches(docs: list[str], bs: int, n: int, cycle: bool) -> list[list[str]]:
    """n consecutive batches of bs docs; with cycle=True wrap around the corpus (in order)."""
    if not cycle:
        assert bs * n <= len(docs), f"need {bs * n} docs, have {len(docs)}"
        return [docs[i * bs:(i + 1) * bs] for i in range(n)]
    return [[docs[(i * bs + j) % len(docs)] for j in range(bs)] for i in range(n)]


def mem_budget_gib() -> float:
    free, total = torch.cuda.mem_get_info()
    return min(MEM_FRACTION * total / 2**30, free / 2**30 - MEM_RESERVE_GIB)


def pick_largest(candidates, peak_fn, budget_gib, log=print):
    """Largest candidate whose measured peak (GiB) fits the budget. peak_fn raises on OOM."""
    probes = []
    for c in sorted(candidates, reverse=True):
        try:
            peak = peak_fn(c)
        except torch.cuda.OutOfMemoryError:
            peak = float("inf")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        probes.append({"candidate": c, "peak_gib": None if peak == float("inf") else round(peak, 2)})
        log(f"probe {c}: peak {peak:.2f} GiB (budget {budget_gib:.2f})")
        if peak <= budget_gib:
            return c, probes
    raise RuntimeError(f"no candidate fits the memory budget {budget_gib:.2f} GiB: {probes}")


def _rand_tokens(tok, bs, length, vocab, device):
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(1000, vocab - 1000, (bs, length), generator=g)
    ids[:, 0] = tok.bos_token_id
    return {"input_ids": ids.to(device), "attention_mask": torch.ones_like(ids).to(device)}


def probe_train_bs(log):
    """One full training step (layer-7 config, 512-token forget + retain batches) per candidate."""
    budget = mem_budget_gib()  # before loading: peaks below include the weights
    frozen, updated = jc.load_lm(HF_NAME), jc.load_lm(HF_NAME)
    tok = jc.load_tok(HF_NAME)
    layers_f, layers_u = _layers(frozen), _layers(updated)
    for p in list(frozen.parameters()) + list(updated.parameters()):
        p.requires_grad_(False)
    params = [layers_u[i].mlp.down_proj.weight for i in UPDATE_LAYERS[7]]
    for p in params:
        p.requires_grad_(True)
    ftap, utap = LayerTap(layers_f[7]), LayerTap(layers_u[7])
    vocab = updated.config.vocab_size

    def peak(bs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        opt = torch.optim.AdamW(params, lr=RECIPE["lr"])
        enc = _rand_tokens(tok, bs, RECIPE["max_len"], vocab, updated.device)
        h_f = utap.run(updated, enc)
        h_r = utap.run(updated, enc)
        with torch.no_grad():
            h_r0 = ftap.run(frozen, enc)
        loss = h_f.float().pow(2).mean() + (h_r - h_r0).float().pow(2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        del opt, h_f, h_r, h_r0, loss, enc
        return torch.cuda.max_memory_reserved() / 2**30

    torch.set_grad_enabled(True)
    try:
        bs, probes = pick_largest(BS_CANDIDATES, peak, budget, log)
    finally:
        ftap.h.remove()
        utap.h.remove()
        del frozen, updated, params, ftap, utap
        torch.cuda.empty_cache()
    return bs, probes


def dev_prompt_lengths():
    """Token lengths of every dev prompt the dev eval scores (counts only; no text leaves memory)."""
    from dsgx.attacks.registry import make_attack
    from dsgx.data.mcq import load_mcq
    from dsgx.data.splits import get_split
    from dsgx.run import expand_datasets

    tok = jc.load_tok(HF_NAME)
    attack = make_attack(None, 0)
    lens = []
    for ds in expand_datasets(["@forget", "@utility"], "bio"):
        items = load_mcq(ds)
        prompts = [attack.prompt(items[i])[0] for i in get_split(ds, "dev")]
        lens += [len(x) for x in tok([p for p in prompts if p is not None], add_special_tokens=False)["input_ids"]]
    return lens


def probe_eval_bs(log):
    """Harness MCQ scoring (answer_probs on the TL model) at the longest dev prompt length."""
    from dsgx.eval.mcq_eval import ANSWER_STRINGS, answer_probs
    from dsgx.models.loader import clear, get_bundle

    lmax = max(dev_prompt_lengths())
    budget = mem_budget_gib()  # before loading: peaks below include the weights
    b = get_bundle("gemma-2-2b-it")
    model = b.model
    ans = model.to_tokens(ANSWER_STRINGS, prepend_bos=False).flatten()

    def peak(bs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        enc = _rand_tokens(model.tokenizer, bs, lmax, model.cfg.d_vocab, model.cfg.device)
        nti = torch.full((bs,), lmax - 1, device=model.cfg.device)
        with torch.no_grad():
            answer_probs(model, enc["input_ids"], nti, ans)
        return torch.cuda.max_memory_reserved() / 2**30

    try:
        bs, probes = pick_largest(EVAL_BS_CANDIDATES, peak, budget, log)
    finally:
        del model, b
        clear()
        torch.cuda.empty_cache()
    return bs, probes, lmax


def _layers(model):
    from dsgx.train.core import decoder_layers

    return decoder_layers(model)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"version": 2, "configs": {}, "best": None, "base_dev": None, "r": {}, "bs": None, "eval_bs": None,
            "grid": [list(g) for g in GRID], "grid_rationale": GRID_RATIONALE, "started": now_iso()}


def save_state(st):
    atomic_write_json(STATE, st)


def dev_eval(weights, tag, eval_bs, hp=None):
    """Harness dev eval (WMDP-Bio dev + full-MMLU utility dev); returns (run_dir, summary)."""
    from dsgx.models.loader import clear
    from dsgx.run import run

    cfg = {"exp_id": "RMU-v2-dev", "case": "bio", "split": "dev", "view": "both", "seed": 0,
           "batch_size": eval_bs, "purpose": "tune", "datasets": ["@forget", "@utility"],
           "dataset_label": f"{tag}-wmdp+mmlu48", "method": {"name": "base"},
           "model": {"weights": str(weights)} if weights else {}, "rmu": hp}
    rd = run(cfg)
    clear()
    m = json.loads((rd / "metrics.json").read_text())["views"]["raw"]
    return rd, {"wmdp": m["forget"], "util_pooled": m["utility"]["pooled"],
                "util_unweighted": m["utility"]["unweighted"], "run_dir": str(rd)}


def measure_r(layer, retain_docs, n_batches=10):
    m, tok = jc.load_lm(HF_NAME), jc.load_tok(HF_NAME)
    tok.padding_side = "left"
    tap = LayerTap(_layers(m)[layer])
    norms = []
    with torch.no_grad():
        for b in make_batches(retain_docs, 4, n_batches, cycle=False):
            enc = tok(b, return_tensors="pt", padding=True, truncation=True, max_length=512).to(m.device)
            h = tap.run(m, enc).float()
            mask = enc["attention_mask"].bool() & (enc["input_ids"] != tok.bos_token_id)
            norms.append(h.norm(dim=-1)[mask].cpu())
    tap.h.remove()
    del m, tap
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return float(torch.cat(norms).median())


def train_config(k, hp, forget_docs, retain_docs, mini, models_dir=None, tmp_dir=None, log=print):
    """Train one config (resumable from models_dir/last); returns (weights dir, train log, u)."""
    from dsgx.train.core import mcq_accuracy_hf

    models_dir, tmp_dir = Path(models_dir or MODELS), Path(tmp_dir or TMP)
    cuda = torch.cuda.is_available() and not jc.TINY
    torch.set_grad_enabled(True)  # the harness eval (get_bundle) turns grad off globally
    torch.manual_seed(hp["seed"])
    if cuda:
        torch.cuda.manual_seed(hp["seed"])
    frozen, updated, tok = jc.load_lm(HF_NAME), jc.load_lm(HF_NAME), jc.load_tok(HF_NAME)
    tok.padding_side = "left"
    frozen.eval()
    for p in list(frozen.parameters()) + list(updated.parameters()):
        p.requires_grad_(False)
    lid = hp["layer_id"] if not jc.TINY else min(hp["layer_id"], updated.config.num_hidden_layers - 1)
    lids = [min(x, lid) for x in hp["layer_ids"]] if jc.TINY else hp["layer_ids"]
    params = []
    for i in dict.fromkeys(lids):
        p = _layers(updated)[i].mlp.down_proj.weight
        p.requires_grad_(True)
        params.append(p)
    opt = torch.optim.AdamW(params, lr=hp["lr"])
    d = updated.config.hidden_size
    u = torch.rand(1, 1, d, dtype=updated.dtype, device=updated.device)
    u = u / torch.norm(u) * hp["steering_coeff"]
    ftap, utap = LayerTap(_layers(frozen)[lid]), LayerTap(_layers(updated)[lid])
    n, bs = hp["max_num_batches"], hp["batch_size"]
    forget_b = make_batches(forget_docs, bs, n, cycle=False)
    retain_b = make_batches(retain_docs, bs, n, cycle=True)

    last = models_dir / "last"
    sp = last / "state.pt"
    start, tlog = 0, []
    if sp.exists():
        s = torch.load(sp, weights_only=False)
        if s.get("cfg") == k:
            with torch.no_grad():
                for p, v in zip(params, s["params"]):
                    p.copy_(v.to(p.device, p.dtype))
            opt.load_state_dict(s["opt"])
            u = s["u"].to(updated.device, updated.dtype)
            start, tlog = s["step"], s["log"]
            log(f"cfg {k}: resumed at step {start}")
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time() - (tlog[-1]["time"] if tlog else 0.0)
    updated.train()
    for step in range(start, n):
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
               "retain_loss": float(retain_loss), "lr": hp["lr"], "grad_norm": gn, "time": time.time() - t0,
               "vram_gb": torch.cuda.max_memory_allocated() / 1e9 if cuda else 0.0,
               "h_forget_norm": float(h_f.float().norm(dim=-1).mean())}
        del h_f, h_r, h_r0, loss
        if (step + 1) % EVAL_EVERY == 0 or step + 1 == n:
            updated.eval()
            with torch.no_grad():
                row["mini_forget_acc"] = float(mcq_accuracy_hf(updated, tok, mini[0], bs=MINI_BS).mean())
                row["mini_mmlu_acc"] = float(mcq_accuracy_hf(updated, tok, mini[1], bs=MINI_BS).mean())
            updated.train()
        tlog.append(row)
        if (step + 1) % 10 == 0:
            log(f"cfg {k} step {step + 1}: unlearn {row['unlearn_loss']:.4g} retain {row['retain_loss']:.4g}"
                + (f" mini_forget {row['mini_forget_acc']:.3f} mini_mmlu {row['mini_mmlu_acc']:.3f}"
                   if "mini_forget_acc" in row else ""))
        if (step + 1) % CKPT_EVERY == 0 or step + 1 == n:
            last.mkdir(parents=True, exist_ok=True)
            tmp = last / "state.pt.tmp"
            torch.save({"cfg": k, "step": step + 1, "params": [p.detach().cpu() for p in params],
                        "opt": opt.state_dict(), "u": u.cpu(), "log": tlog}, tmp)
            tmp.replace(sp)
    updated.eval()
    out = tmp_dir / f"cfg{k}"
    shutil.rmtree(out, ignore_errors=True)
    updated.save_pretrained(out, safe_serialization=True)
    u_np = u.float().cpu().numpy()
    ftap.h.remove()
    utap.h.remove()
    del frozen, updated, opt, params, ftap, utap
    if cuda:
        torch.cuda.empty_cache()
    return out, tlog, u_np


def write_grid_md(st):
    L = ["# RMU v2 DEV grid (pre-registered in cluster/rmu_v2_train.py)", "",
         "| cfg | steering (x r_L) | alpha | layer | update | steps |", "|---|---|---|---|---|---|"]
    for k, (sm, a, layer, steps) in enumerate(GRID):
        L.append(f"| c{k} | {sm} | {a} | {layer} | {UPDATE_LAYERS[layer]} | {steps} |")
    L += ["", "## Pruning (48 -> 16)", ""] + [f"- {x}" for x in GRID_RATIONALE]
    L += ["", f"Batch size: train {st.get('bs')} (probe {st.get('bs_probe')}), dev eval {st.get('eval_bs')} "
          f"(probe {st.get('eval_bs_probe')}); TEST bs=1.",
          f"Selection: lowest WMDP-Bio dev accuracy among configs with pooled utility drop <= {MAX_UTIL_DROP}; "
          "else the smallest drop (same rule as v1)."]
    (JOBDIR / "GRID.md").write_text("\n".join(L) + "\n")


def main():
    import pandas as pd

    from dsgx.data.corpora import load_forget_docs, load_retain_docs
    from dsgx.logging.run_logger import RunLogger

    def log(msg):
        jc.log("rmu-v2", msg)

    JOBDIR.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    st = load_state()
    t_job = time.time()

    forget_docs = load_forget_docs("bio-forget-corpus", min_len=RECIPE["min_len"])
    retain_docs = load_retain_docs("wikitext", min_len=RECIPE["min_len"])
    data_info = {"forget_corpus_sha256": sha256_file(paths.forget_corpus_jsonl()), "n_forget_docs": len(forget_docs),
                 "n_retain_docs": len(retain_docs)}
    log("data: " + json.dumps(data_info))

    if st["bs"] is None:
        st["bs"], st["bs_probe"] = probe_train_bs(log)
        max_steps = max(g[3] for g in GRID)
        while st["bs"] * max_steps > len(forget_docs):  # forget batches never repeat a document
            st["bs"] //= 2
        save_state(st)
    if st["eval_bs"] is None:
        st["eval_bs"], st["eval_bs_probe"], st["dev_max_prompt_tokens"] = probe_eval_bs(log)
        save_state(st)
    bs, eval_bs = st["bs"], st["eval_bs"]
    log(f"batch size: train {bs}, dev eval {eval_bs} (TEST stays at bs=1)")

    for layer in UPDATE_LAYERS:
        if str(layer) not in st["r"]:
            st["r"][str(layer)] = measure_r(layer, retain_docs)
            save_state(st)
        log(f"median token norm at layer {layer}: r = {st['r'][str(layer)]:.2f}")
    write_grid_md(st)

    if st["base_dev"] is None:
        _, st["base_dev"] = dev_eval(None, "base", eval_bs)
        save_state(st)
    base = st["base_dev"]
    log(f"base dev: wmdp {base['wmdp']['mean']:.4f} util {base['util_pooled']['mean']:.4f}")

    mini = mini_eval_sets()
    for k, hp in enumerate(grid_hps(st["r"], bs)):
        key = str(k)
        if st["configs"].get(key, {}).get("status") == "done":
            continue
        log(f"cfg {k}: steering {hp['steering_coeff']} ({hp['steering_mult']} x r{hp['layer_id']}), alpha {hp['alpha']}, "
            f"layer {hp['layer_id']} update {hp['layer_ids']}, steps {hp['max_num_batches']}, bs {bs}")
        st["configs"][key] = {"status": "training", "hp": hp}
        save_state(st)
        t0 = time.time()
        wdir, tlog, u = train_config(k, hp, forget_docs, retain_docs, mini, log=log)
        t_train = time.time() - t0
        rd, summ = dev_eval(wdir, f"rmu-v2-c{k}", eval_bs, hp)
        tcfg = {"exp_id": "RMU-v2-train", "case": "bio", "split": "dev", "seed": hp["seed"],
                "datasets": ["bio-forget-corpus", "wikitext"], "dataset_label": f"c{k}",
                "method": {"name": "rmu", **hp}}
        lg = RunLogger(tcfg)
        lg.write_config({"data": {**data_info, "forget_docs_used": bs * hp["max_num_batches"],
                                  "retain_docs_used_with_cycling": bs * hp["max_num_batches"]},
                         "official_reference": "centerforaisafety/wmdp rmu/unlearn.py",
                         "deviations": ["batch size probed (official 4)", "retain corpus cycled in order"],
                         "slurm_job": os.environ.get("SLURM_JOB_ID")})
        pd.DataFrame(tlog).to_parquet(lg.dir / "train_log.parquet", index=False)
        np.save(lg.dir / "control_vec.npy", u)
        summ.update({"train_seconds": round(t_train, 1), "peak_vram_gb_train": max(x["vram_gb"] for x in tlog),
                     "final": {k2: tlog[-1][k2] for k2 in ("unlearn_loss", "retain_loss", "mini_forget_acc",
                                                           "mini_mmlu_acc")}})
        lg.write_metrics({"dev": summ, "base_dev": base,
                          "util_drop": base["util_pooled"]["mean"] - summ["util_pooled"]["mean"],
                          "selection_key": list(sel_key(summ, base))})
        lg.mark_done({"forget": summ["wmdp"], "utility_pooled": summ["util_pooled"], "view": "raw (dev)"})

        best = st["best"]
        if best is None or sel_key(summ, base) < sel_key(st["configs"][str(best)]["dev"], base):
            shutil.rmtree(MODELS / "best", ignore_errors=True)
            shutil.move(str(wdir), str(MODELS / "best"))
            st["best"] = k
            log(f"cfg {k} is the new best")
        else:
            shutil.rmtree(wdir, ignore_errors=True)
        st["configs"][key] = {"status": "done", "hp": hp, "dev": summ, "train_run_dir": str(lg.dir)}
        save_state(st)
        log(f"cfg {k} dev: wmdp {summ['wmdp']['mean']:.4f} util {summ['util_pooled']['mean']:.4f} "
            f"(drop {base['util_pooled']['mean'] - summ['util_pooled']['mean']:+.4f}), train {t_train:.0f} s")

    shutil.rmtree(MODELS / "last", ignore_errors=True)
    shutil.rmtree(TMP, ignore_errors=True)
    b = st["best"]
    st["selected"] = {"cfg": b, "hp": st["configs"][str(b)]["hp"], "dev": st["configs"][str(b)]["dev"],
                      "rule": f"lowest WMDP-Bio dev acc among configs with pooled utility drop <= {MAX_UTIL_DROP}; "
                              "else smallest drop",
                      "weights": str(MODELS / "best")}
    edge = [x for x in ("steering_mult", "alpha", "max_num_batches")
            if st["selected"]["hp"][x] in (max(g[GRID_KEYS[x]] for g in GRID), min(g[GRID_KEYS[x]] for g in GRID))]
    st["selected"]["at_grid_edge"] = edge
    st["finished"] = now_iso()
    st["job_seconds_last_attempt"] = round(time.time() - t_job, 1)
    save_state(st)
    (MODELS / "best" / "SELECTED.json").write_text(json.dumps(st["selected"], indent=1, default=str))
    log("selected: " + json.dumps({"cfg": b, "hp": st["selected"]["hp"], "at_grid_edge": edge}))
    return 0


GRID_KEYS = {"steering_mult": 0, "alpha": 1, "max_num_batches": 3}

if __name__ == "__main__":
    sys.exit(main())
