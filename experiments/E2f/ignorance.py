"""E2f Step 1 task (GPU): ignorance direction from safe entity questions, forced-on steering check, selection.

Base model: per layer in LAYERS, last-token resid_post of open_prompt(question) for the 300 made-up + 300 real
build questions; direction = mean(made-up) - mean(real), unit-normalised; norm = mean last-token residual norm over
the 600 build questions. Check: greedy generation (max_new 48) on the 100 held-out real-entity questions, base and
forced-on steering with c x norm x direction at every position, c in COEFS (+ DIAG_COEFS, diagnostic): "I don't
know / not familiar" rate (common.IDK), refusal rate, gibberish rate, paired vs base. Selection: common.select_cell.
Then the same recipe on the TOFU model (direction + norm at the selected layer only; no evaluation, no tuning).
Saved: $DSG_CACHE/E2f/<exp>/ignorance_{base,tofu}.npz. Questions and generations are safe; generations are still
kept only under $DSG_PRIVATE (convention), metrics in the run dir.
"""
import json

import numpy as np
import pandas as pd
import torch

from dsgx.eval import stats
from dsgx.eval.openqa import gibberish, open_prompt, text_hash
from dsgx.itemckpt import ItemCheckpoints

from experiments.E2f.common import (ALPHA, COEFS, DIAG_COEFS, LAYERS, dir_path, is_idk, is_refusal, select_cell)
from experiments.E2f.entities import build_sets
from experiments.E2f.switchgen import Response, generate


@torch.no_grad()
def last_resid(model, questions, layers):
    names = [f"blocks.{l}.hook_resid_post" for l in layers]
    out = {l: [] for l in layers}
    for q in questions:
        t = model.to_tokens(open_prompt(q), prepend_bos=False).to(model.cfg.device)
        _, c = model.run_with_cache(t, stop_at_layer=max(layers) + 1, names_filter=lambda n: n in names)
        for l in layers:
            out[l].append(c[f"blocks.{l}.hook_resid_post"][0, -1].float().cpu().numpy())
    return {l: np.stack(v) for l, v in out.items()}


def direction(model, sets, layers):
    R = last_resid(model, [q["question"] for q in sets["real"]], layers)
    F = last_resid(model, [q["question"] for q in sets["fake"]], layers)
    out = {}
    for l in layers:
        d = F[l].mean(0) - R[l].mean(0)
        norm = float(np.linalg.norm(np.concatenate([R[l], F[l]]), axis=1).mean())
        out[l] = {"dir": d / np.linalg.norm(d), "norm": norm, "raw_diff_norm": float(np.linalg.norm(d))}
    return out


def save_dirs(path, dirs, extra):
    np.savez(path, layers=np.array(list(dirs)), **{f"dir_{l}": v["dir"] for l, v in dirs.items()},
             **{f"norm_{l}": np.array(v["norm"]) for l, v in dirs.items()}, info=np.array(json.dumps(extra)))


def load_vec(exp_id, model_key, layer, coef, device):
    with np.load(dir_path(exp_id, model_key)) as z:
        d, n = z[f"dir_{layer}"], float(z[f"norm_{layer}"])
    return torch.tensor(coef * n * d, dtype=torch.float32, device=device)


def task(ctx):
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    a = ctx.args
    sets = build_sets()
    held = sets["heldout"][: int(a["limit"])] if a.get("limit") else sets["heldout"]
    max_new = int(a.get("max_new", 48))
    b = ctx.bundle()
    m = b.model
    dirs = direction(m, sets, LAYERS)
    save_dirs(dir_path(ctx.exp_id, "base"), dirs, {"model": "gemma-2-2b-it", "n_real": 300, "n_fake": 300})
    cells = [(l, c) for l in LAYERS for c in COEFS + DIAG_COEFS]
    ctx.write_config(n_heldout=len(held), layers=LAYERS, coefs=COEFS, diag_coefs=DIAG_COEFS, max_new=max_new,
                     norms={l: dirs[l]["norm"] for l in LAYERS})
    ctx.progress.update(items_total=len(held) * (1 + len(cells)), items_done=0, force=True)
    resp = [Response(f"L{l}c{c:g}", steer=(l, torch.tensor(c * dirs[l]["norm"] * dirs[l]["dir"], dtype=torch.float32,
                                                              device=b.device))) for l, c in cells]
    priv = ctx.private_dir()
    ck = ItemCheckpoints(priv / "partial", ["E2f-ign", [q["id"] for q in held], cells, max_new,
                                            {l: dirs[l]["norm"] for l in LAYERS}])

    def one(q):
        p = open_prompt(q["question"])
        r0 = generate(m, p, b, None, (), max_new)
        rf = generate(m, p, b, None, resp, max_new, forced=True)
        out = {"base": r0["base"]["tokens"]}
        out.update({r.name: rf[r.name]["tokens"] for r in resp})
        ctx.progress.advance(len(resp))
        return out

    done = ck.map(held, one, key=lambda q: q["id"], on_done=ctx.progress.advance)
    rows, gens = [], []
    for q, o in zip(held, done):
        for cond, ids in o.items():
            text = m.tokenizer.decode(ids, skip_special_tokens=True)
            gens.append({"id": q["id"], "cond": cond, "text": text})
            rows.append({"id": q["id"], "type": q["type"], "cond": cond, "idk": is_idk(text), "refusal": is_refusal(text),
                         "n_tokens": len(ids), "text_hash": text_hash(text), **gibberish(text, ids)})
    with (priv / "generations.jsonl").open("w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    base = df[df["cond"] == "base"].set_index("id")
    res = {"base": {k: stats.bootstrap_ci(base[k].astype(float).values) for k in ("idk", "refusal", "gibberish")}}
    sel_in = []
    for (l, c), r in zip(cells, resp):
        d = df[df["cond"] == r.name].set_index("id").loc[base.index]
        pb = stats.paired_bootstrap(d["idk"].astype(float).values, base["idk"].astype(float).values)
        mc = stats.mcnemar(d["idk"].values, base["idk"].values)
        res[r.name] = {"layer": l, "coef": c, "diagnostic": c in DIAG_COEFS,
                       **{k: stats.bootstrap_ci(d[k].astype(float).values) for k in ("idk", "refusal", "gibberish")},
                       "idk_vs_base": {**pb, "mcnemar_p": mc["p"], "sig_higher": bool(pb["lo"] > 0 and mc["p"] < ALPHA)}}
        sel_in.append({"layer": l, "coef": c, "idk": res[r.name]["idk"]["mean"], "gib": res[r.name]["gibberish"]["mean"]})
    sel = select_cell(sel_in)
    sel["works"] = bool(res[f"L{sel['layer']}c{sel['coef']:g}"]["idk_vs_base"]["sig_higher"])
    # same recipe on the TOFU model, selected layer only
    tofu_w = resolve_weights(a.get("tofu_weights", "ckpt:E2f/tofu_full"), ctx.exp_id)
    bt = get_bundle(weights=tofu_w)
    dt = direction(bt.model, sets, [sel["layer"]])
    save_dirs(dir_path(ctx.exp_id, "tofu"), dt, {"model": tofu_w, "selected_from": "base model Step 1"})
    out = {"cells": res, "selection": sel, "norms": {l: dirs[l]["norm"] for l in LAYERS},
           "raw_diff_norms": {l: dirs[l]["raw_diff_norm"] for l in LAYERS},
           "tofu": {"layer": sel["layer"], "norm": dt[sel["layer"]]["norm"]}, "n_heldout": len(held)}
    ctx.write_metrics(out)
    ctx.finish({"view": "e2f-ignorance", "forget": None, "selected": [sel["layer"], sel["coef"]]})
    ck.clear()
    return sel
