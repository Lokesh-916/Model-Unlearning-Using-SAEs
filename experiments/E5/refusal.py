"""E5 Step 1 task (GPU): refusal direction, forced-on steering check on held-out harmless instructions, selection.

Base model gemma-2-2b-it. Per layer in LAYERS: resid_post at the last instruction token of chat(instruction) for the
256 harmful + 256 harmless build instructions; direction = mean(harmful) - mean(harmless), unit-normalised; norm =
mean residual norm at that position over the 512 build prompts. Check: greedy, max_new 96, on the 100 held-out
harmless instructions: base (no steering) and s x norm x direction added at blocks.L.hook_resid_post on all positions
for s in STRENGTHS (all 20 cells + base as rows of one batch per prompt). Refusal (common.is_refusal) and gibberish
(dsgx openqa.gibberish) rates. Selection: common.select_cell. Then the same recipe on the TOFU model at the selected
layer (direction + norm; no evaluation, no re-selection). Harmful instructions and all generations are never printed;
generations only under $DSG_PRIVATE; ids, hashes and rates in the run dir.
"""
import json

import numpy as np
import pandas as pd
import torch

from dsgx.eval import stats
from dsgx.eval.openqa import gibberish, text_hash
from dsgx.itemckpt import ItemCheckpoints

from experiments.E5.batchgen import forced_batch
from experiments.E5.common import LAYERS, STRENGTHS, cache_path, is_refusal, select_cell
from experiments.E5.data import build_sets, chat, last_instruction_pos


@torch.no_grad()
def resid_at_instruction(model, texts, layers):
    names = {f"blocks.{l}.hook_resid_post" for l in layers}
    out = {l: [] for l in layers}
    for t in texts:
        toks = model.to_tokens(chat(t), prepend_bos=False).to(model.cfg.device)
        pos = last_instruction_pos(model, toks)
        _, c = model.run_with_cache(toks, stop_at_layer=max(layers) + 1, names_filter=lambda n: n in names)
        for l in layers:
            out[l].append(c[f"blocks.{l}.hook_resid_post"][0, pos].float().cpu().numpy())
    return {l: np.stack(v) for l, v in out.items()}


def directions(model, sets, layers):
    H = resid_at_instruction(model, [x["text"] for x in sets["harmful"]], layers)
    S = resid_at_instruction(model, [x["text"] for x in sets["harmless"]], layers)
    out = {}
    for l in layers:
        d = H[l].mean(0) - S[l].mean(0)
        out[l] = {"dir": d / np.linalg.norm(d), "norm": float(np.linalg.norm(np.concatenate([H[l], S[l]]), axis=1).mean()),
                  "raw_diff_norm": float(np.linalg.norm(d))}
    return out


def save_dirs(path, dirs, extra):
    np.savez(path, layers=np.array(list(dirs)), **{f"dir_{l}": v["dir"] for l, v in dirs.items()},
             **{f"norm_{l}": np.array(v["norm"]) for l, v in dirs.items()}, info=np.array(json.dumps(extra)))


def load_dir(exp_id, model_key, layer):
    with np.load(cache_path(exp_id, model_key)) as z:
        return z[f"dir_{layer}"].astype(np.float32), float(z[f"norm_{layer}"])


def task(ctx):
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    a = ctx.args
    sets = build_sets()
    held = sets["heldout"][: int(a["limit"])] if a.get("limit") else sets["heldout"]
    max_new = int(a.get("max_new", 96))
    b = ctx.bundle()
    m = b.model
    dirs = directions(m, sets, LAYERS)
    save_dirs(cache_path(ctx.exp_id, "base"), dirs, {"model": "gemma-2-2b-it", "n_harmful": len(sets["harmful"]),
                                                     "n_harmless": len(sets["harmless"])})
    cells = [(l, s) for l in LAYERS for s in STRENGTHS]
    names = ["base"] + [f"L{l}s{s:g}" for l, s in cells]
    vec = {(l, s): torch.tensor(s * dirs[l]["norm"] * dirs[l]["dir"], dtype=torch.float32, device=b.device)
           for l, s in cells}
    rows = [(None, None)] + [(l, vec[(l, s)]) for l, s in cells]
    ctx.write_config(n_heldout=len(held), layers=LAYERS, strengths=STRENGTHS, max_new=max_new,
                     norms={l: dirs[l]["norm"] for l in LAYERS}, heldout_ids=[q["id"] for q in held],
                     build_ids={"harmful": [x["id"] for x in sets["harmful"]],
                                "harmless": [x["id"] for x in sets["harmless"]]})
    ctx.progress.update(items_total=len(held), items_done=0, force=True)
    priv = ctx.private_dir()
    ck = ItemCheckpoints(priv / "partial", ["E5-refusal", [q["id"] for q in held], cells, max_new,
                                            {l: dirs[l]["norm"] for l in LAYERS}])

    def one(q):
        return forced_batch(m, chat(q["text"]), rows, LAYERS, max_new)

    done = ck.map(held, one, key=lambda q: q["id"], on_done=ctx.progress.advance)
    recs, gens = [], []
    for q, outs in zip(held, done):
        for name, ids in zip(names, outs):
            text = m.tokenizer.decode(ids, skip_special_tokens=True)
            gens.append({"id": q["id"], "cond": name, "text": text})
            recs.append({"id": q["id"], "cond": name, "refusal": is_refusal(text), "n_tokens": len(ids),
                         "text_hash": text_hash(text), **gibberish(text, ids)})
    with (priv / "generations.jsonl").open("w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    df = pd.DataFrame(recs)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    base = df[df["cond"] == "base"].set_index("id")
    res = {"base": {k: stats.bootstrap_ci(base[k].astype(float).values) for k in ("refusal", "gibberish")}}
    sel_in = []
    for (l, s), name in zip(cells, names[1:]):
        d = df[df["cond"] == name].set_index("id").loc[base.index]
        res[name] = {"layer": l, "s": s, **{k: stats.bootstrap_ci(d[k].astype(float).values) for k in ("refusal", "gibberish")},
                     "refusal_vs_base_mcnemar_p": stats.mcnemar(d["refusal"].values, base["refusal"].values)["p"]}
        sel_in.append({"layer": l, "s": s, "refusal": res[name]["refusal"]["mean"], "gib": res[name]["gibberish"]["mean"]})
    sel = select_cell(sel_in)
    if ctx.smoke and a.get("smoke_force"):  # smoke only: exercise Steps 2-4 whatever 3 items select
        f = a["smoke_force"]
        c = res[f"L{f['layer']}s{f['s']:g}"]
        sel = {"layer": int(f["layer"]), "s": float(f["s"]), "refusal": c["refusal"]["mean"], "gib": c["gibberish"]["mean"],
               "eligible": True, "smoke_forced": True}
    out = {"cells": res, "selection": sel, "norms": {l: dirs[l]["norm"] for l in LAYERS},
           "raw_diff_norms": {l: dirs[l]["raw_diff_norm"] for l in LAYERS}, "n_heldout": len(held)}
    if sel["eligible"]:
        bt = get_bundle(weights=resolve_weights(a.get("tofu_weights", "ckpt:E2f/tofu_full"), ctx.exp_id))
        dt = directions(bt.model, sets, [sel["layer"]])
        save_dirs(cache_path(ctx.exp_id, "tofu"), dt, {"model": "ckpt:E2f/tofu_full", "selected_from": "base Step 1"})
        out["tofu"] = {"layer": sel["layer"], "norm": dt[sel["layer"]]["norm"],
                       "raw_diff_norm": dt[sel["layer"]]["raw_diff_norm"],
                       "cos_with_base": float(dt[sel["layer"]]["dir"] @ dirs[sel["layer"]]["dir"])}
    ctx.write_metrics(out)
    ctx.finish({"view": "e5-refusal", "forget": None, "selected": [sel.get("layer"), sel.get("s")],
                "eligible": sel["eligible"]})
    ck.clear()
    return sel
