"""E5b: extended calibration of E5 (POST-v2 EXTENSION, EXPLORATORY, DEV only, never TEST, lab PC; not a claim input).
Rules fixed in DEVIATIONS (2026-11-14, E5b row, 2026-11-14T18:51:10+05:30) before any E5b code ran.

Step 1 (task `refusal`, entry refusal_task): E5's base-model refusal directions at layers 10 / 12 reused unchanged;
  s in STRENGTHS x mean residual norm; modes "all" (every position, whole generation) and "start8" (prompt positions +
  the first 8 generated positions, then none); greedy, 96 new tokens on E5's 100 held-out harmless instructions;
  selection (select_b); the chosen setting once on 100 fresh harmless instructions (report only); TOFU-model direction
  at the selected layer.
Step 2 (task `ignorance`, entry ignorance_task): beyond-cutoff ignorance direction (experiments.E5.cutoff), mean
  last-token resid_post of open_prompt(unknowable) - (known), layers 10 / 12; same strengths, modes, selection;
  calibration = IDK rate (cutoff.is_idk) on 100 held-out well-known questions; TOFU-model direction at the selected layer.
Both write runs/<exp>/<task>/metrics.json with "selection" = {layer, s, mode, rate, gib, eligible, tier, label} (the
refusal one also carries "refusal" = rate, read by sae.py / gen.py). Harmful instructions and generations are never
printed; generations only under $DSG_PRIVATE.
"""
import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import torch

from dsgx import paths
from dsgx.eval import stats
from dsgx.eval.openqa import gibberish, open_prompt, text_hash
from dsgx.itemckpt import ItemCheckpoints

from experiments.E5 import cutoff
from experiments.E5.batchgen import forced_batch
from experiments.E5.common import N_BUILD, N_HELD, cache_path, is_refusal

LAYERS_B = [10, 12]
STRENGTHS_B = [0.30, 0.35, 0.40, 0.45]
MODES = ["all", "start8"]
MODE_LIMIT = {"all": None, "start8": 8}
PRIMARY_RATE, GIB_MAX, FALLBACK_RATE = 0.90, 0.02, 0.70
N_FRESH = 100
WEAKER = "weaker setting"
NOT_CONFIRMED = "not confirmed"


def cell_name(layer, s, mode):
    return f"L{layer}s{s:g}{mode}"


def select_b(cells: list[dict]) -> dict:
    """cells: [{layer, s, mode, rate, gib}] point estimates.
    Primary: rate >= 0.90 and gib <= 0.02; smallest s; ties -> higher rate, lower gib, start8 before all, lower layer.
    Fallback: highest rate among gib <= 0.02 (ties -> lower gib, smaller s, start8 before all, lower layer) if rate >= 0.70,
    label 'weaker setting'. Else {'eligible': False}."""
    eps = 1e-12
    mo = {"start8": 0, "all": 1}
    clean = [c for c in cells if c["gib"] <= GIB_MAX + eps]
    prim = [c for c in clean if c["rate"] >= PRIMARY_RATE - eps]
    if prim:
        b = sorted(prim, key=lambda c: (c["s"], -c["rate"], c["gib"], mo[c["mode"]], c["layer"]))[0]
        return {**b, "eligible": True, "tier": "primary", "label": None}
    if clean:
        b = sorted(clean, key=lambda c: (-c["rate"], c["gib"], c["s"], mo[c["mode"]], c["layer"]))[0]
        if b["rate"] >= FALLBACK_RATE - eps:
            return {**b, "eligible": True, "tier": "fallback", "label": WEAKER}
    return {"eligible": False, "tier": None, "label": None}


def confirm_ok(sel: dict, rate: float, gib: float) -> bool:
    need = PRIMARY_RATE if sel["tier"] == "primary" else FALLBACK_RATE
    return bool(rate >= need - 1e-12 and gib <= GIB_MAX + 1e-12)


def ign_cache_path(exp_id: str, model_key: str):
    d = paths.cache_dir() / "E5" / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d / f"ignorance_{model_key}.npz"


def load_npz_dir(path, layer):
    with np.load(path) as z:
        return z[f"dir_{layer}"].astype(np.float32), float(z[f"norm_{layer}"])


def fresh_harmless():
    """The 100 Alpaca instructions after E5's 256 build + 100 held-out in the same default_rng(0) permutation."""
    from experiments.E5.data import harmless_pool

    safe = harmless_pool()
    ps = np.random.default_rng(0).permutation(len(safe))
    a = N_BUILD + N_HELD
    return [{"id": f"alpaca:{i}", "text": safe[i]} for i in ps[a:a + N_FRESH]]


def _grid(dirs, device):
    cells = [(l, s, mo) for l in LAYERS_B for s in STRENGTHS_B for mo in MODES]
    rows = [(None, None)] + [(l, torch.tensor(s * dirs[l][1] * dirs[l][0], dtype=torch.float32, device=device))
                             for l, s, mo in cells]
    limits = [None] + [MODE_LIMIT[mo] for _, _, mo in cells]
    names = ["base"] + [cell_name(*c) for c in cells]
    return cells, rows, limits, names


def _score(df, names, cells, rate_key):
    base = df[df["cond"] == "base"].set_index("id")
    res = {"base": {k: stats.bootstrap_ci(base[k].astype(float).values) for k in (rate_key, "gibberish")}}
    sel_in = []
    for (l, s, mo), name in zip(cells, names[1:]):
        d = df[df["cond"] == name].set_index("id").loc[base.index]
        res[name] = {"layer": l, "s": s, "mode": mo,
                     **{k: stats.bootstrap_ci(d[k].astype(float).values) for k in (rate_key, "gibberish")},
                     "rate_vs_base_mcnemar_p": stats.mcnemar(d[rate_key].values, base[rate_key].values)["p"]}
        sel_in.append({"layer": l, "s": s, "mode": mo, "rate": res[name][rate_key]["mean"], "gib": res[name]["gibberish"]["mean"]})
    return res, sel_in


def _smoke_force(ctx, res, rate_key):
    f = ctx.args["smoke_force"]
    c = res[cell_name(int(f["layer"]), float(f["s"]), f["mode"])]
    return {"layer": int(f["layer"]), "s": float(f["s"]), "mode": f["mode"], "rate": c[rate_key]["mean"],
            "gib": c["gibberish"]["mean"], "eligible": True, "tier": "fallback", "label": WEAKER, "smoke_forced": True}


def _gen_records(m, qs, done, names, key, grader):
    recs, gens = [], []
    for q, outs in zip(qs, done):
        for name, ids in zip(names, outs):
            text = m.tokenizer.decode(ids, skip_special_tokens=True)
            gens.append({"id": q["id"], "cond": name, "text": text})
            recs.append({"id": q["id"], "cond": name, key: grader(text), "refusal": is_refusal(text),
                         "idk": cutoff.is_idk(text), "n_tokens": len(ids), "text_hash": text_hash(text),
                         **gibberish(text, ids)})
    return recs, gens


def _write_gens(path, gens, mode="w"):
    with path.open(mode) as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")


def _confirm(ctx, m, qs, prompt_fn, sel, dirs, rate_key, grader, max_new, tag):
    """The chosen setting once on fresh items: rows base + selected. Report only."""
    L = sel["layer"]
    vec = torch.tensor(sel["s"] * dirs[L][1] * dirs[L][0], dtype=torch.float32, device=m.cfg.device)
    rows, limits, names = [(None, None), (L, vec)], [None, MODE_LIMIT[sel["mode"]]], ["base", "selected"]
    priv = ctx.private_dir()
    ck = ItemCheckpoints(priv / f"partial-{tag}", [tag, [q["id"] for q in qs], L, sel["s"], sel["mode"], max_new])
    done = ck.map(qs, lambda q: forced_batch(m, prompt_fn(q), rows, [L], max_new, limits), key=lambda q: q["id"])
    recs, gens = _gen_records(m, qs, done, names, rate_key, grader)
    _write_gens(priv / f"generations-{tag}.jsonl", gens)
    df = pd.DataFrame(recs)
    df.to_parquet(ctx.run_dir() / f"items-{tag}.parquet", index=False)
    out = {}
    for c in names:
        d = df[df["cond"] == c]
        out[c] = {k: stats.bootstrap_ci(d[k].astype(float).values) for k in (rate_key, "gibberish")}
    r, g = out["selected"][rate_key]["mean"], out["selected"]["gibberish"]["mean"]
    out["ok"] = confirm_ok(sel, r, g)
    out["n"] = len(qs)
    out["ids"] = [q["id"] for q in qs]
    ck.clear()
    return out


def refusal_task(ctx):
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    from experiments.E5.data import build_sets, chat
    from experiments.E5.refusal import directions, save_dirs

    a = ctx.args
    src_exp = a.get("dir_exp", "E5")
    src = cache_path(src_exp, "base")
    sha = hashlib.sha256(src.read_bytes()).hexdigest()
    dst = cache_path(ctx.exp_id, "base")
    shutil.copyfile(src, dst)                      # reuse E5's directions unchanged (gen.py / sae.py read <exp>)
    dirs = {l: load_npz_dir(dst, l) for l in LAYERS_B}
    sets = build_sets()
    lim = int(a["limit"]) if a.get("limit") else None
    held = sets["heldout"][:lim] if lim else sets["heldout"]
    fresh = fresh_harmless()
    fresh = fresh[:lim] if lim else fresh
    assert not ({q["id"] for q in fresh} & {q["id"] for q in sets["heldout"] + sets["harmless"]})
    max_new = int(a.get("max_new", 96))
    b = ctx.bundle()
    m = b.model
    cells, rows, limits, names = _grid(dirs, b.device)
    ctx.write_config(dir_exp=src_exp, dir_sha256=sha, layers=LAYERS_B, strengths=STRENGTHS_B, modes=MODES,
                     max_new=max_new, norms={l: dirs[l][1] for l in LAYERS_B}, heldout_ids=[q["id"] for q in held],
                     fresh_ids=[q["id"] for q in fresh])
    ctx.progress.update(items_total=len(held), items_done=0, force=True)
    priv = ctx.private_dir()
    ck = ItemCheckpoints(priv / "partial", ["E5b-refusal", [q["id"] for q in held], cells, max_new, sha])
    done = ck.map(held, lambda q: forced_batch(m, chat(q["text"]), rows, LAYERS_B, max_new, limits),
                  key=lambda q: q["id"], on_done=ctx.progress.advance)
    recs, gens = _gen_records(m, held, done, names, "refusal", is_refusal)
    _write_gens(priv / "generations.jsonl", gens)
    df = pd.DataFrame(recs)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    res, sel_in = _score(df, names, cells, "refusal")
    sel = select_b(sel_in)
    if ctx.smoke and a.get("smoke_force"):
        sel = _smoke_force(ctx, res, "refusal")
    out = {"cells": res, "selection": sel, "norms": {l: dirs[l][1] for l in LAYERS_B}, "n_heldout": len(held),
           "dir_exp": src_exp, "dir_sha256": sha, "rate_key": "refusal"}
    if sel["eligible"]:
        sel["refusal"] = sel["rate"]
        out["confirm"] = _confirm(ctx, m, fresh, lambda q: chat(q["text"]), sel, dirs, "refusal", is_refusal, max_new,
                                  "confirm")
        sel["confirmed"] = out["confirm"]["ok"]
        bt = get_bundle(weights=resolve_weights(a.get("tofu_weights", "ckpt:E2f/tofu_full"), ctx.exp_id))
        dt = directions(bt.model, sets, [sel["layer"]])
        save_dirs(cache_path(ctx.exp_id, "tofu"), dt, {"model": "ckpt:E2f/tofu_full", "selected_from": "E5b Step 1"})
        L = sel["layer"]
        out["tofu"] = {"layer": L, "norm": dt[L]["norm"], "raw_diff_norm": dt[L]["raw_diff_norm"],
                       "cos_with_base": float(dt[L]["dir"] @ dirs[L][0])}
    ctx.write_metrics(out)
    ctx.finish({"view": "e5b-refusal", "forget": None, "selected": [sel.get("layer"), sel.get("s"), sel.get("mode")],
                "eligible": sel["eligible"], "tier": sel["tier"]})
    ck.clear()
    return {k: sel.get(k) for k in ("layer", "s", "mode", "eligible", "tier")}


@torch.no_grad()
def ign_directions(model, sets, layers):
    from experiments.E2f.ignorance import last_resid

    U = last_resid(model, [q["question"] for q in sets["unknown"]], layers)
    K = last_resid(model, [q["question"] for q in sets["known"]], layers)
    out = {}
    for l in layers:
        d = U[l].mean(0) - K[l].mean(0)
        out[l] = {"dir": d / np.linalg.norm(d), "norm": float(np.linalg.norm(np.concatenate([U[l], K[l]]), axis=1).mean()),
                  "raw_diff_norm": float(np.linalg.norm(d))}
    return out


def _save(path, dirs, extra):
    np.savez(path, layers=np.array(list(dirs)), **{f"dir_{l}": v["dir"] for l, v in dirs.items()},
             **{f"norm_{l}": np.array(v["norm"]) for l, v in dirs.items()}, info=np.array(json.dumps(extra)))


def ignorance_task(ctx):
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    a = ctx.args
    sets = cutoff.build_sets()
    held = sets["heldout"][: int(a["limit"])] if a.get("limit") else sets["heldout"]
    max_new = int(a.get("max_new", 96))
    b = ctx.bundle()
    m = b.model
    dd = ign_directions(m, sets, LAYERS_B)
    _save(ign_cache_path(ctx.exp_id, "base"), dd, {"model": "gemma-2-2b-it", "n_unknown": len(sets["unknown"]),
                                                    "n_known": len(sets["known"])})
    dirs = {l: (dd[l]["dir"].astype(np.float32), dd[l]["norm"]) for l in LAYERS_B}
    cells, rows, limits, names = _grid(dirs, b.device)
    ctx.write_config(layers=LAYERS_B, strengths=STRENGTHS_B, modes=MODES, max_new=max_new,
                     norms={l: dd[l]["norm"] for l in LAYERS_B}, heldout_ids=[q["id"] for q in held],
                     n_unknown=len(sets["unknown"]), n_known=len(sets["known"]))
    ctx.progress.update(items_total=len(held), items_done=0, force=True)
    priv = ctx.private_dir()
    ck = ItemCheckpoints(priv / "partial", ["E5b-ign", [q["id"] for q in held], cells, max_new,
                                            {l: dd[l]["norm"] for l in LAYERS_B}])
    done = ck.map(held, lambda q: forced_batch(m, open_prompt(q["question"]), rows, LAYERS_B, max_new, limits),
                  key=lambda q: q["id"], on_done=ctx.progress.advance)
    recs, gens = _gen_records(m, held, done, names, "idk", cutoff.is_idk)
    _write_gens(priv / "generations.jsonl", gens)
    df = pd.DataFrame(recs)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    res, sel_in = _score(df, names, cells, "idk")
    sel = select_b(sel_in)
    if ctx.smoke and a.get("smoke_force"):
        sel = _smoke_force(ctx, res, "idk")
    out = {"cells": res, "selection": sel, "norms": {l: dd[l]["norm"] for l in LAYERS_B},
           "raw_diff_norms": {l: dd[l]["raw_diff_norm"] for l in LAYERS_B}, "n_heldout": len(held), "rate_key": "idk"}
    if sel["eligible"]:
        L = sel["layer"]
        bt = get_bundle(weights=resolve_weights(a.get("tofu_weights", "ckpt:E2f/tofu_full"), ctx.exp_id))
        dt = ign_directions(bt.model, sets, [L])
        _save(ign_cache_path(ctx.exp_id, "tofu"), dt, {"model": "ckpt:E2f/tofu_full", "selected_from": "E5b Step 2"})
        out["tofu"] = {"layer": L, "norm": dt[L]["norm"], "cos_with_base": float(dt[L]["dir"] @ dirs[L][0])}
    ctx.write_metrics(out)
    ctx.finish({"view": "e5b-ignorance", "forget": None, "selected": [sel.get("layer"), sel.get("s"), sel.get("mode")],
                "eligible": sel["eligible"], "tier": sel["tier"]})
    ck.clear()
    return {k: sel.get(k) for k in ("layer", "s", "mode", "eligible", "tier")}
