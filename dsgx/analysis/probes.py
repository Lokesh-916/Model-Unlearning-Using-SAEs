"""Knowledge-depth tools (A4, D3, A7): residual capture, logit lens, linear answer probes.

capture_task (GPU): for MCQ items of `datasets` on dev+test, capture the residual stream at the
    last prompt token for every layer under a method (base / dsg-faithful / gated / ...), plus a
    per-layer logit-lens readout of the four answer letters. Saved under
    $DSG_CACHE/residuals/<exp>/<tag>/ (float16 [N, n_layers, d]).
probe_task (CPU): per layer, multinomial logistic regression predicting the gold letter, trained
    on dev, tested on test, for several seeds (seed = 80% bootstrap of dev + solver init), plus a
    control task with random labels fixed per item (Hewitt & Liang selectivity).
"""
import json

import numpy as np
import torch

from dsgx import paths
from dsgx.util import atomic_write_json


def _store(ctx, tag):
    d = paths.cache_dir() / "residuals" / ctx.exp_id / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


@torch.no_grad()
def capture_task(ctx):
    """args: datasets, case, limit (per dataset per split), methods: [{tag, method cfg, model}]."""
    from dsgx.data.mcq import FORGET_DATASET, format_prompt, load_mcq
    from dsgx.data.splits import get_split
    from dsgx.eval.mcq_eval import ANSWER_STRINGS
    from dsgx.methods.registry import make_method
    from dsgx.models.loader import get_bundle
    from dsgx.run import ensure_cache, resolve

    a = ctx.args
    case = a.get("case", "bio")
    datasets = a.get("datasets", [FORGET_DATASET[case]])
    plan = []
    for ds in datasets:
        items = load_mcq(ds)
        for sp in ("dev", "test"):
            ids = get_split(ds, sp)
            if a.get("limit"):
                ids = ids[: int(a["limit"])]
            plan += [(items[i], sp) for i in ids]
    ctx.progress.update(items_total=len(plan) * len(a["methods"]), items_done=0, force=True)
    out = {}
    for mspec in a["methods"]:
        tag = mspec["tag"]
        cfg = resolve({"exp_id": ctx.exp_id, "case": case, "split": "dev", "method": mspec["method"],
                       "model": mspec.get("model", a.get("model", {})), "seed": ctx.seed})
        mc = cfg["model"]
        from dsgx.run import resolve_weights

        b = get_bundle(mc["name"], mc["sae_release"], mc["sae_id"], mc["dtype"],
                       weights=resolve_weights(mc.get("weights"), ctx.exp_id))
        ensure_cache(cfg, b)
        method = make_method(cfg["method"], b, ctx.seed)
        model = b.model
        n_layers = model.cfg.n_layers
        names = [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
        ans = model.to_tokens(ANSWER_STRINGS, prepend_bos=False).flatten()
        X = np.zeros((len(plan), n_layers, model.cfg.d_model), dtype=np.float16)
        lens = np.zeros((len(plan), n_layers, 4), dtype=np.float32)
        method.install()
        try:
            for j, (it, sp) in enumerate(plan):
                t = model.to_tokens(format_prompt(it), prepend_bos=False).to(b.device)
                method.set_lengths([t.shape[1]])
                if hasattr(method, "before_forward"):
                    method.before_forward(t, [t.shape[1]])
                _, c = model.run_with_cache(t, names_filter=lambda n: n in names)
                method.pop_records()
                R = torch.stack([c[n][0, -1] for n in names])  # [n_layers, d]
                X[j] = R.float().cpu().numpy()
                lg = model.unembed(model.ln_final(R[:, None, :]))[:, 0, :].float()
                lg = lg[:, ans].reshape(n_layers, 2, 4).max(1).values
                lens[j] = lg.softmax(-1).cpu().numpy()
                ctx.progress.advance(1)
        finally:
            method.remove()
        d = _store(ctx, tag)
        np.save(d / "X.npy", X)
        np.save(d / "logit_lens.npy", lens)
        meta = {"items": [it.item_id for it, _ in plan], "split": [sp for _, sp in plan],
                "gold": [it.answer for it, _ in plan], "method": cfg["method"], "model": mc,
                "method_info": method.info}
        atomic_write_json(d / "meta.json", meta)
        gold = np.array(meta["gold"])
        ll_acc = (lens.argmax(-1) == gold[:, None]).mean(0)
        ctx.write_metrics({"tag": tag, "n": len(plan), "logit_lens_acc_by_layer": ll_acc.tolist()}, tag)
        ctx.finish({"view": f"capture:{tag}"}, tag)
        out[tag] = str(d)
    return out


def probe_task(ctx):
    """args: tags (capture tags), source_exp (default this exp), seeds, C, layers (default all)."""
    from sklearn.linear_model import LogisticRegression

    from dsgx.eval.stats import bootstrap_ci, seed_summary

    a = ctx.args
    src = a.get("source_exp", ctx.exp_id)
    if ctx.smoke and not src.endswith("-smoke"):
        src += "-smoke"
    seeds = a.get("seeds", [0, 1, 2, 3, 4])
    res = {}
    total = len(a["tags"]) * len(seeds)
    ctx.progress.update(items_total=total, items_done=0, force=True)
    for tag in a["tags"]:
        d = paths.cache_dir() / "residuals" / src / tag
        X = np.load(d / "X.npy", mmap_mode="r")
        meta = json.loads((d / "meta.json").read_text())
        y = np.array(meta["gold"])
        sp = np.array(meta["split"])
        tr, te = np.where(sp == "dev")[0], np.where(sp == "test")[0]
        layers = a.get("layers") or list(range(X.shape[1]))
        per_layer = {}
        for s in seeds:
            rng = np.random.default_rng(s)
            sub = rng.choice(tr, size=max(2, int(0.8 * len(tr))), replace=False)
            yc = np.random.default_rng(1000 + s).integers(0, 4, size=len(y))  # control labels
            for L in layers:
                Xtr = np.asarray(X[sub, L], dtype=np.float32)
                Xte = np.asarray(X[te, L], dtype=np.float32)
                mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-4
                accs = {}
                for name, lab in (("probe", y), ("control", yc)):
                    if len(set(lab[sub])) < 2:
                        accs[name] = None
                        continue
                    clf = LogisticRegression(C=float(a.get("C", 0.05)), max_iter=500, random_state=s)
                    clf.fit((Xtr - mu) / sd, lab[sub])
                    pred = clf.predict((Xte - mu) / sd)
                    accs[name] = (pred == lab[te]).astype(float)
                    if name == "probe":
                        np.save(ctx.run_dir(tag) / f"w_L{L}_s{s}.npy", clf.coef_.astype(np.float32))
                per_layer.setdefault(L, []).append(accs)
            ctx.progress.advance(1)
        summ = {}
        for L, runs in per_layer.items():
            p = [r["probe"].mean() for r in runs if r["probe"] is not None]
            c = [r["control"].mean() for r in runs if r["control"] is not None]
            summ[L] = {"probe": seed_summary(p), "control": seed_summary(c),
                       "probe_ci_seed0": bootstrap_ci(runs[0]["probe"]) if runs[0]["probe"] is not None else None,
                       "selectivity": (np.mean(p) - np.mean(c)) if p and c else None, "n_test": int(len(te))}
        ctx.write_metrics({"tag": tag, "source": str(d), "layers": summ, "chance": 0.25}, tag)
        best = max(summ.items(), key=lambda kv: kv[1]["probe"]["mean"] or 0)
        ctx.finish({"forget": {**best[1]["probe_ci_seed0"], "layer": best[0]} if best[1]["probe_ci_seed0"] else None,
                    "view": f"probe:{tag}"}, tag)
        res[tag] = best[0]
    return res
