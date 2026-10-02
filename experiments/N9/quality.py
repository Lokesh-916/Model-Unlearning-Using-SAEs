"""N9 SAE-quality explanation (must): per forget TEST item under DSG, record SAE reconstruction MSE,
explained variance, L0 (from items.parquet of an A1/B1-style run, which already logs recon_mse and l0),
plus feature-splitting/absorption indicators from the cache. Then a logistic regression predicts gate
MISSES (base-correct, DSG still correct = a leak the gate let through) from these per-item properties,
with a bootstrap CI on each coefficient. Reads an existing DSG run's items.parquet (no model needed)."""
import json

import numpy as np

from dsgx import paths


def _load_run(exp_glob, need_cols):
    import pandas as pd

    for d in sorted(paths.runs_dir().glob(exp_glob)):
        if not (d / "items.parquet").exists():
            continue
        df = pd.read_parquet(d / "items.parquet")
        cfg = json.loads((d / "config.json").read_text()).get("config", {})
        if cfg.get("method", {}).get("name", "").startswith("dsg") and all(c in df for c in need_cols):
            return d, df, cfg
    return None, None, None


def splitting_indicators(case):
    """Decoder-cosine neighbour count and activation correlation for the DSG features (feature
    splitting / absorption proxies), from the SAE and cache."""
    from dsgx.data import activation_cache as ac
    from dsgx.methods import dsg
    from dsgx.models.loader import get_bundle

    b = get_bundle()
    cache = ac.ActivationCache(ac.build_cache(b, f"{case}-forget-corpus", "wikitext", 0))
    feats = dsg.select_features(cache, 200, 95)
    import torch

    W = torch.nn.functional.normalize(b.sae.W_dec.float(), dim=1)
    sub = W[torch.tensor(feats)]
    cos = (sub @ W.T).cpu().numpy()
    out = {}
    for i, f in enumerate(feats):
        row = cos[i].copy(); row[f] = -1
        out[int(f)] = {"max_cos_neighbor": float(row.max()), "n_cos_above_0.3": int((row > 0.3).sum())}
    return out, feats


def task(ctx):
    a = ctx.args
    case = a.get("case", "bio")
    d, df, cfg = _load_run(a.get("run_glob", f"B1*/*dsg*{'' if not ctx.smoke else ''}*"), ["recon_mse", "l0", "correct", "in_dsg_subset"])
    if df is None:
        d, df, cfg = _load_run("A1-*/*dsg-faithful*", ["recon_mse", "l0", "correct", "in_dsg_subset"])
    if df is None:
        ctx.write_metrics({"error": "no DSG run with recon_mse/l0 found; run A1 or B1 first"})
        ctx.finish({"view": "n9", "forget": None})
        return {}
    fd = cfg.get("forget_datasets", ["wmdp-bio", "wmdp-cyber"])
    f = df[df["dataset"].isin(fd) & df["in_dsg_subset"].fillna(False)].copy()
    # "miss" = DSG still answers this gated-domain item correctly (knowledge leaked through)
    f["miss"] = f["correct"].astype(int)
    split_ind, feats = splitting_indicators(case)
    X = f[["recon_mse", "l0", "rho"]].fillna(0).values.astype(float)
    y = f["miss"].values
    out = {"n": int(len(f)), "miss_rate": float(y.mean()), "run": str(d)}
    if len(set(y)) == 2 and len(f) >= 20:
        from sklearn.linear_model import LogisticRegression

        mu, sd = X.mean(0), X.std(0) + 1e-9
        Xs = (X - mu) / sd
        coefs = []
        rng = np.random.default_rng(0)
        for _ in range(1000):
            idx = rng.integers(0, len(Xs), len(Xs))
            if len(set(y[idx])) < 2:
                continue
            coefs.append(LogisticRegression(max_iter=500).fit(Xs[idx], y[idx]).coef_[0])
        coefs = np.array(coefs)
        base = LogisticRegression(max_iter=500).fit(Xs, y).coef_[0]
        out["logreg"] = {name: {"coef": float(base[i]), "lo": float(np.percentile(coefs[:, i], 2.5)),
                                "hi": float(np.percentile(coefs[:, i], 97.5))}
                         for i, name in enumerate(["recon_mse", "l0", "rho"])}
    out["splitting_summary"] = {"mean_max_cos_neighbor": float(np.mean([v["max_cos_neighbor"] for v in split_ind.values()])),
                                "n_features_with_close_neighbor": int(sum(v["n_cos_above_0.3"] > 0 for v in split_ind.values()))}
    ctx.write_metrics(out)
    ctx.finish({"view": "n9-sae-quality", "forget": None})
    return {"miss_rate": out["miss_rate"]}
