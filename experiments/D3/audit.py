"""D3 baked-model audit for D1-local and D2: SAE activity on forget-corpus rows (fire rate of the DSG
features, overall per-feature mean activation), and which features changed most vs the base model.
A4-style probes on these models run as capture/probe tasks in the same config."""
import numpy as np
import torch

from dsgx.data import activation_cache as ac
from dsgx.methods import dsg
from dsgx.models.loader import get_bundle
from dsgx.run import resolve_weights


@torch.no_grad()
def _feature_means(b, rows):
    s = None
    n = 0
    for r in rows:
        t = torch.tensor(np.asarray(r)[None].astype(np.int64), device=b.device)
        _, c = b.model.run_with_cache(t, stop_at_layer=b.layer + 1, names_filter=b.hook_name)
        a = b.sae.encode(c[b.hook_name])[0, 1:].float()
        s = a.sum(0) if s is None else s + a.sum(0)
        n += a.shape[0]
    return (s / n).cpu().numpy()


def task(ctx):
    a = ctx.args
    nrows = int(a.get("rows", 64))
    base = get_bundle()
    cache = ac.ActivationCache(ac.build_cache(base, "bio-forget-corpus", "wikitext", 0))
    feats = dsg.select_features(cache, 20, 95)
    rows = np.asarray(cache.tokens("forget"))[:nrows]
    mu_base = _feature_means(base, rows)
    # Release the base model before loading edited models (one 2B model fits on a 16 GB card).
    import gc

    from dsgx.models import loader
    del base
    loader.clear(); gc.collect(); torch.cuda.empty_cache()
    out = {}
    for tag, w in a.get("models", {"d1": "ckpt:D1/undo_a0.3", "d2": "ckpt:D2/nullspace"}).items():
        b = get_bundle(weights=resolve_weights(w, ctx.exp_id))
        w_meta = b.meta["weights"]
        mu = _feature_means(b, rows)
        del b
        loader.clear(); gc.collect(); torch.cuda.empty_cache()
        d = mu - mu_base
        top = np.argsort(-np.abs(d))[:20]
        out[tag] = {"weights": w_meta, "dsg_feature_mean_act": float(mu[feats].mean()),
                    "dsg_feature_mean_act_base": float(mu_base[feats].mean()),
                    "dsg_feature_ratio": float(mu[feats].sum() / max(mu_base[feats].sum(), 1e-9)),
                    "top_changed": [{"feature": int(f), "delta": float(d[f]), "is_dsg": int(f) in set(feats)} for f in top],
                    "overlap_top20_with_dsg": len(set(map(int, top)) & set(feats))}
        ctx.progress.advance(1)
    ctx.write_metrics({"rows": nrows, "features": feats, "models": out})
    ctx.finish({"view": "d3-audit", "forget": None})
    return {k: v["dsg_feature_ratio"] for k, v in out.items()}
