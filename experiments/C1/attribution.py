"""C1 causal feature selection: rank SAE features by attribution patching (activation x gradient of
the correct-answer logprob) on DEV forget items, and by chi-square (fires-anywhere, forget vs
retain) from the activation cache. Compare with the DSG score: feature overlap (Jaccard) and gate
AUROC. Writes feature files to $DSG_CACHE/features/C1_<method>.json for use via gated(features_file)."""
import json

import numpy as np
import torch

from dsgx import paths
from dsgx.data import activation_cache as ac
from dsgx.data.mcq import FORGET_DATASET, format_prompt, load_mcq
from dsgx.data.splits import get_split
from dsgx.methods import dsg


def _feat_file(name, feats, scores, extra):
    p = paths.cache_dir() / "features" / f"{name}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"features": [int(f) for f in feats], "scores": [float(s) for s in scores], **extra}))
    return p


def attribution_scores(bundle, items, topk=200):
    """Mean over items of relu(act) * d logp(correct letter) / d act, at the SAE hook, last token."""
    from dsgx.eval.mcq_eval import ANSWER_STRINGS

    m, sae = bundle.model, bundle.sae
    ans = m.to_tokens(ANSWER_STRINGS, prepend_bos=False).flatten()
    acc = torch.zeros(sae.W_dec.shape[0], device=bundle.device)
    torch.set_grad_enabled(True)
    for it in items:
        t = m.to_tokens(format_prompt(it), prepend_bos=False).to(bundle.device)
        feats = {}

        def hook(resid, hook):
            a = sae.encode(resid)
            a.retain_grad()
            feats["a"] = a
            return sae.decode(a) + (resid - sae.decode(a))

        m.reset_hooks()
        logits = m.run_with_hooks(t, fwd_hooks=[(bundle.hook_name, hook)])[0, -1]
        gold = ans[it.answer]  # "A".."D" index via first 4
        lp = torch.log_softmax(logits.float(), -1)[gold]
        m.zero_grad(set_to_none=True)
        lp.backward()
        a = feats["a"][0]
        g = a.grad[0] if a.grad.dim() == 3 else a.grad
        acc += (torch.relu(a) * a.grad)[-1].abs().detach()
        m.reset_hooks()
    torch.set_grad_enabled(False)
    s = (acc / len(items)).cpu().numpy()
    order = np.argsort(-s)
    return order[:topk], s


def chi2_scores(cache, feats_pool=None, topk=200):
    f = cache.stats("forget")["fire_rate"]
    r = cache.stats("retain")["fire_rate"]
    nf = cache.stats("forget")["n_tokens"].item()
    nr = cache.stats("retain")["n_tokens"].item()
    # 2x2 chi-square of (fires) x (forget vs retain)
    a = f * nf; b = nf - a; c = r * nr; d = nr - c
    n = nf + nr
    denom = (a + b) * (c + d) * (a + c) * (b + d) + 1e-9
    chi = n * (a * d - b * c) ** 2 / denom
    order = np.argsort(-chi)
    return order[:topk], chi


def task(ctx):
    a = ctx.args
    case = a.get("case", "bio")
    n = int(a.get("n_features", 20))
    nfit = int(a.get("n_fit", 64))
    from dsgx.models.loader import get_bundle

    b = get_bundle()
    cache = ac.ActivationCache(ac.build_cache(b, f"{case}-forget-corpus", "wikitext", 0))
    fd = FORGET_DATASET[case]
    items = [load_mcq(fd)[i] for i in get_split(fd, "dev")[:nfit]]
    ctx.progress.update(items_total=nfit + 2, items_done=0, force=True)
    attr_order, attr_s = attribution_scores(b, items, topk=400)
    ctx.progress.advance(nfit)
    chi_order, chi_s = chi2_scores(cache, topk=400)
    ctx.progress.advance(1)
    dsg_feats = dsg.select_features(cache, n, 95)
    attr_top, chi_top = list(attr_order[:n]), list(chi_order[:n])
    _feat_file(f"C1_attr_{case}", attr_top, attr_s[attr_top], {"method": "attribution", "case": case})
    _feat_file(f"C1_chi2_{case}", chi_top, chi_s[chi_top], {"method": "chi2", "case": case})
    _feat_file(f"C1_dsg_{case}", dsg_feats, [0] * n, {"method": "dsg", "case": case})
    ctx.progress.advance(1)

    def jac(x, y):
        x, y = set(map(int, x)), set(map(int, y))
        return len(x & y) / len(x | y)

    m = {"n_features": n, "overlap": {"attr_vs_dsg": jac(attr_top, dsg_feats),
         "chi2_vs_dsg": jac(chi_top, dsg_feats), "attr_vs_chi2": jac(attr_top, chi_top)},
         "features": {"attr": attr_top, "chi2": chi_top, "dsg": [int(x) for x in dsg_feats]}}
    ctx.write_metrics(m)
    ctx.finish({"view": "c1-selection", "forget": None})
    return m["overlap"]
