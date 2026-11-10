"""E3 rank task: per SAE, rank features by attribution patching on WMDP-Bio DEV (fit half) items.

Effect of zero-ablating feature f on the correct-answer logit margin m = s[gold] - max_{j != gold} s[j]
(s = the 4 answer-letter logits after Gemma's softcap, max over "A" / " A" as in the scorer; the softmax normaliser
cancels, so this is also the log-prob margin), linearised: d m ~ -sum_t a_{t,f} * dm/da_{t,f} over positions t > 0
(BOS excluded, the same positions the ablation touches). Causal score = mean over fit items of
sum_t a_{t,f} * dm/da_{t,f} (positive = the feature supports the correct answer); ranked descending. The SAE is
spliced in with its error detached and added back, so the forward pass is unchanged.

Also per SAE: frequency ranking (token fire rate on the WMDP-Bio dev-fit prompts minus the fire rate on an MMLU
utility dev sample), a seeded random set of 50 features among those that fire at least once on the fit prompts,
and (DSG's layer-3 SAE only) DSG's own 20 detection features from the bio activation cache. Feature files go to
$DSG_CACHE/features/E3/<exp>/<tag>__<kind>.json. Ids and metrics only. Item-level resume: dsgx.itemckpt.
"""
import json

import numpy as np
import torch

from dsgx.data import activation_cache as ac
from dsgx.data.mcq import format_prompt, load_mcq, utility_subjects
from dsgx.data.splits import get_split
from dsgx.eval.mcq_eval import ANSWER_STRINGS
from dsgx.itemckpt import ItemCheckpoints
from dsgx.methods import dsg
from dsgx.util import atomic_write_json

from experiments.E3.common import N_RANDOM, RELEASE, SAES, TOPK, features_dir, fit_heldout

N_KEEP = 500  # ranked ids stored per list


def _splice(sae, store, grad: bool):
    def hook(resid, hook=None):
        with torch.no_grad():
            acts = sae.encode(resid)
            err = resid - sae.decode(acts)
        a = acts.detach().clone().requires_grad_(grad)
        store["a"] = a
        return sae.decode(a) + err
    return hook


def _cap(model, x):
    sc = getattr(model.cfg, "output_logits_soft_cap", None)
    return sc * torch.tanh(x / sc) if sc else x


def item_stats(model, sae, hook_name, prompt: str, gold: int | None, ans_tok):
    """-> dict(attr [F] or None, fire [F] counts over t > 0, ntok). gold None = forward only (retain sample)."""
    store = {}
    t = model.to_tokens(prompt, prepend_bos=False).to(model.cfg.device)
    grad = gold is not None
    with torch.set_grad_enabled(grad), model.hooks(fwd_hooks=[(hook_name, _splice(sae, store, grad))]):
        resid = model(t, return_type=None, stop_at_layer=model.cfg.n_layers)
        out = {"ntok": int(t.shape[1] - 1)}
        a = store["a"]
        out["fire"] = (a.detach()[0, 1:] > 0).sum(0).to(torch.int32).cpu().numpy()
        out["attr"] = None
        if grad:
            x = model.ln_final(resid)[0, -1]
            W, b = model.unembed.W_U[:, ans_tok], model.unembed.b_U[ans_tok]
            s = _cap(model, x @ W + b).float().reshape(2, 4).max(0).values
            other = torch.cat([s[:gold], s[gold + 1:]])
            margin = s[gold] - other.max()
            margin.backward()
            out["attr"] = (a.detach()[0, 1:].float() * a.grad[0, 1:].float()).sum(0).cpu().numpy().astype(np.float32)
            out["margin"] = float(margin.detach())
    del store
    return out


def retain_sample(n: int, seed: int = 0):
    pool = [(d, i) for d in utility_subjects("bio") for i in get_split(d, "dev")]
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(pool), size=min(n, len(pool)), replace=False)
    return [pool[k] for k in sorted(pick)]


def _jac(x, y):
    x, y = set(map(int, x)), set(map(int, y))
    return len(x & y) / len(x | y) if (x | y) else float("nan")


def task(ctx):
    from dsgx.models.loader import get_sae

    a = ctx.args
    tags = a.get("saes") or [t for t, _ in SAES]
    n_fit = a.get("n_fit")
    n_retain = int(a.get("n_retain", 640))
    b = ctx.bundle()  # base model (+ DSG's layer-3 SAE; the bundle's SAE is not used here)
    m = b.model
    m.requires_grad_(False)
    ans_tok = m.to_tokens(ANSWER_STRINGS, prepend_bos=False).flatten()
    wmdp = load_mcq("wmdp-bio")
    fit, held = fit_heldout(get_split("wmdp-bio", "dev"))
    if n_fit:
        fit = fit[: int(n_fit)]
    rs = retain_sample(n_retain)
    ctx.write_config(saes=tags, n_fit=len(fit), n_heldout=len(held), n_retain=len(rs))
    total = len(tags) * (len(fit) + len(rs))
    ctx.progress.update(items_total=total, items_done=0, force=True)
    fdir = features_dir(ctx.exp_id)
    summary = {}
    for tag in tags:
        sid = dict(SAES)[tag]
        out_p = fdir / f"{tag}.json"
        if out_p.exists():
            summary[tag] = json.loads(out_p.read_text())["metrics"]
            ctx.progress.advance(len(fit) + len(rs))
            continue
        sae = get_sae(RELEASE, sid, b.dtype, b.device)
        sae.requires_grad_(False)
        hook_name = sae.cfg.metadata.hook_name
        layer = int(hook_name.split(".")[1])
        pdir = ctx.run_dir() / "partial" / tag
        ck_f = ItemCheckpoints(pdir / "fit", ["E3-rank-fit", tag, sid, fit])
        ck_r = ItemCheckpoints(pdir / "retain", ["E3-rank-retain", tag, sid, rs])
        F = sae.W_dec.shape[0]
        fit_out = ck_f.map(fit, lambda i: item_stats(m, sae, hook_name, format_prompt(wmdp[i]), wmdp[i].answer, ans_tok),
                           key=str, on_done=ctx.progress.advance)
        ret_out = ck_r.map(rs, lambda di: item_stats(m, sae, hook_name, format_prompt(load_mcq(di[0])[di[1]]), None, ans_tok),
                           key=lambda di: f"{di[0]}:{di[1]}", on_done=ctx.progress.advance)
        attr = np.mean([o["attr"] for o in fit_out], axis=0)
        fr_f = np.sum([o["fire"] for o in fit_out], axis=0) / max(1, sum(o["ntok"] for o in fit_out))
        fr_r = np.sum([o["fire"] for o in ret_out], axis=0) / max(1, sum(o["ntok"] for o in ret_out))
        causal = np.argsort(-attr, kind="stable")[:N_KEEP]
        freq = np.argsort(-(fr_f - fr_r), kind="stable")[:N_KEEP]
        alive = np.where(fr_f > 0)[0]
        rng = np.random.default_rng(1000 + layer + (100 if tag.endswith("dsg") else 0))
        rand = sorted(int(x) for x in rng.choice(alive, size=min(N_RANDOM, len(alive)), replace=False))
        files = {f"causal{k}": [int(x) for x in causal[:k]] for k in TOPK}
        files.update({f"freq{k}": [int(x) for x in freq[:k]] for k in TOPK})
        files[f"rand{N_RANDOM}"] = rand
        dsg20 = None
        if tag.endswith("dsg"):
            cache = ac.open_cache(ac.model_tag(b), RELEASE, sid, "bio-forget-corpus", "wikitext", 0, 1024, 1024)
            dsg20 = [int(x) for x in dsg.select_features(cache, 20, 95)]
            files["dsg20"] = dsg20
        for kind, feats in files.items():
            atomic_write_json(fdir / f"{tag}__{kind}.json", {"features": feats, "tag": tag, "sae_id": sid, "kind": kind})
        met = {"tag": tag, "sae_id": sid, "layer": layer, "n_features_total": int(F), "n_fit": len(fit),
               "n_retain": len(rs), "n_alive_fit": int(len(alive)),
               "mean_margin_fit": float(np.mean([o["margin"] for o in fit_out])),
               "jaccard_freq_vs_causal": {str(k): _jac(causal[:k], freq[:k]) for k in TOPK},
               "attr_top": [float(attr[i]) for i in causal[:max(TOPK)]],
               "attr_sum_top": {str(k): float(attr[causal[:k]].sum()) for k in TOPK},
               "rand_attr_sum": float(attr[rand].sum())}
        if dsg20 is not None:
            met["jaccard_dsg20_vs_causal"] = {str(k): _jac(causal[:k], dsg20) for k in TOPK}
            met["dsg20_attr_sum"] = float(attr[dsg20].sum())
            met["dsg20_causal_rank"] = sorted(int(np.where(np.argsort(-attr, kind="stable") == f)[0][0]) for f in dsg20)
        atomic_write_json(out_p, {"metrics": met, "causal": [int(x) for x in causal], "freq": [int(x) for x in freq],
                                  "fit_ids": [int(i) for i in fit], "heldout_ids": [int(i) for i in held],
                                  "retain_ids": [f"{d}:{i}" for d, i in rs]})
        summary[tag] = met
        ck_f.clear(); ck_r.clear()
        del sae
        torch.cuda.empty_cache()
    ctx.write_metrics({"saes": summary})
    ctx.finish({"view": "e3-rank", "n_saes": len(summary)})
    return {t: s["jaccard_freq_vs_causal"] for t, s in summary.items()}
