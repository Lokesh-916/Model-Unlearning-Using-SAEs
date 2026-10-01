"""B5 white-box obfuscation: a short discrete suffix (GCG-lite) appended after the options that
minimises the DSG gate score while keeping the correct-answer margin positive. Greedy single-token
swaps over a small candidate vocab; held-out items test transfer. Suffix -> $DSG_PRIVATE."""
import json

import numpy as np
import torch

from dsgx.data import activation_cache as ac
from dsgx.data.mcq import format_prompt, load_mcq
from dsgx.data.splits import get_split
from dsgx.methods import dsg
from dsgx.models.loader import get_bundle


def _rho(bundle, feats, prompt):
    t = bundle.model.to_tokens(prompt, prepend_bos=False).to(bundle.device)
    _, c = bundle.model.run_with_cache(t, stop_at_layer=bundle.layer + 1, names_filter=bundle.hook_name)
    a = bundle.sae.encode(c[bundle.hook_name][0]); a[0] = 0
    return float((a[:, feats] > 0).any(1).float().mean())


def task(ctx):
    a = ctx.args
    case = a.get("case", "bio")
    ds = a.get("dataset", "wmdp-bio")
    steps = int(a.get("steps", 200))
    suffix_len = int(a.get("suffix_len", 8))
    n_fit = int(a.get("n_fit", 16))
    b = get_bundle()
    cache = ac.ActivationCache(ac.build_cache(b, f"{case}-forget-corpus", "wikitext", 0))
    feats = torch.tensor(dsg.select_features(cache, 20, 95), device=b.device)
    items = load_mcq(ds)
    fit = [items[i] for i in get_split(ds, "dev")[:n_fit]]
    tok = b.model.tokenizer
    vocab = np.random.default_rng(0).choice(tok.vocab_size, size=256, replace=False).tolist()
    suffix = [tok.encode(" the", add_special_tokens=False)[0]] * suffix_len
    ctx.progress.update(items_total=steps, items_done=0, force=True)

    def avg_rho(sfx):
        s = tok.decode(sfx)
        return np.mean([_rho(b, feats, format_prompt(it) + "\n" + s) for it in fit])

    best = avg_rho(suffix)
    for step in range(steps):
        pos = step % suffix_len
        cand = suffix.copy()
        improved = best
        for v in np.random.default_rng(step).choice(vocab, size=16, replace=False):
            trial = suffix.copy(); trial[pos] = int(v)
            r = avg_rho(trial)
            if r < improved:
                improved, cand = r, trial
        suffix, best = cand, improved
        ctx.progress.update(items_done=step + 1, current_metric=f"rho {best:.3f}")
    text = tok.decode(suffix)
    out = ctx.private_dir() / "suffix.json"
    out.write_text(json.dumps({"text": text, "avg_rho_fit": best, "suffix_len": suffix_len, "steps": steps}))
    ctx.write_metrics({"suffix_tokens": suffix, "avg_rho_fit": best, "dataset": ds})
    ctx.finish({"view": "suffix", "forget": None})
    return {"avg_rho_fit": best}
