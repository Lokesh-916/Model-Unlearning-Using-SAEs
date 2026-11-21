"""P3 questions task (GPU): every E4a DEV question scored by every P3 fit (batch 1, exact positions).

Prompts: experiments.E4a.common.question_items + the shared attack suite, as E4a / P4 (built in memory only). Per prompt:
model to layer 8, canonical L8 16k SAE, per fit: x = 1[SAE act > 0] on the fit's 20 features (non-BOS positions), score
= cusum_max of x @ w1 + (1 - x) @ w0 (float32 on the device, then python floats), exactly as E4a / P4, so the 100 % fit
reproduces E4a's logged L08 score bit for bit. Item-level resume. items.parquet: ids and scores only, no text.
"""
import json

import pandas as pd
import torch

from dsgx.itemckpt import ItemCheckpoints

from experiments.E4a.common import ATTACKS, question_items
from experiments.E4a.questions import cusum_max
from experiments.P3.common import LAYER, RELEASE, SAE_ID, work_dir


class Scorer:
    def __init__(self, exp, bundle):
        from dsgx.models.loader import get_sae

        fdir = work_dir(exp, "fit")
        self.names = sorted(json.loads((fdir / "index.json").read_text())["fits"])
        dev = bundle.device
        self.sae = get_sae(RELEASE, SAE_ID, bundle.dtype, dev)
        assert self.sae.cfg.metadata.hook_name == f"blocks.{LAYER}.hook_resid_post"
        self.cols, self.alias = {}, {}
        seen = {}
        for nm in self.names:
            f = json.loads((fdir / f"{nm}.json").read_text())
            sig = json.dumps([f["features"], f["w1"], f["w0"]])
            if sig in seen:                       # identical fit (e.g. the three 100 % seeds): same score
                self.alias[nm] = seen[sig]
                continue
            seen[sig] = nm
            self.cols[nm] = (torch.tensor(f["features"], device=dev), torch.tensor(f["w1"], dtype=torch.float32, device=dev),
                             torch.tensor(f["w0"], dtype=torch.float32, device=dev))

    @torch.no_grad()
    def score(self, cache, n):
        acts = self.sae.encode(cache[f"blocks.{LAYER}.hook_resid_post"][0, :n])
        out = {}
        for nm, (feats, w1, w0) in self.cols.items():
            x = (acts[1:, feats] > 0).float()
            out[f"score|{nm}"] = cusum_max((x @ w1 + (1 - x) @ w0).tolist())
        for nm, src in self.alias.items():
            out[f"score|{nm}"] = out[f"score|{src}"]
        return out


def task(ctx):
    from dsgx.attacks.registry import make_attack

    a = ctx.args
    exp = a.get("fit_exp", ctx.exp_id)
    b = ctx.bundle()
    m = b.model
    sc = Scorer(exp, b)
    attacks = {k: make_attack({**v, "_exp_id": "E4a"}) for k, v in ATTACKS.items()}
    items = question_items(a.get("limit"))
    name = f"blocks.{LAYER}.hook_resid_post"
    ctx.write_config(fit_exp=exp, n_items=len(items), fits=sc.names, aliases=sc.alias)
    ctx.progress.update(items_total=len(items), items_done=0, force=True)

    @torch.no_grad()
    def one(it):
        key, cond, (atk, item) = it
        prompt, info = attacks[atk].prompt(item)
        if prompt is None:
            return {"key": key, "cond": cond, "skip": str(info.get("skip"))}
        t = m.to_tokens(prompt, prepend_bos=False).to(b.device)
        n = t.shape[1]
        _, cache = m.run_with_cache(t, stop_at_layer=LAYER + 1, names_filter=lambda nm: nm == name)
        return {"key": key, "cond": cond, "skip": "", "n_tok": int(n), **sc.score(cache, n)}

    ck = ItemCheckpoints(ctx.run_dir() / "partial", ["P3-questions", exp, sc.names, [k for k, _, _ in items]])
    rows = ck.map(items, one, key=lambda it: it[0], on_done=ctx.progress.advance)
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    m_out = {"n": int(len(df)), "n_skipped": int((df["skip"] != "").sum()),
             "by_cond": df[df["skip"] == ""].groupby("cond").size().to_dict(), "n_distinct_fits": len(sc.cols)}
    ctx.write_metrics(m_out)
    ctx.finish({"view": "p3-questions", "forget": None})
    ck.clear()
    return m_out
