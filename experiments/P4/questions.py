"""P4 questions task (GPU): scores of every E4a DEV question for every P4 setting (batch 1, exact positions).

Prompts: experiments.E4a.common.question_items + the shared attack suite, exactly as E4a's questions task (built in
memory only). Per prompt: model to layer 9; for each SAE (L07 / L08 / L09 16k, L08 65k) the per-token fire bits of
its top-40 features (non-BOS positions, stored packed) and the CUSUM scores of the P4 settings, computed as E4a
(float32 matmul on the device over the gathered top-k features, then cusum_max in python floats; drift subtracted
per token). Item-level resume (dsgx.itemckpt). items.parquet: ids, scores and packed bits only, no text.
"""
import json

import numpy as np
import pandas as pd
import torch

from dsgx.itemckpt import ItemCheckpoints

from experiments.E4a.common import ATTACKS, question_items
from experiments.P4.common import RELEASE, SAE16, SAE65, SETTINGS, cusum_max, score_col, work_dir


class Scorer:
    def __init__(self, exp, bundle):
        from dsgx.models.loader import get_sae

        fdir = work_dir(exp, "fit")
        self.delta = json.loads((fdir / "meta.json").read_text())["delta"]
        dev = bundle.device
        saes = {t: (l, s) for t, (l, s) in SAE16.items()}
        saes[SAE65[0]] = (SAE65[1], SAE65[2])
        self.sae, self.fit, self.layer = {}, {}, {}
        for tag, (layer, sid) in saes.items():
            self.sae[tag] = get_sae(RELEASE, sid, bundle.dtype, dev)
            assert self.sae[tag].cfg.metadata.hook_name == f"blocks.{layer}.hook_resid_post"
            self.fit[tag] = json.loads((fdir / f"{tag}.json").read_text())
            self.layer[tag] = layer
        # one entry per distinct score column
        self.cols = {}
        for s, (tag, k, dm, _) in SETTINGS.items():
            f = self.fit[tag]
            if len(f["features"]) < k:
                continue  # smoke only (tiny doc sample)
            self.cols[score_col(s)] = (tag, torch.tensor(f["features"][:k], device=dev),
                                       torch.tensor(f["w1"][:k], dtype=torch.float32, device=dev),
                                       torch.tensor(f["w0"][:k], dtype=torch.float32, device=dev), dm * self.delta)
        self.top = {t: torch.tensor(self.fit[t]["features"], device=dev) for t in self.fit}

    @torch.no_grad()
    def score(self, cache, n):
        acts = {t: self.sae[t].encode(cache[f"blocks.{self.layer[t]}.hook_resid_post"][0, :n]) for t in self.sae}
        out = {}
        for col, (tag, feats, w1, w0, drift) in self.cols.items():
            x = (acts[tag][1:, feats] > 0).float()
            out[col] = cusum_max((x @ w1 + (1 - x) @ w0).tolist(), drift)
        for t, top in self.top.items():
            bits = (acts[t][1:, top] > 0).cpu().numpy()
            out[f"bits|{t}"] = np.packbits(bits, axis=1).tobytes()
            out[f"nbits|{t}"] = int(bits.shape[1])
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
    names = {f"blocks.{l}.hook_resid_post" for l in sc.layer.values()}
    ctx.write_config(fit_exp=exp, n_items=len(items), delta=sc.delta, score_cols=sorted(sc.cols))
    ctx.progress.update(items_total=len(items), items_done=0, force=True)

    @torch.no_grad()
    def one(it):
        key, cond, (atk, item) = it
        prompt, info = attacks[atk].prompt(item)
        if prompt is None:
            return {"key": key, "cond": cond, "skip": str(info.get("skip"))}
        t = m.to_tokens(prompt, prepend_bos=False).to(b.device)
        n = t.shape[1]
        _, cache = m.run_with_cache(t, stop_at_layer=max(sc.layer.values()) + 1, names_filter=lambda nm: nm in names)
        row = {"key": key, "cond": cond, "skip": "", "n_tok": int(n)}
        row.update(sc.score(cache, n))
        return row

    ck = ItemCheckpoints(ctx.run_dir() / "partial", ["P4-questions", exp, sorted(sc.cols), [k for k, _, _ in items]])
    rows = ck.map(items, one, key=lambda it: it[0], on_done=ctx.progress.advance)
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    m_out = {"n": int(len(df)), "n_skipped": int((df["skip"] != "").sum()),
             "by_cond": df[df["skip"] == ""].groupby("cond").size().to_dict()}
    ctx.write_metrics(m_out)
    ctx.finish({"view": "p4-questions", "forget": None})
    ck.clear()
    return m_out
