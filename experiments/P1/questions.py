"""P1 questions task (GPU): SAE-CUSUM score of every DEV question at every layer (batch 1, exact positions).
Conditions cyber_clean / cyber_pad400 / cyber_split3 (WMDP-Cyber dev, prompts from the shared attack registry exactly
as the MCQ runs), benign_cs (5 subjects), mmlu (Cyber utility set). items.parquet: keys, lengths, scores; no text."""
import json

import pandas as pd
import torch

from dsgx.itemckpt import ItemCheckpoints

from experiments.E4a.common import LAYERS, RELEASE, SAES, work_dir
from experiments.E4a.questions import cusum_max
from experiments.P1.common import ATTACKS, question_items


def task(ctx):
    from dsgx.attacks.registry import make_attack
    from dsgx.models.loader import get_sae

    a = ctx.args
    exp = a.get("fit_exp", ctx.exp_id)
    b = ctx.bundle()
    m = b.model
    fdir = work_dir(exp, "fit")
    dets = []
    for tag, layer, sid in SAES:
        rec = json.loads((fdir / f"{tag}.json").read_text())
        if not rec["features"]:
            continue
        dets.append((tag, layer, get_sae(RELEASE, sid, b.dtype, b.device),
                     torch.tensor(rec["features"], device=b.device),
                     torch.tensor(rec["w1"], dtype=torch.float32, device=b.device),
                     torch.tensor(rec["w0"], dtype=torch.float32, device=b.device)))
    attacks = {k: make_attack({**v, "_exp_id": ctx.exp_id}) for k, v in ATTACKS.items()}
    items = question_items(a.get("limit"))
    names = {f"blocks.{l}.hook_resid_post" for l in LAYERS}
    ctx.write_config(fit_exp=exp, n_items=len(items), tags=[d[0] for d in dets])
    ctx.progress.update(items_total=len(items), items_done=0, force=True)

    @torch.no_grad()
    def one(it):
        key, cond, (atk, item) = it
        prompt, info = attacks[atk].prompt(item)
        if prompt is None:
            return {"key": key, "cond": cond, "item_id": item.item_id, "skip": str(info.get("skip"))}
        t = m.to_tokens(prompt, prepend_bos=False).to(b.device)
        n = t.shape[1]
        _, cache = m.run_with_cache(t, stop_at_layer=max(LAYERS) + 1, names_filter=lambda nm: nm in names)
        row = {"key": key, "cond": cond, "item_id": item.item_id, "skip": "", "n_tok": int(n)}
        for tag, layer, sae, feats, w1, w0 in dets:
            x = (sae.encode(cache[f"blocks.{layer}.hook_resid_post"][0, :n])[1:, feats] > 0).float()
            row[f"a-sae-cusum|{tag}"] = cusum_max((x @ w1 + (1 - x) @ w0).tolist())
        return row

    ck = ItemCheckpoints(ctx.run_dir() / "partial", ["P1-questions", exp, [k for k, _, _ in items]])
    rows = ck.map(items, one, key=lambda it: it[0], on_done=ctx.progress.advance)
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    out = {"n": int(len(df)), "n_skipped": int((df["skip"] != "").sum()),
           "by_cond": df[df["skip"] == ""].groupby("cond").size().to_dict()}
    ctx.write_metrics(out)
    ctx.finish({"view": "p1-questions", "forget": None})
    ck.clear()
    return out
