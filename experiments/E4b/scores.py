"""E4b Step B scores task (GPU): per-prompt gate scores on WMDP-Bio dev under the Step B attacks (batch 1).

Per prompt: L08 SAE-CUSUM (E4a's fitted detector, unchanged: fit artifacts of exp E4a), DSG's layer-3 rho score and
StreamGuard v1's CUSUM score (references; their gated runs decide independently). Prompts are built in memory by
the shared attack registry exactly as the MCQ runs build them; items.parquet stores ids, lengths and scores only.
Item-level resume (dsgx.itemckpt).
"""
import json

import pandas as pd
import torch

from dsgx.itemckpt import ItemCheckpoints

from experiments.E4a.common import DSG_SAE, RELEASE, SAES, work_dir
from experiments.E4a.questions import cusum_max
from experiments.E4b.common import E4A_EXP, GATE_COL, REF_COLS, STEP_B_ATTACKS

L08 = next(s for s in SAES if s[0] == "L08")


def items(limit=None):
    from dsgx.data.mcq import load_mcq
    from dsgx.data.splits import get_split

    wm = load_mcq("wmdp-bio")
    ids = get_split("wmdp-bio", "dev")
    if limit:
        ids = ids[:limit]
    return [(f"{a}|{wm[i].item_id}", a, wm[i]) for a in STEP_B_ATTACKS for i in ids]


def task(ctx):
    from dsgx.attacks.registry import make_attack
    from dsgx.methods import dsg, gates
    from dsgx.models.loader import get_sae

    a = ctx.args
    b = ctx.bundle(sae_id=DSG_SAE)
    m = b.model
    tag, layer, sid = L08
    rec = json.loads((work_dir(a.get("fit_exp", E4A_EXP), "fit") / f"{tag}.json").read_text())
    sae8 = get_sae(RELEASE, sid, b.dtype, b.device)
    feats = torch.tensor(rec["features"], device=b.device)
    w1 = torch.tensor(rec["w1"], dtype=torch.float32, device=b.device)
    w0 = torch.tensor(rec["w0"], dtype=torch.float32, device=b.device)
    g_rho = gates.Gate({"type": "rho", "n_features": 20, "retain_pct": 95}, b, 0)
    g_sg = gates.Gate({"type": "cusum", "n_features": 20, "retain_pct": 95}, b, 0)
    tau_rho = dsg.calibrate_tau(g_rho.cache, g_rho.features, 95)
    sg_rec = gates.calibrate(b, g_sg, "bio", 0.05, 1000, "mmlu-dev")
    attacks = {k: make_attack({**v, "_exp_id": ctx.exp_id}) for k, v in STEP_B_ATTACKS.items()}
    its = items(a.get("limit"))
    names = {f"blocks.{layer}.hook_resid_post", b.hook_name}
    ctx.write_config(fit_exp=a.get("fit_exp", E4A_EXP), n_items=len(its), tau_dsg=tau_rho,
                     sg1_threshold=sg_rec["threshold"], l08_features=rec["features"], attacks=STEP_B_ATTACKS)
    ctx.progress.update(items_total=len(its), items_done=0, force=True)

    @torch.no_grad()
    def one(it):
        key, atk, item = it
        prompt, info = attacks[atk].prompt(item)
        if prompt is None:
            return {"key": key, "attack": atk, "skip": str(info.get("skip"))}
        t = m.to_tokens(prompt, prepend_bos=False).to(b.device)
        n = t.shape[1]
        _, cache = m.run_with_cache(t, stop_at_layer=max(layer, 3) + 1, names_filter=lambda nm: nm in names)
        r8 = cache[f"blocks.{layer}.hook_resid_post"][0, :n]
        x = (sae8.encode(r8)[1:, feats] > 0).float()
        row = {"key": key, "attack": atk, "skip": "", "n_tok": int(n), "pad_len": int(info.get("pad_len") or 0),
               GATE_COL: cusum_max((x @ w1 + (1 - x) @ w0).tolist())}
        r3 = cache[b.hook_name][0]
        acts = b.sae.encode(r3)
        acts[0] = 0
        row[REF_COLS["dsg"]] = g_rho.score(r3, acts, n)["score"]
        row[REF_COLS["sg1"]] = g_sg.score(r3, acts, n)["score"]
        return row

    ck = ItemCheckpoints(ctx.run_dir() / "partial", ["E4b-scores", [k for k, _, _ in its]])
    rows = ck.map(its, one, key=lambda it: it[0], on_done=ctx.progress.advance)
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    out = {"n": int(len(df)), "n_skipped": int((df["skip"] != "").sum()),
           "by_attack": df[df["skip"] == ""].groupby("attack").size().to_dict(),
           "n_tok_median": df[df["skip"] == ""].groupby("attack")["n_tok"].median().to_dict(),
           "ref_own_thresholds": {REF_COLS["dsg"]: tau_rho, REF_COLS["sg1"]: sg_rec["threshold"]}}
    ctx.write_metrics(out)
    ctx.finish({"view": "e4b-scores", "forget": None})
    ck.clear()
    return out
