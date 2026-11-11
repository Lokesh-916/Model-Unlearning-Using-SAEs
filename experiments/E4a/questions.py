"""E4a questions task (GPU): detector scores of every DEV question, every layer (batch 1, exact positions).

Conditions: wmdp_clean / wmdp_pad400 / wmdp_split3 / wmdp_hindi (WMDP-Bio dev, prompts built by the shared attack
suite exactly as the MCQ runs; Hindi from the B3/E2d translation cache, chrF >= 40), benign_bio (5 subjects, dev),
mmlu (utility subjects, dev). Per prompt and layer: a-sae-cusum (per SAE tag), b-probe-mean, b-probe-cusum; plus
references ref-dsg-rho (DSG's 20 features, own SAE) and ref-sg1-cusum (StreamGuard v1: CUSUM on DSG's features, LLR
from the DSG activation cache). Prompts are built in memory only; items.parquet stores ids and scores, no text.
Item-level resume: dsgx.itemckpt (chunks under the run dir).
"""
import json

import numpy as np
import pandas as pd
import torch

from dsgx.itemckpt import ItemCheckpoints

from experiments.E4a.common import DSG_SAE, LAYERS, RELEASE, SAES, question_items, work_dir


def cusum_max(v) -> float:
    s, best = 0.0, 0.0
    for x in v:
        s = max(0.0, s + float(x))
        best = max(best, s)
    return best


class Detectors:
    def __init__(self, exp, bundle):
        from dsgx.models.loader import get_sae

        fdir = work_dir(exp, "fit")
        dev = bundle.device
        self.a = []
        for tag, layer, sid in SAES:
            rec = json.loads((fdir / f"{tag}.json").read_text())
            sae = get_sae(RELEASE, sid, bundle.dtype, dev)
            self.a.append((tag, layer, sae, torch.tensor(rec["features"], device=dev),
                           torch.tensor(rec["w1"], dtype=torch.float32, device=dev),
                           torch.tensor(rec["w0"], dtype=torch.float32, device=dev)))
        self.b = {}
        for layer in LAYERS:
            with np.load(fdir / f"probe_L{layer:02d}.npz") as z:
                self.b[layer] = {k: torch.tensor(z[k], dtype=torch.float32, device=dev)
                                 for k in ("mean_mu", "mean_sd", "mean_w", "mean_b", "tok_mu", "tok_sd", "tok_w", "tok_b")}

    @torch.no_grad()
    def score(self, cache, n) -> dict:
        out = {}
        for tag, layer, sae, feats, w1, w0 in self.a:
            r = cache[f"blocks.{layer}.hook_resid_post"][0, :n]
            acts = sae.encode(r)
            x = (acts[1:, feats] > 0).float()
            out[f"a-sae-cusum|{tag}"] = cusum_max((x @ w1 + (1 - x) @ w0).tolist())
        for layer, P in self.b.items():
            r = cache[f"blocks.{layer}.hook_resid_post"][0, 1:n].float()
            m = r.mean(0)
            out[f"b-probe-mean|L{layer:02d}"] = float(((m - P["mean_mu"]) / P["mean_sd"]) @ P["mean_w"] + P["mean_b"])
            z = ((r - P["tok_mu"]) / P["tok_sd"]) @ P["tok_w"] + P["tok_b"]
            out[f"b-probe-cusum|L{layer:02d}"] = cusum_max(z.tolist())
        return out


def task(ctx):
    from dsgx.attacks.registry import make_attack
    from dsgx.methods import dsg, gates

    from experiments.E4a.common import ATTACKS

    a = ctx.args
    exp = a.get("fit_exp", ctx.exp_id)
    b = ctx.bundle(sae_id=DSG_SAE)
    m = b.model
    det = Detectors(exp, b)
    # references at DSG's layer 3 (bundle SAE = DSG's l0_142)
    g_rho = gates.Gate({"type": "rho", "n_features": 20, "retain_pct": 95}, b, 0)
    g_sg = gates.Gate({"type": "cusum", "n_features": 20, "retain_pct": 95}, b, 0)
    tau_rho = dsg.calibrate_tau(g_rho.cache, g_rho.features, 95)
    sg_rec = gates.calibrate(b, g_sg, "bio", 0.05, 1000, "mmlu-dev")
    attacks = {k: make_attack({**v, "_exp_id": ctx.exp_id}) for k, v in ATTACKS.items()}
    items = question_items(a.get("limit"))
    names = [f"blocks.{l}.hook_resid_post" for l in LAYERS]
    ctx.write_config(fit_exp=exp, n_items=len(items), tau_dsg=tau_rho, sg1_threshold=sg_rec["threshold"],
                     dsg_features=g_rho.features)
    ctx.progress.update(items_total=len(items), items_done=0, force=True)

    @torch.no_grad()
    def one(it):
        key, cond, (atk, item) = it
        prompt, info = attacks[atk].prompt(item)
        if prompt is None:
            return {"key": key, "cond": cond, "skip": str(info.get("skip"))}
        t = m.to_tokens(prompt, prepend_bos=False).to(b.device)
        n = t.shape[1]
        _, cache = m.run_with_cache(t, stop_at_layer=max(LAYERS) + 1, names_filter=lambda nm: nm in names)
        row = {"key": key, "cond": cond, "skip": "", "n_tok": int(n)}
        row.update(det.score(cache, n))
        r3 = cache[b.hook_name][0]
        acts = b.sae.encode(r3)
        acts[0] = 0
        row["ref-dsg-rho|L03dsg"] = g_rho.score(r3, acts, n)["score"]
        row["ref-sg1-cusum|L03dsg"] = g_sg.score(r3, acts, n)["score"]
        return row

    ck = ItemCheckpoints(ctx.run_dir() / "partial", ["E4a-questions", exp, [k for k, _, _ in items]])
    rows = ck.map(items, one, key=lambda it: it[0], on_done=ctx.progress.advance)
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    refs = {"ref-dsg-rho|L03dsg": tau_rho, "ref-sg1-cusum|L03dsg": sg_rec["threshold"]}
    m_out = {"n": int(len(df)), "n_skipped": int((df["skip"] != "").sum()),
             "by_cond": df[df["skip"] == ""].groupby("cond").size().to_dict(), "ref_own_thresholds": refs}
    ctx.write_metrics(m_out)
    ctx.finish({"view": "e4a-questions", "forget": None})
    ck.clear()
    return m_out
