"""E5 Step 2 task (GPU): Gemma Scope latents closest to the refusal direction, and the activation level of the top
latent that gives the same refusal rate on the same 100 held-out harmless instructions.

SAE = canonical Gemma Scope 16k at the selected layer (common.CANONICAL_L0). Cosine(W_dec row, unit direction) for
every latent; top 3 by cosine. Steering with latent j at level c = adding c x W_dec[j] at blocks.L.hook_resid_post on
all positions (the residual the latent writes when it is active at c). Levels c_k = k x a* x norm / ||W_dec[j]||,
k in SAE_K (k = 1: same vector norm as the direction at a*). Chosen level: common.select_level with target = the
Step-1 refusal rate at a*. Also the direction at a* is regenerated in the same batch (row "dir@a*", a numerics check).
Neuronpedia labels: best effort (network), public auto-interp explanations of public latents only.
Latent activation context: mean / max activation of the top 3 at the last instruction token of the build sets.
"""
import json
import urllib.request

import numpy as np
import pandas as pd
import torch

from dsgx import paths
from dsgx.eval import stats
from dsgx.eval.openqa import gibberish, text_hash
from dsgx.itemckpt import ItemCheckpoints

from experiments.E5.batchgen import forced_batch
from experiments.E5.common import CANONICAL_L0, SAE_K, SAE_RELEASE, is_refusal, select_level
from experiments.E5.data import build_sets, chat
from experiments.E5.refusal import load_dir, resid_at_instruction


def step1(exp_id):
    return json.loads((paths.runs_dir() / exp_id / "refusal" / "metrics.json").read_text())


def neuronpedia_label(layer, idx, timeout=10):
    url = f"https://www.neuronpedia.org/api/feature/gemma-2-2b/{layer}-gemmascope-res-16k/{idx}"
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "research-client"}),
                                    timeout=timeout) as r:
            j = json.loads(r.read().decode())
        ex = [e.get("description") for e in (j.get("explanations") or []) if e.get("description")]
        return {"label": ex[0] if ex else None, "source": url, "ok": True}
    except Exception as e:  # noqa: BLE001 - best effort
        return {"label": None, "source": url, "ok": False, "error": type(e).__name__}


def task(ctx):
    from dsgx.models.loader import get_sae

    a = ctx.args
    s1 = step1(a.get("step1_exp", ctx.exp_id))
    sel = s1["selection"]
    if not sel["eligible"]:
        ctx.write_metrics({"stopped": True, "reason": "Step 1: no cell with refusal >= 0.90 and gibberish <= 0.02"})
        ctx.finish({"view": "e5-sae", "forget": None, "stopped": True})
        return {"stopped": True}
    L, s_star = int(sel["layer"]), float(sel["s"])
    b = ctx.bundle()
    m = b.model
    d, norm = load_dir(a.get("step1_exp", ctx.exp_id), "base", L)
    sae_id = f"layer_{L}/width_16k/average_l0_{CANONICAL_L0[L]}"
    sae = get_sae(SAE_RELEASE, sae_id, "float32", b.device)
    W = sae.W_dec.detach().float()
    wn = W.norm(dim=1)
    cos = (W @ torch.tensor(d, device=W.device)) / wn
    top = torch.topk(cos, 3)
    tops = [{"latent": int(i), "cosine": float(c), "dec_norm": float(wn[i])} for c, i in zip(top.values, top.indices)]
    sets = build_sets()
    H = resid_at_instruction(m, [x["text"] for x in sets["harmful"]], [L])[L]
    S = resid_at_instruction(m, [x["text"] for x in sets["harmless"]], [L])[L]
    with torch.no_grad():
        aH = sae.encode(torch.tensor(H, device=W.device)).float().cpu().numpy()
        aS = sae.encode(torch.tensor(S, device=W.device)).float().cpu().numpy()
    for t in tops:
        j = t["latent"]
        t["act_harmful"] = {"mean": float(aH[:, j].mean()), "max": float(aH[:, j].max()), "frac_on": float((aH[:, j] > 0).mean())}
        t["act_harmless"] = {"mean": float(aS[:, j].mean()), "max": float(aS[:, j].max()), "frac_on": float((aS[:, j] > 0).mean())}
        t["neuronpedia"] = neuronpedia_label(L, j) if not ctx.smoke else {"label": None, "ok": False, "error": "smoke"}
    j0 = tops[0]["latent"]
    base_level = s_star * norm / tops[0]["dec_norm"]
    levels = [k * base_level for k in SAE_K]
    wj = W[j0].to(b.device)
    vdir = torch.tensor(s_star * norm * d, dtype=torch.float32, device=b.device)
    rows = [(L, vdir)] + [(L, c * wj) for c in levels]
    names = ["dir@a*"] + [f"lat{j0}k{k:g}" for k in SAE_K]
    held = sets["heldout"][: int(a["limit"])] if a.get("limit") else sets["heldout"]
    max_new = int(a.get("max_new", 96))
    ctx.write_config(layer=L, s_star=s_star, norm=norm, sae_id=sae_id, top=tops, levels=levels, max_new=max_new,
                     heldout_ids=[q["id"] for q in held])
    ctx.progress.update(items_total=len(held), items_done=0, force=True)
    priv = ctx.private_dir()
    ck = ItemCheckpoints(priv / "partial", ["E5-sae", [q["id"] for q in held], L, s_star, j0, levels, max_new])
    done = ck.map(held, lambda q: forced_batch(m, chat(q["text"]), rows, [L], max_new), key=lambda q: q["id"],
                  on_done=ctx.progress.advance)
    recs, gens = [], []
    for q, outs in zip(held, done):
        for name, ids in zip(names, outs):
            text = m.tokenizer.decode(ids, skip_special_tokens=True)
            gens.append({"id": q["id"], "cond": name, "text": text})
            recs.append({"id": q["id"], "cond": name, "refusal": is_refusal(text), "n_tokens": len(ids),
                         "text_hash": text_hash(text), **gibberish(text, ids)})
    with (priv / "generations.jsonl").open("w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    df = pd.DataFrame(recs)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    res, lv = {}, []
    for name, (k, c) in zip(names, [(None, None)] + list(zip(SAE_K, levels))):
        dd = df[df["cond"] == name]
        res[name] = {"k": k, "level": c, **{x: stats.bootstrap_ci(dd[x].astype(float).values) for x in ("refusal", "gibberish")}}
        if k is not None:
            lv.append({"k": k, "level": c, "refusal": res[name]["refusal"]["mean"], "gib": res[name]["gibberish"]["mean"]})
    target = s1["cells"][f"L{L}s{s_star:g}"]["refusal"]["mean"]
    chosen = select_level(lv, target)
    out = {"layer": L, "s_star": s_star, "norm": norm, "sae_id": sae_id, "top": tops, "levels": res,
           "target_refusal": target, "chosen": chosen, "n_heldout": len(held),
           "latent_vec_norm_over_resid_norm": float(chosen["level"] * tops[0]["dec_norm"] / norm)}
    ctx.write_metrics(out)
    ctx.finish({"view": "e5-sae", "forget": None, "latent": j0, "level": chosen["level"], "matched": chosen["matched"]})
    ck.clear()
    return {"latent": j0, "matched": chosen["matched"]}
