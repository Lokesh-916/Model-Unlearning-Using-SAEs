"""E5 Step 3/4 task (GPU): open-ended generation with the frozen v2 gate (prompt + every generated token) and the
arms A-E (+ the exploratory curve) after a fire.

args: items (wmdp-bio-open | mmlu-open:<5 benign-bio subjects> | tofu-forget10 | tofu-retain_perturbed), split (dev;
TOFU sets are used whole, as in E2f), attack (none | pad400 | split3), n_items (150), subset_seed (0), max_new (96),
model {weights} (TOFU: ckpt:E2f/tofu_full), direction (base | tofu), curve (bool), limit (smoke).
Gate: frozen StreamGuard v2 (experiments.V2.common.METHOD; fit sha256 checked by experiments.V2.method.load_fit).
Arms (common.ARMS): A clamp (v2's response); B direction at a*; C top SAE latent at its Step-2 level; D clamp +
direction a*; E direction at a* x (0.25 + 0.75 p) with p from the fire-time score (common.prop_strength; reference =
E4a L08 scores above the threshold on clean WMDP-Bio dev MCQ questions); curve: direction at 0.25 / 0.5 / 0.75 a*.
Per item and condition (base + arms): match vs reference (openqa.grade; leak on hazard / forget sets, correctness
otherwise), gibberish, refusal, fired, fire step, fire-time score. The prompt score of the streaming detector is
checked against experiments.V2.method.l08_score on every item. Generations only under $DSG_PRIVATE; items.parquet
has ids, hashes and scores. Item-level resume (chunks in the private dir).
"""
import json

import numpy as np
import pandas as pd
import torch

from dsgx import paths
from dsgx.eval.openqa import gibberish, grade, item_set, text_hash
from dsgx.itemckpt import ItemCheckpoints

from experiments.E5.batchgen import L08Detector, continue_batch, phase1
from experiments.E5.common import ARMS, CURVE, CURVE_ARMS, CANONICAL_L0, SAE_RELEASE, is_refusal, prop_strength
from experiments.E5.data import open_attack_prompt
from experiments.E5.refusal import load_dir
from experiments.V2.common import DSG_FEATURES, METHOD, THRESHOLD


def ref_scores() -> np.ndarray:
    q = pd.read_parquet(paths.runs_dir() / "E4a" / "questions" / "items.parquet")
    q = q[(q["skip"] == "") & (q["cond"] == "wmdp_clean")]
    s = q["a-sae-cusum|L08"].astype(float).values
    return np.sort(s[s > THRESHOLD])


def load_gate(b):
    from dsgx.models.loader import get_sae

    from experiments.V2.method import load_fit

    g = METHOD["gate"]
    rec = load_fit(g)
    dev = b.device
    sae8 = get_sae(g["release"], g["sae_id"], b.dtype, dev)
    feats = torch.tensor(rec["features"], device=dev)
    w1 = torch.tensor(rec["w1"], dtype=torch.float32, device=dev)
    w0 = torch.tensor(rec["w0"], dtype=torch.float32, device=dev)
    return sae8, feats, w1, w0


def stopped(ctx, why):
    ctx.write_metrics({"stopped": True, "reason": why})
    ctx.finish({"view": f"e5-gen:{ctx.args.get('items')}", "forget": None, "stopped": True})
    return {"stopped": True}


def task(ctx):
    from dsgx.models.loader import get_bundle, get_sae
    from dsgx.run import resolve_weights

    from experiments.V2.method import l08_score

    a = ctx.args
    s1exp = a.get("step1_exp", ctx.exp_id)
    s1 = json.loads((paths.runs_dir() / s1exp / "refusal" / "metrics.json").read_text())
    if not s1["selection"]["eligible"]:
        return stopped(ctx, "Step 1: no eligible cell")
    s2 = json.loads((paths.runs_dir() / s1exp / "sae" / "metrics.json").read_text())
    L, s_star = int(s1["selection"]["layer"]), float(s1["selection"]["s"])
    items = item_set(a["items"], a.get("split", "dev"))
    if a.get("n_items") and len(items) > int(a["n_items"]):
        keep = np.sort(np.random.default_rng(int(a.get("subset_seed", 0))).choice(len(items), int(a["n_items"]), replace=False))
        items = [items[i] for i in keep]
    if a.get("limit"):
        items = items[: int(a["limit"])]
    attack = a.get("attack", "none")
    max_new = int(a.get("max_new", 96))
    curve = bool(a.get("curve", False))
    mc = a.get("model", {})
    b = get_bundle(weights=resolve_weights(mc.get("weights"), ctx.exp_id))
    m = b.model
    sae8, feats, w1, w0 = load_gate(b)
    dkey = a.get("direction", "base")
    d, norm = load_dir(s1exp, dkey, L)
    unit = torch.tensor(d, dtype=torch.float32, device=b.device)
    v_star = s_star * norm * unit
    sae_l = get_sae(SAE_RELEASE, f"layer_{L}/width_16k/average_l0_{CANONICAL_L0[L]}", "float32", b.device)
    j0, level = int(s2["top"][0]["latent"]), float(s2["chosen"]["level"])
    v_lat = level * sae_l.W_dec[j0].detach().float().to(b.device)
    ref = ref_scores()
    clamp = (b.sae, DSG_FEATURES, 500.0, b.hook_name)
    conds = ["base"] + ARMS + (CURVE_ARMS if curve else [])
    ctx.write_config(items=a["items"], attack=attack, n_items=len(items), max_new=max_new, model=mc, direction=dkey,
                     layer=L, s_star=s_star, norm=norm, latent=j0, latent_level=level, threshold=THRESHOLD,
                     n_ref_scores=len(ref), conds=conds, item_ids=[it.item_id for it in items])
    ctx.progress.update(items_total=len(items), items_done=0, force=True)
    priv = ctx.private_dir()
    ck = ItemCheckpoints(priv / "partial", ["E5-gen", a["items"], attack, [it.item_id for it in items], max_new, mc,
                                            dkey, L, s_star, norm, j0, level, THRESHOLD, conds])

    def one(it):
        p = open_attack_prompt(it.question, attack, it.item_id)
        toks = m.to_tokens(p, prepend_bos=False).to(b.device)
        m.reset_hooks()
        ref_score = l08_score(m, sae8, 8, feats, w1, w0, toks)
        det = L08Detector(sae8, feats, w1, w0)
        out, stop, fire, fscore, pscore = phase1(m, toks, "blocks.8.hook_resid_post", det, max_new)
        res = {"prompt_len": int(toks.shape[1]), "fire_step": fire, "fire_score": fscore, "prompt_score": pscore,
               "prompt_score_ref": ref_score, "max_score": det.best, "base": out, "strength_E": None, "p_E": None}
        if fire is None:
            for c in conds[1:]:
                res[c] = list(out)
            return res
        sE, pE = prop_strength(fscore, ref, s_star)
        res["strength_E"], res["p_E"] = sE, pE
        rows = [{"clamp": True, "vec": None}, {"clamp": False, "vec": v_star}, {"clamp": False, "vec": v_lat},
                {"clamp": True, "vec": v_star}, {"clamp": False, "vec": (sE / s_star) * v_star}]
        if curve:
            rows += [{"clamp": False, "vec": c * v_star} for c in CURVE]
        prefix = out[:fire]
        seq = torch.cat([toks, torch.tensor([prefix], dtype=torch.long, device=b.device)], dim=1) if prefix else toks
        for c, (new, _) in zip(conds[1:], continue_batch(m, seq, max_new - len(prefix), rows, clamp, L)):
            res[c] = prefix + new
        return res

    done = ck.map(items, one, key=lambda it: it.item_id, on_done=ctx.progress.advance)
    m.reset_hooks()
    recs, gens = [], []
    for it, o in zip(items, done):
        for c in conds:
            text = m.tokenizer.decode(o[c], skip_special_tokens=True)
            gens.append({"item_id": it.item_id, "cond": c, "text": text})
            recs.append({"item_id": it.item_id, "cond": c, "fired": o["fire_step"] is not None, "fire_step": o["fire_step"],
                         "fire_score": o["fire_score"], "prompt_score": o["prompt_score"],
                         "prompt_score_ref": o["prompt_score_ref"], "max_score": o["max_score"],
                         "strength_E": o["strength_E"], "p_E": o["p_E"], "prompt_len": o["prompt_len"],
                         "n_tokens": len(o[c]), "text_hash": text_hash(text), "refusal": is_refusal(text),
                         **gibberish(text, o[c]), "_text": text, "_ref": it.reference})
    with (priv / "generations.jsonl").open("w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    grades = grade([r["_text"] for r in recs], [r["_ref"] for r in recs]) if recs else []
    for r, gr in zip(recs, grades):
        r.update(gr)
        del r["_text"], r["_ref"]
    df = pd.DataFrame(recs)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    base = df[df["cond"] == "base"]
    diff = (base["prompt_score"].astype(float) - base["prompt_score_ref"].astype(float)).abs()
    dec = ((base["prompt_score"].astype(float) > THRESHOLD) != (base["prompt_score_ref"].astype(float) > THRESHOLD))
    met = {"n": len(items), "attack": attack, "fire_rate": float(base["fired"].mean()) if len(base) else None,
           "prompt_fire_rate": float((base["fire_step"] == 0).mean()) if len(base) else None,
           "check": {"prompt_score_bit_identical": int((diff == 0).sum()), "max_abs_diff": float(diff.max()) if len(diff) else 0.0,
                     "prompt_decision_diff": int(dec.sum()), "n": int(len(base))}}
    for c in conds:
        dd = df[(df["cond"] == c) & df["fired"]]
        met[c] = {k: float(dd[k].astype(float).mean()) if len(dd) else None for k in ("match", "gibberish", "refusal")}
    ctx.write_metrics(met)
    ctx.finish({"view": f"e5-gen:{a['items']}:{attack}", "forget": met["A"]["match"], "check": met["check"]})
    ck.clear()
    return met["check"]
