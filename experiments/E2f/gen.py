"""E2f Steps 2-3 task (GPU): open-ended generation with the streaming CUSUM gate and five responses on fire.

args: items (wmdp-bio-open | mmlu-open:<5 benign-bio subjects> | tofu-forget10 | tofu-retain_perturbed), split
(dev; TOFU sets have no split in this project and are used whole, as in X1-suite), max_new, model {weights},
gate {forget_corpus, retain_corpus} (default bio), direction ("base" | "tofu"), check_n (items also run through
dsgx.gen.stream.generate with the clamp, to check that the switch generator reproduces StreamGuard v1 exactly).
Gate = StreamGuard v1 (cusum, DSG's 20 features / retain_pct 95 on the gate's corpora, 5 % MMLU-dev FPR).
Conditions: base (no gate), clamp, R6, ign, R6+ign, R5+ign; ign = the Step-1 selected layer and coefficient, or
args.ign {layer, coef, dir_exp} (E2f-L12 override, exploratory). n_items / subset_seed: seeded random item subset.
Per item and condition: match vs reference (openqa.grade; leak on hazard / forget sets, correctness otherwise),
gibberish (openqa.gibberish), IDK (common.IDK), refusal (common.REFUSAL), gate fire step. Generations only under
$DSG_PRIVATE; items.parquet has ids, hashes and scores. Item-level resume (chunks in the private dir).
"""
import json

import numpy as np
import pandas as pd
import torch

from dsgx import paths
from dsgx.eval import stats
from dsgx.eval.openqa import gibberish, grade, item_set, open_prompt, text_hash
from dsgx.itemckpt import ItemCheckpoints

from experiments.E2f.common import DELTAS, RESPONSES, is_idk, is_refusal
from experiments.E2f.ignorance import load_vec
from experiments.E2f.switchgen import Response, generate, patched_weights

CONDS = ["base"] + RESPONSES


def load_delta(exp_id, rid):
    from safetensors.torch import load_file

    src_exp, r = DELTAS[rid]  # the real (non-smoke) deltas, also in smoke runs
    return load_file(str(paths.cache_dir() / "models" / src_exp / r / "delta.safetensors"))


def task(ctx):
    from dsgx.data import activation_cache as ac
    from dsgx.gen.stream import generate as sg_generate
    from dsgx.methods import gates
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    a = ctx.args
    items = item_set(a["items"], a.get("split", "dev"))
    if a.get("n_items") and len(items) > int(a["n_items"]):  # seeded random subset, original order kept (E2f-L12)
        keep = np.sort(np.random.default_rng(int(a.get("subset_seed", 0))).choice(len(items), int(a["n_items"]), replace=False))
        items = [items[i] for i in keep]
    if a.get("limit"):
        items = items[: int(a["limit"])]
    max_new = int(a.get("max_new", 64))
    mc = a.get("model", {})
    b = get_bundle(weights=resolve_weights(mc.get("weights"), ctx.exp_id))
    m = b.model
    gspec = {"type": "cusum", "n_features": 20, "retain_pct": 95, "case": "bio", **a.get("gate", {})}
    ac.build_cache(b, gspec.get("forget_corpus", "bio-forget-corpus"), gspec.get("retain_corpus", "wikitext"), 0)
    g = gates.Gate(gspec, b, 0)
    cal = gates.calibrate(b, g, "bio", 0.05, 1000, "mmlu-dev")
    gate = gates.stream_gate(g)
    ov = a.get("ign")  # override of the Step-1 selection (E2f-L12, DEVIATIONS 2026-11-12: exploratory)
    if ov:
        sel = {"layer": int(ov["layer"]), "coef": float(ov["coef"]), "source": "override", "label": ov.get("label", "exploratory")}
        dexp = ov.get("dir_exp", ctx.exp_id)
    else:
        sel = json.loads((paths.runs_dir() / a.get("ign_exp", ctx.exp_id) / "ignorance" / "metrics.json").read_text())["selection"]
        dexp = a.get("ign_exp", ctx.exp_id)
    ign = (sel["layer"], load_vec(dexp, a.get("direction", "base"), sel["layer"], sel["coef"], b.device))
    W = {r: patched_weights(m, load_delta(ctx.exp_id, r)) for r in ("R5", "R6")}
    resp = [Response("clamp", b.hook_name, clamp=(b.sae, g.features, 500.0)),
            Response("R6", weights=W["R6"]), Response("ign", steer=ign),
            Response("R6+ign", weights=W["R6"], steer=ign), Response("R5+ign", weights=W["R5"], steer=ign)]
    assert [r.name for r in resp] == RESPONSES
    check_n = int(a.get("check_n", 20))
    ctx.write_config(items=a["items"], n_items=len(items), gate=gspec, calib=cal, features=g.features,
                     ignorance=sel, direction=a.get("direction", "base"), dir_exp=dexp, max_new=max_new, model=mc,
                     item_ids=[it.item_id for it in items])
    ctx.progress.update(items_total=len(items), items_done=0, force=True)
    priv = ctx.private_dir()
    ck = ItemCheckpoints(priv / "partial", ["E2f-gen", a["items"], [it.item_id for it in items], max_new, mc, gspec,
                                            cal["threshold"], sel, check_n])

    def one(k_it):
        k, it = k_it
        p = open_prompt(it.question)
        r = generate(m, p, b, gate, resp, max_new)
        out = {"fire_step": r["fire_step"], **{c: r[c]["tokens"] for c in CONDS}}
        if k < check_n:
            ref = sg_generate(m, p, b, gate[0], 500.0, gate[1], "stream", max_new, score_fn=gate[2], token_fn=gate[3])
            out["check_equal"] = ref.tokens == r["clamp"]["tokens"]
            out["check_fire_equal"] = (ref.first_gate_step == r["fire_step"])
        return out

    done = ck.map(list(enumerate(items)), one, key=lambda kit: kit[1].item_id, on_done=ctx.progress.advance)
    m.reset_hooks()
    rows, gens = [], []
    for it, o in zip(items, done):
        for c in CONDS:
            text = m.tokenizer.decode(o[c], skip_special_tokens=True)
            gens.append({"item_id": it.item_id, "cond": c, "text": text})
            rows.append({"item_id": it.item_id, "cond": c, "fired": o["fire_step"] is not None, "fire_step": o["fire_step"],
                         "n_tokens": len(o[c]), "text_hash": text_hash(text), "idk": is_idk(text),
                         "refusal": is_refusal(text), **gibberish(text, o[c]), "_text": text, "_ref": it.reference})
    with (priv / "generations.jsonl").open("w") as f:
        for gg in gens:
            f.write(json.dumps(gg) + "\n")
    grades = grade([r["_text"] for r in rows], [r["_ref"] for r in rows])
    for r, gr in zip(rows, grades):
        r.update(gr)
        r["idk_or_refusal"] = bool(r["idk"] or r["refusal"])
        del r["_text"], r["_ref"]
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    checks = [o for o in done if "check_equal" in o]
    met = {"n": len(items), "threshold": cal["threshold"], "fire_rate": float(np.mean([o["fire_step"] is not None for o in done])),
           "check": {"n": len(checks), "clamp_tokens_equal": int(sum(o["check_equal"] for o in checks)),
                     "fire_step_equal": int(sum(o["check_fire_equal"] for o in checks))}}
    for c in CONDS:
        d = df[df["cond"] == c]
        met[c] = {k: stats.bootstrap_ci(d[k].astype(float).values) for k in ("match", "gibberish", "idk", "refusal", "idk_or_refusal")}
    ctx.write_metrics(met)
    ctx.finish({"view": f"e2f-gen:{a['items']}", "forget": met["clamp"]["match"], "check": met["check"]})
    ck.clear()
    return met["check"]
