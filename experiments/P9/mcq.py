"""P9 MCQ task (GPU): per DEV item, every condition in one pass.

args: cond (clean | pad400 | split3 | hindi), limit (smoke). clean = WMDP-Bio dev (@forget) + benign biology (5 MMLU
subjects) + MMLU utility dev (bio set); the attacked conditions = WMDP-Bio dev only (as V2-devcheck).
Per item: v2 prompt score (experiments.V2.method.l08_score, exactly v2's pass 1) -> fired. Not fired: one base
forward (every condition answers with the base model). Fired: base; A = v2 clamp (experiments.V2.method's own hook,
forced on); B = v2-R refusal direction at layer 10 (0.45 x norm, every prompt position: an MCQ answer is one
prompt forward, so start8 = all); R-a..R-d via experiments.P9.core (fallback -> A's answer).
items.parquet: ids, predictions, correctness, fired, scores, counts, region shares, rounds, fallbacks, timings.
NO token positions or token ids of redactions are stored. Item-level resume (chunks in the private dir).
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from dsgx.itemckpt import ItemCheckpoints

from experiments.P9.common import (BENIGN_BIO, MCQ_CONDS, N_ATTACK, SUBSET_SEED, NEUTRAL_ID, NEUTRAL_TEXT, V2R, VARIANTS, content_spans,
                                   label_tokens, region_spans)
from experiments.P9.core import AttnBlock, Redactor, Scorer, pos_mask, replaced, resolve_prompt
from experiments.V2.common import METHOD, THRESHOLD

REGIONS = ["question", "options", "preamble", "filler"]


def load_v2r(b):
    """v2-R's refusal vector with E5b's frozen-setting guard (selection + direction sha256)."""
    import hashlib

    from dsgx import paths
    from experiments.E5.common import cache_path
    from experiments.E5.refusal import load_dir

    s1 = json.loads((paths.runs_dir() / V2R["step1_exp"] / "refusal" / "metrics.json").read_text())["selection"]
    got = (int(s1["layer"]), float(s1["s"]), s1.get("mode", "all"))
    if got != (V2R["layer"], V2R["s"], V2R["mode"]):
        raise RuntimeError(f"E5b selection {got} != frozen v2-R setting")
    h = hashlib.sha256(cache_path(V2R["step1_exp"], "base").read_bytes()).hexdigest()
    if h != V2R["direction_sha256"]:
        raise RuntimeError(f"direction sha256 {h} != frozen")
    d, norm = load_dir(V2R["step1_exp"], "base", V2R["layer"])
    return V2R["s"] * norm * torch.tensor(d, dtype=torch.float32, device=b.device), norm


def mcq_items(cond: str, limit=None, n_attack=N_ATTACK):
    """[(dataset, item, prompt)] for the condition, in the harness order (attack.prompt; Hindi below chrF skipped).
    Attacked conditions use the fixed seeded subset of n_attack WMDP-Bio dev items (default_rng(SUBSET_SEED))."""
    from dsgx.attacks.registry import make_attack
    from dsgx.data.mcq import load_mcq, utility_subjects
    from dsgx.data.splits import get_split

    att = make_attack({**MCQ_CONDS[cond], "_exp_id": "P9"}, 0)
    dsets = ["wmdp-bio"] + ((BENIGN_BIO + utility_subjects("bio")) if cond == "clean" else [])
    out = []
    for ds in dict.fromkeys(dsets):
        items = load_mcq(ds)
        ids = get_split(ds, "dev")
        if cond != "clean" and n_attack and len(ids) > n_attack:
            keep = np.sort(np.random.default_rng(SUBSET_SEED).choice(len(ids), int(n_attack), replace=False))
            ids = [ids[i] for i in keep]
        if limit:
            ids = ids[: int(limit)]
        for i in ids:
            p, _ = att.prompt(items[i])
            if p is not None:
                out.append((ds, items[i], p))
    return out


def setup(b):
    from experiments.E5.gen import load_gate
    from experiments.V2.method import StreamGuardV2

    m = b.model
    assert m.to_tokens(NEUTRAL_TEXT, prepend_bos=False)[0].tolist() == [NEUTRAL_ID]
    sae8, feats, w1, w0 = load_gate(b)
    sc = Scorer(m, sae8, feats, w1, w0)
    v2 = StreamGuardV2(METHOD, b, 0)      # v2's own pass-1 + clamp hook (forced per item)
    vec, norm = load_v2r(b)
    return sc, v2, vec, norm


def tokens_and_candidates(m, prompt):
    tok = m.tokenizer
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    toks = m.to_tokens(prompt, prepend_bos=False)
    if enc["input_ids"] != toks[0].tolist():
        raise RuntimeError("offset tokenisation differs from model.to_tokens")
    spans = content_spans(prompt)
    special = set(tok.all_special_ids) | {tok.convert_tokens_to_ids("<start_of_turn>"), tok.convert_tokens_to_ids("<end_of_turn>")}
    offs = enc["offset_mapping"]
    cand = [i for i, ((s, e), t) in enumerate(zip(offs, enc["input_ids"])) if i > 0 and t not in special
            and e > s and any(a <= s < b for a, b in spans)]
    return toks, cand, offs, spans


@torch.no_grad()
def answer(m, toks, ans_tokens, hooks=()):
    from dsgx.eval.mcq_eval import answer_probs

    m.reset_hooks()
    for name, h in hooks:
        m.add_hook(name, h)
    try:
        p = answer_probs(m, toks, torch.tensor([toks.shape[1] - 1], device=toks.device), ans_tokens)
    finally:
        m.reset_hooks()
    return int(p[0].float().argmax())


class _Add:
    def __init__(self, v):
        self.v = v

    def __call__(self, resid, hook=None):
        return resid + self.v.to(resid.dtype)


def task(ctx):
    from dsgx.data.mcq import pre_question
    from dsgx.eval.mcq_eval import ANSWER_STRINGS
    from experiments.V2.method import l08_score

    a = ctx.args
    cond = a["cond"]
    b = ctx.bundle()
    m = b.model
    sc, v2, vec, norm = setup(b)
    ans_tokens = m.to_tokens(ANSWER_STRINGS, prepend_bos=False).flatten()
    rows_in = mcq_items(cond, a.get("limit"), int(a.get("n_attack", N_ATTACK)))
    ctx.write_config(cond=cond, attack=MCQ_CONDS[cond], n_items=len(rows_in), threshold=THRESHOLD, neutral=NEUTRAL_ID,
                     v2r_norm=norm, item_ids=[f"{ds}:{it.item_id}" for ds, it, _ in rows_in])
    ctx.progress.update(items_total=len(rows_in), items_done=0, force=True)
    every = int(a.get("resume_every", 25 if cond != "clean" else 200))
    ck = ItemCheckpoints(ctx.private_dir() / "partial", ["P9-mcq", cond, len(rows_in), THRESHOLD, NEUTRAL_ID, V2R],
                         every=every)

    def clamp_answer(toks):
        v2.hook.records = []
        v2.set_lengths([toks.shape[1]])
        v2.hook.forced = [True]
        r = answer(m, toks, ans_tokens, [(b.hook_name, v2.hook)])
        v2.hook.records = []
        return r

    def one(x):
        ds, it, prompt = x
        key = f"{cond}|{ds}:{it.item_id}"
        toks, cand, offs, spans = tokens_and_candidates(m, prompt)
        toks = toks.to(b.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.reset_hooks()
        s0 = l08_score(m, sc.sae8, 8, sc.feats, sc.w1, sc.w0, toks)
        torch.cuda.synchronize()
        t_gate = time.perf_counter() - t0
        t0 = time.perf_counter()
        base = answer(m, toks, ans_tokens)
        torch.cuda.synchronize()
        t_base = time.perf_counter() - t0
        fired = s0 > THRESHOLD
        r = {"key": key, "dataset": ds, "item_id": it.item_id, "gold": int(it.answer), "fired": bool(fired),
             "score": float(s0), "prompt_len": int(toks.shape[1]), "n_cand": len(cand), "t_gate": t_gate,
             "t_base": t_base, "pred_base": base}
        if not fired:
            for c in ["A", "B"] + VARIANTS:
                r[f"pred_{c}"] = base
            return r
        r["pred_A"] = clamp_answer(toks)
        r["pred_B"] = answer(m, toks, ans_tokens, [(f"blocks.{V2R['layer']}.hook_resid_post", _Add(vec))])
        red = Redactor(sc, toks, cand, s0, key)
        st = resolve_prompt(red, red.first_round())
        labels = label_tokens(offs, spans, region_spans(prompt, cond, pre_question(ds)))
        r.update({"n_pos_blame": len(red.order_pos), "blame_max": float(red.blame.max()) if len(cand) else 0.0,
                  "blame_pos_sum": float(red.blame[red.blame > 0].sum()) if len(cand) else 0.0,
                  "ev_max": float(red.ev.max()) if len(cand) else 0.0, "k1": int(red.k1 or 0),
                  "exhausted1": bool(st["R-a"].get("exhausted1", False)), "t_blame": red.t_blame,
                  "t_ra": red.timing.get("R-a")})
        top = red.order_pos[:1]
        r["top1_region"] = labels[top[0]] if top else ""
        for reg in REGIONS:
            r[f"cand_{reg}"] = sum(labels[p] == reg for p in cand)
        for v in VARIANTS:
            s = st[v]
            r[f"rounds_{v}"] = s["rounds"]
            r[f"fallback_{v}"] = bool(s["fallback"])
            r[f"n_red_{v}"] = len(s["pos"])
            r[f"score_{v}"] = float(s["score"])
            for reg in REGIONS:
                r[f"red_{v}_{reg}"] = sum(labels[p] == reg for p in s["pos"])
            if s["fallback"]:
                r[f"pred_{v}"] = r["pred_A"]
            elif s["mode"] == "replace":
                r[f"pred_{v}"] = answer(m, replaced(toks, s["pos"]), ans_tokens)
            else:
                h = AttnBlock(pos_mask(toks.shape[1], s["pos"], toks.device))
                r[f"pred_{v}"] = answer(m, toks, ans_tokens,
                                        [(f"blocks.{l}.attn.hook_attn_scores", h) for l in range(m.cfg.n_layers)])
        sa, sc_ = set(st["R-a"]["pos"]), set(st["R-c"]["pos"])
        r["jaccard_ac"] = len(sa & sc_) / max(1, len(sa | sc_))
        return r

    t_start = time.time()
    out = ck.map(rows_in, one, key=lambda x: f"{x[0]}:{x[1].item_id}", on_done=ctx.progress.advance)
    m.reset_hooks()
    df = pd.DataFrame(out)
    for c in ["base", "A", "B"] + VARIANTS:
        df[f"correct_{c}"] = df[f"pred_{c}"] == df["gold"]
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    met = {"cond": cond, "n": len(df), "wall_s": time.time() - t_start}
    for ds_group, sel in (("wmdp", df["dataset"] == "wmdp-bio"), ("benign", df["dataset"].isin(BENIGN_BIO)),
                          ("mmlu", ~df["dataset"].isin(["wmdp-bio"] + BENIGN_BIO))):
        d = df[sel]
        if len(d):
            met[ds_group] = {"n": len(d), "fire": float(d["fired"].mean()),
                             **{c: float(d[f"correct_{c}"].mean()) for c in ["base", "A", "B"] + VARIANTS}}
    ctx.write_metrics(met)
    ctx.finish({"view": f"p9-mcq:{cond}", "forget": met.get("wmdp", {}).get("R-a")})
    ck.clear()
    return met
