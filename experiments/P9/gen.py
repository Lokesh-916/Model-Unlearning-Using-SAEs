"""P9 open-ended task (GPU): base / A clamp / B v2-R / R-a..R-d on DEV open-ended items.

args: set (wmdp | benign), n_items (150, seeded subset default_rng(0) as E5), max_new (96), limit (smoke).
Gate: frozen v2 L08 detector on the prompt and after every generated token (experiments.E5.batchgen.phase1; the
prompt score is checked against experiments.V2.method.l08_score on every item). After a fire at step s the sequence
(prompt + base tokens[:s]) continues under A (clamp) and B (v2-R) as one batch (E5 generator), and under each
redaction variant: candidates = user-content tokens of the prompt + the s base tokens; round 1 + prompt re-check
(experiments.P9.core); then greedy generation with the UNMODIFIED model on the redacted sequence (R-b: attention
blocks at every layer), streaming the gate (blocked positions' own increments dropped). A re-fire mid-generation
starts the next round on the whole current sequence (generated tokens become candidates); after MAX_ROUNDS rounds a
fire switches the continuation to v2's clamp (fallback, counted). Answer text = base prefix + new tokens.
Graders: leak / correct (openqa.grade), gibberish, refusal (E5), clarification (P9 regex). Generations only under
$DSG_PRIVATE; items.parquet has ids, hashes, scores, counts. Item-level resume.
"""
import json
import time

import numpy as np
import pandas as pd
import torch

from dsgx.eval.openqa import gibberish, grade, item_set, open_prompt, text_hash
from dsgx.itemckpt import ItemCheckpoints

from experiments.E5.batchgen import L08Detector, continue_batch, end_ids, phase1
from experiments.E5.common import is_refusal
from experiments.P9.common import (MAX_NEW, N_OPEN, OPEN_SETS, SUBSET_SEED, V2R, VARIANTS, is_clarification)
from experiments.P9.core import AttnBlock, Redactor, pos_mask, replaced, resolve_prompt
from experiments.P9.mcq import setup, tokens_and_candidates
from experiments.V2.common import DSG_FEATURES, THRESHOLD


class BatchSkipDetector:
    """Per-row streaming L08 CUSUM for a batch of R sequences; row r skips the increments of excludes[r]."""

    def __init__(self, sae8, feats, w1, w0, excludes, threshold=THRESHOLD):
        self.sae8, self.feats, self.w1, self.w0 = sae8, feats, w1, w0
        self.excl = [set(int(p) for p in e) for e in excludes]
        R = len(excludes)
        self.s, self.best = [0.0] * R, [0.0] * R
        self.pos, self.first, self.threshold = 0, True, float(threshold)

    @torch.no_grad()
    def __call__(self, resid, hook=None):
        acts = self.sae8.encode(resid)
        n = acts.shape[1]
        positions = list(range(self.pos, self.pos + n))
        self.pos += n
        if self.first:
            acts, positions = acts[:, 1:], positions[1:]
            self.first = False
        x = (acts[:, :, self.feats] > 0).float()
        l = (x @ self.w1 + (1 - x) @ self.w0).tolist()
        for r in range(len(self.s)):
            for p, v in zip(positions, l[r]):
                if p in self.excl[r]:
                    continue
                self.s[r] = max(0.0, self.s[r] + float(v))
                self.best[r] = max(self.best[r], self.s[r])
        return resid

    def fired(self, r):
        return self.best[r] > self.threshold


@torch.no_grad()
def stream_batch(m, inps, blocks, gate, n_left):
    """Greedy continuation of R equal-length rows (unmodified model; row r with attention blocks on blocks[r]) with a
    per-row streaming gate. A row stops at EOS, at the budget, or at a re-fire. -> [(tokens, stop, refire | None)]."""
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

    dev = m.cfg.device
    end = end_ids(m)
    R, n = inps.shape
    det = BatchSkipDetector(*gate, excludes=blocks)
    m.reset_hooks()
    m.add_hook("blocks.8.hook_resid_post", det)
    if any(len(bp) for bp in blocks):
        mask = torch.cat([pos_mask(n, bp, dev) for bp in blocks])
        h = AttnBlock(mask)
        for l in range(m.cfg.n_layers):
            m.add_hook(f"blocks.{l}.attn.hook_attn_scores", h)
    outs, stops, refire = [[] for _ in range(R)], ["max_new"] * R, [None] * R
    live = [True] * R
    try:
        cache = TransformerLensKeyValueCache.init_cache(m.cfg, dev, R)
        last = m(inps, past_kv_cache=cache)[:, -1]
        for r in range(R):
            if det.fired(r):
                live[r], stops[r], refire[r] = False, "refire", 0
        for step in range(1, n_left + 1):
            if not any(live):
                break
            nxt = last.argmax(dim=-1)
            for r, t in enumerate(nxt.tolist()):
                if live[r]:
                    outs[r].append(t)
                    if t in end:
                        live[r], stops[r] = False, "eos"
            if not any(live) or step == n_left:
                break
            last = m(nxt.view(R, 1), past_kv_cache=cache)[:, -1]
            for r in range(R):
                if live[r] and det.fired(r):
                    live[r], stops[r], refire[r] = False, "refire", step
    finally:
        m.reset_hooks()
    return list(zip(outs, stops, refire))


def finish_variant(m, v, red, st, seq, cand, n_left, gate, clamp, new, info):
    """Continue a variant after a re-fire (sequentially): next round on the whole current sequence; after the last
    round, v2's clamp continues. -> (new tokens, info)."""
    while True:
        info["refires"] += 1
        cur = torch.cat([seq, torch.tensor([new], dtype=torch.long, device=seq.device)], 1) if new else seq
        cand2 = list(cand) + list(range(seq.shape[1], cur.shape[1]))
        st = red.next_round(v, st, cur, cand2)
        left = n_left - len(new)
        if st["fallback"]:
            info.update(fallback=True, fallback_where="generation")
            inp = replaced(cur, st["pos"]) if st["mode"] == "replace" else cur
            if left > 0:
                (c_new, _), = continue_batch(m, inp, left, [{"clamp": True, "vec": None}], clamp, None)
                new = new + c_new
            break
        if left <= 0:
            break
        inp = replaced(cur, st["pos"]) if st["mode"] == "replace" else cur
        (out, stop, _), = stream_batch(m, inp, [st["pos"] if st["mode"] == "block" else []], gate, left)
        new = new + out
        if stop != "refire":
            break
    return new, {**info, "rounds": st["rounds"], "n_red": len(st["pos"])}


def run_variants(m, red, states, seq, cand, n_left, gate, clamp, a_tokens):
    """All non-fallback variants as rows of one batch; re-fired rows are finished one by one. -> {v: (new, info)}."""
    res = {}
    live = [v for v in VARIANTS if not states[v]["fallback"]]
    for v in VARIANTS:
        if states[v]["fallback"]:
            res[v] = (list(a_tokens), {"fallback": True, "fallback_where": "prompt", "refires": 0,
                                        "rounds": states[v]["rounds"], "n_red": len(states[v]["pos"])})
    if live:
        inps = torch.cat([replaced(seq, states[v]["pos"]) if states[v]["mode"] == "replace" else seq for v in live])
        blocks = [states[v]["pos"] if states[v]["mode"] == "block" else [] for v in live]
        for v, (out, stop, _) in zip(live, stream_batch(m, inps, blocks, gate, n_left)):
            info = {"fallback": False, "fallback_where": "", "refires": 0}
            if stop == "refire":
                res[v] = finish_variant(m, v, red, states[v], seq, cand, n_left, gate, clamp, list(out), info)
            else:
                res[v] = (list(out), {**info, "rounds": states[v]["rounds"], "n_red": len(states[v]["pos"])})
    return res


def task(ctx):
    from experiments.V2.method import l08_score

    a = ctx.args
    which = a["set"]
    items = item_set(OPEN_SETS[which], "dev")
    n_items = int(a.get("n_items", N_OPEN))
    if len(items) > n_items:
        keep = np.sort(np.random.default_rng(SUBSET_SEED).choice(len(items), n_items, replace=False))
        items = [items[i] for i in keep]
    if a.get("limit"):
        items = items[: int(a["limit"])]
    max_new = int(a.get("max_new", MAX_NEW))
    b = ctx.bundle()
    m = b.model
    sc, v2, vec, norm = setup(b)
    gate = (sc.sae8, sc.feats, sc.w1, sc.w0)
    clamp = (b.sae, DSG_FEATURES, 500.0, b.hook_name)
    conds = ["base", "A", "B"] + VARIANTS
    ctx.write_config(set=which, items=OPEN_SETS[which], n_items=len(items), max_new=max_new, threshold=THRESHOLD,
                     v2r=V2R, v2r_norm=norm, item_ids=[it.item_id for it in items])
    ctx.progress.update(items_total=len(items), items_done=0, force=True)
    priv = ctx.private_dir()
    ck = ItemCheckpoints(priv / "partial", ["P9-gen", which, [it.item_id for it in items], max_new, THRESHOLD, V2R],
                         every=int(a.get("resume_every", 10)))

    def one(it):
        prompt = open_prompt(it.question)
        toks, cand, _, _ = tokens_and_candidates(m, prompt)
        toks = toks.to(b.device)
        m.reset_hooks()
        ref = l08_score(m, sc.sae8, 8, sc.feats, sc.w1, sc.w0, toks)
        det = L08Detector(*gate)
        t0 = time.perf_counter()
        out, stop, fire, fscore, pscore = phase1(m, toks, "blocks.8.hook_resid_post", det, max_new)
        res = {"prompt_len": int(toks.shape[1]), "fire_step": fire, "prompt_score": pscore, "prompt_score_ref": ref,
               "n_cand_prompt": len(cand), "base": out, "t_base": time.perf_counter() - t0, "info": {}}
        if fire is None:
            for c in conds[1:]:
                res[c] = list(out)
            return res
        prefix = out[:fire]
        seq = torch.cat([toks, torch.tensor([prefix], dtype=torch.long, device=b.device)], 1) if prefix else toks
        rows = [{"clamp": True, "vec": None}, {"clamp": False, "vec": vec, "limit": V2R["limit"]}]
        (a_new, _), (b_new, _) = continue_batch(m, seq, max_new - len(prefix), rows, clamp, V2R["layer"])
        res["A"], res["B"] = prefix + a_new, prefix + b_new
        cand_s = list(cand) + list(range(toks.shape[1], seq.shape[1]))
        s_seq = float(sc.scores(seq)[0])
        red = Redactor(sc, seq, cand_s, s_seq, f"{which}|{it.item_id}")
        states = resolve_prompt(red, red.first_round())
        res["n_cand"] = len(cand_s)
        res["k1"] = int(red.k1 or 0)
        res["t_blame"] = red.t_blame
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for v, (new, info) in run_variants(m, red, states, seq, cand_s, max_new - len(prefix), gate, clamp,
                                           a_new).items():
            res[v] = prefix + new
            res["info"][v] = info
        torch.cuda.synchronize()
        res["t_variants"] = time.perf_counter() - t0
        return res

    done = ck.map(items, one, key=lambda it: it.item_id, on_done=ctx.progress.advance)
    m.reset_hooks()
    recs, gens = [], []
    for it, o in zip(items, done):
        for c in conds:
            text = m.tokenizer.decode(o[c], skip_special_tokens=True)
            gens.append({"item_id": it.item_id, "cond": c, "text": text})
            inf = o["info"].get(c, {})
            recs.append({"item_id": it.item_id, "cond": c, "fired": o["fire_step"] is not None,
                         "fire_step": o["fire_step"], "prompt_score": o["prompt_score"],
                         "prompt_score_ref": o["prompt_score_ref"], "prompt_len": o["prompt_len"],
                         "n_cand": o.get("n_cand"), "k1": o.get("k1"), "t_blame": o.get("t_blame"),
                         "t_base": o["t_base"], "n_red": inf.get("n_red"), "rounds": inf.get("rounds"),
                         "fallback": inf.get("fallback"), "fallback_where": inf.get("fallback_where"),
                         "refires": inf.get("refires"), "t_variants": o.get("t_variants"),
                         "n_tokens": len(o[c]), "text_hash": text_hash(text), "refusal": is_refusal(text),
                         "clarify": is_clarification(text), **gibberish(text, o[c]), "_text": text, "_ref": it.reference})
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
    met = {"set": which, "n": len(items), "fire_rate": float(base["fired"].mean()) if len(base) else None,
           "check": {"prompt_score_bit_identical": int((diff == 0).sum()), "n": int(len(base)),
                     "max_abs_diff": float(diff.max()) if len(diff) else 0.0}}
    for c in conds:
        dd = df[(df["cond"] == c) & df["fired"]]
        met[c] = {k: float(dd[k].astype(float).mean()) if len(dd) else None
                  for k in ("match", "gibberish", "refusal", "clarify")}
    ctx.write_metrics(met)
    ctx.finish({"view": f"p9-gen:{which}", "forget": met["R-a"]["match"], "check": met["check"]})
    ck.clear()
    return met["check"]
