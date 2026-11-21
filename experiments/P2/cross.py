"""P2 cross task (GPU, bs 1, item-level resume): only what no logged DEV run has.

Items (prompts built in memory exactly as the E4a / P1 questions tasks; ids only are stored):
  bio/wmdp_{clean,pad400,split3}     both scores; the bio score must equal E4a's logged L08 score (else STOP);
                                     predictions with clamp set (cyber,) if the cyber gate fires, (bio, cyber) if both
  cyber/cyber_{clean,pad400,split3}  both scores; the cyber score must equal P1's logged L08 score (else STOP);
                                     predictions (bio,) if the bio gate fires, (bio, cyber) if both
  bio/{benign_bio,mmlu}              both scores are logged; only items where both logged gates fire (prediction
                                     (bio, cyber)) or check items go to the GPU; recomputed scores must equal both
                                     logged scores (else STOP)
Check items (sha256(key) % 20 == 0): also the prediction of the item's logged clamp set ((), (bio,) or (cyber,)),
compared with the logged base / sgF / clampF prediction in the summary (wiring check, must be 100 %).

items.parquet: key, side, cond, dataset, item_id, n_tok, bio_score, cyber_score, logged scores, set, pred_<set> for
every computed set, check flag. No text.
"""
import numpy as np
import pandas as pd
import torch

from dsgx import paths
from dsgx.itemckpt import ItemCheckpoints

from experiments.P2.common import (BIO_CONDS, BIO_THRESHOLD, CYBER_CONDS, CYBER_THRESHOLD, E4A_EXP, GATE_COL,
                                   P1_EXP, is_check, set_name)


def logged_scores():
    out = {}
    for side, exp in (("bio", E4A_EXP), ("cyber", P1_EXP)):
        q = pd.read_parquet(paths.runs_dir() / exp / "questions" / "items.parquet", columns=["key", "cond", "skip", GATE_COL])
        q = q[q["skip"] == ""].copy()
        q["item_id"] = q["key"].str.split("|", n=1).str[1]
        out[side] = q
    return out


def _by_item(q, conds):
    g = q[q["cond"].isin(conds)]
    s = g.set_index("item_id")[GATE_COL].astype(float)
    if s.index.duplicated().any():
        raise RuntimeError(f"duplicate items in logged scores {conds}")
    return s


def plan(limit=None):
    """[(key, side, cond, (attack, item), logged_bio, logged_cyber)]"""
    from experiments.E4a.common import question_items as bio_items
    from experiments.P1.common import question_items as cyber_items

    L = logged_scores()
    bq, cq = L["bio"], L["cyber"]
    b_logged = bq.set_index("key")[GATE_COL].astype(float)
    c_logged = cq.set_index("key")[GATE_COL].astype(float)
    c_clean = _by_item(cq, ["mmlu", "benign_cs"])        # cyber score on clean MMLU items (cyber @utility + benign cs)
    out = []
    for key, cond, ai in bio_items(limit):
        if cond in BIO_CONDS:
            if key in b_logged.index:
                out.append((f"bio/{key}", "bio", cond, ai, float(b_logged[key]), None))
        elif cond in ("benign_bio", "mmlu"):
            iid = key.split("|", 1)[1]
            if key in b_logged.index and iid in c_clean.index and (
                    (b_logged[key] > BIO_THRESHOLD and c_clean[iid] > CYBER_THRESHOLD) or is_check(f"bio/{key}")):
                out.append((f"bio/{key}", "bio", cond, ai, float(b_logged[key]), float(c_clean[iid])))
    for key, cond, ai in cyber_items(limit):
        if cond in CYBER_CONDS and key in c_logged.index:
            out.append((f"cyber/{key}", "cyber", cond, ai, None, float(c_logged[key])))
    return out


def task(ctx):
    from dsgx.attacks.registry import make_attack
    from dsgx.eval.mcq_eval import ANSWER_STRINGS, answer_probs

    from experiments.E4a.common import ATTACKS as BIO_ATT
    from experiments.P1.common import ATTACKS as CY_ATT
    from experiments.P2.method import TwoGates

    a = ctx.args
    b = ctx.bundle()
    m = b.model
    tg = TwoGates(b)
    atk = {"bio": {k: make_attack({**v, "_exp_id": E4A_EXP}) for k, v in BIO_ATT.items()},
           "cyber": {k: make_attack({**v, "_exp_id": P1_EXP}) for k, v in CY_ATT.items()}}
    items = plan(a.get("limit"))
    ans = m.to_tokens(ANSWER_STRINGS, prepend_bos=False).flatten()
    ctx.write_config(n_items=len(items), gates=tg.info, check_mod=20)
    ctx.progress.update(items_total=len(items), items_done=0, force=True)

    @torch.no_grad()
    def predict(tb, n, s):
        tg.install(s, [n])
        try:
            p = answer_probs(m, tb, torch.tensor([n - 1], device=tb.device), ans).float().cpu().numpy()[0]
        finally:
            tg.clear()
        return int(np.argmax(p))

    @torch.no_grad()
    def one(it):
        key, side, cond, (an, item), lb, lc = it
        prompt, info = atk[side][an].prompt(item)
        if prompt is None:
            return {"key": key, "side": side, "cond": cond, "skip": str(info.get("skip"))}
        tb = m.to_tokens([prompt], padding_side="right", prepend_bos=False).to(b.device)
        n = len(m.to_tokens(prompt, prepend_bos=False)[0])
        sb, sc = tg.scores(tb, n)
        for name, got, want in (("bio", sb, lb), ("cyber", sc, lc)):
            if want is not None and got != want:
                raise RuntimeError(f"STOP: {key}: recomputed {name} score differs from the logged one "
                                   f"(|diff| {abs(got - want):.3g})")
        s = tg.decide(sb, sc)
        need = []
        if cond in BIO_CONDS:
            logged_set = ("bio",) if "bio" in s else ()
            if "cyber" in s:
                need.append(("cyber",))
        elif cond in CYBER_CONDS:
            logged_set = ("cyber",) if "cyber" in s else ()
            if "bio" in s:
                need.append(("bio",))
        else:
            logged_set = s if len(s) < 2 else None
        if len(s) == 2:
            need.append(("bio", "cyber"))
        chk = is_check(key) and logged_set is not None
        if chk and logged_set not in need:
            need.append(logged_set)
        row = {"key": key, "side": side, "cond": cond, "skip": "", "item_id": key.split("|", 1)[1],
               "n_tok": int(n), "bio_score": sb, "cyber_score": sc,
               "logged_bio": np.nan if lb is None else lb, "logged_cyber": np.nan if lc is None else lc,
               "set": set_name(s), "check": bool(chk), "check_set": set_name(logged_set) if chk else ""}
        for x in need:
            row[f"pred_{set_name(x)}"] = predict(tb, n, x)
        return row

    ck = ItemCheckpoints(ctx.run_dir() / "partial", ["P2-cross", [k for k, *_ in items]])
    rows = ck.map(items, one, key=lambda it: it[0], on_done=ctx.progress.advance)
    df = pd.DataFrame(rows)
    for c in ["pred_none", "pred_bio", "pred_cyber", "pred_bio+cyber"]:
        if c not in df:
            df[c] = np.nan
    df.to_parquet(ctx.run_dir() / "items.parquet", index=False)
    ok = df[df["skip"] == ""]
    m_out = {"n": int(len(df)), "n_skipped": int((df["skip"] != "").sum()),
             "by_cond_set": {f"{c}|{s}": int(v) for (c, s), v in ok.groupby(["cond", "set"]).size().items()},
             "n_pred": {c: int(ok[c].notna().sum()) for c in ok.columns if c.startswith("pred_")},
             "n_check": int(ok["check"].sum())}
    ctx.write_metrics(m_out)
    ctx.finish({"view": "p2-cross", "forget": None})
    ck.clear()
    return m_out
