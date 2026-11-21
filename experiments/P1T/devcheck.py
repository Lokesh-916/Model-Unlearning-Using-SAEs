"""P1T-devcheck check task (CPU): the frozen `p1t-l08` method replayed on DEV, compared with P1.

For each P1T-devcheck MCQ run (DEV, bs 1; WMDP-Cyber clean / pad400 / split3 + benign computing): per item, the online
pass-1 L08 score vs P1's logged score (runs/P1/questions `a-sae-cusum|L08`), the online decision vs P1's
(logged > THRESHOLD), and the online prediction vs P1's composite (P1 clampF prediction if fired, else P1 base).
FAILS (so every P1T TEST job stays blocked) if any decision differs, any logged prompt is missing, or the composite
prediction mismatch share exceeds DEVCHECK_MAX_PRED_MISMATCH. Bit-identity of scores is reported, not required.
"""
import json

import numpy as np
import pandas as pd

from dsgx import paths

from experiments.P1T.common import BENIGN_CS, DEVCHECK_MAX_PRED_MISMATCH, THRESHOLD

COL = "a-sae-cusum|L08"


def logged() -> pd.DataFrame:
    q = pd.read_parquet(paths.runs_dir() / "P1" / "questions" / "items.parquet")
    q = q[(q["skip"] == "") & q["cond"].isin(["cyber_clean", "cyber_pad400", "cyber_split3", "benign_cs"])]
    return q[["cond", "item_id", COL]].rename(columns={COL: "logged"})


def p1_run(cond: str, atk: str) -> pd.DataFrame:
    hits = [h for h in sorted((paths.runs_dir() / "P1").glob(f"*__cyber-p1-{cond}-{atk}__*/items.parquet"))
            if (h.parent / "DONE").exists()]
    if not hits:
        raise RuntimeError(f"P1 run {cond}-{atk} missing")
    return pd.read_parquet(hits[-1])[["item_id", "dataset", "pred"]]


def online(exp_id: str) -> pd.DataFrame:
    rows = []
    for d in sorted((paths.runs_dir() / exp_id).glob(f"{exp_id}__*")):
        if not (d / "DONE").exists():
            continue
        lab = json.loads((d / "config.json").read_text())["config"]["dataset_label"]   # p1tdev-<atk>
        atk = lab.split("-", 1)[1]
        it = pd.read_parquet(d / "items.parquet")[["item_id", "dataset", "gate_score", "gate_fired", "pred"]]
        it["atk"] = atk
        it["cond"] = np.where(it["dataset"].isin(BENIGN_CS), "benign_cs", f"cyber_{atk}")
        rows.append(it)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def composite_pred(on: pd.DataFrame) -> pd.Series:
    out = pd.Series(index=on.index, dtype="float64")
    for atk, g in on.groupby("atk"):
        b = p1_run("base", atk).set_index(["dataset", "item_id"])["pred"]
        f = p1_run("clampF", atk).set_index(["dataset", "item_id"])["pred"]
        k = list(zip(g["dataset"], g["item_id"]))
        bp, fp = b.reindex(k).values, f.reindex(k).values
        out.loc[g.index] = np.where(g["gate_fired"].astype(bool).values, fp, bp)
    return out


def compare(lg: pd.DataFrame, on: pd.DataFrame) -> dict:
    on = on.copy()
    on["p1_composite"] = composite_pred(on)
    m = lg.merge(on, on=["cond", "item_id"], how="left", indicator=True)
    missing = m[m["_merge"] != "both"]
    m = m[m["_merge"] == "both"].copy()
    m["dec_p1"] = m["logged"].astype(float) > THRESHOLD
    m["dec_on"] = m["gate_score"].astype(float) > THRESHOLD
    m["flag"] = m["gate_fired"].astype(bool)
    m["pred_eq"] = m["pred"].astype(float) == m["p1_composite"].astype(float)
    per = {}
    for c, g in m.groupby("cond"):
        diff = (g["gate_score"].astype(float) - g["logged"].astype(float)).abs()
        per[c] = {"n": int(len(g)), "score_bit_identical": int((diff == 0).sum()), "max_abs_diff": float(diff.max()),
                  "decision_diff": int((g["dec_p1"] != g["dec_on"]).sum()),
                  "flag_vs_threshold_diff": int((g["dec_on"] != g["flag"]).sum()),
                  "pred_mismatch": int((~g["pred_eq"]).sum()), "fire_rate": float(g["dec_on"].mean())}
    tot = {k: sum(v[k] for v in per.values()) for k in ("n", "score_bit_identical", "decision_diff",
                                                         "flag_vs_threshold_diff", "pred_mismatch")}
    share = tot["pred_mismatch"] / max(1, tot["n"])
    ok = (tot["decision_diff"] == 0 and tot["flag_vs_threshold_diff"] == 0 and len(missing) == 0
          and share <= DEVCHECK_MAX_PRED_MISMATCH)
    return {"ok": bool(ok), "n_logged": int(len(lg)), "n_missing": int(len(missing)), "totals": tot,
            "pred_mismatch_share": share, "per_cond": per, "threshold": THRESHOLD}


def render(r: dict) -> str:
    L = [f"P1T-devcheck (DEV only, labpc): frozen `p1t-l08` replayed on {r['n_logged']} logged P1 DEV prompts. "
         f"Threshold {r['threshold']!r}. Missing: {r['n_missing']}. **{'PASS' if r['ok'] else 'FAIL'}**: decision "
         f"differences {r['totals']['decision_diff']}, scores bit-identical {r['totals']['score_bit_identical']} / "
         f"{r['totals']['n']}, predictions differing from P1's composite {r['totals']['pred_mismatch']} "
         f"({r['pred_mismatch_share']:.4f}).", "",
         "| condition | n | scores bit-identical | max abs diff | decisions differing | predictions differing | fire rate |",
         "|---|---|---|---|---|---|---|"]
    for c, v in sorted(r["per_cond"].items()):
        L.append(f"| {c} | {v['n']} | {v['score_bit_identical']} | {v['max_abs_diff']:.3g} | {v['decision_diff']} | "
                 f"{v['pred_mismatch']} | {v['fire_rate']:.4f} |")
    return "\n".join(L)


def task(ctx):
    from dsgx.util import atomic_write_json

    lg = logged()
    on = online(ctx.exp_id)
    if ctx.args.get("smoke") or ctx.smoke:   # smoke runs score only a few items per condition
        lg = lg.merge(on[["cond", "item_id"]].drop_duplicates(), on=["cond", "item_id"])
    r = compare(lg, on)
    d = ctx.run_dir()
    atomic_write_json(d / "DEVCHECK.json", r)
    (d / "DEVCHECK.md").write_text(render(r) + "\n")
    ctx.write_metrics({"ok": r["ok"], "totals": r["totals"], "n_missing": r["n_missing"],
                       "pred_mismatch_share": r["pred_mismatch_share"]})
    if not r["ok"]:
        raise RuntimeError(f"P1T-devcheck FAILED: {r['totals']}, missing {r['n_missing']} -- do not run TEST")
    ctx.finish({"view": "p1t-devcheck", "forget": None, "ok": True})
    return r["totals"]
