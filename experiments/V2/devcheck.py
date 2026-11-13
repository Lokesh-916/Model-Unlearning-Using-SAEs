"""V2-devcheck check task (CPU): the frozen StreamGuard v2 method replayed on every DEV prompt E4a / E4b scored.

For each V2-devcheck MCQ run (DEV split, bs 1): per item, the online pass-1 L08 score (items.parquet gate_score) vs
the logged score (E4a questions `a-sae-cusum|L08` for wmdp_clean / pad400 / split3 / hindi, benign_bio, mmlu; E4b
Step B scores for bbio400 / bbio1600 / midpad400 / interleave2), and the online decision (score > frozen threshold)
vs E4b's (logged score > 7.06202507019043). FAILS (exit != 0, so every V2-test job stays blocked) if any decision
differs or any logged prompt is missing. Bit-identity of the scores is reported, not required.
"""
import json

import numpy as np
import pandas as pd

from dsgx import paths

from experiments.V2.common import THRESHOLD, THRESHOLD_E4B

COL = "a-sae-cusum|L08"
E4A_CONDS = {"wmdp_clean", "wmdp_pad400", "wmdp_split3", "wmdp_hindi", "benign_bio", "mmlu"}


def logged() -> pd.DataFrame:
    rd = paths.runs_dir()
    q = pd.read_parquet(rd / "E4a" / "questions" / "items.parquet")
    s = pd.read_parquet(rd / "E4b" / "scores" / "items.parquet")
    q = q[q["skip"] == ""][["key", "cond", COL]]
    s = s[s["skip"] == ""][["key", "attack", COL]].rename(columns={"attack": "cond"})
    out = pd.concat([q, s], ignore_index=True).rename(columns={COL: "logged"})
    out["item_id"] = out["key"].str.split("|", n=1).str[1]
    return out


def online(exp_id: str) -> pd.DataFrame:
    rows = []
    for d in sorted((paths.runs_dir() / exp_id).glob(f"{exp_id}__*")):
        if not (d / "DONE").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())["config"]
        lab = cfg["dataset_label"]          # sg2dev-<cond> or sg2dev-clean
        it = pd.read_parquet(d / "items.parquet")[["item_id", "dataset", "gate_score", "gate_fired"]]
        cond = lab.split("-", 1)[1]
        if cond == "clean":
            it["cond"] = np.where(it["dataset"] == "wmdp-bio", "wmdp_clean",
                                  np.where(it["dataset"].isin(["college_biology", "high_school_biology", "anatomy",
                                                                "virology", "medical_genetics"]), "benign_bio", "mmlu"))
        else:
            it["cond"] = cond
        rows.append(it)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def compare(lg: pd.DataFrame, on: pd.DataFrame) -> dict:
    m = lg.merge(on, on=["cond", "item_id"], how="left", indicator=True)
    missing = m[m["_merge"] != "both"]
    m = m[m["_merge"] == "both"].copy()
    m["dec_e4b"] = m["logged"].astype(float) > THRESHOLD_E4B
    m["dec_v2"] = m["gate_score"].astype(float) > THRESHOLD
    m["dec_v2_online_flag"] = m["gate_fired"].astype(bool)
    per = {}
    for c, g in m.groupby("cond"):
        diff = (g["gate_score"].astype(float) - g["logged"].astype(float)).abs()
        per[c] = {"n": int(len(g)), "score_bit_identical": int((diff == 0).sum()), "max_abs_diff": float(diff.max()),
                  "decision_diff": int((g["dec_e4b"] != g["dec_v2"]).sum()),
                  "flag_vs_threshold_diff": int((g["dec_v2"] != g["dec_v2_online_flag"]).sum()),
                  "fire_rate": float(g["dec_v2"].mean())}
    tot = {k: sum(v[k] for v in per.values()) for k in ("n", "score_bit_identical", "decision_diff",
                                                         "flag_vs_threshold_diff")}
    ok = tot["decision_diff"] == 0 and tot["flag_vs_threshold_diff"] == 0 and len(missing) == 0
    return {"ok": bool(ok), "n_logged": int(len(lg)), "n_missing": int(len(missing)), "totals": tot, "per_cond": per,
            "threshold": THRESHOLD, "threshold_e4b": THRESHOLD_E4B}


def render(r: dict) -> str:
    L = [f"V2-devcheck (DEV only, labpc): frozen StreamGuard v2 replayed on {r['n_logged']} logged DEV prompts. "
         f"Threshold {r['threshold']!r} (E4b {r['threshold_e4b']!r}). Missing: {r['n_missing']}. "
         f"**{'PASS' if r['ok'] else 'FAIL'}**: decision differences {r['totals']['decision_diff']}, "
         f"scores bit-identical {r['totals']['score_bit_identical']} / {r['totals']['n']}.", "",
         "| condition | n | scores bit-identical | max abs diff | decisions differing | fire rate |",
         "|---|---|---|---|---|---|"]
    for c, v in sorted(r["per_cond"].items()):
        L.append(f"| {c} | {v['n']} | {v['score_bit_identical']} | {v['max_abs_diff']:.3g} | {v['decision_diff']} | "
                 f"{v['fire_rate']:.4f} |")
    return "\n".join(L)


def task(ctx):
    from dsgx.util import atomic_write_json

    lg = logged()
    on = online(ctx.exp_id)
    if ctx.args.get("smoke"):  # smoke runs score only a few items per condition
        lg = lg.merge(on[["cond", "item_id"]].drop_duplicates(), on=["cond", "item_id"])
    r = compare(lg, on)
    d = ctx.run_dir()
    atomic_write_json(d / "DEVCHECK.json", r)
    (d / "DEVCHECK.md").write_text(render(r) + "\n")
    ctx.write_metrics({"ok": r["ok"], "totals": r["totals"], "n_missing": r["n_missing"]})
    if not r["ok"]:
        raise RuntimeError(f"V2-devcheck FAILED: {r['totals']}, missing {r['n_missing']} -- do not run TEST")
    ctx.finish({"view": "v2-devcheck", "forget": None, "ok": True})
    return r["totals"]
