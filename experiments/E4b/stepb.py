"""E4b Step B summary (CPU): attacks a benign-biology-quiet L08 gate could open. DEV only; ids and metrics only.

Inputs: E4b runs (base, dsg, sg1, sgF = clamp forced on, R6 forced on; WMDP-Bio dev, bs 1) per Step B attack, the
E4b scores task (per-prompt L08 / reference scores), the Step 0 thresholds (recomputed from E4a's MMLU calibration
scores with the same rule). Composites: L08 -> clamp = sgF prediction where L08 fires, else base; L08 -> R6 likewise.
DSG and StreamGuard v1 = their own gated runs. Recall = fire share. Paired bootstrap + exact McNemar vs DSG and
StreamGuard v1. "Opened" = an L08 composite significantly above StreamGuard v1 on that attack (reported only;
Step B is not part of rule v2).
"""
import numpy as np
import pandas as pd

from dsgx import paths

from experiments.E2d.summary import _ci, _load_runs, _paired
from experiments.E2e.analysis import _d, _f
from experiments.E4b.common import (ALPHAS, GATE_COL, OP_TEXT, REF_COLS, STEP_B_ATTACKS, STEP_B_CONDS,
                                    conformal_threshold)
from experiments.E4b.stepa import calibration_scores, load_scores

NAMES = {"base": "base (no gate)", "dsg": "DSG (rho + clamp)", "sg1": "StreamGuard v1 (CUSUM + clamp)"}


def collect(exp="E4b", nb=2000, e4a_exp="E4a"):
    runs = _load_runs(exp)
    miss = [(c, a) for c in STEP_B_CONDS for a in STEP_B_ATTACKS if (c, a) not in runs]
    if miss:
        raise RuntimeError(f"missing runs: {miss}")
    cal = calibration_scores(load_scores(e4a_exp)["clean"])
    ops = {k: conformal_threshold(cal, a) for k, a in ALPHAS.items()}
    sd = pd.read_parquet(paths.runs_dir() / exp / "scores" / "items.parquet")
    sd = sd[sd["skip"] == ""].copy()
    sd["item_id"] = sd["key"].str.split("|", n=1).str[1]
    import json
    sg1_tau = json.loads((paths.runs_dir() / exp / "scores" / "metrics.json").read_text())["ref_own_thresholds"][REF_COLS["sg1"]]
    out = {"hardware": "labpc", "ops": ops, "attacks": STEP_B_ATTACKS, "rows": {}, "checks": {}}
    for a in STEP_B_ATTACKS:
        it = {c: runs[(c, a)]["items"] for c in STEP_B_CONDS}
        base = it["base"]
        idx = base.index
        for c, x in it.items():
            if not x.index.equals(idx):
                raise RuntimeError(f"{c}/{a}: items differ from base")
        s = sd[sd["attack"] == a].assign(dataset="wmdp-bio").set_index(["dataset", "item_id"]).reindex(idx)
        if s[GATE_COL].isna().any():
            raise RuntimeError(f"{a}: {int(s[GATE_COL].isna().sum())} items without scores")
        gold = base["gold"].values
        corr = {c: (it[c]["pred"].values == gold) for c in STEP_B_CONDS}
        fired = {"dsg": it["dsg"]["gate_fired"].astype(bool).values, "sg1": it["sg1"]["gate_fired"].astype(bool).values}
        # checks: the scores task rebuilt the same prompts (StreamGuard v1 decision), forced clamp == gated clamp
        f_sg = fired["sg1"]
        e4 = s[REF_COLS["sg1"]].values > sg1_tau
        out["checks"][a] = {"n": int(len(idx)), "sg1_decision_agree": float((e4 == f_sg).mean()),
                            "sgF_eq_sg1_on_fired": float((it["sgF"]["pred"].values[f_sg] == it["sg1"]["pred"].values[f_sg]).mean()) if f_sg.any() else None,
                            "n_sg1_fired": int(f_sg.sum())}
        rows = {}
        for c in ["base", "dsg", "sg1"]:
            rows[NAMES[c]] = {"fire": _ci(fired[c].astype(float), nb) if c in fired else None,
                              "forget": _ci(corr[c], nb), "vs_dsg": None if c == "dsg" else _paired(corr[c], corr["dsg"], nb),
                              "vs_sg1": None if c == "sg1" else _paired(corr[c], corr["sg1"], nb)}
        for op, v in ops.items():
            f = s[GATE_COL].values > v["threshold"]
            for r, rn in (("sgF", "clamp"), ("R6", "R6")):
                p = np.where(f, corr[r], corr["base"])
                rows[f"L08 {op} → {rn}"] = {"fire": _ci(f.astype(float), nb), "forget": _ci(p, nb),
                                           "vs_dsg": _paired(p, corr["dsg"], nb), "vs_sg1": _paired(p, corr["sg1"], nb),
                                           "a_fire": float(p[f].mean()) if f.any() else None,
                                           "base_miss": float(corr["base"][~f].mean()) if (~f).any() else None,
                                           "n_fire": int(f.sum())}
        rows["R6 forced on every prompt"] = {"fire": None, "forget": _ci(corr["R6"], nb), "vs_dsg": _paired(corr["R6"], corr["dsg"], nb),
                                             "vs_sg1": _paired(corr["R6"], corr["sg1"], nb)}
        rows["clamp forced on every prompt"] = {"fire": None, "forget": _ci(corr["sgF"], nb), "vs_dsg": _paired(corr["sgF"], corr["dsg"], nb),
                                                "vs_sg1": _paired(corr["sgF"], corr["sg1"], nb)}
        out["rows"][a] = rows
        out["checks"][a]["median_tokens"] = float(s["n_tok"].median())
    out["opened"] = {a: [k for k, v in out["rows"][a].items() if k.startswith("L08") and v["vs_sg1"]["sig_higher"]]
                     for a in STEP_B_ATTACKS}
    out["runs"] = {f"{c}|{a}": runs[(c, a)]["run"] for c in STEP_B_CONDS for a in STEP_B_ATTACKS}
    return out


ATT_TEXT = {"bbio400": "B1 dilution, benign-biology filler 400 (before)",
            "bbio1600": "B1 dilution, benign-biology filler 1600 (before)",
            "midpad400": "B7 mid-question pad 400 (WikiText)", "interleave2": "B7 interleave k=2 (WikiText)"}


def render(s):
    L = ["Hardware: labpc. WMDP-Bio DEV (@forget), bs 1, identical items per attack. Recall = fire share of the gate; "
         "forget = accuracy after the gate's response (lower is better). 95 % bootstrap CIs; paired = row − comparator, "
         "paired bootstrap [95 % CI] and exact McNemar p; ↑ / ↓ significant. L08 thresholds: "
         + ", ".join(f"{OP_TEXT[k]} t = {v['threshold']:.4f}" for k, v in s["ops"].items()) + ".", ""]
    for a, rows in s["rows"].items():
        ck = s["checks"][a]
        L += [f"### {ATT_TEXT[a]}", "",
              f"n {ck['n']}, median prompt length {ck['median_tokens']:.0f} tokens. Scores task vs StreamGuard v1 run, same "
              f"gate decision: {ck['sg1_decision_agree']:.4f}. Clamp forced on == StreamGuard v1's "
              f"gated prediction on its {ck['n_sg1_fired']} fired prompts: {ck['sgF_eq_sg1_on_fired'] if ck['sgF_eq_sg1_on_fired'] is None else round(ck['sgF_eq_sg1_on_fired'], 4)}.", "",
              "| condition | recall (fire share) | forget | vs DSG | vs StreamGuard v1 |", "|---|---|---|---|---|"]
        for k, v in rows.items():
            L.append(f"| {k} | {_f(v['fire']) if v['fire'] else '–'} | {_f(v['forget'])} | {_d(v['vs_dsg'])} | {_d(v['vs_sg1'])} |")
        L.append("")
    op = {a: v for a, v in s["opened"].items() if v}
    L.append("**Opened (an L08 composite significantly above StreamGuard v1):** "
             + ("; ".join(f"{ATT_TEXT[a]}: {', '.join(v)}" for a, v in op.items()) if op else "none") + ".")
    return "\n".join(L)


def task(ctx):
    from dsgx.util import atomic_write_json

    exp = ctx.exp_id
    s = collect(exp, int(ctx.args.get("n_boot", 2000)), ctx.args.get("e4a_exp", "E4a"))
    d = ctx.run_dir()
    atomic_write_json(d / "E4B_STEPB.json", s)
    (d / "E4B_STEPB.md").write_text(render(s) + "\n")
    ctx.write_metrics({"opened": s["opened"]})
    ctx.finish({"view": "e4b-stepb", "forget": None, "opened": s["opened"]})
    return s["opened"]
