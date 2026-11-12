"""E4b Step A (CPU, offline): L08 gate -> clamp / R5 / R6 composites from existing per-item DEV predictions.

Inputs (all DEV, bs 1, identical items): E2d runs base / dsg / sg1 / R5, E2e runs R6 / sgF (clamp forced on), the
E4a questions run (per-prompt L08 SAE-CUSUM score and the DSG / StreamGuard v1 reference scores). Composite:
pred = responder's forced-on prediction where L08 fires (score > t_op), else base. Rule v2 unchanged
(experiments.E2e.common.select_v2). Ids and metrics only.

Run: python -m experiments.E4b.stepa [--n-boot 2000]  -> $DSG_RESULTS/runs/E4b/stepA/{E4B_STEPA.json,.md}
"""
import json
import subprocess

import numpy as np
import pandas as pd

from dsgx import paths

from experiments.E2d.summary import COLS, _ci, _parts, calib_ids
from experiments.E2e.analysis import ATT, CN, Data, _d, _f, ref_cols, rule_input, summarise
from experiments.E2e.common import FPR_MAX, UTIL_MARGIN, select_v2
from experiments.E4b.common import (ALPHAS, E4A_EXP, GATE_COL, OP_TEXT, REF_COLS, RESP_TEXT, RESPONDERS,
                                    conformal_threshold)

E4A_ATT = {"wmdp_clean": "clean", "wmdp_pad400": "pad400", "wmdp_split3": "split3", "wmdp_hindi": "hindi",
           "benign_bio": "clean", "mmlu": "clean"}


def load_scores(e4a_exp=E4A_EXP):
    """{attack: DataFrame indexed (dataset, item_id) with the L08 and reference scores}; skipped rows dropped."""
    q = pd.read_parquet(paths.runs_dir() / e4a_exp / "questions" / "items.parquet")
    q = q[q["skip"] == ""].copy()
    q["item_id"] = q["key"].str.split("|", n=1).str[1]
    q["dataset"] = q["item_id"].str.rsplit(":", n=1).str[0]
    q["attack"] = q["cond"].map(E4A_ATT)
    cols = [GATE_COL] + list(REF_COLS.values())
    out = {}
    for a, g in q.groupby("attack"):
        d = g.set_index(["dataset", "item_id"])[cols].astype(float).sort_index()
        if d.index.duplicated().any():
            raise RuntimeError(f"{a}: duplicate items in E4a scores")
        out[a] = d
    return out


def calibration_scores(sc_clean):
    cal = calib_ids()
    m = sc_clean.index.isin(list(cal))
    if m.sum() != len(cal):
        raise RuntimeError(f"calibration prompts: {m.sum()} of {len(cal)} found in the E4a scores")
    return sc_clean[GATE_COL].values[m]


class Gate:
    """L08 fire masks aligned to the E2d base index per attack."""

    def __init__(self, D, scores, thr):
        self.thr = thr
        self.fired, self.score = {}, {}
        for a in ATT:
            idx = D.base[a].index
            s = scores[a]
            miss = idx.difference(s.index)
            if len(miss):
                raise RuntimeError(f"{a}: {len(miss)} E2d items without an E4a score")
            self.score[a] = s.reindex(idx)
        self.set(thr)

    def set(self, thr):
        self.thr = thr
        self.fired = {a: self.score[a][GATE_COL].values > thr for a in ATT}


def candidate(D, G, name, resp):
    col, tiers = {}, {}
    for a in ATT:
        f = G.fired[a]
        p = D.pred("base", a).copy()
        p[f] = D.pred(resp, a)[f]
        col.update({c: v.values for c, v in D.columns(a, p).items()})
        mh = D.masks(a, f)
        for c in mh:
            tiers[c] = {"hard": float(mh[c].mean()), "soft": 0.0, "any": float(mh[c].mean()),
                        "n": int(len(mh[c])), "n_hard": int(mh[c].sum()), "n_soft": 0}
    return {"name": name, "soft": None, "hard": resp, "col": col, "tiers": tiers}


def rfd(D, name, fired, col):
    """E1/E2d style: R (fire share on forget per attack), F (fire share on utility / benign biology),
    a_fire / a_miss (forget clean, composite on fired / base on missed), damage per fire = base − composite
    accuracy on the fired prompts. fired: {attack: mask on the base index}; col: correct-vectors per column."""
    row = {"name": name}
    for a in ATT:
        fm = D.masks(a, fired[a])
        bm = D.columns(a, D.pred("base", a))
        for c in fm:
            f = fm[c]
            b = bm[c].values
            o = np.asarray(col[c], bool)
            row[f"fire_{c}"] = float(f.mean())
            row[f"n_fire_{c}"] = int(f.sum())
            row[f"acc_fire_{c}"] = float(o[f].mean()) if f.any() else None
            row[f"base_fire_{c}"] = float(b[f].mean()) if f.any() else None
            row[f"acc_miss_{c}"] = float(b[~f].mean()) if (~f).any() else None
            row[f"damage_per_fire_{c}"] = float(b[f].mean() - o[f].mean()) if f.any() else None
    return row


def collect(nb=2000, e2e_exp="E2e", e2d_exp="E2d", e4a_exp=E4A_EXP):
    D = Data("B", e2e_exp, e2d_exp)
    scores = load_scores(e4a_exp)
    cal = calibration_scores(scores["clean"])
    ops = {k: conformal_threshold(cal, a) for k, a in ALPHAS.items()}
    mm = scores["clean"].index.get_level_values(0) != "wmdp-bio"
    # Step 0 report: distribution of the L08 score on MMLU utility dev (the 6076 @utility items)
    uidx = _parts(D.base["clean"], "clean")["utility"].index
    us = scores["clean"].reindex(uidx)[GATE_COL].values
    vc = pd.Series(us).round(4).value_counts()
    step0 = {"n_utility": int(len(us)), "n_distinct": int(len(vc)), "mode": float(vc.index[0]),
             "n_at_mode": int(vc.iloc[0]), "share_at_mode": float(vc.iloc[0] / len(us)),
             "n_below_mode": int((us < vc.index[0] - 1e-6).sum()),
             "quantiles": {str(p): float(np.quantile(us, p)) for p in [0.5, 0.9, 0.95, 0.985, 0.99, 0.995, 1.0]},
             "e4a_threshold_q95_all_mmlu": float(np.quantile(scores["clean"][GATE_COL].values[mm], 0.95, method="higher")),
             "cal_top20": [float(x) for x in np.sort(cal)[::-1][:20]],
             "ops": {k: {**v, "fire_utility_all": float((us > v["threshold"]).mean())} for k, v in ops.items()}}
    G = Gate(D, scores, ops["op5"]["threshold"])
    # consistency: E4a's StreamGuard v1 reference score vs the E2d sg1 run's gate decision (same prompts?)
    agree = {}
    for a in ATT:
        e4 = G.score[a][REF_COLS["sg1"]].values > D.tau
        agree[a] = {"n": int(len(e4)), "agree": float((e4 == D.fired[a]).mean()),
                    "n_disagree": int((e4 != D.fired[a]).sum())}
    refs = {"dsg": ref_cols(D, "dsg"), "sg1": ref_cols(D, "sg1"), "base": ref_cols(D, "base")}
    lv_ref = {r: {c: _ci(refs[r][c], nb) for c in COLS} for r in refs}
    cands, rfds = [], []
    dsg_fired = {a: D.items[("dsg", a)]["gate_fired"].astype(bool).values for a in ATT}
    rfds.append(rfd(D, "DSG (rho + clamp)", dsg_fired, refs["dsg"]))
    rfds.append(rfd(D, "StreamGuard v1 (CUSUM + clamp)", D.fired, refs["sg1"]))
    for op in ALPHAS:
        G.set(ops[op]["threshold"])
        for r in RESPONDERS:
            c = candidate(D, G, f"L08 {op} → {RESP_TEXT[r]}", r)
            c["op"], c["resp"] = op, r
            s = summarise(D, c, refs, nb)
            s["op"], s["resp"] = op, r
            cands.append(s)
            rfds.append(rfd(D, c["name"], G.fired, c["col"]))
    # utility without the 960 calibration prompts (they set the threshold)
    calset = calib_ids()
    keep = ~uidx.isin(list(calset))
    util_ex = {"n": int(keep.sum())}
    for r in ["base", "dsg", "sg1"]:
        util_ex[r] = _ci(np.asarray(refs[r]["utility"])[keep], nb)
    for op in ALPHAS:
        G.set(ops[op]["threshold"])
        for r in RESPONDERS:
            c = candidate(D, G, "", r)
            util_ex[f"L08 {op} → {RESP_TEXT[r]}"] = _ci(np.asarray(c["col"]["utility"])[keep], nb)
    sel = select_v2([rule_input(s) for s in cands], lv_ref["dsg"]["utility"]["mean"])
    return {"hardware": "labpc", "step0": step0, "sg1_tau": D.tau, "sg1_ref_agreement": agree,
            "ref_levels": lv_ref, "candidates": cands, "rfd": rfds, "util_excl_calib": util_ex,
            "selection": {"dsg_utility": sel["dsg_utility"],
                          "rows": [{k: r[k] for k in ("name", "order", "elig", "worst_forget", "n_failed", "utility",
                                                      "fpr", "benign_bio", "forget")} for r in sel["rows"]],
                          "selected": sel["selected"]["name"] if sel["selected"] else None,
                          "closest": sel["closest"]["name"] if sel["closest"] else None},
            "runs": {f"{c}|{a}": r["run"] for (c, a), r in D.runs.items()}}


def _p(x, nd=3):
    return "–" if x is None else f"{x:.{nd}f}"


def render(s):
    z = s["step0"]
    L = ["Hardware: labpc. DEV split only, bs 1, identical items; composites are exact per-prompt recombinations of "
         "forced-on predictions. 95 % bootstrap CIs (n = items); paired = composite − comparator, paired bootstrap "
         "[95 % CI] and exact McNemar p; ↑ / ↓ = significant (CI excludes 0 and p < 0.05).", "",
         "### Step 0: L08 SAE-CUSUM score on MMLU utility dev", "",
         f"n {z['n_utility']}; {z['n_distinct']} distinct values; {z['n_at_mode']} ({100 * z['share_at_mode']:.2f} %) "
         f"equal {z['mode']:.4f}, {z['n_below_mode']} are lower. Quantiles: "
         + ", ".join(f"q{k} {v:.3f}" for k, v in z["quantiles"].items())
         + f". E4a's threshold (95th percentile of all MMLU dev, 'higher') = {z['e4a_threshold_q95_all_mmlu']:.4f}.", "",
         "Top 20 of the 960 calibration scores: " + ", ".join(f"{x:.2f}" for x in z["cal_top20"]) + ".", "",
         "| operating point | alpha | threshold t | calibration prompts above t | (1 + k) / 961 | tied at t | fire share, all MMLU utility dev |",
         "|---|---|---|---|---|---|---|"]
    for k, v in z["ops"].items():
        L.append(f"| {OP_TEXT[k]} | {v['alpha']} | {v['threshold']:.4f} | {v['n_cal_above']} / {v['n_cal']} "
                 f"({v['fpr_cal']:.4f}) | {v['bound']:.4f} | {v['n_cal_tied_at_t']} | {v['fire_utility_all']:.4f} |")
    ag = s["sg1_ref_agreement"]
    L += ["", "Prompt identity check (E4a's StreamGuard v1 reference score > tau vs the E2d StreamGuard v1 run's own "
          "gate decision): " + ", ".join(f"{a} {v['agree']:.4f} ({v['n_disagree']} of {v['n']} differ)" for a, v in ag.items()) + ".",
          "", "### Levels", "", "| condition | " + " | ".join(CN[c] for c in COLS) + " |", "|---|" + "---|" * len(COLS)]
    for r, nm in [("base", "base"), ("dsg", "DSG (rho + clamp)"), ("sg1", "StreamGuard v1 (CUSUM + clamp)")]:
        L.append(f"| {nm} | " + " | ".join(_f(s["ref_levels"][r][c]) for c in COLS) + " |")
    for c in s["candidates"]:
        L.append(f"| {c['name']} | " + " | ".join(_f(c["levels"][k]) for k in COLS) + " |")
    for ref, nm in [("dsg", "DSG (rho + clamp)"), ("sg1", "StreamGuard v1 (CUSUM + clamp)")]:
        L += ["", f"### Paired vs {nm}", "", "| candidate | " + " | ".join(CN[c] for c in COLS) + " |",
              "|---|" + "---|" * len(COLS)]
        for c in s["candidates"]:
            L.append(f"| {c['name']} | " + " | ".join(_d(c["paired"][ref][k]) for k in COLS) + " |")
    L += ["", "### R, F and damage per fire (E1 / E2d style)", "",
          "R = fire share on forget (clean / pad400 / split3 / Hindi); F = fire share on utility and benign biology; "
          "a_fire = composite forget accuracy on fired clean items, a_miss = base accuracy on missed clean items; "
          "damage / fire = base − composite accuracy on the fired prompts.", "",
          "| gate → responder | R clean | R pad400 | R split3 | R Hindi | a_fire | a_miss | F utility (n) | damage/fire utility | F benign bio (n) | damage/fire benign bio |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in s["rfd"]:
        L.append(f"| {r['name']} | {_p(r['fire_forget_clean'])} | {_p(r['fire_forget_pad400'])} | {_p(r['fire_forget_split3'])} | "
                 f"{_p(r['fire_forget_hindi'])} | {_p(r['acc_fire_forget_clean'])} | {_p(r['acc_miss_forget_clean'])} | "
                 f"{_p(r['fire_utility'])} ({r['n_fire_utility']}) | {_p(r['damage_per_fire_utility'])} | "
                 f"{_p(r['fire_benign_bio'])} ({r['n_fire_benign_bio']}) | {_p(r['damage_per_fire_benign_bio'])} |")
    ue = s["util_excl_calib"]
    L += ["", f"Utility without the 960 calibration prompts (n {ue['n']}): "
          + "; ".join(f"{k} {_f(v)}" for k, v in ue.items() if k != "n") + "."]
    sel = s["selection"]
    L += ["", "### Selection rule v2 (DEVIATIONS 2026-11-11, unchanged)", "",
          f"Eligible iff utility ≥ DSG {sel['dsg_utility']:.4f} − {UTIL_MARGIN} = {sel['dsg_utility'] - UTIL_MARGIN:.4f}; "
          f"benign FPR (fire share on `@utility`) ≤ {FPR_MAX}; benign biology significantly above DSG; forget not "
          "significantly above StreamGuard v1 on clean, pad400, split3, Hindi. Pick: lowest worst-case forget, then "
          "higher benign biology.", "",
          "| candidate | utility | FPR | benign bio | worst-case forget | (1) utility | (2) FPR | (3) bio > DSG | (4) forget ≤ SG v1 | eligible |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    yn = (lambda b: "yes" if b else "**no**")
    for r in sel["rows"]:
        e = r["elig"]
        L.append(f"| {r['name']} | {r['utility']:.4f} | {r['fpr']:.4f} | {r['benign_bio']:.4f} | {r['worst_forget']:.4f} | "
                 f"{yn(e['utility'])} | {yn(e['fpr'])} | {yn(e['benign_bio_sig_above_dsg'])} | "
                 f"{yn(e['forget_not_sig_above_sg1'])} | {'**yes**' if e['eligible'] else 'no'} |")
    L.append("")
    L.append(f"**Selected by rule v2: {sel['selected']}.**" if sel["selected"] else
             f"**No candidate is eligible under rule v2.** Closest: {sel['closest']}. The rule is not relaxed.")
    return "\n".join(L)


def main():
    import argparse

    from dsgx.util import atomic_write_json

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    s = collect(a.n_boot)
    d = paths.runs_dir() / "E4b" / "stepA"
    d.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    atomic_write_json(d / "config.json", {"task": "E4b-stepA", "commit": commit, "dirty": dirty, "n_boot": a.n_boot,
                                          "hardware": {"label": "labpc"}, "split": "dev"})
    atomic_write_json(d / "E4B_STEPA.json", s)
    (d / "E4B_STEPA.md").write_text(render(s) + "\n")
    atomic_write_json(d / "metrics.json", {"selected": s["selection"]["selected"], "closest": s["selection"]["closest"],
                                           "ops": {k: v["threshold"] for k, v in s["step0"]["ops"].items()}})
    (d / "DONE").write_text("ok\n")
    print(d / "E4B_STEPA.md")


if __name__ == "__main__":
    main()
