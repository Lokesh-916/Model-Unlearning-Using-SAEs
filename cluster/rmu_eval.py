"""TEST evaluation of the selected RMU model vs base, DSG and the unverified third-party RMU.

All runs: harness, split=test, bs=1, view=both, datasets WMDP-Bio + full-MMLU utility (48 subjects,
hazard-adjacent excluded). Primary utility = full-MMLU pooled; secondary = legacy 4-subject set
(MMLU-u = unweighted mean of the 4 subject accuracies), both views. 95% bootstrap CIs (10k),
paired bootstrap + exact McNemar for RMU vs each comparator on identical items.
DSG = the paper config (N=20, retain 95th pct, multiplier 500, seed 0); A1's tuned DSG comes later.
Writes $DSG_RESULTS/jobs/rmu/summary.json and SUMMARY.md (metrics only; no item text).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

from dsgx import paths
from dsgx.data.mcq import DSG_UTILITY_SUBJECTS
from dsgx.eval import stats
from dsgx.labels import reference_label
from dsgx.models.loader import clear
from dsgx.run import run
from dsgx.util import atomic_write_json, now_iso

JOBDIR = paths.results_dir() / "jobs" / "rmu"
BEST = paths.cache_dir() / "models" / "RMU-cluster" / "best"
THIRD = "AMindToThink/gemma-2-2b-it_RMU_s200_a300_layer3"
COMMON = {"exp_id": "RMU-cluster-test", "case": "bio", "split": "test", "view": "both", "seed": 0,
          "batch_size": 1, "datasets": ["@forget", "@utility"]}
DSG4 = DSG_UTILITY_SUBJECTS["bio"]


def runs():
    sel = json.loads((JOBDIR / "grid_state.json").read_text())["selected"]
    return {
        "base": {**COMMON, "dataset_label": "base", "method": {"name": "base"}},
        "rmu_cluster": {**COMMON, "dataset_label": f"rmu-cluster-c{sel['cfg']}", "method": {"name": "base"},
                        "model": {"weights": str(BEST)}, "rmu": sel["hp"]},
        "dsg_paper": {**COMMON, "dataset_label": "dsg-paper", "method": {"name": "dsg-faithful", "n_features": 20,
                                                                         "retain_pct": 95, "multiplier": 500}},
        "rmu_third_party_unverified": {**COMMON, "dataset_label": "rmu-unverified", "method": {"name": "base"},
                                       "model": {"weights": THIRD}},
    }, sel


def view_block(df, view):
    d = df if view == "raw" else df[df["in_dsg_subset"] == True]  # noqa: E712
    out = {}
    f = d[d["dataset"] == "wmdp-bio"]["correct"].to_numpy(float)
    out["wmdp_bio"] = stats.bootstrap_ci(f)
    util = d[d["dataset"] != "wmdp-bio"]
    groups = {s: g["correct"].to_numpy(float) for s, g in util.groupby("dataset")}
    if view == "raw":
        out["utility_full"] = stats.stratified_bootstrap(groups)
    leg = {s: groups[s] for s in DSG4 if s in groups and len(groups[s])}
    out["utility_legacy4"] = stats.stratified_bootstrap(leg)
    out["utility_legacy4"]["per_subject"] = {s: {"acc": float(v.mean()), "n": int(len(v))} for s, v in leg.items()}
    return out


def paired(a, b, view):
    """RMU (a) vs comparator (b) on identical items."""
    out = {}
    for name, sel in (("wmdp_bio", lambda d: d["dataset"] == "wmdp-bio"),
                      ("utility_full", lambda d: d["dataset"] != "wmdp-bio"),
                      ("utility_legacy4", lambda d: d["dataset"].isin(DSG4))):
        if view == "dsg_subset" and name == "utility_full":
            continue
        x, y = a[sel(a)], b[sel(b)]
        if view == "dsg_subset":
            x, y = x[x["in_dsg_subset"] == True], y[y["in_dsg_subset"] == True]  # noqa: E712
        m = x[["item_id", "correct"]].merge(y[["item_id", "correct"]], on="item_id", suffixes=("_a", "_b"))
        out[name] = {**stats.paired_bootstrap(m["correct_a"].to_numpy(float), m["correct_b"].to_numpy(float)),
                     "mcnemar": stats.mcnemar(m["correct_a"], m["correct_b"])}
    return out


def fmt(x):
    return "-" if not x or x.get("mean") is None else f"{x['mean']:.4f} [{x['lo']:.4f}, {x['hi']:.4f}] (n={x['n']})"


def main():
    JOBDIR.mkdir(parents=True, exist_ok=True)
    cfgs, sel = runs()
    res, items = {}, {}
    for name, cfg in cfgs.items():
        print(f"[eval] {name}", flush=True)
        rd = run(cfg)
        clear()
        m = json.loads((rd / "metrics.json").read_text())
        items[name] = pd.read_parquet(rd / "items.parquet", columns=["item_id", "dataset", "correct", "in_dsg_subset"])
        res[name] = {"run_dir": str(rd), "raw": view_block(items[name], "raw"),
                     "dsg_subset": view_block(items[name], "dsg_subset"), "timing": m["timing"],
                     "label": reference_label(cfg)}
        print(f"[eval] {name}: WMDP raw {fmt(res[name]['raw']['wmdp_bio'])} | util full "
              f"{fmt(res[name]['raw']['utility_full']['pooled'])}", flush=True)
    comp = {f"rmu_cluster_vs_{b}": {v: paired(items["rmu_cluster"], items[b], v) for v in ("raw", "dsg_subset")}
            for b in ("base", "dsg_paper", "rmu_third_party_unverified")}
    out = {"time": now_iso(), "slurm_job": os.environ.get("SLURM_JOB_ID"), "selected": sel, "runs": res,
           "paired": comp, "notes": [
               "DSG-subset view: legacy base-correct ids exist only for WMDP-Bio and the 4 legacy subjects; "
               "full-MMLU DSG-subset ids are the lab-PC Wave-1 job (not yet available), so utility_full is raw only.",
               "bs=1 for all TEST runs. Dev selection used bs=16 (decision 1)."]}
    atomic_write_json(JOBDIR / "summary.json", out)

    L = [f"# RMU (cluster) TEST summary — {out['time']}", "",
         f"Selected config c{sel['cfg']}: steering {sel['hp']['steering_coeff']} ({sel['hp']['steering_mult']} x r={sel['hp']['r']}), "
         f"alpha {sel['hp']['alpha']}, layer {sel['hp']['layer_id']}, update {sel['hp']['layer_ids']}", ""]
    for v in ("raw", "dsg_subset"):
        L += [f"## View: {v}", "", "| run | WMDP-Bio | MMLU utility full (pooled) | legacy-4 MMLU-u (unweighted) |", "|---|---|---|---|"]
        for name, r in res.items():
            uf = r[v].get("utility_full", {}).get("pooled") if v == "raw" else None
            L.append(f"| {name} | {fmt(r[v]['wmdp_bio'])} | {fmt(uf)} | {fmt(r[v]['utility_legacy4']['unweighted'])} |")
        L.append("")
    L += ["## Paired: RMU-cluster minus comparator (diff [95% CI], bootstrap p, McNemar p)", ""]
    for k, vv in comp.items():
        for v, blk in vv.items():
            for metric, p in blk.items():
                L.append(f"- {k} / {v} / {metric}: {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}], p={p['p']:.4f}, "
                         f"McNemar p={p['mcnemar']['p']:.4g} (n={p['n']})")
    (JOBDIR / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
