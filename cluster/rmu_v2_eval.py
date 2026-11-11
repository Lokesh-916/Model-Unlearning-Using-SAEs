"""TEST evaluation of the selected RMU-v2 model vs base, DSG (paper config) and the unverified third-party RMU.

Same protocol as cluster/rmu_eval.py (v1): harness, split=test, bs=1, view=both, WMDP-Bio + full-MMLU
utility (48 subjects); 95% bootstrap CIs; paired bootstrap + exact McNemar of RMU-v2 vs each comparator on
identical items. All four conditions are run fresh here on gpuws (exp RMU-v2-test). RMU-v1 (rmu_cluster) is
an extra paired comparator read from its existing gpuws TEST run (jobs/rmu/summary.json; same items, bs=1);
it is skipped with a note if that run is not on disk. Never mix these numbers with lab-PC runs.
Writes $DSG_RESULTS/jobs/rmu-v2/summary.json and SUMMARY.md (metrics only; no item text).
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd

from cluster.rmu_eval import THIRD, fmt, paired, view_block
from dsgx import paths
from dsgx.labels import reference_label
from dsgx.models.loader import clear
from dsgx.run import run
from dsgx.util import atomic_write_json, gpu_info, hardware_label, now_iso

JOBDIR = paths.results_dir() / "jobs" / "rmu-v2"
V1 = paths.results_dir() / "jobs" / "rmu" / "summary.json"
BEST = paths.cache_dir() / "models" / "RMU-v2" / "best"
COMMON = {"exp_id": "RMU-v2-test", "case": "bio", "split": "test", "view": "both", "seed": 0,
          "batch_size": 1, "datasets": ["@forget", "@utility"]}
ITEM_COLS = ["item_id", "dataset", "correct", "in_dsg_subset"]


def runs():
    sel = json.loads((JOBDIR / "grid_state.json").read_text())["selected"]
    return {
        "base": {**COMMON, "dataset_label": "base", "method": {"name": "base"}},
        "rmu_v2": {**COMMON, "dataset_label": f"rmu-v2-c{sel['cfg']}", "method": {"name": "base"},
                   "model": {"weights": str(BEST)}, "rmu": sel["hp"]},
        "dsg_paper": {**COMMON, "dataset_label": "dsg-paper", "method": {"name": "dsg-faithful", "n_features": 20,
                                                                         "retain_pct": 95, "multiplier": 500}},
        "rmu_third_party_unverified": {**COMMON, "dataset_label": "rmu-unverified", "method": {"name": "base"},
                                       "model": {"weights": THIRD}},
    }, sel


def v1_items(hw):
    """RMU-v1 TEST items from its gpuws run, or (None, reason)."""
    if not V1.exists():
        return None, f"{V1} missing"
    r = json.loads(V1.read_text())["runs"].get("rmu_cluster", {})
    rd = Path(r.get("run_dir", ""))
    if r.get("hardware") != hw or not (rd / "items.parquet").exists():
        return None, f"v1 run not usable here (hardware {r.get('hardware')!r}, run dir present: {rd.exists()})"
    return pd.read_parquet(rd / "items.parquet", columns=ITEM_COLS), str(rd)


def main():
    JOBDIR.mkdir(parents=True, exist_ok=True)
    hw = hardware_label()
    cfgs, sel = runs()
    res, items = {}, {}
    for name, cfg in cfgs.items():
        print(f"[eval] {name}", flush=True)
        rd = run(cfg)
        clear()
        m = json.loads((rd / "metrics.json").read_text())
        rhw = json.loads((rd / "config.json").read_text()).get("hardware", {}).get("label")
        assert rhw == hw, f"{name}: run hardware {rhw!r} != {hw!r}; never mix hardware in one table"
        items[name] = pd.read_parquet(rd / "items.parquet", columns=ITEM_COLS)
        res[name] = {"run_dir": str(rd), "raw": view_block(items[name], "raw"),
                     "dsg_subset": view_block(items[name], "dsg_subset"), "timing": m["timing"],
                     "label": reference_label(cfg), "hardware": hw}
        print(f"[eval] {name}: WMDP raw {fmt(res[name]['raw']['wmdp_bio'])} | util full "
              f"{fmt(res[name]['raw']['utility_full']['pooled'])}", flush=True)
    notes = ["bs=1 for all TEST runs; dev selection used the probed dev batch size (grid_state.json eval_bs).",
             "DSG-subset view: full-MMLU DSG-subset ids are not available, so utility_full is raw only."]
    v1, v1_src = v1_items(hw)
    if v1 is not None:
        items["rmu_v1"] = v1
        res["rmu_v1"] = {"run_dir": v1_src, "raw": view_block(v1, "raw"), "dsg_subset": view_block(v1, "dsg_subset"),
                         "hardware": hw, "label": "RMU v1 (cluster/rmu_train.py, read from its TEST run)"}
    else:
        notes.append(f"RMU-v1 comparison skipped: {v1_src}")
    comps = [b for b in ("base", "dsg_paper", "rmu_third_party_unverified", "rmu_v1") if b in items]
    comp = {f"rmu_v2_vs_{b}": {v: paired(items["rmu_v2"], items[b], v) for v in ("raw", "dsg_subset")} for b in comps}
    out = {"time": now_iso(), "hardware": {"label": hw, **gpu_info()}, "slurm_job": os.environ.get("SLURM_JOB_ID"),
           "selected": sel, "runs": res, "paired": comp, "notes": notes}
    atomic_write_json(JOBDIR / "summary.json", out)

    hp = sel["hp"]
    L = [f"# RMU v2 (cluster) TEST summary — {out['time']}", "",
         f"Hardware: **{hw}** ({out['hardware'].get('gpu')}, driver {out['hardware'].get('driver')}). "
         "All rows and paired tests are on this hardware only; do not combine with other hardware.", "",
         f"Selected config c{sel['cfg']}: steering {hp['steering_coeff']} ({hp['steering_mult']} x r={hp['r']}), "
         f"alpha {hp['alpha']}, layer {hp['layer_id']}, update {hp['layer_ids']}, steps {hp['max_num_batches']}, "
         f"train bs {hp['batch_size']}; at grid edge: {sel.get('at_grid_edge') or 'none'}", ""]
    for v in ("raw", "dsg_subset"):
        L += [f"## View: {v}", "", "| run | WMDP-Bio | MMLU utility full (pooled) | legacy-4 MMLU-u (unweighted) |",
              "|---|---|---|---|"]
        for name, r in res.items():
            uf = r[v].get("utility_full", {}).get("pooled") if v == "raw" else None
            L.append(f"| {name} | {fmt(r[v]['wmdp_bio'])} | {fmt(uf)} | {fmt(r[v]['utility_legacy4']['unweighted'])} |")
        L.append("")
    L += ["## Paired: RMU-v2 minus comparator (diff [95% CI], bootstrap p, McNemar p)", ""]
    for k, vv in comp.items():
        for v, blk in vv.items():
            for metric, p in blk.items():
                L.append(f"- {k} / {v} / {metric}: {p['diff']:+.4f} [{p['lo']:+.4f}, {p['hi']:+.4f}], p={p['p']:.4f}, "
                         f"McNemar p={p['mcnemar']['p']:.4g} (n={p['n']})")
    L += ["", "## Notes", ""] + [f"- {n}" for n in notes]
    (JOBDIR / "SUMMARY.md").write_text("\n".join(L) + "\n")
    print("\n".join(L), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
