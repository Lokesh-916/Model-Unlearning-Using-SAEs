"""Aggregation for the final report: per-run table, across-seed summaries and paired tests.

* run_table(runs)    one row per finished MCQ run: both views, forget/utility mean + 95% CI + n,
                     gate FPR/FNR, seed, batch size, hardware, reference label, commit.
* seed_table(df)     per condition (all fields except seed): mean and t-based 95% CI across seeds.
* paired_tests(runs) for every TEST MCQ run, paired bootstrap + exact McNemar against the DSG run
                     and the base run of the same experiment, case, attack and seed (base: any seed),
                     on identical items, separately for forget and utility items.
"""
import json

import numpy as np
import pandas as pd

from dsgx.eval import stats
from dsgx.labels import reference_label

DSG_NAMES = ("dsg-faithful",)


def _ci(x, key):
    x = x or {}
    return {f"{key}": x.get("mean"), f"{key}_lo": x.get("lo"), f"{key}_hi": x.get("hi"), f"{key}_n": x.get("n")}


def run_table(runs) -> pd.DataFrame:
    rows = []
    for r in runs:
        if not r.is_mcq:
            continue
        c = r.cfg
        row = {"exp": r.base_exp, "run": r.name, "condition": r.label(), "method": r.method, "case": r.case,
               "split": r.split, "seed": r.seed, "attack": r.attack.get("name", "none"),
               "attack_params": json.dumps({k: v for k, v in r.attack.items() if k != "name"}, sort_keys=True),
               "model": (c.get("model") or {}).get("name"), "weights": r.weights or "",
               "reference": reference_label(c) or "", "batch_size": r.metrics.get("batch_size"),
               "hardware": r.hardware, "commit": ((r.config.get("git") or {}).get("commit") or "")[:10],
               "wall_s": r.config.get("wall_seconds"),
               "peak_vram_gb": (r.metrics.get("timing") or {}).get("peak_vram_gb"), "dir": str(r.dir)}
        for view in ("raw", "dsg_subset"):
            row.update(_ci(r.forget(view), f"{view}_forget"))
            row.update(_ci(r.utility(view), f"{view}_util"))
        g = r.gate
        row.update(_ci(g.get("benign_fpr"), "benign_fpr"))
        row.update(_ci(g.get("hazard_fnr"), "hazard_fnr"))
        row["tau"] = g.get("tau")
        rows.append(row)
    return pd.DataFrame(rows)


def seed_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    keys = ["exp", "condition", "method", "case", "split", "attack", "attack_params", "model", "weights",
            "reference", "hardware"]
    out = []
    for k, g in df.groupby(keys, dropna=False):
        row = dict(zip(keys, k))
        row["seeds"] = sorted(int(s) for s in g["seed"].dropna().unique())
        for col in ("raw_forget", "raw_util", "dsg_subset_forget", "dsg_subset_util", "benign_fpr"):
            s = stats.seed_summary(g[col].dropna().tolist())
            row[col] = s.get("mean")
            row[f"{col}_lo"] = s.get("lo")
            row[f"{col}_hi"] = s.get("hi")
            row[f"{col}_n_seeds"] = s.get("n_seeds")
            row[f"{col}_n_items"] = int(g[f"{col}_n"].dropna().iloc[0]) if g[f"{col}_n"].notna().any() else None
        out.append(row)
    return pd.DataFrame(out)


def _keyed(items: pd.DataFrame) -> pd.Series:
    it = items.copy()
    it["key"] = it["dataset"].astype(str) + ":" + it["item_id"].astype(str)
    return it.drop_duplicates("key").set_index("key")


def compare(a, b, forget_datasets, n_boot=stats.N_BOOT) -> dict:
    """Paired comparison of run a vs run b (a - b) on identical items, forget and utility separately."""
    ia, ib = a.items, b.items
    if ia is None or ib is None:
        return {}
    ka, kb = _keyed(ia), _keyed(ib)
    common = ka.index.intersection(kb.index)
    out = {}
    for part, mask in (("forget", ka.loc[common, "dataset"].isin(forget_datasets)),
                       ("utility", ~ka.loc[common, "dataset"].isin(forget_datasets))):
        idx = common[mask.values]
        if len(idx) == 0:
            continue
        x = ka.loc[idx, "correct"].astype(float).values
        y = kb.loc[idx, "correct"].astype(float).values
        out[part] = {"paired_bootstrap": stats.paired_bootstrap(x, y, n_boot=n_boot),
                     "mcnemar": stats.mcnemar(x, y), "n": int(len(idx))}
    return out


def paired_tests(runs, n_boot=stats.N_BOOT, split="test") -> list[dict]:
    mcq = [r for r in runs if r.is_mcq and r.split == split]
    out = []
    by = {}
    for r in mcq:
        by.setdefault((r.base_exp, r.case, json.dumps(r.attack, sort_keys=True)), []).append(r)
    for (exp, case, att), rs in by.items():
        bases = [r for r in rs if r.is_base]
        dsgs = [r for r in rs if r.method in DSG_NAMES]
        for r in rs:
            if r.is_base:
                continue
            fd = r.cfg.get("forget_datasets") or ["wmdp-bio", "wmdp-cyber"]
            refs = []
            if bases:
                refs.append(("base", bases[0]))
            if r.method not in DSG_NAMES:
                same_seed = [d for d in dsgs if d.seed == r.seed] or dsgs
                if same_seed:
                    refs.append(("dsg", same_seed[0]))
            for tag, ref in refs:
                res = compare(r, ref, fd, n_boot)
                if res:
                    out.append({"exp": exp, "case": case, "attack": json.loads(att), "run": r.name,
                                "condition": r.label(), "vs": tag, "ref_run": ref.name,
                                "reference": reference_label(r.cfg) or "", "hardware": r.hardware, **res})
    return out
