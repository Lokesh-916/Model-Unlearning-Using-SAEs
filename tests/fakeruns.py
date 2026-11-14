"""Fake finished runs in the real run-directory format (config.json, metrics.json, items.parquet, DONE).

No model, no GPU: correctness vectors are given directly. Used by the final-report and combine tests.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from dsgx import paths
from dsgx.eval import stats
from dsgx.logging.run_logger import make_run_id

UTIL = ["high_school_geography", "human_aging"]


def mcq_run(exp, method, *, case="bio", split="test", seed=0, attack=None, forget=(), util=(), fired=None,
            model=None, root=None, n_boot=200, extra_cfg=None, hardware="labpc"):
    """forget/util: 0/1 correctness per item (item ids 0..n-1; util split over UTIL subjects)."""
    cfg = {"exp_id": exp, "case": case, "split": split, "seed": seed, "method": dict(method),
           "attack": dict(attack or {"name": "none"}), "datasets": ["wmdp-bio"] + UTIL,
           "forget_datasets": ["wmdp-bio"], "model": {"name": "gemma-2-2b-it", **(model or {})}, "batch_size": 1,
           **(extra_cfg or {})}
    rid = make_run_id(cfg)
    d = Path(root or paths.runs_dir()) / exp / rid
    d.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, c in enumerate(forget):
        rows.append({"item_id": i, "dataset": "wmdp-bio", "correct": bool(c), "split": split})
    for i, c in enumerate(util):
        rows.append({"item_id": i, "dataset": UTIL[i % 2], "correct": bool(c), "split": split})
    df = pd.DataFrame(rows)
    gated = method.get("name") not in ("base",)
    if gated:
        f = np.asarray(fired if fired is not None else [i < len(forget) for i in range(len(rows))], dtype=bool)
        df["gate_fired"] = f
        df["rho"] = f.astype(float) * 0.5 + 0.01
    df.to_parquet(d / "items.parquet", index=False)
    fc = df[df.dataset == "wmdp-bio"]["correct"].astype(float).values
    uc = {s: df[df.dataset == s]["correct"].astype(float).values for s in UTIL}
    view = {"per_dataset": {}, "forget": stats.bootstrap_ci(fc, n_boot), "utility": stats.stratified_bootstrap(uc, n_boot)}
    met = {"run_id": rid, "n_items": len(df), "batch_size": 1, "views": {"raw": view}}
    if gated:
        bg = df[df.dataset != "wmdp-bio"]["gate_fired"].values
        met["gate"] = {"tau": 0.5, "benign_fpr": stats.bootstrap_ci(bg, n_boot),
                       "hazard_fnr": stats.bootstrap_ci(~df[df.dataset == "wmdp-bio"]["gate_fired"].values, n_boot)}
    (d / "metrics.json").write_text(json.dumps(met))
    (d / "config.json").write_text(json.dumps({"run_id": rid, "config": cfg, "hardware": {"label": hardware},
                                               "method_info": {"features": [1, 2, 3]}, "wall_seconds": 10}))
    (d / "DONE").write_text(json.dumps({"headline": {}}))
    return d


def task_run(exp, name, metrics, root=None, files=None):
    d = Path(root or paths.runs_dir()) / exp / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(metrics))
    (d / "DONE").write_text(json.dumps({"headline": {}}))
    for fn, obj in (files or {}).items():
        (d / fn).write_text(json.dumps(obj))
    return d


def bern(p, n, seed):
    return (np.random.default_rng(seed).random(n) < p).astype(int).tolist()
