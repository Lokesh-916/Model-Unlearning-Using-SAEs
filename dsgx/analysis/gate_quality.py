"""Gate quality: per-run AUROC of the gate score separating forget items from benign (utility)
items, from items.parquet (gate_score column, falling back to rho). Used by C2/C3/C6/N7.

Task entry `task` with args {exp_id, forget_datasets}.
"""
import json

import numpy as np

from dsgx import paths


def auroc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUROC: P(score(forget) > score(benign))."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    # average ranks for ties
    _, inv, counts = np.unique(allv, return_inverse=True, return_counts=True)
    csum = np.concatenate([[0], counts.cumsum()])
    avg = {i: (csum[i] + csum[i + 1] + 1) / 2 for i in range(len(counts))}
    ranks = np.array([avg[i] for i in inv])
    r_pos = ranks[: len(pos)].sum()
    return float((r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def task(ctx):
    import pandas as pd

    exp = ctx.args.get("exp_id", ctx.exp_id)
    if ctx.smoke and not exp.endswith("-smoke"):
        exp += "-smoke"
    fd = set(ctx.args.get("forget_datasets", ["wmdp-bio", "wmdp-cyber"]))
    out = []
    for d in sorted((paths.runs_dir() / exp).glob("*__*")):
        if not (d / "DONE").exists() or not (d / "items.parquet").exists():
            continue
        cfg = json.loads((d / "config.json").read_text())["config"]
        if cfg["method"]["name"] == "base":
            continue
        it = pd.read_parquet(d / "items.parquet")
        if "gate_score" in it and it["gate_score"].notna().any():
            col = "gate_score"
        elif "rho" in it and it["rho"].notna().any():
            col = "rho"
        else:
            continue
        clean = it[it["attack"].isin(["none"]) | (it.get("pad_len", 0) == 0)]
        pos = clean[clean["dataset"].isin(fd)][col].dropna().values
        neg = clean[~clean["dataset"].isin(fd)][col].dropna().values
        info = cfg["method"].get("gate", {})
        out.append({"run": d.name, "layer": info.get("sae_id", cfg["model"]["sae_id"]),
                    "gate_type": info.get("type", "dsg"), "auroc": auroc(pos, neg),
                    "n_pos": int(len(pos)), "n_neg": int(len(neg)),
                    "threshold": json.loads((d / "metrics.json").read_text()).get("gate", {}).get("tau")})
    out.sort(key=lambda r: -(r["auroc"] if r["auroc"] == r["auroc"] else -1))
    (ctx.run_dir() / "gate_quality.json").write_text(json.dumps(out, indent=1))
    ctx.write_metrics({"source_exp": exp, "n_runs": len(out), "ranked": out,
                       "top3": [r["run"] for r in out[:3]]})
    ctx.finish({"view": "gate-auroc", "forget": None})
    return {"n": len(out)}
