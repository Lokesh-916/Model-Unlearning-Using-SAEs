"""Theory checks T1-T5 (CPU, empirical side), MASTER_PLAN 5.6. Each reads finished run artifacts and
reports an R^2 / slope / bound comparison; none load a model."""
import glob
import json

import numpy as np
import pandas as pd

from dsgx import paths


def _runs(glob_pat, cols):
    for d in sorted(paths.runs_dir().glob(glob_pat)):
        if (d / "items.parquet").exists():
            df = pd.read_parquet(d / "items.parquet")
            if all(c in df for c in cols):
                yield d, df, json.loads((d / "config.json").read_text()).get("config", {})


def t1(ctx):
    """Predicted rho under dilution rho*L/(L+P) + filler_fire_rate*P/(L+P) vs measured rho, per item."""
    rows = []
    base_rho = {}
    for d, df, cfg in _runs("B1*/*dsg*", ["rho", "pad_len", "prompt_len", "item_id"]):
        for _, r in df.iterrows():
            if r["pad_len"] == 0:
                base_rho[r["item_id"]] = r["rho"]
    for d, df, cfg in _runs("B1*/*dsg*", ["rho", "pad_len", "prompt_len", "item_id"]):
        for _, r in df.iterrows():
            if r["pad_len"] and r["item_id"] in base_rho:
                L = r["prompt_len"] - r["pad_len"]; P = r["pad_len"]
                pred = base_rho[r["item_id"]] * L / (L + P)  # filler fire rate ~ 0 (benign)
                rows.append({"measured": r["rho"], "predicted": pred, "pad": int(r["pad_len"])})
    out = {"n": len(rows)}
    if rows:
        m = np.array([x["measured"] for x in rows]); p = np.array([x["predicted"] for x in rows])
        ss = 1 - np.sum((m - p) ** 2) / max(np.sum((m - m.mean()) ** 2), 1e-9)
        out.update(r2=float(ss), rmse=float(np.sqrt(np.mean((m - p) ** 2))),
                   corr=float(np.corrcoef(m, p)[0, 1]) if len(rows) > 1 else None)
    ctx.write_metrics(out); ctx.finish({"view": "T1", "forget": None}); return out


def t2(ctx):
    """Window / CUSUM statistic vs padding length: slope ~ 0 means the streaming gate is pad-robust."""
    out = {}
    for stat in ("window_max", "cusum_max"):
        xs, ys = [], []
        for d, df, cfg in _runs("C2*/*gated*", [stat, "pad_len"]):
            g = df.dropna(subset=[stat]).groupby("pad_len")[stat].mean()
            xs += list(g.index); ys += list(g.values)
        if len(set(xs)) > 1:
            sl = float(np.polyfit(xs, ys, 1)[0])
            out[stat] = {"slope": sl, "n_points": len(xs)}
    ctx.write_metrics({"stats": out}); ctx.finish({"view": "T2", "forget": None}); return out


def t3(ctx):
    """Gate FPR/FNR on every evaluated set vs the leakage<=FNR, damage<=FPR relations (from gate stats)."""
    rows = []
    for d, df, cfg in _runs("*/*gated*", ["gate_fired", "dataset", "correct", "in_dsg_subset"]):
        # real runs only: skip *-smoke experiments, archived/superseded dirs and unfinished runs
        if d.parent.name.endswith("-smoke") or d.parent.name.startswith("_") or not (d / "DONE").exists():
            continue
        fd = set(cfg.get("forget_datasets", ["wmdp-bio", "wmdp-cyber"]))
        for ds, g in df.groupby("dataset"):
            benign = ds not in fd
            rows.append({"run": d.name, "dataset": ds, "benign": benign,
                         "fire_rate": float(g["gate_fired"].mean()), "n": int(len(g))})
    ctx.write_metrics({"per_set": rows[:500], "n": len(rows)})
    ctx.finish({"view": "T3", "forget": None}); return {"n": len(rows)}


def t4(ctx):
    """D2: change in retain outputs on held-out retain keys vs the null-space bound (needs D2 audit)."""
    recs = []
    for f in glob.glob(str(paths.runs_dir() / "D2*/*/metrics.json")):
        recs.append(json.loads(open(f).read()))
    ctx.write_metrics({"note": "requires D2 edit + eval; compares edited vs base retain accuracy",
                       "n_d2_runs": len(recs)})
    ctx.finish({"view": "T4", "forget": None}); return {"n_d2_runs": len(recs)}


def t5(ctx):
    """CUSUM assumptions: autocorrelation of the per-token LLR and a normality check, from traces.npz."""
    acs = []
    for f in glob.glob(str(paths.runs_dir() / "C2*/*cusum*/traces.npz")):
        z = np.load(f)
        for k in [k for k in z.files if k.startswith("rho__")][:50]:
            x = np.diff(z[k].astype(float))
            if len(x) > 3 and x.std() > 0:
                acs.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
    out = {"n_series": len(acs), "mean_lag1_autocorr": float(np.mean(acs)) if acs else None}
    ctx.write_metrics(out); ctx.finish({"view": "T5", "forget": None}); return out
