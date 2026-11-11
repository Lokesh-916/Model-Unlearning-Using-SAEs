"""E4a summary task (CPU): per layer and detector, on DEV questions only.

AUROC hazard (WMDP-Bio clean) vs benign biology and vs MMLU utility (bootstrap CI, positives and negatives resampled
separately). Threshold = 95th percentile ("higher") of the detector's MMLU-dev scores (5 % MMLU false-fire rate; the
threshold's own sampling error is not propagated). At it: recall (fire rate) on WMDP-Bio clean / pad400 / split3 /
Hindi and fire rate on benign biology, bootstrap CIs over items. References: DSG's layer-3 rho gate and StreamGuard v1,
at their own thresholds and at the same 5 % MMLU threshold. Verdict line fixed in common.verdict (point estimates).
Writes metrics.json and SUMMARY.md (ids and metrics only).
"""
import numpy as np
import pandas as pd

from dsgx.eval import stats

from experiments.E4a.common import BENIGN_LINE, DETECTORS, FPR, RECALL_LINE, WMDP_CONDS, verdict
from experiments.E4a.fit import auroc

CONDS = WMDP_CONDS + ["benign_bio", "mmlu"]


def auroc_ci(pos, neg, nb=1000, seed=0):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    rng = np.random.default_rng(seed)
    bs = [auroc(pos[rng.integers(0, len(pos), len(pos))], neg[rng.integers(0, len(neg), len(neg))]) for _ in range(nb)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"mean": auroc(pos, neg), "lo": float(lo), "hi": float(hi), "n_pos": len(pos), "n_neg": len(neg)}


def evaluate(df, col, nb, own_thr=None):
    s = {c: df.loc[df["cond"] == c, col].astype(float).values for c in CONDS}
    thr = float(np.quantile(s["mmlu"], 1 - FPR, method="higher"))
    r = {"threshold": thr,
         "auroc_vs_benign": auroc_ci(s["wmdp_clean"], s["benign_bio"], nb),
         "auroc_vs_mmlu": auroc_ci(s["wmdp_clean"], s["mmlu"], nb),
         "fire": {c: stats.bootstrap_ci((s[c] > thr).astype(float), nb) for c in CONDS}}
    if own_thr is not None:
        r["own_threshold"] = float(own_thr)
        r["fire_own"] = {c: stats.bootstrap_ci((s[c] > own_thr).astype(float), nb) for c in CONDS}
    return r


def collect(df, own, nb=2000):
    df = df[df["skip"] == ""]
    out = {"n": df.groupby("cond").size().to_dict(), "rows": [], "refs": {}}
    for col in [c for c in df.columns if "|" in c]:
        det, tag = col.split("|")
        r = evaluate(df, col, nb, own.get(col))
        r.update(detector=det, layer=tag)
        if det.startswith("ref-"):
            out["refs"][det] = r
        else:
            out["rows"].append(r)
    flat = [{"layer": r["layer"], "detector": r["detector"],
             "recall": {c: r["fire"][c]["mean"] for c in WMDP_CONDS}, "benign_fire": r["fire"]["benign_bio"]["mean"]}
            for r in out["rows"]]
    out["verdict"] = verdict(flat)
    # closest candidates (descriptive)
    out["best_min_recall_at_benign_line"] = max(
        (f for f in flat if f["benign_fire"] <= BENIGN_LINE), key=lambda f: min(f["recall"].values()), default=None)
    out["lowest_benign_at_recall_line"] = min(
        (f for f in flat if min(f["recall"].values()) >= RECALL_LINE), key=lambda f: f["benign_fire"], default=None)
    return out


def _ci(d, nd=3):
    return "-" if d is None or d.get("mean") is None else f"{d['mean']:.{nd}f} [{d['lo']:.{nd}f}, {d['hi']:.{nd}f}]"


def render(s, hardware="labpc"):
    L = [f"Hardware: {hardware}. DEV split only. Detectors trained on documents only; evaluated on dev questions. "
         f"Threshold = 5 % MMLU-dev false-fire rate (95th percentile of MMLU scores). 95 % bootstrap CIs over items.",
         "", "n per condition: " + ", ".join(f"{c} {s['n'].get(c, 0)}" for c in CONDS), ""]
    hdr = ("| detector | AUROC vs benign bio | AUROC vs MMLU | recall clean | recall pad400 | recall split3 | "
           "recall Hindi | benign-bio fire | MMLU fire |")
    sep = "|" + "---|" * 9

    def line(name, r, key="fire"):
        f = r[key]
        return (f"| {name} | {_ci(r['auroc_vs_benign'])} | {_ci(r['auroc_vs_mmlu'])} | "
                + " | ".join(_ci(f[c]) for c in WMDP_CONDS) + f" | {_ci(f['benign_bio'])} | {_ci(f['mmlu'])} |")

    L += ["### References (layer 3, DSG's own SAE)", "", hdr, sep]
    for det, r in s["refs"].items():
        L.append(line(f"{det} @ 5 % MMLU", r))
        if "fire_own" in r:
            L.append(line(f"{det} @ own threshold {r['own_threshold']:.4g}", r, "fire_own"))
    L.append("")
    layers = sorted({r["layer"] for r in s["rows"]}, key=lambda t: (int(t[1:3]), t))
    for tag in layers:
        L += [f"### {tag}" + (" (DSG's own l0_142 SAE; detector (a) only)" if tag == "L03dsg" else ""), "", hdr, sep]
        for det in DETECTORS:
            for r in s["rows"]:
                if r["layer"] == tag and r["detector"] == det:
                    L.append(line(det, r))
        L.append("")
    v = s["verdict"]
    L += ["### Verdict", "", f"Line (fixed before any result): some layer and detector with {v['line']}.",
          f"**{'MET' if v['met'] else 'NOT MET'}**" + (f": {v['passing']}" if v["met"] else "."), ""]
    return "\n".join(L)


def task(ctx):
    from dsgx.util import atomic_write_json  # noqa: F401
    import json

    from dsgx import paths

    exp = ctx.args.get("questions_exp", ctx.exp_id)
    qd = paths.runs_dir() / exp / "questions"
    df = pd.read_parquet(qd / "items.parquet")
    own = json.loads((qd / "metrics.json").read_text())["ref_own_thresholds"]
    s = collect(df, own, int(ctx.args.get("n_boot", 2000)))
    ctx.write_config(questions_exp=exp)
    ctx.write_metrics(s)
    (ctx.run_dir() / "SUMMARY.md").write_text(render(s))
    ctx.finish({"view": "e4a-summary", "forget": None, "verdict_met": s["verdict"]["met"]})
    return s["verdict"]
