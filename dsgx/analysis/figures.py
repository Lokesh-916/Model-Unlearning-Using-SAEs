"""Every figure of MASTER_PLAN section 7, regenerated from logs (PNG + PDF).

Each function takes (runs, outdir, ctx) and returns a list of written stems, or raises Skip(reason)
when its data does not exist yet. Colours: the validated categorical palette in fixed order, with a
marker / line style per series as a second encoding (print and colour-blind safe).
"""
import json
from pathlib import Path

import numpy as np

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


class Skip(Exception):
    pass


PAPER = False          # set by paper_assets: resize every figure to a paper column width, PDF only
COL_IN, WIDE_IN = 3.4, 6.75   # one column; full text width of two-column venues (ICML/ACL)


def setup(paper=False):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
        "font.size": 8 if paper else 9, "axes.titlesize": 9 if paper else 10, "legend.fontsize": 7 if paper else 8,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    return plt


def series_style(i):
    return {"color": PALETTE[i % len(PALETTE)], "marker": MARKERS[i % len(MARKERS)]}


def save(fig, outdir: Path, stem: str, size=None):
    outdir.mkdir(parents=True, exist_ok=True)
    if size:
        fig.set_size_inches(*size)
    exts = ("png", "pdf")
    if PAPER:  # single column unless the figure was designed wide; keep the aspect ratio
        w, h = fig.get_size_inches()
        tw = WIDE_IN if w > 6 else COL_IN
        fig.set_size_inches(tw, h * tw / w)
        exts = ("pdf",)
    fig.tight_layout()
    for ext in exts:
        fig.savefig(outdir / f"{stem}.{ext}", dpi=200 if ext == "png" else None)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return stem


def _forget_acc(r):
    f = r.forget("raw") or {}
    return f.get("mean"), f.get("lo"), f.get("hi")


# ----------------------------------------------------------------------------- 1. accuracy vs padding
def acc_vs_padding(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.base_exp == "B1" and r.is_mcq and r.attack.get("name") == "dilution"]
    if not rs:
        raise Skip("no B1 runs")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    groups = {}
    for r in rs:
        key = "base" if r.is_base else f"DSG, {r.attack.get('position', 'before')}, {r.attack.get('source', 'wikitext')}"
        groups.setdefault(key, []).append(r)
    for i, (k, g) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0] != "base", kv[0]))):
        g = sorted(g, key=lambda r: int(r.attack.get("pad", 0) or 0))
        x = [int(r.attack.get("pad", 0) or 0) for r in g]
        m = np.array([_forget_acc(r) for r in g], dtype=float)
        st = series_style(i)
        ax.plot(x, m[:, 0], label=k, linestyle="--" if k == "base" else "-", **st)
        ax.fill_between(x, m[:, 1], m[:, 2], color=st["color"], alpha=0.12, linewidth=0)
    ax.axhline(0.25, color=INK2, linewidth=1, linestyle=":")
    pads = sorted({int(r.attack.get("pad", 0) or 0) for r in rs})
    ax.set_xscale("symlog", linthresh=50, linscale=0.5)
    ax.set_xticks(pads, [str(p) for p in pads])
    ax.minorticks_off()
    ax.set_xlabel("padding tokens (symlog scale)")
    ax.set_ylabel("WMDP-Bio TEST accuracy (raw)")
    ax.set_title("Accuracy vs padding (B1); dotted = chance")
    ax.legend(ncol=2 if len(groups) > 4 else 1)
    return [save(fig, out, "b1_accuracy_vs_padding")]


# ----------------------------------------------------------------------------- 2. rho distributions
def rho_distributions(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.is_mcq and r.method == "dsg-faithful" and r.split == "test" and r.is_clean
          and r.items is not None and "rho" in r.items]
    if not rs:
        raise Skip("no clean DSG TEST runs with rho")
    r = sorted(rs, key=lambda r: (r.base_exp != "A1-test", r.seed))[0]
    it = r.items
    fd = r.cfg.get("forget_datasets") or ["wmdp-bio"]
    fig, ax = plt.subplots(figsize=(5, 3.2))
    bins = np.linspace(0, max(float(it["rho"].max()), 1e-3), 40)
    ax.hist(it.loc[it["dataset"].isin(fd), "rho"], bins=bins, color=PALETTE[0], alpha=0.75, label="forget (WMDP)")
    ax.hist(it.loc[~it["dataset"].isin(fd), "rho"], bins=bins, color=PALETTE[1], alpha=0.6, label="utility (MMLU)")
    tau = r.gate.get("tau")
    if tau is not None:
        ax.axvline(tau, color=INK, linewidth=1.2, linestyle="--", label=f"tau = {tau:.3f}")
    ax.set_xlabel("rho (fraction of tokens firing a selected feature)")
    ax.set_ylabel("items")
    ax.set_title(f"rho distributions ({r.base_exp}, {r.case}, seed {r.seed})")
    ax.legend()
    return [save(fig, out, "rho_distributions")]


# ----------------------------------------------------------------------------- 3. gate ROC
def _roc(pos, neg):
    s = np.concatenate([pos, neg])
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    o = np.argsort(-s, kind="mergesort")
    y = y[o]
    tpr = np.r_[0, np.cumsum(y) / max(len(pos), 1)]
    fpr = np.r_[0, np.cumsum(1 - y) / max(len(neg), 1)]
    return fpr, tpr


def gate_roc(runs, out, ctx):
    from dsgx.analysis.gate_quality import auroc

    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.is_mcq and r.base_exp in ("C1", "C2", "C3", "C6", "N7", "A1-test", "X1") and r.is_clean
          and not r.is_base and r.items is not None and r.split == "test"]
    curves = []
    for r in rs:
        it = r.items
        col = "gate_score" if "gate_score" in it and it["gate_score"].notna().any() else "rho"
        if col not in it or it[col].isna().all():
            continue
        fd = r.cfg.get("forget_datasets") or ["wmdp-bio"]
        pos = it.loc[it["dataset"].isin(fd), col].dropna().values
        neg = it.loc[~it["dataset"].isin(fd), col].dropna().values
        if len(pos) and len(neg):
            curves.append((auroc(pos, neg), f"{r.base_exp}: {r.label()[:40]}", pos, neg))
    if not curves:
        raise Skip("no gate runs with scores on forget and benign items")
    curves = sorted(curves, key=lambda c: -c[0])[:8]
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for i, (a, name, pos, neg) in enumerate(curves):
        f, t = _roc(pos, neg)
        ax.plot(f, t, color=PALETTE[i], label=f"{name} (AUROC {a:.3f})", linewidth=1.6)
    ax.plot([0, 1], [0, 1], color=INK2, linewidth=1, linestyle=":")
    ax.axvline(0.05, color=INK2, linewidth=0.8, linestyle="--")
    ax.set_xlabel("benign false-positive rate")
    ax.set_ylabel("forget true-positive rate")
    ax.set_title("Gate ROC (TEST; dashed = 5% FPR); top 8 by AUROC")
    ax.legend(fontsize=6, loc="lower right")
    return [save(fig, out, "gate_roc")]


# ----------------------------------------------------------------------------- 4. Pareto
def pareto(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.is_mcq and r.base_exp == "A1-dev" and r.utility("raw") and r.forget("raw")]
    if not rs:
        raise Skip("no A1-dev runs")
    fig, ax = plt.subplots(figsize=(5, 3.6))
    cases = sorted({(r.case, r.cfg["method"].get("retain_corpus", "")) for r in rs}, key=str)
    for i, c in enumerate(cases):
        g = [r for r in rs if (r.case, r.cfg["method"].get("retain_corpus", "")) == c]
        st = series_style(i)
        base = [r for r in g if r.is_base]
        gg = [r for r in g if not r.is_base]
        ax.scatter([r.utility("raw")["mean"] for r in gg], [r.forget("raw")["mean"] for r in gg], s=22,
                   color=st["color"], marker=st["marker"], label=f"{c[0]} DSG" + (f" ({c[1]})" if c[1] else ""))
        for b in base:
            ax.scatter([b.utility("raw")["mean"]], [b.forget("raw")["mean"]], s=70, facecolor="white",
                       edgecolor=st["color"], marker=st["marker"], linewidth=1.8, label=f"{c[0]} base")
    ax.set_xlabel("full-MMLU utility (DEV, raw, pooled)")
    ax.set_ylabel("forget accuracy (DEV, raw)")
    ax.set_title("Forget vs utility, A1 dev grid (lower-right is better)")
    ax.legend(fontsize=6)
    return [save(fig, out, "pareto_forget_utility")]


# ----------------------------------------------------------------------------- 5. probes by layer
def probes_by_layer(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    pr = [r for r in runs if r.base_exp in ("A4", "D3") and r.name.startswith("probe__") and r.metrics.get("layers")]
    if not pr:
        raise Skip("no A4/D3 probe results")
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    for i, r in enumerate(sorted(pr, key=lambda r: r.name)):
        L = sorted(int(k) for k in r.metrics["layers"])
        m = [r.metrics["layers"][str(l)]["probe"]["mean"] for l in L]
        c = [r.metrics["layers"][str(l)]["control"]["mean"] for l in L]
        st = series_style(i)
        ax.plot(L, m, label=f"{r.base_exp} {r.name.split('__', 1)[1]} probe", **st)
        ax.plot(L, c, color=st["color"], linestyle=":", linewidth=1.2, label=f"{r.name.split('__', 1)[1]} control")
    ax.axhline(0.25, color=INK2, linewidth=1, linestyle="--")
    ax.set_xlabel("layer")
    ax.set_ylabel("answer-probe TEST accuracy")
    ax.set_title("Linear answer probes by layer (dashed = chance)")
    ax.legend(fontsize=6, ncol=2)
    return [save(fig, out, "probe_accuracy_by_layer")]


# ----------------------------------------------------------------------------- 6. relearning curves
def relearning_curves(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rl = [r for r in runs if r.base_exp in ("A6", "A6-full") and r.metrics.get("curve")]
    if not rl:
        raise Skip("no A6 relearning results")
    ks = sorted({(r.metrics.get("k"), r.metrics.get("rank")) for r in rl}, key=str)
    n = len(ks)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.5 * rows), squeeze=False, sharey=True)
    conds = sorted({r.metrics["condition"] for r in rl})
    for ai, (k, rank) in enumerate(ks):
        ax = axs[ai // cols][ai % cols]
        for ci, cnd in enumerate(conds):
            g = [r for r in rl if r.metrics["condition"] == cnd and (r.metrics.get("k"), r.metrics.get("rank")) == (k, rank)]
            for r in g[:1]:
                m = r.metrics
                x = [0] + [p["step"] for p in m["curve"]]
                y = [m["before"]["forget_acc"]] + [p["forget_acc"] for p in m["curve"]]
                ax.plot(x, y, label=cnd, **series_style(ci))
        ax.set_title(f"k={k}, rank={rank}")
        ax.set_xlabel("relearn steps")
    axs[0][0].set_ylabel("forget accuracy")
    for j in range(n, rows * cols):
        axs[j // cols][j % cols].axis("off")
    axs[0][0].legend(fontsize=6)
    return [save(fig, out, "relearning_curves")]


# ----------------------------------------------------------------------------- 7. conformal coverage
def conformal_coverage(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.base_exp == "N6" and r.metrics.get("coverage")]
    if not rs:
        raise Skip("no N6 results")
    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    i = 0
    for r in rs:
        cov = r.metrics["coverage"]
        a = sorted(float(x) for x in cov)
        for which, ls in (("empirical_fpr_heldout", "-"), ("empirical_fpr_shifted", "--")):
            st = series_style(i)
            ax.plot(a, [cov[str(x) if str(x) in cov else repr(x)][which] for x in a], linestyle=ls,
                    label=f"{r.name} {which.split('_')[-1]}", **st)
            i += 1
    lim = max(0.12, ax.get_ylim()[1])
    ax.plot([0, lim], [0, lim], color=INK2, linewidth=1, linestyle=":")
    ax.set_xlabel("target alpha")
    ax.set_ylabel("empirical benign FPR")
    ax.set_title("Split-conformal threshold (N6); dotted = target")
    ax.legend(fontsize=6)
    return [save(fig, out, "conformal_coverage")]


# ----------------------------------------------------------------------------- 8. per-language bars
def per_language(runs, out, ctx):
    from dsgx.analysis.claims import attack_success_records

    plt = setup(ctx.get("paper"))
    recs = [r for r in attack_success_records(runs, "B3") if r["method_d"].get("name") == "dsg-faithful"]
    if not recs:
        raise Skip("no B3 attack-success results")
    recs = sorted(recs, key=lambda r: (r["attack"].get("name"), str(r["attack"].get("lang", r["attack"].get("encoding")))))
    lab = [str(r["attack"].get("lang") or r["attack"].get("encoding") or r["attack"].get("name")) for r in recs]
    dsg = [r["acc_under_attack"]["mean"] for r in recs]
    base = [(r.get("base_acc_same_transform") or {}).get("mean", np.nan) for r in recs]
    x = np.arange(len(lab))
    fig, ax = plt.subplots(figsize=(max(5, 0.5 * len(lab) + 1.5), 3.2))
    ax.bar(x - 0.2, base, 0.38, color=PALETTE[1], label="base model, same transform", edgecolor="white", linewidth=1)
    ax.bar(x + 0.2, dsg, 0.38, color=PALETTE[0], label="DSG", edgecolor="white", linewidth=1)
    ax.axhline(0.25, color=INK2, linewidth=1, linestyle=":")
    ax.set_xticks(x, lab, rotation=0)
    ax.set_ylabel("WMDP-Bio TEST accuracy")
    ax.set_title("Cross-lingual and encoded attacks (B3); missing bar = no base control run")
    ax.legend()
    return [save(fig, out, "b3_per_language")]


# ----------------------------------------------------------------------------- 9. N5 rounds
def n5_rounds(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.base_exp == "N5" and r.metrics.get("rounds")]
    if not rs:
        raise Skip("no N5 results")
    rd = rs[0].metrics["rounds"]
    fig, ax = plt.subplots(figsize=(4.4, 3.2))
    ax.plot([x["round"] for x in rd], [x["attack_success_test_proxy"] for x in rd], **series_style(0))
    ax.set_xlabel("hardening round")
    ax.set_ylabel("attack success (TEST)")
    ax.set_title("Adaptive hardening loop (N5)")
    return [save(fig, out, "n5_rounds")]


# ----------------------------------------------------------------------------- 10. T1
def t1_rho(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.base_exp == "T" and r.name == "T1"]
    if not rs:
        raise Skip("no T1 results")
    p = rs[0].dir / "t1_points.json"
    if not p.exists():
        raise Skip(f"T1 stores only summary stats (R2={rs[0].metrics.get('r2')}); no per-item points")
    pts = json.loads(p.read_text())
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(pts["predicted"], pts["measured"], s=10, color=PALETTE[0])
    lim = [0, max(max(pts["predicted"]), max(pts["measured"]))]
    ax.plot(lim, lim, color=INK2, linestyle=":")
    ax.set_xlabel("predicted rho")
    ax.set_ylabel("measured rho")
    ax.set_title(f"T1: R2 = {rs[0].metrics.get('r2'):.3f}")
    return [save(fig, out, "t1_predicted_vs_measured")]


# ----------------------------------------------------------------------------- 11. feature overlap
def feature_overlap(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    sets = {}
    for r in runs:
        if not r.is_mcq or r.is_base:
            continue
        feats = (r.config.get("method_info") or {}).get("features")
        if not feats or r.case != "bio":
            continue
        if r.base_exp == "A1-test" and r.method == "dsg-faithful":
            sets.setdefault(f"A1 seed {r.seed}", set(feats))
        if r.base_exp == "C1":
            ff = (r.cfg["method"].get("gate") or {}).get("features_file", "C1")
            sets.setdefault(f"C1 {Path(ff).stem.replace('C1_', '')}", set(feats))
    if len(sets) < 2:
        raise Skip("fewer than two feature sets")
    names = sorted(sets)
    M = np.array([[len(sets[a] & sets[b]) / max(len(sets[a] | sets[b]), 1) for b in names] for a in names])
    fig, ax = plt.subplots(figsize=(0.45 * len(names) + 2.2, 0.45 * len(names) + 1.6))
    im = ax.imshow(M, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right")
    ax.set_yticks(range(len(names)), names)
    ax.grid(False)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=6,
                    color="white" if M[i, j] > 0.6 else INK)
    fig.colorbar(im, ax=ax, label="Jaccard")
    ax.set_title("Selected-feature overlap (Bio)")
    return [save(fig, out, "feature_overlap")]


# ----------------------------------------------------------------------------- 12. gibberish
def gibberish_by_gate(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.base_exp in ("B6", "C5", "N8", "A3") and isinstance(r.metrics.get("gibberish"), dict)]
    if not rs:
        raise Skip("no generation runs with gibberish metrics")
    rs = sorted(rs, key=lambda r: (r.base_exp, r.name))
    lab = [f"{r.base_exp}:{r.name.split('__', 1)[-1]}" for r in rs]
    m = np.array([[r.metrics["gibberish"]["mean"], r.metrics["gibberish"]["lo"], r.metrics["gibberish"]["hi"]] for r in rs])
    fig, ax = plt.subplots(figsize=(5.5, 0.32 * len(rs) + 1.2))
    y = np.arange(len(rs))
    ax.barh(y, m[:, 0], color=PALETTE[0], height=0.6, edgecolor="white")
    ax.errorbar(m[:, 0], y, xerr=[m[:, 0] - m[:, 1], m[:, 2] - m[:, 0]], fmt="none", ecolor=INK2, elinewidth=1)
    ax.set_yticks(y, lab)
    ax.set_xlabel("gibberish rate (95% CI)")
    ax.set_title("Gibberish rate after clamping, by gate")
    return [save(fig, out, "gibberish_by_gate")]


ALL = [acc_vs_padding, rho_distributions, gate_roc, pareto, probes_by_layer, relearning_curves,
       conformal_coverage, per_language, n5_rounds, t1_rho, feature_overlap, gibberish_by_gate]


def make_all(runs, outdir: Path, paper=False) -> dict:
    res = {}
    for f in ALL:
        try:
            res[f.__name__] = {"written": f(runs, Path(outdir), {"paper": paper})}
        except Skip as e:
            res[f.__name__] = {"skipped": str(e)}
        except Exception as e:  # never let one figure stop the report
            res[f.__name__] = {"error": f"{type(e).__name__}: {e}"}
    return res
