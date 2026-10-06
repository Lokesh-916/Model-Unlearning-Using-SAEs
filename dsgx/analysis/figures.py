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
    if i >= len(PALETTE):  # beyond the palette: neutral ink with a marker no coloured series uses
        return {"color": INK2, "marker": ["h", "p", "d", "<", ">"][(i - len(PALETTE)) % 5]}
    return {"color": PALETTE[i], "marker": MARKERS[i % len(MARKERS)]}


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
        _paper_layout(fig)
    fig.tight_layout()
    for ext in exts:
        fig.savefig(outdir / f"{stem}.{ext}", dpi=200 if ext == "png" else None)
    import matplotlib.pyplot as plt

    plt.close(fig)
    return stem


def _paper_layout(fig):
    """Paper mode: the LaTeX caption replaces the title of a single-panel figure, and every legend moves below the
    panels (one figure legend), so no legend covers data or runs off the canvas at column width."""
    axes = [a for a in fig.axes if a.get_visible() and a.axison]
    if len(axes) == 1:
        axes[0].set_title("")
    handles, labels = [], []
    for a in axes:
        leg = a.get_legend()
        if leg is None:
            continue
        for h, l in zip(*a.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
        leg.remove()
    if not labels:
        return
    w, h = fig.get_size_inches()
    ncol = max(1, min(len(labels), 3 if w < 5 else 4, max(1, int(w / max(len(l) for l in labels) * 13))))
    rows = (len(labels) + ncol - 1) // ncol
    extra = 0.16 * rows + 0.1
    fig.set_size_inches(w, h + extra)
    fig.legend(handles, labels, loc="lower center", ncol=ncol, fontsize=6.5, frameon=False, handlelength=1.8,
               columnspacing=1.0, borderaxespad=0.2)
    fig.tight_layout(rect=(0, extra / (h + extra), 1, 1))
    fig.tight_layout = lambda *a, **k: None  # keep the reserved legend band in save()


def _forget_acc(r):
    f = r.forget("raw") or {}
    return f.get("mean"), f.get("lo"), f.get("hi")


# ----------------------------------------------------------------------------- 1. accuracy vs padding
def acc_vs_padding(runs, out, ctx):
    """One panel per padding position (before / after / around the question), one line per filler source, the
    unguarded base model (dashed) in every panel; pads are evenly spaced (they roughly double)."""
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.base_exp == "B1" and r.is_mcq and r.attack.get("name") == "dilution"]
    if not rs:
        raise Skip("no B1 runs")
    pads = sorted({int(r.attack.get("pad", 0) or 0) for r in rs})
    pos = {p: i for i, p in enumerate(pads)}
    base = sorted([r for r in rs if r.is_base], key=lambda r: int(r.attack.get("pad", 0) or 0))
    dsg = [r for r in rs if not r.is_base]
    positions = [p for p in ("before", "after", "around") if any(r.attack.get("position", "before") == p for r in dsg)]
    sources = sorted({r.attack.get("source", "wikitext") for r in dsg})
    fig, axs = plt.subplots(1, len(positions), figsize=(2.3 * len(positions), 2.4), squeeze=False, sharey=True)

    def line(ax, g, label, **kw):
        g = sorted(g, key=lambda r: int(r.attack.get("pad", 0) or 0))
        x = [pos[int(r.attack.get("pad", 0) or 0)] for r in g]
        m = np.array([_forget_acc(r) for r in g], dtype=float)
        ax.plot(x, m[:, 0], label=label, markersize=3.5, linewidth=1.3, **kw)
        ax.fill_between(x, m[:, 1], m[:, 2], color=kw["color"], alpha=0.10, linewidth=0)

    for ax, p in zip(axs[0], positions):
        if base:
            line(ax, base, "base model (no gate)", color=INK2, marker="o", linestyle="--")
        for i, src in enumerate(sources):
            g = [r for r in dsg if r.attack.get("position", "before") == p and r.attack.get("source", "wikitext") == src]
            if g:
                line(ax, g, f"DSG, {src.replace('benign_bio', 'benign bio')} filler", **series_style(i))
        ax.axhline(0.25, color=INK2, linewidth=0.8, linestyle=":")
        ax.set_xticks(range(len(pads)), [str(q) for q in pads], rotation=45)
        ax.minorticks_off()
        ax.set_title(f"padding {p} the question")
        ax.set_xlabel("padding tokens")
    axs[0][0].set_ylabel("WMDP-Bio accuracy")
    axs[0][0].legend(fontsize=6)
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


# ----------------------------------------------------------------------------- 4b. Cyber Pareto (Wave-1 decision)
# No DSG config keeps Cyber full-MMLU utility within 1 point of base (DEV grid), so Cyber is reported as a
# forget-vs-utility Pareto curve. The DSG paper's utility metric (4 legacy subjects, base-correct items =
# view "dsg_subset") is shown next to full MMLU, so the size of each utility drop is visible under both metrics.
RETAIN_NAME = {"mmlu-aux-chat": "chat-retain", "wikitext": "WikiText retain"}


def cyber_pareto_rows(runs) -> dict:
    """{'dev': [...], 'test': [...], 'base': {'dev': row, 'test': row}}: per A1 Cyber run forget (raw), full-MMLU
    utility (raw, pooled), 4-subject utility (dsg_subset, pooled), drops vs base in points, DEV Pareto flag."""
    def row(r):
        m = r.cfg.get("method") or {}
        f, u, u4 = r.forget("raw") or {}, r.utility("raw") or {}, r.utility("dsg_subset") or {}
        return {"run": r.name, "split": r.split, "seed": r.seed, "base": r.is_base,
                "retain": RETAIN_NAME.get(m.get("retain_corpus"), m.get("retain_corpus") or "-"),
                "N": m.get("n_features"), "pct": m.get("retain_pct"), "c": m.get("multiplier"),
                "forget": f, "full": u, "util4": u4}
    out = {"dev": [], "test": [], "base": {}}
    for r in runs:
        if r.base_exp not in ("A1-dev", "A1-test") or not r.is_mcq or r.case != "cyber" or not r.forget("raw") or not r.utility("raw"):
            continue
        if "rmu" in (r.cfg.get("dataset_label") or ""):
            continue  # unverified third-party reference, not part of the curve
        x = row(r)
        sp = "dev" if r.base_exp == "A1-dev" else "test"
        if x["base"]:
            out["base"][sp] = x
        else:
            out[sp].append(x)
    for sp in ("dev", "test"):
        b = out["base"].get(sp)
        for x in out[sp]:
            x["d_full"] = 100 * (x["full"]["mean"] - b["full"]["mean"]) if b else None
            x["d_util4"] = 100 * (x["util4"]["mean"] - b["util4"]["mean"]) if (b and x["util4"] and b["util4"]) else None
    dev = out["dev"]
    for x in dev:  # Pareto-optimal over all DEV configs: no other config has lower-or-equal forget AND higher-or-equal utility, one strictly
        fx, ux = x["forget"]["mean"], x["full"]["mean"]
        x["pareto"] = not any((y["forget"]["mean"] <= fx and y["full"]["mean"] >= ux) and
                              (y["forget"]["mean"] < fx or y["full"]["mean"] > ux) for y in dev)
    return out


def cyber_pareto(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    d = cyber_pareto_rows(runs)
    if not d["dev"] or "dev" not in d["base"]:
        raise Skip("no A1-dev Cyber runs with a base run")
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.9, 3.0))
    b = d["base"]["dev"]
    names = sorted({x["retain"] for x in d["dev"]})
    for i, nm in enumerate(names):
        st = series_style(i)
        g = [x for x in d["dev"] if x["retain"] == nm]
        ax.scatter([x["full"]["mean"] for x in g], [x["forget"]["mean"] for x in g], s=20, color=st["color"],
                   marker=st["marker"], edgecolor="white", linewidth=0.6, label=f"DSG, {nm}", zorder=3)
        g4 = [x for x in g if x["d_util4"] is not None]  # runs without the dsg_subset view have no 4-subject point
        bx.scatter([x["d_full"] for x in g4], [x["d_util4"] for x in g4], s=20, color=st["color"], marker=st["marker"],
                   edgecolor="white", linewidth=0.6, label=f"DSG, {nm}", zorder=3)
    front = sorted([x for x in d["dev"] if x["pareto"]], key=lambda x: x["full"]["mean"])
    ax.plot([x["full"]["mean"] for x in front], [x["forget"]["mean"] for x in front], color=INK2, linewidth=1.2,
            drawstyle="steps-post", label="Pareto front (DEV)", zorder=2)
    ax.scatter([b["full"]["mean"]], [b["forget"]["mean"]], s=60, facecolor="white", edgecolor=INK, linewidth=1.5,
               marker="o", label="base", zorder=4)
    ax.axvline(b["full"]["mean"] - 0.01, color=INK2, linewidth=0.8, linestyle="--")
    ax.text(b["full"]["mean"] - 0.012, ax.get_ylim()[1], "base $-$1 pt", ha="right", va="top", fontsize=6, color=INK2)
    ax.set_xlabel("full-MMLU utility (DEV, raw)")
    ax.set_ylabel("WMDP-Cyber accuracy (DEV, raw)")
    ax.set_title("(a) Cyber forget vs utility (lower-right better)")
    ax.legend(fontsize=6, loc="upper left")
    lo = min([x["d_full"] for x in d["dev"]] + [x["d_util4"] for x in d["dev"] if x["d_util4"] is not None] + [0])
    bx.plot([lo, 0], [lo, 0], color=INK2, linewidth=0.8, linestyle=":", label="equal drop")
    for sp, mk in (("test", "*"),):
        for x in d[sp]:
            if x["d_util4"] is not None:
                bx.scatter([x["d_full"]], [x["d_util4"]], s=45, marker=mk, facecolor="none",
                           edgecolor=series_style(names.index(x["retain"]) if x["retain"] in names else 2)["color"], linewidth=0.9, zorder=4)
    if d["test"]:
        bx.scatter([], [], s=45, marker="*", facecolor="none", edgecolor=INK2, label="selected config, TEST (5 seeds)")
    bx.set_xlabel("full-MMLU utility change vs base (points)")
    bx.set_ylabel("4-subject utility change (points)")
    bx.set_title("(b) Utility drop: full MMLU vs DSG's 4 subjects")
    bx.legend(fontsize=6, loc="upper left")
    return [save(fig, out, "cyber_pareto")]


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
        ax.plot(L, m, label=f"{r.name.split('__', 1)[1]} probe", markersize=4, **st)
        ax.plot(L, c, color=st["color"], linestyle=":", linewidth=1.2, label=f"{r.name.split('__', 1)[1]} control")
    ax.axhline(0.25, color=INK2, linewidth=1, linestyle="--")
    ax.set_xlabel("layer")
    ax.set_ylabel("probe accuracy (test)")
    ax.set_title("Linear answer probes by layer (dashed = chance)")
    ax.legend(fontsize=6, ncol=2)
    return [save(fig, out, "probe_accuracy_by_layer")]


# ----------------------------------------------------------------------------- 6. relearning curves
def relearning_curves(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rl = [r for r in runs if (r.base_exp == "A6" or r.base_exp.startswith("A6-full")) and r.metrics.get("curve")]
    if ctx.get("full_only"):
        rl = [r for r in rl if str(r.metrics.get("rank")) == "full"]
    if not rl:
        raise Skip("no A6 relearning results")
    rank_order = {"full": 0, "64": 1, "8": 2}
    ks = sorted({(r.metrics.get("k"), r.metrics.get("rank")) for r in rl},
                key=lambda kr: (rank_order.get(str(kr[1]), 9), int(kr[0] or 0)))
    n = len(ks)
    cols = min(n, 4)
    rows = (n + cols - 1) // cols
    fig, axs = plt.subplots(rows, cols, figsize=(3.0 * cols, 2.5 * rows), squeeze=False, sharey=True)
    conds = sorted({r.metrics["condition"] for r in rl}, key=lambda c: (list(A6_NAMES).index(c) if c in A6_NAMES else 99, c))
    for ai, (k, rank) in enumerate(ks):
        ax = axs[ai // cols][ai % cols]
        for ci, cnd in enumerate(conds):
            g = [r for r in rl if r.metrics["condition"] == cnd and (r.metrics.get("k"), r.metrics.get("rank")) == (k, rank)]
            for r in g[:1]:
                m = r.metrics
                x = [0] + [p["step"] for p in m["curve"]]
                y = [m["before"]["forget_acc"]] + [p["forget_acc"] for p in m["curve"]]
                name = A6_NAMES.get(cnd, cnd)
                if cnd == "d1":
                    name = "D1-full, beta 0.3 (utility at chance)" if str(rank) == "full" else "D1 low-rank, beta 0.3"
                ax.plot(x, y, label=name, markersize=4, linewidth=1.5, **series_style(ci))
        ax.set_title(f"k = {k}, " + ("full FT" if str(rank) == "full" else f"LoRA r{rank}"))
        ax.set_xlabel("relearn steps")
    axs[0][0].set_ylabel("forget accuracy")
    for j in range(n, rows * cols):
        axs[j // cols][j % cols].axis("off")
    for a in (axs.flat if ctx.get("paper") else [axs[0][0]]):
        if a.lines:
            a.legend(fontsize=6)  # paper mode merges every panel's entries into one legend below
    return [save(fig, out, "relearning_full" if ctx.get("full_only") else "relearning_curves")]


A6_NAMES = {"dsg-hook": "DSG, hook kept", "dsg-nohook": "DSG, hook removed", "d1-a0.1": "D1-full, beta 0.1",
            "d1v2": "D1 v2", "d1": "D1, beta 0.3", "d2": "D2 null-space edit", "student": "distilled control (beta 0)",
            "rmu": "RMU v1", "rmu-v2": "RMU v2"}


def relearning_full(runs, out, ctx):
    """Main-text version of the relearning curves: full fine-tuning panels only (gpuws)."""
    return relearning_curves(runs, out, {**ctx, "full_only": True})


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


ALL = [acc_vs_padding, rho_distributions, gate_roc, pareto, cyber_pareto, probes_by_layer, relearning_curves, relearning_full,
       conformal_coverage, per_language, n5_rounds, t1_rho, feature_overlap, gibberish_by_gate]


# ============================================================================= DSG figure parity (session 8)
# Every figure type of the DSG paper, for DSG and for our methods. Inputs: harness runs (lab or gpuws) and
# the gpuws figure-parity job (cluster/figparity.py: FP-* experiments).
SCORE_COL = {"window": "window_max", "cusum": "cusum_max", "probe_sae": "probe_score", "probe_resid": "probe_score"}


def _score(r):
    """(column, values Series) of the gate score in a run's items (gate type aware), or (None, None)."""
    it = r.items
    if it is None:
        return None, None
    g = (r.cfg.get("method") or {}).get("gate") or {}
    for col in (SCORE_COL.get(g.get("type", "rho")), "gate_score", "rho"):
        if col and col in it and it[col].notna().any():
            return col, it[col]
    return None, None


def _method_tag(r):
    m = r.cfg.get("method") or {}
    if m.get("name") == "gated":
        g = m.get("gate") or {}
        return f"gate {g.get('type', 'rho')}" + (f"-w{g['w']}" if "w" in g else "")
    return {"dsg-faithful": "DSG"}.get(m.get("name"), m.get("name", "?"))


# ----------------------------------------------------------------------------- P1. gate-score distributions
def gate_score_distributions(runs, out, ctx):
    """Per method: gate score on forget items (clean), retain/utility items (clean) and forget items under attack."""
    plt = setup(ctx.get("paper"))
    clean, attacked = {}, {}
    for r in runs:
        if not r.is_mcq or r.is_base or r.split != "test" or r.items is None:
            continue
        col, _ = _score(r)
        if not col:
            continue
        key = _method_tag(r)
        if r.is_clean:
            clean.setdefault(key, r)
        else:
            attacked.setdefault(key, []).append(r)
    keys = [k for k in sorted(clean) if k in attacked] or sorted(clean)
    if not keys:
        raise Skip("no gated TEST runs with per-item gate scores")
    keys = keys[:4]
    fig, axs = plt.subplots(1, len(keys), figsize=(3.0 * len(keys), 2.8), squeeze=False)
    for ax, k in zip(axs[0], keys):
        r = clean[k]
        col, sc = _score(r)
        it = r.items
        fd = r.cfg.get("forget_datasets") or ["wmdp-bio"]
        parts = [("forget (clean)", sc[it["dataset"].isin(fd)], PALETTE[0]),
                 ("retain / utility (clean)", sc[~it["dataset"].isin(fd)], PALETTE[1])]
        if k in attacked:
            a = max(attacked[k], key=lambda x: int(x.attack.get("pad", 0) or 0))
            ac, asc = _score(a)
            parts.append((f"forget, {a.attack.get('name')} {a.attack.get('pad', a.attack.get('lang', ''))}",
                          asc[a.items["dataset"].isin(fd)], PALETTE[2]))
        hi = max(float(np.nanmax(v.values)) for _, v, _ in parts if len(v)) or 1e-3
        bins = np.linspace(0, hi, 40)
        for lab, v, c in parts:
            ax.hist(v.dropna(), bins=bins, color=c, alpha=0.55, label=lab, density=True)
        thr = r.gate.get("tau") if r.gate.get("tau") is not None else (r.config.get("method_info") or {}).get("calib", {}).get("threshold")
        if thr is not None and np.isfinite(thr) and thr >= 0:
            ax.axvline(thr, color=INK, linestyle="--", linewidth=1.1, label=f"threshold {thr:.3f}")
        ax.set_title(f"{k} ({r.base_exp})")
        ax.set_xlabel(col)
    axs[0][0].set_ylabel("density")
    axs[0][0].legend(fontsize=6)
    return [save(fig, out, "gate_score_distributions")]


# ----------------------------------------------------------------------------- P2. forget vs utility (TEST)
def forget_utility_test(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.is_mcq and r.split == "test" and r.is_clean and r.case in (None, "bio")
          and r.forget("raw") and r.utility("raw") and (r.cfg.get("forget_datasets") or ["wmdp-bio"]) == ["wmdp-bio"]
          and len(r.cfg.get("datasets", [])) > 10]  # full-MMLU utility only (comparable x axis)
    if not rs:
        raise Skip("no clean TEST runs with full-MMLU utility")
    groups = {}
    for r in rs:
        tag = "base" if r.is_base else (f"{_method_tag(r)}" if not r.weights else f"weights: {Path(str(r.weights)).name}")
        groups.setdefault(tag, []).append(r)
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for i, (k, g) in enumerate(sorted(groups.items(), key=lambda kv: (kv[0] != "base", kv[0]))):
        st = series_style(i)
        x = [r.utility("raw")["mean"] for r in g]
        y = [r.forget("raw")["mean"] for r in g]
        ax.scatter(x, y, s=60 if k == "base" else 26, color=st["color"], marker=st["marker"], label=f"{k} ({len(g)})",
                   facecolor="white" if k == "base" else st["color"], linewidth=1.6)
    ax.axhline(0.25, color=INK2, linewidth=1, linestyle=":")
    ax.set_xlabel("full-MMLU utility (TEST, raw, pooled)")
    ax.set_ylabel("WMDP-Bio forget accuracy (TEST, raw)")
    ax.set_title("Forget vs utility, every clean TEST condition (lower right is better)")
    ax.legend(fontsize=6, ncol=2)
    return [save(fig, out, "forget_utility_test")]


# ----------------------------------------------------------------------------- P3. relearning by epochs
def relearning_epochs(runs, out, ctx):
    """Relearning curves with x = epochs over the k relearn passages (steps x batch / k); full FT and LoRA."""
    plt = setup(ctx.get("paper"))
    rl = [r for r in runs if (r.base_exp in ("A6",) or r.base_exp.startswith("A6-full")) and r.metrics.get("curve")]
    if not rl:
        raise Skip("no A6 relearning results")
    ks = sorted({r.metrics.get("k") for r in rl})
    fig, axs = plt.subplots(1, len(ks), figsize=(2.6 * len(ks), 2.6), squeeze=False, sharey=True)
    conds = sorted({(r.metrics["condition"], str(r.metrics.get("rank"))) for r in rl})
    for ax, k in zip(axs[0], ks):
        for ci, (cnd, rank) in enumerate(conds):
            g = [r for r in rl if (r.metrics["condition"], str(r.metrics.get("rank"))) == (cnd, rank) and r.metrics.get("k") == k]
            for r in g[:1]:
                m = r.metrics
                bs = m.get("batch_size") or (r.config.get("args") or {}).get("bs") or 4
                x = [0] + [p["step"] * bs / k for p in m["curve"]]
                y = [m["before"]["forget_acc"]] + [p["forget_acc"] for p in m["curve"]]
                ax.plot(x, y, label=f"{cnd} ({'full' if rank == 'full' else 'LoRA r' + rank})", **series_style(ci))
        ax.set_xscale("symlog", linthresh=1)
        ax.set_title(f"k = {k} passages")
        ax.set_xlabel("epochs over the k passages")
    axs[0][0].set_ylabel("WMDP-Bio forget accuracy")
    axs[0][0].legend(fontsize=5.5)
    return [save(fig, out, "relearning_epochs")]


def _fp(runs, exp):
    return [r for r in runs if r.base_exp == exp and r.is_mcq]


def _fpp(r):
    return r.cfg.get("fp_params") or {}


# ----------------------------------------------------------------------------- P4. clamp strength x N
def clamp_grid(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in _fp(runs, "FP-clamp") if _fpp(r)]
    if not rs:
        raise Skip("no FP-clamp runs (gpuws job figs)")
    base = next((r for r in _fp(runs, "FP-clamp") if r.is_base), None)
    meths = sorted({_fpp(r)["method"] for r in rs})
    fig, axs = plt.subplots(2, len(meths), figsize=(3.3 * len(meths), 4.6), squeeze=False, sharex=True)
    for j, me in enumerate(meths):
        ns = sorted({_fpp(r)["n"] for r in rs if _fpp(r)["method"] == me})
        for i, n in enumerate(ns):
            g = sorted([r for r in rs if _fpp(r)["method"] == me and _fpp(r)["n"] == n], key=lambda r: _fpp(r)["c"])
            x = [_fpp(r)["c"] for r in g]
            st = series_style(i)
            axs[0][j].plot(x, [r.forget("raw")["mean"] for r in g], label=f"N={n}", **st)
            axs[1][j].plot(x, [(r.utility("raw") or {}).get("mean") for r in g], **st)
        for row, getter in ((0, "forget"), (1, "utility")):
            if base:
                v = (base.forget("raw") if getter == "forget" else base.utility("raw")) or {}
                axs[row][j].axhline(v.get("mean", np.nan), color=INK2, linestyle="--", linewidth=1, label="base" if row == 0 else None)
            axs[row][j].set_xscale("log")
        axs[0][j].set_title({"dsg": "DSG", "ours": "our gate"}.get(me, me))
        axs[1][j].set_xlabel("clamp strength c")
    axs[0][0].set_ylabel("WMDP-Bio DEV accuracy")
    axs[1][0].set_ylabel("MMLU (4 subj.) DEV accuracy")
    axs[0][0].legend(fontsize=6, ncol=2)
    return [save(fig, out, "clamp_strength_grid")]


# ----------------------------------------------------------------------------- P5. static vs dynamic
def static_dynamic(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = {r.cfg.get("dataset_label"): r for r in _fp(runs, "FP-static")}
    order = [k for k in ("base", "dsg-dynamic", "dsg-static", "ours-dynamic", "ours-static") if k in rs]
    if len(order) < 2:
        raise Skip("no FP-static runs (gpuws job figs)")
    fig, ax = plt.subplots(figsize=(4.8, 3.0))
    x = np.arange(len(order))
    for off, (lab, get, c) in zip((-0.2, 0.2), (("forget (WMDP-Bio)", "forget", PALETTE[0]), ("utility (full MMLU)", "utility", PALETTE[1]))):
        v = [((rs[k].forget("raw") if get == "forget" else rs[k].utility("raw")) or {}) for k in order]
        m = np.array([d.get("mean", np.nan) for d in v])
        err = np.array([[d.get("mean", 0) - d.get("lo", 0), d.get("hi", 0) - d.get("mean", 0)] for d in v]).T
        ax.bar(x + off, m, 0.38, yerr=err, color=c, label=lab, edgecolor="white", capsize=2, error_kw={"elinewidth": 0.8})
    ax.set_xticks(x, order, rotation=15)
    ax.set_ylabel("TEST accuracy (raw)")
    ax.set_title("Static (always clamp) vs dynamic (gated) clamping")
    ax.legend()
    return [save(fig, out, "static_vs_dynamic")]


# ----------------------------------------------------------------------------- P6. data efficiency
def data_efficiency(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in _fp(runs, "FP-dataeff") if _fpp(r)]
    if not rs:
        raise Skip("no FP-dataeff runs (gpuws job figs)")
    base = next((r for r in _fp(runs, "FP-dataeff") if r.is_base), None)
    fig, axs = plt.subplots(1, 2, figsize=(6.2, 2.8))
    for i, me in enumerate(sorted({_fpp(r)["method"] for r in rs})):
        st = series_style(i)
        ms = sorted({_fpp(r)["m"] for r in rs if _fpp(r)["method"] == me})
        for ax, get in zip(axs, ("forget", "utility")):
            vals = [[((r.forget("raw") if get == "forget" else r.utility("raw")) or {}).get("mean", np.nan)
                     for r in rs if _fpp(r)["method"] == me and _fpp(r)["m"] == m] for m in ms]
            mean = [np.nanmean(v) for v in vals]
            ax.plot(ms, mean, label={"dsg": "DSG", "ours": "our gate"}.get(me, me), **st)
            ax.fill_between(ms, [np.nanmin(v) for v in vals], [np.nanmax(v) for v in vals], color=st["color"], alpha=0.15, linewidth=0)
    for ax, get in zip(axs, ("forget", "utility")):
        if base:
            v = (base.forget("raw") if get == "forget" else base.utility("raw")) or {}
            ax.axhline(v.get("mean", np.nan), color=INK2, linestyle="--", linewidth=1, label="base")
        ax.set_xscale("log", base=2)
        ax.set_xlabel("feature-selection corpus (rows of 1024 tokens per side)")
    axs[0].set_ylabel("WMDP-Bio DEV accuracy")
    axs[1].set_ylabel("MMLU (4 subj.) DEV accuracy")
    axs[0].legend(fontsize=6)
    fig.suptitle("Data efficiency: mean over seeds, band = min-max", fontsize=8)
    return [save(fig, out, "data_efficiency")]


def per_dataset_acc(r, ds):
    it = r.items
    if it is None:
        return np.nan
    g = it[it["dataset"] == ds]["correct"]
    return float(g.mean()) if len(g) else np.nan


# ----------------------------------------------------------------------------- P7. multi-topic
def multitopic(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = {r.cfg.get("dataset_label"): r for r in _fp(runs, "FP-multitopic")}
    if not rs:
        raise Skip("no FP-multitopic runs (gpuws job figs)")
    order = [k for k in ("base", "dsg-bio-only", "dsg-cyber-only", "dsg-bio+cyber", "ours-bio+cyber") if k in rs]
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    x = np.arange(len(order))
    for j, (lab, fn) in enumerate((("WMDP-Bio", lambda r: per_dataset_acc(r, "wmdp-bio")),
                                   ("WMDP-Cyber", lambda r: per_dataset_acc(r, "wmdp-cyber")),
                                   ("MMLU utility", lambda r: (r.utility("raw") or {}).get("mean", np.nan)))):
        ax.bar(x + (j - 1) * 0.27, [fn(rs[k]) for k in order], 0.26, color=PALETTE[j], label=lab, edgecolor="white")
    ax.axhline(0.25, color=INK2, linewidth=1, linestyle=":")
    ax.set_xticks(x, order, rotation=15)
    ax.set_ylabel("TEST accuracy (raw)")
    ax.set_title("Multi-topic unlearning (bio + cyber at once)")
    ax.legend(fontsize=6, ncol=3)
    return [save(fig, out, "multitopic")]


# ----------------------------------------------------------------------------- P8. latency
def latency_by_length(runs, out, ctx):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.base_exp == "FP-latency" and r.metrics.get("latency")]
    if not rs:
        raise Skip("no FP-latency results (gpuws job figs)")
    lat = rs[0].metrics["latency"]
    Ls = sorted(int(k) for k in lat)
    names = [k for k in lat[str(Ls[0])] if k != "base"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    for i, nm in enumerate(names):
        ax.plot(Ls, [100 * lat[str(L)][nm]["overhead_vs_base"] for L in Ls], label=nm, **series_style(i))
    ax.axhline(0, color=INK2, linewidth=1)
    ax.set_xscale("log", base=2)
    ax.set_xlabel("sequence length (tokens), batch size 1")
    ax.set_ylabel("latency overhead vs base (%)")
    ax.set_title(f"Forward latency overhead ({rs[0].hardware})")
    ax.legend(fontsize=6)
    return [save(fig, out, "latency_by_length")]


# ----------------------------------------------------------------------------- P9. TOFU feature highlights
def tofu_highlight(runs, out, ctx, n_show=2, max_tok=90):
    plt = setup(ctx.get("paper"))
    rs = [r for r in runs if r.base_exp == "FP-highlight" and r.metrics.get("examples")]
    if not rs:
        raise Skip("no FP-highlight results (TOFU only; gpuws job figs)")
    m = rs[0].metrics
    ex = [e for e in m["examples"] if e["part"] == "forget"][:n_show] + [e for e in m["examples"] if e["part"] == "retain"][:n_show]
    import matplotlib.colors as mcolors

    vmax = max(max(e["sel_max_act"][1:] or [0]) for e in ex) or 1.0
    cmap = mcolors.LinearSegmentedColormap.from_list("hl", ["#ffffff", PALETTE[1]])
    fig, axs = plt.subplots(len(ex), 1, figsize=(6.9, 1.05 * len(ex)), squeeze=False)
    for ax, e in zip(axs[:, 0], ex):
        ax.axis("off")
        x, y = 0.0, 0.85
        toks = e["tokens"][1:max_tok]
        acts = e["sel_max_act"][1:max_tok]
        for t, a in zip(toks, acts):
            s = t.replace("▁", " ").replace("\n", " \\n ")
            if s.startswith("<") and s.endswith(">"):
                continue
            w = 0.0105 * max(len(s), 1)
            if x + w > 1.0:
                x, y = 0.0, y - 0.3
            ax.text(x, y, s, fontsize=6, family="monospace", va="top", transform=ax.transAxes,
                    bbox={"boxstyle": "square,pad=0.05", "facecolor": cmap(min(max(a, 0) / vmax, 1.0)), "edgecolor": "none"})
            x += w
        ax.set_title(f"{e['part']}: rho {e['rho']:.3f} ({'fires' if e['gate_rho_fires'] else 'no fire'}), "
                     f"window {e['window']:.3f} ({'fires' if e['gate_window_fires'] else 'no fire'})", fontsize=7, loc="left")
    fig.suptitle("TOFU: max activation of the 20 selected SAE features per token (darker = higher)", fontsize=8)
    return [save(fig, out, "tofu_feature_highlight")]


ALL += [gate_score_distributions, forget_utility_test, relearning_epochs, clamp_grid, static_dynamic,
        data_efficiency, multitopic, latency_by_length, tofu_highlight]


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
