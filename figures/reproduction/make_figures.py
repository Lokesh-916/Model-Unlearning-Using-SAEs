"""Reproduction figures for the DSG paper (COLM 2025), built from our own
independently-reproduced numbers (see the *_REPRODUCTION.md / ABLATION_*.md
reports in the project root). Not pixel-identical to the paper's figures -
same claims, our data.

Run from anywhere: python3 figures/reproduction/make_figures.py
Outputs PNGs into this same directory.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
OUT = HERE

# ---- palette (validated categorical set, see dataviz skill references/palette.md) ----
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
RED = "#e34948"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": GRID,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "font.size": 11,
})


def style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.grid(axis="y", visible=True)
    ax.grid(axis="x", visible=False)
    ax.tick_params(length=0)


def load(path):
    with open(os.path.join(ROOT, path)) as f:
        return json.load(f)


def savefig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print("wrote", path)


# ============================================================
# Figure 1 - Data efficiency (paper Section 4.4 / Figure 6A)
# ============================================================
def fig_data_efficiency():
    d = load("data_efficiency_reproduction/dsg_dataefficiency_bio_results.json")
    fracs = sorted((int(k) for k in d.keys()), reverse=True)
    wmdp = [d[str(f)]["wmdp-bio"] * 100 for f in fracs]
    mmlu = [d[str(f)]["mmlu_avg"] * 100 for f in fracs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.axhspan(0, 40, color=AQUA, alpha=0.06, zorder=0)
    ax.plot(fracs, wmdp, marker="o", markersize=6, linewidth=2, color=BLUE, label="WMDP-Bio accuracy (lower = better forgetting)")
    ax.plot(fracs, mmlu, marker="o", markersize=6, linewidth=2, color=ORANGE, label="MMLU accuracy (higher = better retained utility)")
    ax.axhline(40, color=INK_MUTED, linestyle="--", linewidth=1)
    ax.text(0.98, 41.5, "paper's 40% WMDP ceiling", fontsize=9, color=INK_SECONDARY,
            ha="right", transform=ax.get_yaxis_transform())
    ax.set_xlabel("Feature-selection data used (% of forget/retain corpus)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Data-efficiency reproduction: DSG holds below 40% WMDP\ndown to 20% of the training data (paper Section 4.4)")
    ax.set_xticks(fracs)
    ax.set_ylim(0, 105)
    ax.invert_xaxis()
    style_axes(ax)
    ax.legend(frameon=False, loc="lower left", fontsize=9)
    savefig(fig, "fig1_data_efficiency.png")


# ============================================================
# Figure 2 - Zero-shot tau sweep, bio + cyber (Section 4.4 / Figure 6B)
# ============================================================
def fig_zeroshot():
    bio = load("zeroshot_reproduction/dsg_zeroshot_bio_results.json")
    cyber = load("zeroshot_reproduction/dsg_zeroshot_cyber_results.json")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, data, key, title, tau_star in (
        (axes[0], bio, "wmdp-bio", "Bio", 0.6),
        (axes[1], cyber, "wmdp-cyber", "Cyber", 0.2),
    ):
        taus = sorted(float(t) for t in data.keys())
        wmdp = [data[f"{t:.1f}"][key] * 100 for t in taus]
        mmlu = [data[f"{t:.1f}"]["mmlu_avg"] * 100 for t in taus]
        ax.plot(taus, wmdp, marker="o", markersize=5, linewidth=2, color=BLUE, label=f"WMDP-{title}")
        ax.plot(taus, mmlu, marker="o", markersize=5, linewidth=2, color=ORANGE, label="MMLU avg")
        ax.axvline(tau_star, color=INK_MUTED, linestyle="--", linewidth=1)
        ax.text(tau_star, 103, f"paper-optimal τ={tau_star}", fontsize=8.5, color=INK_SECONDARY, ha="center")
        ax.set_xlabel("Fixed threshold τ")
        ax.set_title(f"WMDP-{title}", fontsize=12)
        style_axes(ax)
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_ylim(0, 108)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Zero-shot DSG (no forget/retain data): fixed-τ sweep\n(paper Section 4.4, Figure 6B)", y=1.06, fontsize=13)
    savefig(fig, "fig2_zeroshot_tau_sweep.png")


# ============================================================
# Figure 3 - Clamp strength ablation (Section 4.5 / Figure 11)
# ============================================================
def fig_clamp():
    d = load("ablations/dsg_clamp_ablation_results.json")["results"]
    cs = sorted(int(c) for c in d.keys())
    wmdp = [d[str(c)]["wmdp-bio"] * 100 for c in cs]
    mmlu = [d[str(c)]["mmlu_avg"] * 100 for c in cs]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(cs, wmdp, marker="o", markersize=6, linewidth=2, color=BLUE, label="WMDP-Bio accuracy")
    ax.plot(cs, mmlu, marker="o", markersize=6, linewidth=2, color=ORANGE, label="MMLU avg")
    ax.set_xscale("log")
    ax.set_xticks(cs)
    ax.set_xticklabels([str(c) for c in cs], rotation=45, ha="right")
    ax.set_xlabel("Clamp strength c")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Clamp-strength ablation: stable from c=50 to c=500\n(paper Section 4.5, Figure 11)")
    style_axes(ax)
    ax.legend(frameon=False, loc="center right", fontsize=9)
    savefig(fig, "fig3_clamp_ablation.png")


# ============================================================
# Figure 4 - p_ratio / p_dyn threshold ablations (Section 4.5)
# ============================================================
def fig_thresholds():
    d = load("ablations/dsg_threshold_ablations_results.json")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    pr = d["p_ratio_sweep"]
    xs = sorted(int(k) for k in pr.keys())
    wmdp = [pr[str(x)]["wmdp-bio"] * 100 for x in xs]
    mmlu = [pr[str(x)]["mmlu_avg"] * 100 for x in xs]
    axes[0].plot(xs, wmdp, marker="o", markersize=6, linewidth=2, color=BLUE, label="WMDP-Bio")
    axes[0].plot(xs, mmlu, marker="o", markersize=6, linewidth=2, color=ORANGE, label="MMLU avg")
    axes[0].set_xlabel("p_ratio percentile")
    axes[0].set_title("Feature-selection percentile (p_ratio)")
    axes[0].set_ylim(0, 105)

    pd = d["p_dyn_sweep"]
    xs2 = sorted(int(k) for k in pd.keys())
    wmdp2 = [pd[str(x)]["wmdp-bio"] * 100 for x in xs2]
    mmlu2 = [pd[str(x)]["mmlu_avg"] * 100 for x in xs2]
    axes[1].plot(xs2, wmdp2, marker="o", markersize=6, linewidth=2, color=BLUE, label="WMDP-Bio")
    axes[1].plot(xs2, mmlu2, marker="o", markersize=6, linewidth=2, color=ORANGE, label="MMLU avg")
    axes[1].axvspan(90, 95, color=AQUA, alpha=0.08)
    axes[1].text(92.5, 103, "paper's\nsweet spot", fontsize=8, color=INK_SECONDARY, ha="center")
    axes[1].set_xlabel("p_dyn percentile (retain calibration)")
    axes[1].set_title("Dynamic threshold percentile (p_dyn)")
    axes[1].set_ylim(0, 105)

    for ax in axes:
        ax.set_ylabel("Accuracy (%)")
        style_axes(ax)
        ax.legend(frameon=False, loc="lower left", fontsize=9)
    fig.suptitle("Threshold sensitivity ablations (paper Section 4.5)", y=1.03, fontsize=13)
    savefig(fig, "fig4_threshold_ablations.png")


# ============================================================
# Figure 5 - rho vs rho_raw discriminability (Section 4.5 / Appendix K.1, Figure 12)
# ============================================================
def fig_rho():
    d = load("ablations/dsg_rho_comparison_results.json")
    comparisons = ["WikiText vs MMLU\n(want low)", "WikiText vs WMDP-Bio\n(want high)"]
    rho_mean = [d["wikitext_vs_mmlu"]["rho"]["mean"], d["wikitext_vs_wmdp_bio"]["rho"]["mean"]]
    rho_std = [d["wikitext_vs_mmlu"]["rho"]["std"], d["wikitext_vs_wmdp_bio"]["rho"]["std"]]
    raw_mean = [d["wikitext_vs_mmlu"]["rho_raw"]["mean"], d["wikitext_vs_wmdp_bio"]["rho_raw"]["mean"]]
    raw_std = [d["wikitext_vs_mmlu"]["rho_raw"]["std"], d["wikitext_vs_wmdp_bio"]["rho_raw"]["std"]]

    x = range(len(comparisons))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.bar([i - w / 2 for i in x], rho_mean, width=w, yerr=rho_std, capsize=3,
           color=BLUE, label="ρ (activation percentage)")
    ax.bar([i + w / 2 for i in x], raw_mean, width=w, yerr=raw_std, capsize=3,
           color=ORANGE, label="ρ_raw (raw firing count)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(comparisons)
    ax.set_ylabel("Total variation distance")
    ax.set_ylim(0, 1.05)
    ax.set_title("ρ discriminates retain-vs-forget better than ρ_raw\n(paper Section 4.5 / Appendix K.1, Figure 12)")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper center", fontsize=9)
    savefig(fig, "fig5_rho_vs_rho_raw.png")


# ============================================================
# Figure 6 - inference latency overhead, ours vs paper (Appendix L, Table 15)
# ============================================================
def fig_latency():
    d = load("latency_benchmark/dsg_latency_results.json")
    seqs = sorted(int(k) for k in d.keys())
    ours = [d[str(s)]["overhead_pct"] for s in seqs]
    paper = {256: 7.0, 512: 2.5, 1024: 3.1}
    papers = [paper[s] for s in seqs]

    x = range(len(seqs))
    w = 0.32
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar([i - w / 2 for i in x], ours, width=w, color=BLUE, label="Our hardware (RTX 2000 Ada)")
    ax.bar([i + w / 2 for i in x], papers, width=w, color=ORANGE, label="Paper (A6000, Table 15)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(s) for s in seqs])
    ax.set_xlabel("Sequence length")
    ax.set_ylabel("Inference overhead vs. base model (%)")
    ax.set_title("DSG inference overhead stays small on both GPUs\n(paper Appendix L, Table 15)")
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    savefig(fig, "fig6_latency_overhead.png")


# ============================================================
# Figure 7 - headline comparison vs RMU / Farrell et al. (Table 1)
# ============================================================
def fig_main_comparison():
    methods = ["RMU", "Farrell et al.", "DSG (zero-shot)", "DSG (full)"]
    bio = [50.00, 59.22, 31.04, 29.64]
    cyber = [88.00, 52.73, 41.45, 26.74]
    colors = [INK_MUTED, INK_MUTED, YELLOW, BLUE]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True)
    for ax, vals, title in ((axes[0], bio, "WMDP-Bio"), (axes[1], cyber, "WMDP-Cyber")):
        bars = ax.bar(methods, vals, color=colors)
        ax.axhline(25, color=RED, linestyle="--", linewidth=1)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%", ha="center", fontsize=9, color=INK)
        ax.set_title(title)
        ax.set_ylim(0, 100)
        style_axes(ax)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=20, ha="right")
    axes[0].set_ylabel("Accuracy (%) — lower is better forgetting")
    axes[0].text(3.5, 27, "random guess (25%)", fontsize=8, color=RED, ha="right")
    fig.suptitle("Our reproduction reproduces DSG's headline result:\nbeats RMU and Farrell et al. on both domains (paper Table 1)", y=1.05, fontsize=13)
    savefig(fig, "fig7_main_comparison.png")


if __name__ == "__main__":
    fig_data_efficiency()
    fig_zeroshot()
    fig_clamp()
    fig_thresholds()
    fig_rho()
    fig_latency()
    fig_main_comparison()
