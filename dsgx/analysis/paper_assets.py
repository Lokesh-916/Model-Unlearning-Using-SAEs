"""Paper assets in one command: every final table as LaTeX (booktabs) and every figure as a PDF at
paper size (single column 3.4 in, double 6.9 in; fonts 8 pt; Type-42 fonts embedded).

    python -m dsgx.analysis.paper_assets                       # from $DSG_RESULTS (runs + summary)
    python -m dsgx.analysis.paper_assets --smoke --out DIR     # test on the smoke runs
    python -m dsgx.analysis.paper_assets --hardware gpuws --runs $P/dsg_results_cluster/runs --out $P/dsg_results_cluster/paper

Writes <out>/tables/*.tex (each a complete `table` float with caption and label; needs \\usepackage{booktabs})
and <out>/figures/*.pdf (the 12 report figures + the qualitative figures when their inputs exist), plus
<out>/paper_assets.tex that \\input{}s every table, and INDEX.md. Numbers: mean [95% CI], n; across-seed CIs
where seeds > 1; non-significant paired differences are marked with a dagger.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from dsgx import paths
from dsgx.analysis import catalog, claims, figures
from dsgx.analysis.collect import load_all
from dsgx.util import atomic_write_text

TEX_ESC = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}", "~": r"\textasciitilde{}",
           "^": r"\^{}", "\\": r"\textbackslash{}", "<": r"$<$", ">": r"$>$", "|": r"\textbar{}"}


def esc(s) -> str:
    return "".join(TEX_ESC.get(c, c) for c in str(s))


def num(m, lo=None, hi=None, n=None, d=3) -> str:
    if m is None or (isinstance(m, float) and np.isnan(m)):
        return "--"
    s = f"{m:.{d}f}"
    if lo is not None and hi is not None and not (isinstance(lo, float) and np.isnan(lo)):
        s += f" [{lo:.{d}f}, {hi:.{d}f}]"
    if n is not None and not (isinstance(n, float) and np.isnan(n)):
        s += f" {{\\scriptsize n={int(n)}}}"
    return s


def table(rows, header, caption, label, align=None, note=None) -> str:
    align = align or ("l" * len(header))
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{" + caption + "}", r"\label{" + label + "}",
         r"\resizebox{\linewidth}{!}{%", r"\begin{tabular}{" + align + "}", r"\toprule",
         " & ".join(header) + r" \\", r"\midrule"]
    L += [" & ".join(r) + r" \\" for r in rows] or [r"\multicolumn{" + str(len(header)) + r"}{c}{(no results)} \\"]
    L += [r"\bottomrule", r"\end{tabular}}"]
    if note:
        L.append(r"\par\smallskip{\footnotesize " + note + "}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def _seed(r, col):
    return num(r.get(col), r.get(f"{col}_lo"), r.get(f"{col}_hi"), r.get(f"{col}_n_items"))


def build_tables(st, paired, cl, comp, runs, hw) -> dict:
    T = {}
    hwn = f" Hardware: {esc(hw)}."
    a1 = st[st["exp"] == "A1-test"] if not st.empty else st
    T["baselines"] = table([[esc(r["condition"]) + (r" $^\ast$" if r["reference"] else ""), esc(r["case"]), _seed(r, "raw_forget"),
                             _seed(r, "dsg_subset_forget"), _seed(r, "raw_util"), _seed(r, "benign_fpr")] for _, r in a1.iterrows()],
                           ["Condition", "Case", "Forget (raw)", "Forget (DSG subset)", "Utility (full MMLU)", "Benign FPR"],
                           "Baselines on TEST (A1): mean across seeds with 95\\% CI." + hwn, "tab:baselines", "llcccc",
                           r"$^\ast$ unverified third-party RMU reference, not a main comparison.")
    T["claims"] = table([[esc(c["id"]), esc(c["claim"]), r"\textbf{" + esc(c["verdict"]) + "}", esc(c["evidence"][0] if c["evidence"] else "")[:120]]
                         for c in cl], ["ID", "Claim", "Verdict", "Key evidence"], "Claims and verdicts (fixed rules)." + hwn,
                        "tab:claims", "p{0.06\\linewidth}p{0.3\\linewidth}lp{0.45\\linewidth}")
    att = []
    for exp in ("B1", "B2", "B3", "B4", "B5"):
        recs = [r for r in claims.attack_success_records(runs, exp) if r["method_d"].get("name") == "dsg-faithful" and r["attack_success"]]
        if recs:
            b = max(recs, key=lambda r: r["attack_success"]["mean"])
            a = b["attack_success"]
            ctrl = b.get("base_acc_same_transform") or {}
            params = ", ".join(f"{k}={v}" for k, v in b["attack"].items() if k not in ("name", "_exp_id"))
            att.append([exp, esc(b["attack"].get("name")), esc(params)[:40], num(a["mean"], a["lo"], a["hi"], a["n"]),
                        num(b["acc_under_attack"]["mean"]), num(ctrl.get("mean"))])
    T["attacks"] = table(att, ["Exp.", "Attack", "Setting", "Attack success", "Acc. under attack", "Base, same transform"],
                         "Strongest attack per axis against DSG (TEST, gated items)." + hwn, "tab:attacks", "lllccc")
    for e in sorted(set(st["exp"])) if not st.empty else []:
        if e == "A1-test":
            continue
        g = st[st["exp"] == e]
        T[f"exp_{e}"] = table([[esc(r["condition"])[:60], esc(r["split"]), _seed(r, "raw_forget"), _seed(r, "raw_util"), _seed(r, "benign_fpr")]
                               for _, r in g.iterrows()], ["Condition", "Split", "Forget (raw)", "Utility", "Benign FPR"],
                              f"{esc(e)}: {esc(catalog.EXPERIMENTS.get(e, (e,))[0])}." + hwn, f"tab:exp-{e.lower()}", "llccc")
    pt = [p for p in paired if "forget" in p]
    rows = []
    for p in pt:
        b, m = p["forget"]["paired_bootstrap"], p["forget"]["mcnemar"]
        dag = "" if (b["p"] < 0.05 and m["p"] < 0.05) else r"$^\dagger$"
        rows.append([esc(p["exp"]), esc(p["condition"])[:50], esc(p["vs"]), f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]{dag}",
                     f"{b['p']:.3g}", f"{m['p']:.3g}", str(p["forget"]["n"])])
    T["paired_tests"] = table(rows, ["Exp.", "Condition", "vs", "$\\Delta$ forget acc.", "boot. $p$", "McNemar $p$", "$n$"],
                              "Paired comparisons on identical TEST forget items." + hwn, "tab:paired", "lllcccr",
                              r"$^\dagger$ not significant (bootstrap or McNemar $p \ge 0.05$).")
    T["completeness"] = table([[esc(k), esc(v["name"]), esc(v["priority"]), esc(v["status"])] for k, v in comp.items()],
                              ["ID", "Experiment", "Priority", "Status"], "Experiment completeness." + hwn, "tab:completeness", "llll")
    sel = paths.runs_dir() / "X1-screen" / "select" / "COMBINE_SELECTION.json"
    if sel.exists():
        s = json.loads(sel.read_text())
        T["combine_selection"] = table([[esc(r["slot"]), esc(r["name"]), num(r.get("diff")), num(r.get("mcnemar_p"), d=4),
                                         num(r.get("utility_cost")), num(r.get("benign_fpr")), "yes" if r["eligible"] else "no"]
                                        for r in s["candidates"]],
                                       ["Slot", "Candidate", "$\\Delta$ target", "McNemar $p$", "Util. cost", "Benign FPR", "Eligible"],
                                       "DEV selection of the combined method (X1; rule fixed in code before screening).", "tab:combine", "llccccc")
    return T


# ----------------------------------------------------------------------------- DSG figure parity (session 8)
# DSG paper figure/table type -> (our figure stem, our table key, data source). INDEX.md lists each with its status,
# so it is visible what still lacks data (never a re-run just to plot: every number is logged by the jobs).
PARITY = [
    ("gate-score distributions (forget / retain / attacked)", "gate_score_distributions", None, "TEST gated runs + attacks (lab B suite; gpuws FP-static)"),
    ("forget vs utility scatter", "forget_utility_test", None, "every clean TEST run with full-MMLU utility"),
    ("forget vs utility, dev sweep (Pareto)", "pareto_forget_utility", "sweep_a1_dev", "A1-dev"),
    ("relearning curves across epochs", "relearning_epochs", None, "A6 (LoRA), A6-full, A6-full-d1v2"),
    ("clamp strength x feature count", "clamp_strength_grid", "sweep_clamp", "FP-clamp (gpuws, DEV)"),
    ("static vs dynamic clamping", "static_vs_dynamic", "static_dynamic", "FP-static (gpuws, TEST)"),
    ("data efficiency (feature-selection corpus size)", "data_efficiency", "data_efficiency", "FP-dataeff (gpuws, DEV)"),
    ("multi-topic unlearning", "multitopic", "multitopic", "FP-multitopic (gpuws, TEST)"),
    ("latency (batch size 1, by sequence length)", "latency_by_length", "latency", "FP-latency (gpuws)"),
    ("feature-activation highlights (TOFU only)", "tofu_feature_highlight", None, "FP-highlight (gpuws, TOFU)"),
    ("hyperparameter sweep tables", None, "sweep_a1_dev, sweep_clamp, sweep_rmu_v2, sweep_d1_v2", "A1-dev, FP-clamp, RMU-v2-dev, D1-v2"),
]


def _ci_cell(d):
    return num((d or {}).get("mean"), (d or {}).get("lo"), (d or {}).get("hi")) if d else "--"


def _pct(x):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:+.1f}\\%"


def build_parity_tables(runs, hw) -> dict:
    from dsgx.analysis.figures import per_dataset_acc

    T, hwn = {}, f" Hardware: {esc(hw)}."
    fp = lambda e: [r for r in runs if r.base_exp == e and r.is_mcq]  # noqa: E731
    pp = lambda r: r.cfg.get("fp_params") or {}  # noqa: E731
    # A1 dev sweep (DSG hyperparameters)
    a1 = sorted([r for r in runs if r.base_exp == "A1-dev" and r.is_mcq and not r.is_base],
                key=lambda r: (r.case or "", str(r.cfg["method"].get("retain_corpus")), r.cfg["method"].get("n_features", 0),
                               r.cfg["method"].get("retain_pct", 0), r.cfg["method"].get("multiplier", 0)))
    if a1:
        T["sweep_a1_dev"] = table([[esc(r.case), esc(r.cfg["method"].get("retain_corpus", "")), str(r.cfg["method"].get("n_features")),
                                    str(r.cfg["method"].get("retain_pct")), str(r.cfg["method"].get("multiplier")),
                                    _ci_cell(r.forget("raw")), _ci_cell(r.utility("raw"))] for r in a1],
                                  ["Case", "Retain corpus", "$N$", "Retain pct.", "$c$", "Forget (DEV)", "Utility (DEV)"],
                                  "DSG hyperparameter sweep on DEV (A1)." + hwn, "tab:sweep-a1", "llrrrcc")
    cl = sorted([r for r in fp("FP-clamp") if pp(r)], key=lambda r: (pp(r)["method"], pp(r)["n"], pp(r)["c"]))
    if cl:
        T["sweep_clamp"] = table([[{"dsg": "DSG", "ours": "our gate"}.get(pp(r)["method"], esc(pp(r)["method"])), str(pp(r)["n"]), str(pp(r)["c"]),
                                   _ci_cell(r.forget("raw")), _ci_cell(r.utility("raw")), _ci_cell(r.utility("dsg_subset"))] for r in cl],
                                 ["Method", "$N$", "$c$", "Forget (DEV)", "MMLU-4 (DEV, raw)", "MMLU-4 (DEV, DSG subset)"],
                                 "Clamp strength $c$ $\\times$ number of features $N$ (DEV; WMDP-Bio and the 4 legacy MMLU subjects)." + hwn,
                                 "tab:sweep-clamp", "lrrccc")
    de = sorted([r for r in fp("FP-dataeff") if pp(r)], key=lambda r: (pp(r)["method"], pp(r)["m"], pp(r)["seed"]))
    if de:
        T["data_efficiency"] = table([[{"dsg": "DSG", "ours": "our gate"}.get(pp(r)["method"], esc(pp(r)["method"])), str(pp(r)["m"]),
                                       str(pp(r)["seed"]), str(pp(r).get("n_features", "")), _ci_cell(r.forget("raw")), _ci_cell(r.utility("raw"))]
                                      for r in de], ["Method", "Rows / side", "Seed", "Features", "Forget (DEV)", "MMLU-4 (DEV)"],
                                     "Data efficiency: feature-selection corpus size (DEV)." + hwn, "tab:data-efficiency", "lrrrcc")
    st = {r.cfg.get("dataset_label"): r for r in fp("FP-static")}
    if st:
        T["static_dynamic"] = table([[esc(k), _ci_cell(st[k].forget("raw")), _ci_cell(st[k].utility("raw")),
                                      _ci_cell((st[k].gate or {}).get("benign_fpr"))] for k in sorted(st)],
                                    ["Condition", "Forget (TEST)", "Utility (TEST, full MMLU)", "Benign fire rate"],
                                    "Static (always clamp) vs dynamic (gated) clamping." + hwn, "tab:static-dynamic", "lccc")
    mt = {r.cfg.get("dataset_label"): r for r in fp("FP-multitopic")}
    if mt:
        T["multitopic"] = table([[esc(k), num(per_dataset_acc(mt[k], "wmdp-bio")), num(per_dataset_acc(mt[k], "wmdp-cyber")),
                                  _ci_cell(mt[k].utility("raw"))] for k in sorted(mt)],
                                ["Condition", "WMDP-Bio", "WMDP-Cyber", "MMLU utility"],
                                "Multi-topic unlearning: bio and cyber features at once (TEST, raw)." + hwn, "tab:multitopic", "lccc")
    lat = [r for r in runs if r.base_exp == "FP-latency" and r.metrics.get("latency")]
    if lat:
        L = lat[0].metrics["latency"]
        names = list(L[sorted(L, key=int)[0]])
        T["latency"] = table([[str(k)] + [f"{L[k][n]['ms_median']:.1f}" + ("" if n == "base" else f" ({_pct(L[k][n]['overhead_vs_base'])})")
                                          for n in names] for k in sorted(L, key=int)],
                             ["Tokens"] + [esc(n) for n in names],
                             "Forward latency at batch size 1, median ms (overhead vs base)." + f" Hardware: {esc(lat[0].hardware)}.",
                             "tab:latency", "r" + "r" * len(names))
    rmu = [r for r in runs if r.base_exp == "RMU-v2-dev" and r.is_mcq]
    if rmu:
        T["sweep_rmu_v2"] = table([[esc(r.cfg.get("dataset_label") or r.name)[:40], _ci_cell(r.forget("raw")), _ci_cell(r.utility("raw"))]
                                   for r in sorted(rmu, key=lambda r: r.name)], ["Config", "Forget (DEV)", "Utility (DEV)"],
                                  "RMU v2 DEV grid." + hwn, "tab:sweep-rmu", "lcc")
    d1 = [r for r in runs if r.base_exp == "D1-v2" and r.is_mcq and r.split == "dev"]
    if d1:
        T["sweep_d1_v2"] = table([[esc(r.cfg.get("dataset_label")), _ci_cell(r.forget("raw")), _ci_cell(r.utility("raw"))]
                                  for r in sorted(d1, key=lambda r: r.cfg.get("dataset_label", ""))],
                                 ["Student", "Forget (DEV)", "Utility (DEV)"], "D1 v2 noise sweep on DEV (selection input)." + hwn,
                                 "tab:sweep-d1v2", "lcc")
    return T


def parity_index(T, pdfs) -> list[str]:
    L = ["", "## DSG figure parity", "", "| DSG figure type | our figure | our table | data | status |", "|---|---|---|---|---|"]
    for name, fig, tab, src in PARITY:
        have_f = fig is None or f"{fig}.pdf" in pdfs
        have_t = tab is None or all(t.strip() in T for t in tab.split(","))
        L.append(f"| {name} | {fig or '-'} | {tab or '-'} | {src} | {'ready' if have_f and have_t else 'waiting for data'} |")
    return L


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="LaTeX tables + paper-size PDF figures")
    ap.add_argument("--runs")
    ap.add_argument("--out")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--hardware", default="auto")
    ap.add_argument("--n-boot", type=int, default=10_000)
    a = ap.parse_args(argv)
    from dsgx.analysis.final_report import compute_summary

    out = Path(a.out) if a.out else paths.results_dir() / "paper"
    runs = load_all(Path(a.runs) if a.runs else None, smoke=a.smoke)
    labels = Counter(r.hardware for r in runs)
    hw = a.hardware
    if hw == "auto":
        if len(labels) > 1:
            print(f"several hardware labels {dict(labels)}; choose --hardware", file=sys.stderr)
            return 2
        hw = next(iter(labels), "labpc")
    runs = [r for r in runs if r.hardware == hw]
    comp, df, st, paired, cl = compute_summary(runs, {}, {}, a.n_boot)
    T = build_tables(st, paired, cl, comp, runs, hw)
    T.update(build_parity_tables(runs, hw))
    td = out / "tables"
    td.mkdir(parents=True, exist_ok=True)
    for k, v in T.items():
        atomic_write_text(td / f"{k}.tex", v)
    atomic_write_text(out / "paper_assets.tex", "% \\usepackage{booktabs,graphicx}\n" + "".join(f"\\input{{tables/{k}}}\n" for k in T))
    figures.PAPER = True
    try:
        figs = figures.make_all(runs, out / "figures", paper=True)
        from dsgx.analysis.qual import q5_geometry, q6_trajectory

        for mod in (q5_geometry, q6_trajectory):
            try:
                mod.main((["--smoke"] if a.smoke else []) + ["--out", str(out / "figures")])
            except Exception as e:  # qualitative inputs may not exist yet
                figs[mod.__name__] = {"error": str(e)}
    finally:
        figures.PAPER = False
    for junk in list((out / "figures").glob("*.json")) + list((out / "figures").glob("Q*.md")):
        junk.unlink()  # keep only PDFs in the paper figure folder
    pdfs = sorted(p.name for p in (out / "figures").glob("*.pdf"))
    idx = ["# Paper assets", "", f"Hardware: {hw}. Tables (booktabs): " + ", ".join(f"`tables/{k}.tex`" for k in T), "",
           "Figures (PDF, paper size): " + ", ".join(f"`figures/{p}`" for p in pdfs), "",
           "Skipped figures: " + "; ".join(f"{k}: {v.get('skipped') or v.get('error')}" for k, v in figs.items() if "written" not in v)]
    idx += parity_index(T, set(pdfs))
    atomic_write_text(out / "INDEX.md", "\n".join(idx) + "\n")
    print(f"wrote {len(T)} tables, {len(pdfs)} figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
