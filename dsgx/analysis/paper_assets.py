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
    atomic_write_text(out / "INDEX.md", "\n".join(idx) + "\n")
    print(f"wrote {len(T)} tables, {len(pdfs)} figures -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
