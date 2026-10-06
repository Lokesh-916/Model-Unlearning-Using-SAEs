"""P4 as one command: completeness, aggregation, CIs and paired tests, all figures, summary.json,
FINAL_REPORT.md (MASTER_PLAN section 8 order) with the claims table, and the N10 audit cards.

    python -m dsgx.analysis.final_report                     # full results in $DSG_RESULTS
    python -m dsgx.analysis.final_report --interim           # allowed before everything is done
    python -m dsgx.analysis.final_report --smoke --out DIR   # test on the smoke runs
    python -m dsgx.analysis.final_report --hardware gpuws --runs $P/dsg_results_cluster/runs --out $P/dsg_results_cluster/report

Outputs (in --out, default $DSG_RESULTS): FINAL_REPORT.md, summary.json, tables/*.csv, figures/*.{png,pdf},
audit_cards/*.{md,json}. Read-only on runs; never reads $DSG_PRIVATE; prints ids and metrics only.
Per-GPU baselines: runs of one hardware label only (lab PC and gpuws are never mixed).
"""
import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from dsgx import paths
from dsgx.analysis import aggregate, catalog, claims, figures
from dsgx.analysis.collect import fmt_ci, load_all
from dsgx.util import atomic_write_json, atomic_write_text, now_iso

PARTS = {"I (Break)": ["B1", "B2", "B3", "B4", "B5", "B6", "A2", "A3"],
         "II (Explain)": ["A4", "N9", "C1", "C3", "T", "D3"],
         "III (Fix)": ["C2", "C4", "C5", "C6", "N5", "N6", "N7", "N8", "D1", "D2", "A6", "X1"]}


def _md_table(df: pd.DataFrame, cols, fmt=None) -> str:
    if df is None or df.empty:
        return "_no finished runs_\n"
    fmt = fmt or {}
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = fmt[c](r) if c in fmt else r.get(c)
            cells.append("" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _ci_cell(prefix, nkey="n"):
    def f(r):
        m = r.get(prefix)
        if m is None or (isinstance(m, float) and np.isnan(m)):
            return "n/a"
        lo, hi = r.get(f"{prefix}_lo"), r.get(f"{prefix}_hi")
        n = r.get(f"{prefix}_{nkey}")
        s = f"{m:.3f}"
        if lo is not None and hi is not None and not (isinstance(lo, float) and np.isnan(lo)):
            s += f" [{lo:.3f}, {hi:.3f}]"
        if n is not None and not (isinstance(n, float) and np.isnan(n)):
            s += f" n={int(n)}"
        return s
    return f


def _seed_ci(prefix):
    def f(r):
        m = r.get(prefix)
        if m is None or (isinstance(m, float) and np.isnan(m)):
            return "n/a"
        lo, hi, k, n = r.get(f"{prefix}_lo"), r.get(f"{prefix}_hi"), r.get(f"{prefix}_n_seeds"), r.get(f"{prefix}_n_items")
        s = f"{m:.3f}"
        if lo is not None and not (isinstance(lo, float) and np.isnan(lo)):
            s += f" ± [{lo:.3f}, {hi:.3f}]"
        s += f" ({int(k)} seed{'s' if k != 1 else ''}"
        s += f", n={int(n)})" if n is not None and not (isinstance(n, float) and np.isnan(n)) else ")"
        return s
    return f


def anomalies(df: pd.DataFrame) -> list[str]:
    out = []
    if df.empty:
        return out
    for _, r in df.iterrows():
        n = r.get("raw_forget_n")
        if n and n >= 10 and r.get("raw_forget_lo") is not None and r["raw_forget_lo"] == r["raw_forget_hi"]:
            out.append(f"zero-width forget CI with n={int(n)}: {r['run']}")
        if r.get("method") == "base" and not r.get("weights") and r.get("raw_forget") is not None \
                and r["raw_forget"] < 0.25 and (r.get("raw_forget_n") or 0) >= 50:
            out.append(f"base model below chance on forget set ({r['raw_forget']:.3f}): {r['run']}")
        if r.get("benign_fpr") is not None and r["benign_fpr"] > 0.10 and r.get("split") == "test":
            out.append(f"benign FPR {r['benign_fpr']:.3f} > 0.10 (target 0.05): {r['run']}")
    test = df[(df["split"] == "test") & (df["attack"] == "none")]
    for (exp, case), g in test.groupby(["exp", "case"]):
        b = g[(g["method"] == "base") & (g["weights"] == "")]
        if b.empty or b["raw_util"].isna().all():
            continue
        bu = b["raw_util"].iloc[0]
        for _, r in g[g["method"] != "base"].iterrows():
            if r.get("raw_util") is not None and bu - r["raw_util"] > 0.02:
                out.append(f"utility drop {bu - r['raw_util']:.3f} > 2 points vs base ({exp}/{case}): {r['run']}")
    return out


def compute_summary(runs, jobs, states, n_boot):
    comp = catalog.completeness(runs, jobs, states)
    df = aggregate.run_table(runs)
    st = aggregate.seed_table(df)
    paired = aggregate.paired_tests(runs, n_boot=n_boot)
    cl = claims.evaluate(runs, paired)
    return comp, df, st, paired, cl


def methods_for_cards(runs, st: pd.DataFrame, cl) -> dict:
    """Card data for DSG and the best fixes (summary.json 'methods' block, read by N10 too)."""
    out = {}

    def pick(exp, method_pred):
        if st.empty:
            return None
        g = st[(st["exp"] == exp) & (st["split"] == "test") & (st["attack"] == "none") & (st["reference"] == "")]
        g = g[g.apply(method_pred, axis=1)]
        return g.iloc[0] if len(g) else None

    def ci_of(row, prefix):
        if row is None or row.get(prefix) is None or (isinstance(row.get(prefix), float) and np.isnan(row[prefix])):
            return None
        return {"mean": row[prefix], "lo": row.get(f"{prefix}_lo"), "hi": row.get(f"{prefix}_hi"),
                "n": row.get(f"{prefix}_n_items"), "n_seeds": row.get(f"{prefix}_n_seeds")}

    as_b1 = claims.attack_success_records(runs, "B1")
    as_b3 = claims.attack_success_records(runs, "B3")

    def max_as(recs):
        recs = [r for r in recs if r["method_d"].get("name") == "dsg-faithful" and r["attack_success"]]
        return max((r["attack_success"] for r in recs), key=lambda a: a["mean"], default=None)

    probe = [r for r in runs if r.base_exp == "A4" and r.name == "probe__dsg"]
    pb = None
    if probe:
        lay = probe[0].metrics.get("layers") or {}
        if lay:
            L = max(lay, key=lambda k: lay[k]["probe"]["mean"])
            pb = {**lay[L]["probe"], "layer": int(L)}
    lat = [r for r in runs if r.base_exp == "A8" and r.name == "latency"]
    lat_dsg = ((lat[0].metrics.get("forward") or {}).get("dsg-faithful") or {}).get("overhead_vs_base") if lat else None
    ch6 = next((c for c in cl if c["id"] == "C-H6"), {})
    dsg = pick("A1-test", lambda r: r["method"] == "dsg-faithful" and r["case"] == "bio")
    out["dsg-faithful"] = {"model": "gemma-2-2b-it / gemma-scope-2b-pt-res L3 (16k, l0 142)",
                           "forget": ci_of(dsg, "dsg_subset_forget") or ci_of(dsg, "raw_forget"),
                           "utility": ci_of(dsg, "raw_util"), "benign_fpr": ci_of(dsg, "benign_fpr"),
                           "hazard_fnr": None, "attack_dilution": max_as(as_b1), "attack_crosslingual": max_as(as_b3),
                           "probe_best": pb, "relearn_verdict": "baseline (see C-H6)",
                           "latency_overhead": f"{lat_dsg:+.1%}" if isinstance(lat_dsg, float) else "n/a",
                           "runs": [] if dsg is None else [f"A1-test/{dsg['condition']}"]}
    x1 = pick("X1", lambda r: str(r["condition"]).startswith(("gated", "composite")) and "loo" not in str(r["condition"]))
    if x1 is not None:
        out["combined-gate (X1)"] = {"model": "gemma-2-2b-it", "forget": ci_of(x1, "dsg_subset_forget") or ci_of(x1, "raw_forget"),
                                     "utility": ci_of(x1, "raw_util"), "benign_fpr": ci_of(x1, "benign_fpr"),
                                     "relearn_verdict": "n/a (inference-time gate)", "runs": [f"X1/{x1['condition']}"]}
    for exp, tag in (("D1", "undo"), ("D2", "nullspace")):
        row = pick(exp, lambda r, t=tag: t in str(r["condition"]))
        if row is not None:
            out[f"{exp}-{tag}"] = {"model": "gemma-2-2b-it (baked)", "forget": ci_of(row, "dsg_subset_forget") or ci_of(row, "raw_forget"),
                                   "utility": ci_of(row, "raw_util"), "relearn_verdict": f"C-H6: {ch6.get('verdict', 'n/a')}",
                                   "runs": [f"{exp}/{row['condition']}"]}
    return out


CARD = """# Audit Card: {name}

| Field | Value |
|---|---|
| Method | {name} |
| Model / SAE | {model} |
| Hardware baseline | {hardware} |
| Forget accuracy (TEST; DSG-subset view if available) | {forget} |
| Utility (full MMLU, pooled, raw) | {utility} |
| Benign FPR (target 5%) | {fpr} |
| Attack success, dilution (max over B1) | {dilution} |
| Attack success, cross-lingual / encoded (max over B3) | {crosslingual} |
| Knowledge retained internally (best-layer probe acc) | {probe} |
| Resists LoRA relearning | {relearn} |
| Latency overhead / token | {latency} |
| Source runs | {runs} |

_Generated {date} from summary.json. Numbers: mean [95% CI] (n); across-seed CIs where seeds > 1. Not published._
"""


def write_cards(methods: dict, outdir: Path, hardware: str) -> list[str]:
    outdir.mkdir(parents=True, exist_ok=True)
    out = []
    for name, m in methods.items():
        card = {"name": name, "model": m.get("model", "n/a"), "hardware": hardware,
                "forget": fmt_ci(m.get("forget")), "utility": fmt_ci(m.get("utility")), "fpr": fmt_ci(m.get("benign_fpr")),
                "dilution": fmt_ci(m.get("attack_dilution")), "crosslingual": fmt_ci(m.get("attack_crosslingual")),
                "probe": fmt_ci(m.get("probe_best")) + (f" @ layer {m['probe_best']['layer']}" if m.get("probe_best") else ""),
                "relearn": m.get("relearn_verdict", "n/a"), "latency": m.get("latency_overhead", "n/a"),
                "runs": ", ".join(m.get("runs") or ["(pending)"]), "date": now_iso()}
        safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "-")
        atomic_write_text(outdir / f"{safe}.md", CARD.format(**card))
        atomic_write_json(outdir / f"{safe}.json", {**card, "raw": m})
        out.append(str(outdir / f"{safe}.md"))
    return out


def render(comp, df, st, paired, cl, figs, cards, meta) -> str:
    L = [f"# FINAL REPORT — DSG unlearning capstone{' (INTERIM)' if meta['interim'] else ''}", "",
         f"Generated {meta['time']} by `python -m dsgx.analysis.final_report`. Hardware baseline: **{meta['hardware']}** "
         f"(results of other hardware are reported separately and never mixed). Runs: {meta['n_runs']} finished "
         f"({meta['n_mcq']} MCQ). Source: `{meta['runs_root']}`.", ""]
    v = Counter(c["verdict"] for c in cl)
    status = Counter(x["status"].split(" ")[0] for x in comp.values())
    L += ["## 1. Executive summary", ""]
    ex = [f"- Experiments: " + ", ".join(f"{k} {n}" for k, n in sorted(status.items())) + ".",
          f"- Claims: {v.get('Supported', 0)} supported, {v.get('Not supported', 0)} not supported, "
          f"{v.get('Inconclusive', 0)} inconclusive (rules in `dsgx/analysis/claims.py`)."]
    for c in cl:
        ex.append(f"- {c['id']} {c['claim']} **{c['verdict']}** — {c['evidence'][0] if c['evidence'] else ''}")
    L += ex[:10] + [""]
    L += ["## 2. Setup used and deviations", "",
          f"- Model gemma-2-2b-it, Gemma Scope 16k layer-3 SAE (DSG default); TEST numbers bs=1; 95% bootstrap CIs over "
          f"items ({meta['n_boot']} resamples), paired bootstrap + exact McNemar for comparisons; 3–5 seeds.",
          f"- Package versions: {json.dumps(meta.get('versions', {}))}",
          f"- Deviations: {meta['n_deviations']} rows in `DEVIATIONS.md` (all departures from MASTER_PLAN are listed there).", ""]
    L += ["## 3. Baseline reproduction", ""]
    if meta.get("sanity"):
        s = meta["sanity"]
        L.append(f"Sanity gate (Bio N=20, seed 0, DSG-subset view): WMDP-Bio {s.get('wmdp', {}).get('correct')}/"
                 f"{s.get('wmdp', {}).get('n')} = {s.get('wmdp', {}).get('acc', float('nan')):.4f} (paper/legacy 0.2937), "
                 f"MMLU-u {s.get('mmlu_u', float('nan')):.4f} (0.9941), tau {s.get('tau', float('nan')):.4f} (0.5458); pass={s.get('pass')}.")
        L.append("")
    a1 = st[(st["exp"] == "A1-test")] if not st.empty else st
    L.append(_md_table(a1, ["condition", "case", "reference", "raw_forget", "raw_util", "dsg_subset_forget", "dsg_subset_util", "benign_fpr"],
                       {c: _seed_ci(c) for c in ("raw_forget", "raw_util", "dsg_subset_forget", "dsg_subset_util", "benign_fpr")}))
    L += ["Rows with a `reference` label (third-party RMU) are an **unverified reference**, never a main comparison.", ""]
    L += ["## 4. Claims", "", "| claim | verdict | criterion | evidence |", "|---|---|---|---|"]
    for c in cl:
        L.append(f"| {c['id']} {c['claim']} | **{c['verdict']}** | {c['criterion']} | {'<br>'.join(c['evidence'][:6])} |")
    L.append("")
    sec = 5
    for part, exps in PARTS.items():
        L += [f"## {sec}. Part {part} results", ""]
        for e in exps:
            g = st[st["exp"] == e] if not st.empty else st
            tasks = [r for r in meta["task_runs"] if r[0] == e]
            if (g is None or g.empty) and not tasks:
                continue
            L += [f"### {e} — {catalog.EXPERIMENTS.get(e, ('',))[0]} ({comp.get(e, {}).get('status', '?')})", ""]
            if g is not None and not g.empty:
                L.append(_md_table(g, ["condition", "case", "split", "raw_forget", "raw_util", "dsg_subset_forget", "benign_fpr"],
                                   {c: _seed_ci(c) for c in ("raw_forget", "raw_util", "dsg_subset_forget", "benign_fpr")}))
            pt = [p for p in paired if p["exp"] == e and "forget" in p]
            if pt:
                L += ["Paired tests (forget items, TEST; diff = condition − reference):", "",
                      "| condition | vs | diff [95% CI] | bootstrap p | McNemar p | n |", "|---|---|---|---|---|---|"]
                for p in pt[:40]:
                    b, m = p["forget"]["paired_bootstrap"], p["forget"]["mcnemar"]
                    sig = "" if (b["p"] < 0.05 and m["p"] < 0.05) else " (n.s.)"
                    L.append(f"| {p['condition']} | {p['vs']} | {b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]{sig} | "
                             f"{b['p']:.4f} | {m['p']:.4f} | {p['forget']['n']} |")
                L.append("")
            for _, name, head in tasks[:12]:
                L.append(f"- task `{name}`: {head}")
            L.append("")
        sec += 1
    L += [f"## 8. Generality", ""]
    for e in ("A7", "A2"):
        g = st[st["exp"] == e] if not st.empty else st
        if g is not None and not g.empty:
            L.append(_md_table(g, ["condition", "model", "attack", "raw_forget", "raw_util"],
                               {c: _seed_ci(c) for c in ("raw_forget", "raw_util")}))
    c7 = next(c for c in cl if c["id"] == "C-H7")
    L += [f"C-H7: **{c7['verdict']}** — " + "; ".join(c7["evidence"]), ""]
    L += ["## 9. Surprises and anomalies", ""]
    an = meta["anomalies"]
    L += [f"- {a}" for a in an[:40]] or ["- none flagged automatically"]
    if meta.get("anomalies_md"):
        L.append(f"- See also `ANOMALIES.md` ({meta['anomalies_md']} lines).")
    L += ["", "## 10. Failed, skipped and partial items", "", "| id | name | priority | status | jobs | failed jobs |", "|---|---|---|---|---|---|"]
    for k, x in comp.items():
        if x["status"] != "done":
            L.append(f"| {k} | {x['name']} | {x['priority']} | {x['status']} | {dict(x['jobs']) or ''} | {' '.join(x['failed_jobs'][:4])} |")
    L += ["", "## 11. Compute used", ""]
    if not df.empty:
        cu = df.groupby("exp").agg(runs=("run", "count"), wall_h=("wall_s", lambda s: round(np.nansum(s.astype(float)) / 3600, 2)),
                                   peak_vram_gb=("peak_vram_gb", "max")).reset_index()
        L.append(_md_table(cu, ["exp", "runs", "wall_h", "peak_vram_gb"]))
        L.append(f"Total MCQ wall time: {cu['wall_h'].sum():.1f} h (task jobs not included; see queue PROGRESS.md).")
    L += ["", "## 12. Next experiments", ""]
    for c in cl:
        if c["verdict"] == "Inconclusive":
            miss = [e for e in c["evidence"] if e.startswith(("missing", "only", "A7", "A2"))]
            L.append(f"- {c['id']}: " + ("; ".join(miss) if miss else "collect the evidence named in the criterion"))
    L += ["", "## 13. Index of figures and run directories", ""]
    for k, f in figs.items():
        L.append(f"- {k}: " + (", ".join(f"`figures/{s}.png`" for s in f["written"]) if "written" in f
                               else f"skipped ({f.get('skipped') or f.get('error')})"))
    L += [f"- audit cards: {len(cards)} in `audit_cards/`", f"- per-run table: `tables/runs.csv`; per-condition: "
          f"`tables/conditions.csv`; paired tests: `tables/paired_tests.csv`; run directories: `{meta['runs_root']}/<exp>/`", ""]
    return "\n".join(L)


def with_run_cis(st: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Single-seed conditions get the run's own item-level bootstrap CI (the seed table has none), exactly as
    the results digest reports them; multi-seed rows keep the across-seed CI."""
    cols = ("raw_forget", "raw_util", "dsg_subset_forget", "dsg_subset_util", "benign_fpr")
    out = st.copy()
    for i, r in st.iterrows():
        if len(r.get("seeds") or []) > 1:
            continue
        m = df[(df["exp"] == r["exp"]) & (df["condition"] == r["condition"]) & (df["split"] == r["split"]) &
               (df["case"] == r["case"]) & (df["attack_params"] == r["attack_params"]) & (df["hardware"] == r["hardware"])]
        if len(m):
            for c in cols:
                for suf in ("_lo", "_hi"):
                    if c + suf in m.columns and c + suf in out.columns:
                        out.at[i, c + suf] = m.iloc[0][c + suf]
    return out


def paper_extras(runs, root: Path) -> dict:
    """Numbers the paper quotes that are not MCQ condition rows (read by dsgx.analysis.paper_numbers):
    strongest DSG attack per axis (attack-success tasks), A4/D3 probe best layers, A2 TOFU conditions,
    A6 relearning ranges. Descriptive only; verdicts stay with claims.py."""
    from dsgx.analysis import results_digest as rd

    ex = {"attack_success_max": {}, "probes": {}, "tofu": {}, "a6": {}}
    for exp in ("B1", "B2", "B3", "B4", "B5"):
        recs = [r for r in claims.attack_success_records(runs, exp) if claims._is_dsg_rec(r) and r.get("attack_success")
                and r["attack"].get("name") != "none" and (exp != "B1" or int(r["attack"].get("pad", 0) or 0) > 0)]
        if recs:
            b = max(recs, key=lambda r: r["attack_success"].get("mean") or -1)
            ex["attack_success_max"][exp] = {**b["attack_success"], "n_conditions": len(recs),
                                             "attack": {k: v for k, v in b["attack"].items() if k != "_exp_id"}}
    for r in runs:
        if r.base_exp in ("A4", "D3") and r.name.startswith("probe__") and r.metrics.get("layers"):
            lay = {int(k): v for k, v in r.metrics["layers"].items() if isinstance(v, dict) and isinstance(v.get("probe"), dict)}
            if lay:
                bl = max(lay, key=lambda k: lay[k]["probe"]["mean"])
                ex["probes"][f"{r.base_exp}/{r.name.split('__', 1)[1]}"] = {"best_layer": bl, "probe": lay[bl]["probe"],
                                                                           "control": lay[bl].get("control")}
        if r.base_exp == "A2" and r.name == "tofu-metrics":
            for cond, c in (r.metrics.get("conditions") or {}).items():
                if isinstance(c, dict):
                    ex["tofu"][cond] = {k: c.get(k) for k in ("model_utility", "forget_quality_ks_p")}
    if root.exists():
        for (exp, cond), xs in rd.a6_section(root)[1].items():
            d = [after - bef for bef, after, _, _ in xs]
            ex["a6"][f"{exp}/{cond}"] = {"before": [min(x[0] for x in xs), max(x[0] for x in xs)],
                                         "after": [min(x[1] for x in xs), max(x[1] for x in xs)],
                                         "delta": [min(d), max(d)], "n_cells": len(xs),
                                         "k": sorted({x[2] for x in xs if x[2] is not None})}
    return ex


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="P4 final report in one command")
    ap.add_argument("--runs", help="runs root (default $DSG_RESULTS/runs)")
    ap.add_argument("--out", help="output dir (default $DSG_RESULTS)")
    ap.add_argument("--smoke", action="store_true", help="use the *-smoke experiments")
    ap.add_argument("--interim", action="store_true", help="allow an incomplete queue")
    ap.add_argument("--hardware", default="auto", help="labpc | gpuws | auto (only one label per report)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--no-figures", action="store_true")
    a = ap.parse_args(argv)
    t0 = time.time()
    root = Path(a.runs) if a.runs else paths.runs_dir()
    out = Path(a.out) if a.out else paths.results_dir()
    runs = load_all(root, smoke=a.smoke)
    labels = Counter(r.hardware for r in runs)
    hw = a.hardware
    if hw == "auto":
        if len(labels) > 1:
            print(f"runs carry several hardware labels {dict(labels)}; choose one with --hardware", file=sys.stderr)
            return 2
        hw = next(iter(labels), "labpc")
    runs = [r for r in runs if r.hardware == hw]
    from dsgx.queue import common as q

    jobs = {j: v for j, v in q.load_jobs().items() if bool(v.get("smoke")) == a.smoke} if not a.runs else {}
    states = {j: q.load_state(j) for j in jobs}
    comp, df, st, paired, cl = compute_summary(runs, jobs, states, a.n_boot)
    status = Counter(x["status"].split(" ")[0] for x in comp.values())
    print("completeness: " + " / ".join(f"{k} {n}" for k, n in sorted(status.items())))
    unfinished = [k for k, x in comp.items() if x["status"] in ("partial", "not-run") and x["priority"] != "deferred"]
    interim = a.interim or a.smoke or bool(unfinished)
    if unfinished and not a.interim and not a.smoke:
        print(f"note: {len(unfinished)} experiment(s) unfinished ({' '.join(unfinished[:10])}); writing an INTERIM report")
    tables = out / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    df.to_csv(tables / "runs.csv", index=False)
    st.to_csv(tables / "conditions.csv", index=False)
    pd.json_normalize(paired).to_csv(tables / "paired_tests.csv", index=False)
    figs = {} if a.no_figures else figures.make_all(runs, out / "figures")
    methods = methods_for_cards(runs, st, cl)
    cards = write_cards(methods, out / "audit_cards", hw)
    sanity = None
    sp = paths.results_dir() / "sanity" / "latest.json"
    if sp.exists() and hw == "labpc":
        sanity = json.loads(sp.read_text())
    devp, anp = out / "DEVIATIONS.md", out / "ANOMALIES.md"
    if not devp.exists():
        devp = paths.results_dir() / "DEVIATIONS.md"
    versions = next((r.config.get("versions") for r in runs if r.config.get("versions")), {})
    task_runs = [(r.base_exp, r.name, json.dumps(json.loads((r.dir / "DONE").read_text()).get("headline"), default=str)[:160])
                 for r in runs if not r.is_mcq and (r.dir / "DONE").exists()]
    meta = {"time": now_iso(), "hardware": hw, "interim": interim, "n_runs": len(runs),
            "n_mcq": sum(r.is_mcq for r in runs), "runs_root": str(root), "n_boot": a.n_boot, "versions": versions,
            "n_deviations": sum(1 for l in devp.read_text().splitlines() if l.startswith("| 20")) if devp.exists() else 0,
            "sanity": sanity, "anomalies": anomalies(df), "task_runs": task_runs,
            "anomalies_md": len(anp.read_text().splitlines()) if anp.exists() else 0}
    summary = {"generated": meta["time"], "hardware": hw, "interim": interim, "smoke": a.smoke,
               "completeness": comp, "claims": cl, "methods": methods,
               "experiments": {e: g.to_dict("records") for e, g in with_run_cis(st, df).groupby("exp")} if not st.empty else {},
               "paired_tests": paired, "figures": figs, "anomalies": meta["anomalies"],
               "paper": paper_extras(runs, root)}
    atomic_write_json(out / "summary.json", summary)
    atomic_write_text(out / "FINAL_REPORT.md", render(comp, df, st, paired, cl, figs, cards, meta))
    nfig = sum(len(f.get("written", [])) for f in figs.values())
    print(f"claims: " + ", ".join(f"{c['id']} {c['verdict']}" for c in cl))
    print(f"wrote {out / 'FINAL_REPORT.md'}, {out / 'summary.json'}, {nfig} figures, {len(cards)} audit cards "
          f"({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
