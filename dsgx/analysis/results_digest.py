"""RESULTS_DIGEST.md: every finished result so far, both machines, aggregates only (regenerate every session).

    python -m dsgx.analysis.results_digest                  # -> $DSG_RESULTS/RESULTS_DIGEST.md
    python -m dsgx.analysis.results_digest --out FILE --cluster-runs DIR --n-boot 2000

Reads lab runs ($DSG_RESULTS/runs, hardware labpc) and fetched server runs (dsg_results_cluster/runs + jobs,
hardware gpuws). The two machines are separate hardware baselines: every table is per machine, nothing is
pooled or compared across them. Numbers are mean [95% CI] n (bootstrap over items unless stated; A6 stores
accuracies only, so its intervals are Wilson binomial). TEST numbers are reported, never used for a choice.
No item text, prompts or generations are read or written; only metrics files.
The claims section applies the fixed rules of dsgx/analysis/claims.py per machine, then lists the evidence
that exists so far for each claim (descriptive; the verdict is the rule's).
"""
import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from dsgx import paths
from dsgx.analysis import aggregate, claims
from dsgx.analysis.collect import load_all
from dsgx.util import atomic_write_text

CLUSTER = Path(os.environ.get("DSG_RESULTS_CLUSTER", Path.home() / "projects/mechunlearn-project/dsg_results_cluster"))
SKIP_EXP = ("_archive", "canary", "watchdog-test", "sanity_job73_backup")
# Lab A2 TOFU result accepted as a C-H7 input of the gpuws verdict (2026-10-05, DEVIATIONS) only when its two
# fine-tunes and the metrics job ran at the fixed Trainer loop with the same optimizer (exp/A2 ef5eb17), the
# metrics job started after both fine-tunes ended, and all three are DONE. Since 2026-10-06 (session 23) the metrics job
# must be the re-run at exp/A2 1b9cbb2 (= ef5eb17 + the activation-cache tag fix): the ef5eb17 metrics run read DSG
# features from the smoke model's cache (superseded, runs/A2/_superseded/smoke-cache-2026-10-06).
A2_FAIR_COMMIT = "ef5eb17"
A2_METRICS_COMMIT = "1b9cbb2"
A2_FAIR_JOBS = ("A2-tofu-finetune-full", "A2-tofu-finetune-retain", "A2-tofu-metrics")


def lab_a2_tofu_fair(lab) -> tuple[list, str]:
    """(lab A2 tofu-metrics runs usable as the gpuws C-H7 TOFU input, one-line provenance note)."""
    qd = paths.results_dir() / "queue"
    jobs = {j: (read(qd / "jobs" / f"{j}.json") or {}, read(qd / "state" / f"{j}.json") or {}) for j in A2_FAIR_JOBS}
    want = {j: A2_METRICS_COMMIT if j == "A2-tofu-metrics" else A2_FAIR_COMMIT for j in jobs}
    bad = [j for j, (job, st) in jobs.items()
           if st.get("status") != "DONE" or not str(job.get("commit", "")).startswith(want[j])]
    if bad:
        return [], "lab A2 TOFU not used: " + ", ".join(f"{j} not DONE at exp/A2 {want[j]}" for j in bad)
    ft_end = max(jobs[j][1].get("end", 0) for j in A2_FAIR_JOBS[:2])
    if jobs["A2-tofu-metrics"][1].get("start", 0) < ft_end:
        return [], "lab A2 TOFU not used: tofu-metrics started before a fine-tune ended"
    runs = [r for r in lab if r.base_exp == "A2" and r.name == "tofu-metrics"]
    return runs, (f"lab A2 TOFU (exp/A2 {A2_FAIR_COMMIT}: fixed Trainer loop, 8-bit AdamW for full and retain; "
                  f"metrics re-run at {A2_METRICS_COMMIT} with features from the real model) "
                  f"{'used' if runs else 'missing'}")


def ci(d, digits=3) -> str:
    if not d or d.get("mean") is None or (isinstance(d.get("mean"), float) and math.isnan(d["mean"])):
        return "–"
    s = f"{d['mean']:.{digits}f}"
    if d.get("lo") is not None and d.get("hi") is not None and not (isinstance(d["lo"], float) and math.isnan(d["lo"])):
        s += f" [{d['lo']:.{digits}f}, {d['hi']:.{digits}f}]"
    if d.get("n") is not None and not (isinstance(d["n"], float) and math.isnan(d["n"])):
        s += f" n={int(d['n'])}"
    return s


def wilson(p, n, z=1.96) -> dict:
    if p is None or not n:
        return {}
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return {"mean": p, "lo": max(0.0, c - h), "hi": min(1.0, c + h), "n": n}


def boot(values, n_boot=2000) -> dict:
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if not len(v):
        return {}
    rng = np.random.default_rng(0)
    m = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(1)
    return {"mean": float(v.mean()), "lo": float(np.quantile(m, 0.025)), "hi": float(np.quantile(m, 0.975)), "n": len(v)}


def read(p: Path):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def md_table(header, rows) -> list[str]:
    if not rows:
        return ["(none yet)", ""]
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)] + ["| " + " | ".join(str(c) for c in r) + " |" for r in rows] + [""]


def _sci(row, col):
    m = row.get(col)
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "–"
    n = row.get(f"{col}_n_items")
    return ci({"mean": m, "lo": row.get(f"{col}_lo"), "hi": row.get(f"{col}_hi"), "n": n})


# ----------------------------------------------------------------------------- MCQ conditions (both machines)
def mcq_section(runs, hw, n_boot) -> tuple[list[str], list, list]:
    L = []
    mcq = [r for r in runs if r.is_mcq]
    if not mcq:
        return [f"No MCQ runs on {hw}.", ""], [], []
    df = aggregate.run_table(mcq)
    st = aggregate.seed_table(df)
    paired = aggregate.paired_tests(mcq, n_boot=n_boot)
    for exp in sorted(set(st["exp"])):
        g = st[st["exp"] == exp].sort_values(["split", "case", "condition"])
        L += [f"#### {exp} ({hw})", ""]
        if any(claims.is_posthoc(r) for r in mcq if r.base_exp == exp):
            L += ["**POST-HOC, EXPLORATORY** (decided after the X1 TEST verdict; DEVIATIONS 2026-10-07): "
                  "not a claim-rule input, reported only as a labelled exploratory comparison.", ""]
        rows = []
        for _, r in g.iterrows():
            seeds = r.get("seeds") or []
            if len(seeds) <= 1:  # one run: item-level bootstrap CI from the run itself
                m = df[(df["exp"] == r["exp"]) & (df["condition"] == r["condition"]) & (df["split"] == r["split"]) &
                       (df["case"] == r["case"]) & (df["attack_params"] == r["attack_params"]) & (df["hardware"] == r["hardware"])]
                if len(m):
                    r = m.iloc[0].copy()
                    for col in ("raw_forget", "raw_util", "dsg_subset_forget", "dsg_subset_util", "benign_fpr"):
                        r[f"{col}_n_items"] = r.get(f"{col}_n")
                    r["raw_forget_n_seeds"] = 1
            att = r["attack"] if r["attack"] != "none" else ""
            if r.get("attack_params") and r["attack_params"] not in ("{}", "", None) and att:
                att += f" {r['attack_params']}"
            rows.append([r["condition"] + (" *(ref.)*" if r.get("reference") else ""), r["case"], r["split"], att or "–",
                         int(r.get("raw_forget_n_seeds") or 1), _sci(r, "raw_forget"), _sci(r, "raw_util"),
                         _sci(r, "dsg_subset_forget"), _sci(r, "dsg_subset_util"), _sci(r, "benign_fpr")])
        L += md_table(["condition", "case", "split", "attack", "seeds (>1: CI across seeds)", "forget acc (raw)", "utility (full MMLU)",
                       "forget (DSG subset)", "utility (DSG 4 subj.)", "benign FPR"], rows)
    pt = [p for p in paired if "forget" in p]
    if pt:
        L += [f"#### Paired tests on identical items ({hw})", ""]
        rows = []
        for p in pt:
            b, m = p["forget"]["paired_bootstrap"], p["forget"]["mcnemar"]
            seed = p["run"].split("__s")[-1].split("__")[0] if "__s" in p["run"] else "?"
            rows.append([p["exp"], p["condition"][:70], seed, p["vs"], f"{b['diff']:+.3f} [{b['lo']:+.3f}, {b['hi']:+.3f}]",
                         f"{b['p']:.3g}", f"{m['p']:.3g}", p["forget"]["n"]])
        L += md_table(["exp", "condition", "seed", "vs", "Δ forget acc", "bootstrap p", "McNemar p", "n"], rows)
    return L, st, paired


# ----------------------------------------------------------------------------- task results (generic)
def _flat(o, pre=""):
    if isinstance(o, dict):
        if "mean" in o and isinstance(o.get("mean"), (int, float)) and ("lo" in o or "n" in o):
            yield pre, o
            return
        for k, v in o.items():
            if k.endswith("_values") or k in ("per_item", "items", "curve"):
                continue
            yield from _flat(v, f"{pre}.{k}" if pre else k)


def task_section(runs, hw) -> list[str]:
    rows = []
    for r in runs:
        if r.is_mcq or r.base_exp in ("A6", "A6-full", "A6-full-d1v2", "A2-tofu-full", "FP-latency", "C3"):
            continue
        for k, d in _flat(r.metrics):
            rows.append([r.base_exp, r.name[:60], k[:60], ci(d)])
    if not rows:
        return []
    return [f"#### Task results with CIs ({hw})", ""] + md_table(["exp", "run", "metric", "mean [95% CI] n"], rows)


def conformal_section(runs, hw) -> list[str]:
    """N6: conformal threshold per target alpha; empirical FPR on held-out and shifted benign text (Wilson CI)."""
    rows = []
    for r in runs:
        cov, m = r.metrics.get("coverage"), r.metrics
        if r.base_exp != "N6" or not isinstance(cov, dict):
            continue
        g = m.get("gate", {})
        gate = g.get("type", "?") + (f" w{g['w']}" if "w" in g else "")
        for alpha, c in sorted(cov.items(), key=lambda kv: float(kv[0])):
            rows.append([gate, alpha, f"{c['tau']:.4f}", f"{c['calib_fpr']:.3f} n={m.get('n_calib')}",
                         ci(wilson(c["empirical_fpr_heldout"], m.get("n_heldout"))),
                         ci(wilson(c["empirical_fpr_shifted"], m.get("n_shifted")))])
    if not rows:
        return []
    return [f"#### N6 conformal gate thresholds ({hw})", "",
            "Benign false-positive rate at the conformal threshold for target alpha; held-out = same benign "
            "distribution as calibration, shifted = other benign text. Wilson 95% CI.", ""] + md_table(
        ["gate", "target alpha", "tau", "calib FPR", "held-out FPR", "shifted FPR"], rows)


# ----------------------------------------------------------------------------- A6 relearning / tampering (gpuws)
def a6_section(root: Path) -> tuple[list[str], dict]:
    L, summ = [], {}
    rows = []
    for exp, label in (("A6", "LoRA (lab A6 jobs run on gpuws, pinned exp/A6 code)"), ("A6-full", "full fine-tune"),
                       ("A6-full-d1v2", "full fine-tune, D1 v2 student")):
        for f in sorted((root / exp).glob("*/metrics.json")):
            m = read(f) or {}
            if "curve" not in m or not m.get("curve"):
                continue
            b, last = m["before"], m["curve"][-1]
            n, nu = m.get("n_eval", 300), m.get("n_util", 100)
            rows.append([exp, m["condition"], m.get("k"), m.get("rank"), m.get("steps"), ci(wilson(b["forget_acc"], n)),
                         ci(wilson(last["forget_acc"], n)), f"{last['forget_acc'] - b['forget_acc']:+.3f}",
                         ci(wilson(b.get("util_acc"), nu)), ci(wilson(last.get("util_acc"), nu))])
            summ.setdefault((exp, m["condition"]), []).append((b["forget_acc"], last["forget_acc"], m.get("k"), m.get("rank")))
    L += ["Relearning: forget accuracy before → after the last step (A6 stores accuracies only: Wilson 95% CI). "
          "Rank `full` = full-parameter fine-tune; `hook` = DSG clamp kept on during evaluation.", ""]
    L += md_table(["exp", "target", "k", "rank", "steps", "forget before", "forget after", "Δ", "utility before", "utility after"], rows)
    q = []
    for f in sorted((root / "A6").glob("quant*/metrics.json")):
        m = read(f) or {}
        q.append([f"{m.get('bits')}-bit", m.get("condition"), ci(wilson(m.get("forget_acc"), m.get("n_eval", 300))),
                  ci(wilson(m.get("util_acc"), 100))])
    for f in sorted((root / "A6").glob("steer*/metrics.json")):
        m = read(f) or {}
        for k, v in m.items():
            if isinstance(v, dict) and "forget_acc" in v:
                q.append([f"steering {k}", m.get("condition"), ci(wilson(v["forget_acc"], m.get("n_eval", 300))), ci(wilson(v.get("util_acc"), 100))])
    if q:
        L += ["Quantisation and steering (A6, gpuws):", ""] + md_table(["attack", "target", "forget acc", "utility"], q)
    bn = []
    for f in sorted((root / "A6").glob("benign*/metrics.json")):
        m = read(f) or {}
        if "after" in m:
            bn.append([m.get("condition"), m.get("steps"), ci(wilson(m["before"]["forget_acc"], 300)), ci(wilson(m["after"]["forget_acc"], 300)),
                       ci(wilson(m["before"].get("util_acc"), 100)), ci(wilson(m["after"].get("util_acc"), 100))])
    if bn:
        L += ["Benign fine-tune (alpaca; A6, gpuws): does ordinary fine-tuning undo the guard? (n = 300 forget, 100 utility: the task defaults, not overridden by the job; "
              "Wilson CI)", ""]
        L += md_table(["target", "steps", "forget before", "forget after", "utility before", "utility after"], bn)
    st = read(root.parent / "jobs" / "labjobs-a6-lora" / "status.json") or {}
    left = st.get("left") or st.get("failed") or []
    if left:
        L += [f"Not finished yet in the A6 LoRA group: {', '.join(left)} (resubmitted, session 10).", ""]
    return L, summ


# ----------------------------------------------------------------------------- TOFU (gpuws)
TOFU_SETS = ("forget", "retain", "real_authors", "world_facts")
# what the A6-full target names mean (cluster/a6_full.py; DEVIATIONS 2026-10-03)
A6_TARGET = {"d1": "D1-full α 0.3: MMLU collapsed to chance (0.238), so its low forget accuracy is not resistance",
             "d1-a0.1": "D1-full α 0.1", "d1v2": "D1 v2 selected student (α 0.1, 4000 steps)", "rmu": "RMU v1 (under-tuned)",
             "rmu-v2": "RMU v2 (c14)", "dsg-hook": "base weights + DSG clamp kept on", "dsg-nohook": "base weights, clamp removed"}


def tofu_section(root: Path, n_boot) -> tuple[list[str], dict]:
    m = read(root / "A2-tofu-full" / "tofu-metrics" / "metrics.json")
    if not m:
        return [], {}
    L = ["Full TOFU fine-tune of gemma-2-2b-it (forget10). Answer probability with bootstrap CI over questions; "
         "truth-ratio score and ROUGE-L recall are means (metric version 2; v1 numbers are invalid). "
         "Caveat (CLAUDE.md session 9): under cached generation the gate scores single new tokens, so ROUGE for "
         "full+dsg / full+best-gate measures that behaviour.", ""]
    rows, out = [], {}
    for cond, c in (m.get("conditions") or {}).items():
        out[cond] = c
        for s in TOFU_SETS:
            d = c.get(s) or {}
            ap = boot(d.get("answer_prob_values") or [], n_boot) if d.get("answer_prob_values") else {"mean": d.get("answer_prob"), "n": d.get("n")}
            rows.append([cond, s, ci(ap), f"{d.get('truth_ratio_score', float('nan')):.3f}", f"{d.get('rougeL_recall', float('nan')):.3f}"])
        rows.append([cond, "**model utility**", f"{c.get('model_utility', float('nan')):.3f}", "", ""])
        if c.get("forget_quality_ks_p") is not None:
            rows.append([cond, "forget quality (KS p vs retain model)", f"{c['forget_quality_ks_p']:.3g}", "", ""])
    L += md_table(["condition", "set", "answer prob", "truth-ratio score", "ROUGE-L recall"], rows)
    return L, out


# ----------------------------------------------------------------------------- other gpuws jobs
def auc_ci(a, n1, n2) -> dict:
    """Hanley-McNeil standard error of the AUROC."""
    q1, q2 = a / (2 - a), 2 * a * a / (1 + a)
    se = math.sqrt(max(0.0, (a * (1 - a) + (n1 - 1) * (q1 - a * a) + (n2 - 1) * (q2 - a * a)) / (n1 * n2)))
    return {"mean": a, "lo": max(0.0, a - 1.96 * se), "hi": min(1.0, a + 1.96 * se), "n": n1 + n2}


def q2_section(root: Path) -> list[str]:
    m = read(root / "Q2-graphs" / "tofu" / "metrics.json")
    if not m:
        return []
    L = ["**Q2 attribution graphs (TOFU mode, gpuws; circuit-tracer, Gemma Scope transcoders).** Qualitative case studies: one "
         "graph per row, so no CI. `gate` = DSG decision on that prompt (rho vs tau "
         f"{(m.get('gate') or {}).get('tau', float('nan')):.3f}); D2 = null-space edit on the TOFU features; "
         f"{len(m.get('matched_ids') or [])} of 20 DSG features matched a layer-3 transcoder feature (cos ≥ {m.get('match_cos')}).", ""]
    rows = []
    for tag, g in sorted((m.get("graphs") or {}).items()):
        gt = g.get("gate")
        rows.append([tag, f"{g.get('p_key', float('nan')):.3f}", f"{g.get('replacement_score', float('nan')):.3f}",
                     f"{g.get('completeness_score', float('nan')):.3f}", f"{g.get('feature_influence_share', float('nan')):.3f}",
                     f"{g.get('error_influence_share', float('nan')):.3f}", "–" if gt is None else f"rho {gt['rho']:.3f}, fired {gt['fired']}"])
    L += md_table(["graph", "P(key token)", "replacement", "completeness", "feature infl.", "error infl.", "gate"], rows)
    return L


def server_jobs_section(root: Path) -> list[str]:
    L = q2_section(root)
    jobs = root.parent / "jobs"
    g = read(root / "C3" / "auroc" / "metrics.json")
    if g:
        L += ["**C3 layer sweep (lab C3 jobs on gpuws):** gate AUROC forget vs benign by layer "
              "(rho gate, canonical 16k SAEs; Hanley–McNeil CI).", ""]
        L += md_table(["layer", "AUROC", "threshold"], [[x["layer"], ci(auc_ci(x["auroc"], x["n_pos"], x["n_neg"])), f"{x['threshold']:.3f}"]
                                                      for x in g.get("ranked", [])])
    for rd in sorted((root / "FP-latency").glob("*/metrics.json")):
        lat = (read(rd) or {}).get("latency") or {}
        if lat:
            names = list(lat[sorted(lat, key=int)[0]])
            L += ["**Latency (FP-latency v2, interleaved timing, batch size 1):** median ms (overhead vs base). No CI: median of repeats.", ""]
            L += md_table(["tokens"] + names, [[k] + [f"{lat[k][n]['ms_median']:.1f}" + ("" if n == "base" else f" ({100 * lat[k][n]['overhead_vs_base']:+.1f}%)")
                                                     for n in names] for k in sorted(lat, key=int)])
    s = read(jobs / "rmu-v2" / "summary.json")
    if s:
        L += [f"**RMU v2 selection (DEV):** `{json.dumps(s.get('selected'))[:300]}`", ""]
        rows = []
        for k, v in (s.get("paired") or {}).items():
            for view, d in v.items():
                for met, x in d.items():
                    rows.append([k, view, met, f"{x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}]", f"{x['p']:.3g}",
                                 f"{(x.get('mcnemar') or {}).get('p', float('nan')):.3g}", x["n"]])
        L += ["RMU v2 paired TEST comparisons (gpuws):", ""] + md_table(["comparison", "view", "metric", "Δ", "bootstrap p", "McNemar p", "n"], rows)
    sel = read(jobs / "d1-v2" / "SELECTION.json")
    if sel:
        rows = [[a, f"{d['forget']:.3f} n={d['n_forget']}", f"{d['utility']:.3f} n={d['n_util']}", f"{d['util_drop']:+.3f}", "yes" if d["eligible"] else "no"]
                for a, d in sel.get("students_dev", {}).items()]
        L += [f"**D1 v2 DEV selection** (rule: {sel.get('rule')}): selected α = {sel.get('selected_alpha')}, "
              f"bound met: {sel.get('bound_met')}. Base DEV forget {sel['base_dev']['forget']:.3f}, utility {sel['base_dev']['utility']:.3f}.", ""]
        L += md_table(["α", "DEV forget", "DEV utility (excl. hs_geography)", "utility drop", "eligible"], rows)
    L += mtbench_section(jobs)
    v = read(jobs / "validate" / "validate.json")
    if v:
        L += [f"**Server sanity gate (sanity-gpuws, exact):** last validate `{json.dumps(v.get('got', v))[:260]}`", ""]
    return L


def mtbench_paired(jobs: Path, n_boot=2000, ref: str = "base") -> dict:
    """{condition: {mean, lo, hi, p, n_differ, n}}: paired judge-score difference vs `ref` (default base) on the same
    (question, turn), two-sided bootstrap p. Only the `score` field of the judgment files is read."""
    by = {}
    for f in sorted((jobs / "mtbench").glob("judgments_*.jsonl")):
        cond = f.name[len("judgments_"):].split("__")[0]
        d = {}
        for line in f.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r.get("score"), (int, float)) and r["score"] >= 0:
                d[(r["question_id"], r["turn"])] = float(r["score"])
        by[cond] = d
    out = {}
    for cond in sorted(c for c in by if c != ref):
        keys = sorted(set(by[cond]) & set(by.get(ref, {})))
        diffs = np.asarray([by[cond][k] - by[ref][k] for k in keys])
        if len(diffs):
            b = boot(diffs, n_boot)
            m = diffs[np.random.default_rng(0).integers(0, len(diffs), (n_boot, len(diffs)))].mean(1)
            p = float(min(1.0, 2 * min((m <= 0).mean(), (m >= 0).mean())))
            out[cond] = {"mean": b["mean"], "lo": b["lo"], "hi": b["hi"], "p": p, "n_differ": int((diffs != 0).sum()),
                         "n": len(diffs)}
    return out


def mtbench_section(jobs: Path, n_boot=2000) -> list[str]:
    """BM3 MT-Bench: judge score per condition (summary.json) and paired differences vs base on the same
    (question, turn); only the `score` field of the judgment files is read."""
    s = read(jobs / "mtbench" / "summary.json")
    if not s:
        return []
    L = [f"**MT-Bench (BM3, gpuws):** judge `{s.get('judge')}`, same-family judge: {s.get('same_family_judge')} "
         f"(self-preference possible; absolute scores not comparable with the paper's GPT-4-judged 7.78). Score 1–10, "
         "bootstrap CI over (question, turn).", ""]
    sc = s.get("scores") or {}
    L += md_table(["condition", "all", "turn 1", "turn 2", "unparsed"],
                  [[k, ci(v.get("all"), 2), ci(v.get("turn1"), 2), ci(v.get("turn2"), 2), v.get("unparsed")] for k, v in sc.items()
                   if isinstance(v, dict) and "all" in v])
    rows = [[f"{cond} − base", f"{d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]", f"{d['p']:.3g}", d["n_differ"], d["n"]]
            for cond, d in mtbench_paired(jobs, n_boot).items()]
    if rows:
        L += ["Paired difference vs base (same question and turn; two-sided bootstrap p):", ""]
        L += md_table(["comparison", "Δ score", "bootstrap p", "pairs that differ", "n"], rows)
    return L


def x1_status() -> tuple[str, dict]:
    """(one-line X1 combination-wave status, selected slots) from the lab DEV selection, if it exists."""
    sel = read(paths.results_dir() / "runs" / "X1-screen" / "select" / "COMBINE_SELECTION.json")
    if not sel:
        return "X1 (combination wave): DEV selection not run yet.", {}
    slots = sel.get("slots") or {}
    picked = ", ".join(f"{k} = {v or 'default'}" for k, v in slots.items())
    return (f"X1 DEV selection done {str(sel.get('time', ''))[:16]} (labpc): {picked}. The combined method's TEST runs "
            "decide each machine's verdict (X1 on labpc; the gpuws replica of the same jobs on gpuws)."), slots


X1_TEST_RUNS = 192  # configs/experiments/X1.yaml: base 12 + (DSG, default gate, combined gate) x 5 seeds x 12 rows


def x1_progress(runs) -> dict:
    """X1 TEST progress of ONE machine's runs: finished MCQ runs, attack-success task, complete."""
    n = sum(1 for r in runs if r.base_exp == "X1" and r.is_mcq)
    succ = any(r.base_exp == "X1" and r.name == "attack-success" for r in runs)
    return {"runs": n, "attack_success": succ, "complete": n >= X1_TEST_RUNS and succ}


def x1_machine_lines(lab, gpu, vl=None, vg=None) -> list[str]:
    """Per-machine X1 TEST status (labpc and gpuws separately, never pooled) and, once both machines have the full
    X1 TEST wave, the "X1 replicated on two machines" line with each machine's own C-H5 verdict."""
    out, prog = [], {"labpc": x1_progress(lab or []), "gpuws": x1_progress(gpu or [])}
    for hw, p in prog.items():
        if p["runs"] or p["attack_success"]:
            out.append(f"X1 TEST ({hw}): {p['runs']}/{X1_TEST_RUNS} runs, attack-success "
                       f"{'done' if p['attack_success'] else 'not yet'}" + (" (complete)." if p["complete"] else "."))
        else:
            out.append(f"X1 TEST ({hw}): no run yet.")
    if all(p["complete"] for p in prog.values()):
        v = {hw: (d or {}).get("C-H5", {}).get("verdict", "?") for hw, d in (("labpc", vl), ("gpuws", vg))}
        out.append(f"**X1 replicated on two machines** (labpc and gpuws, same 33 jobs and pinned code): C-H5 is "
                   f"{v['labpc']} on labpc and {v['gpuws']} on gpuws ("
                   + ("same verdict" if v["labpc"] == v["gpuws"] else "verdicts differ") +
                   "); each machine is its own hardware baseline, numbers are never pooled or paired across machines.")
    return out


# ----------------------------------------------------------------------------- claims
def claims_section(lab, gpu, lab_paired, gpu_paired, a6, tofu, st_lab, st_gpu) -> list[str]:
    a2_lab, a2_note = lab_a2_tofu_fair(lab)
    L = ["## What this means for claims C-H1..C-H7", "",
         "Verdicts are the fixed rules of `dsgx/analysis/claims.py`, applied per machine (never pooled). "
         "\"Evidence so far\" lists finished results bearing on the claim; it is descriptive and does not override a verdict.", ""]
    vl = {c["id"]: c for c in claims.evaluate(lab, lab_paired)}
    vg = {c["id"]: c for c in claims.evaluate(gpu, gpu_paired, tofu_extra=a2_lab)}
    rows = [[cid, claims.CLAIMS[cid], f"{vl[cid]['verdict']}: {vl[cid]['evidence'][0][:90] if vl[cid]['evidence'] else ''}",
             f"{vg[cid]['verdict']}: {vg[cid]['evidence'][0][:90] if vg[cid]['evidence'] else ''}"] for cid in claims.CLAIMS]
    L += md_table(["claim", "statement", "lab PC (labpc)", "server (gpuws)"], rows)
    rep = [x for x in x1_machine_lines(lab, gpu, vl, vg) if x.startswith("**X1 replicated")]
    L += rep + [""] if rep else []

    def row(st, exp, cond_sub, split="test"):
        if st is None or len(st) == 0:
            return None
        g = st[(st["exp"] == exp) & (st["split"] == split) & (st["condition"].str.contains(cond_sub, regex=False))]
        return g.iloc[0] if len(g) else None

    ev = {cid: [] for cid in claims.CLAIMS}
    allr = [("labpc", r) for r in lab] + [("gpuws", r) for r in gpu]

    def have(*exps):
        return sorted({hw for hw, r in allr if r.base_exp in exps})

    for cid, exps, what in (("C-H1", ("B1",), "B1 (dilution) TEST results"), ("C-H2", ("B2", "B3"), "B2/B3 results"),
                            ("C-H4", ("N9",), "N9 results")):
        h = have(*exps)
        ev[cid].append(f"{what}: present on {', '.join(h)} (see the tables above)." if h else f"No {what} yet on either machine.")
    q2 = read(Path(CLUSTER / "runs" / "Q2-graphs" / "tofu" / "metrics.json")) if CLUSTER.exists() else None
    if q2:
        fr = (q2.get("graphs") or {}).get("fact0-attack-fr") or {}
        en = (q2.get("graphs") or {}).get("fact0-dsg") or {}
        if fr and en:
            ev["C-H2"].append(f"Q2 case study (gpuws, TOFU, n = 1, not a rule input): fact 0 in French, gate {fr.get('gate')}, "
                              f"P(key) {fr.get('p_key', float('nan')):.3f}; same fact in English under DSG: gate {en.get('gate')}.")
    if not have("B3"):
        ev["C-H2"].append("The A7 translate condition (gpuws) waits for the lab B3 translation cache.")
    for hw, r in allr:
        if r.base_exp in ("A4", "D3") and r.name.startswith("probe__") and r.metrics.get("layers"):
            lay = {int(k): v for k, v in r.metrics["layers"].items() if isinstance(v, dict) and isinstance(v.get("probe"), dict)}
            if lay:
                bl = max(lay, key=lambda k: lay[k]["probe"]["mean"])
                ev["C-H3"].append(f"{r.base_exp} {r.name} ({hw}): best layer {bl}, probe acc {ci(lay[bl]['probe'])} "
                                  f"vs control {ci(lay[bl].get('control'))} (chance 0.25).")
    if not ev["C-H3"]:
        ev["C-H3"].append("No A4/D3 probe result yet.")
    if not any("dsg" in x for x in ev["C-H3"]):
        ev["C-H3"].append("The rule also needs the DSG-guarded probe (A4/D3, not finished yet).")
    g = [r for r in (gpu or []) if r.base_exp == "C3" and r.name == "auroc"]
    x1_line, _ = x1_status()
    ev["C-H5"].append(x1_line)
    x1_lines = x1_machine_lines(lab, gpu, vl, vg)
    ev["C-H5"] += x1_lines
    ev["C-H5"].append("Ingredients so far (gpuws): C3 gate AUROC by layer (table above); RMU v2 "
                      "matches DSG's TEST forget accuracy without a gate (paired Δ in the RMU v2 table) at a significant "
                      "full-MMLU cost; FP-static / FP-multitopic tables above.")
    for (exp, cond), xs in sorted(a6.items()):
        if exp == "A6":
            continue
        d = [after - bef for bef, after, _, _ in xs]
        ev["C-H6"].append(f"{exp} {cond} ({A6_TARGET.get(cond, cond)}): forget {min(x[0] for x in xs):.3f}–{max(x[0] for x in xs):.3f} before → "
                          f"{min(x[1] for x in xs):.3f}–{max(x[1] for x in xs):.3f} after relearning (Δ {min(d):+.3f} to {max(d):+.3f}, k ∈ {sorted({x[2] for x in xs})}).")
    lora = {c: xs for (e, c), xs in a6.items() if e == "A6"}
    for c, xs in sorted(lora.items()):
        d = [after - bef for bef, after, _, _ in xs]
        ev["C-H6"].append(f"A6 LoRA {c}: Δ forget {min(d):+.3f} to {max(d):+.3f} over {len(xs)} (k, rank) cells.")
    if not {"d1", "d2", "student"} <= set(lora):
        ev["C-H6"].append("The rule needs matched d1/d2 + student cells (`a6-baked`, staged when the lab D1/D2 weights exist); "
                          "until then C-H6 stays Inconclusive.")
    if tofu:
        f = {k: (v.get("forget") or {}).get("answer_prob") for k, v in tofu.items()}
        u = {k: v.get("model_utility") for k, v in tofu.items()}
        ev["C-H7"].append("TOFU (gpuws, full FT): forget answer prob " + ", ".join(f"{k} {v:.3f}" for k, v in f.items() if v is not None)
                          + "; model utility " + ", ".join(f"{k} {v:.3f}" for k, v in u.items() if v is not None)
                          + ". The best gate here is the default window w16 (no X1 selection yet).")
    ev["C-H7"].append("The C-H7 rule reads the lab `A2` and the gpuws `A2-tofu-full` TOFU results (input change "
                      "2026-10-04, DEVIATIONS); since 2026-10-05 the gpuws verdict (the only machine with A7) also reads "
                      "the lab A2 result, each comparison within its own machine (DEVIATIONS; criterion unchanged). "
                      + a2_note + ".")
    for r in a2_lab:
        c = r.metrics.get("conditions") or {}
        ev["C-H7"].append("TOFU (labpc, A2): model utility " + ", ".join(
            f"{k} {v.get('model_utility'):.3f}" for k, v in c.items() if isinstance(v, dict) and v.get("model_utility") is not None)
            + f"; conditions {', '.join(c)}"
            + ("." if any("gate" in k or "fix" in k for k in c) else " (no best-fix condition yet: X1 TEST not finished)."))
    a7 = have("A7", "A7-1b", "A7-4b", "A7-12b")
    ev["C-H7"].append(f"A7 (Gemma 3) results present on {', '.join(a7)}." if a7 else
                      "No A7 (Gemma 3) result yet (job 110 failed offline; fixed and queued on gpuws in session 10).")
    L += ["### Evidence so far, per claim", ""]
    for cid in claims.CLAIMS:
        L.append(f"- **{cid}** ({claims.CLAIMS[cid]})")
        L += [f"  - {e}" for e in ev[cid]]
    L.append("")
    return L


# ----------------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--cluster-runs", default=str(CLUSTER / "runs"))
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args(argv)
    out = Path(a.out) if a.out else paths.results_dir() / "RESULTS_DIGEST.md"
    croot = Path(a.cluster_runs)
    lab = [r for r in load_all(None) if r.hardware == "labpc" and not r.smoke and r.base_exp not in SKIP_EXP]
    gpu = [r for r in load_all(croot) if r.hardware == "gpuws" and r.base_exp not in SKIP_EXP] if croot.exists() else []

    L = ["# RESULTS DIGEST", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M %Z')} by `python -m dsgx.analysis.results_digest` (regenerate at the end of every session).",
         "", "Rules: aggregates only (no item text); every number is mean [95% CI] n; the lab PC (`labpc`, RTX 2000 Ada) and the "
         "server (`gpuws`, RTX 6000 Ada) are separate hardware baselines and are never mixed in one table or test; "
         "TEST numbers are for reporting only. DEV rows are selection inputs.", "",
         f"Sources: lab `{paths.results_dir() / 'runs'}` ({len(lab)} runs), server `{croot}` ({len(gpu)} runs, fetched and sha256-verified).", ""]
    L += ["## 1. Lab PC (labpc)", ""]
    ml, st_lab, lab_paired = mcq_section(lab, "labpc", a.n_boot)
    L += ml + task_section(lab, "labpc") + conformal_section(lab, "labpc")
    L += ["## 2. Server (gpuws)", "", "### 2.1 MCQ conditions (harness runs)", ""]
    mg, st_gpu, gpu_paired = mcq_section(gpu, "gpuws", a.n_boot)
    L += mg + task_section(gpu, "gpuws") + conformal_section(gpu, "gpuws")
    a6l, a6s = a6_section(croot)
    L += ["### 2.2 A6 tampering (gpuws)", ""] + a6l
    tl, tofu = tofu_section(croot, a.n_boot)
    L += ["### 2.3 TOFU, full fine-tune (gpuws)", ""] + (tl or ["(none yet)", ""])
    L += ["### 2.4 Other server jobs (gpuws)", ""] + server_jobs_section(croot)
    L += claims_section(lab, gpu, lab_paired, gpu_paired, a6s, tofu, st_lab, st_gpu)
    atomic_write_text(out, "\n".join(L) + "\n")
    print(f"wrote {out} ({len(lab)} lab runs, {len(gpu)} gpuws runs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
