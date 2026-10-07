"""StreamGuard (proposal name of the X1 combined gate: DSG's gate with the CUSUM detector) vs DSG, one machine.

Descriptive summary for RESULTS_DIGEST (it decides nothing; C-H5 is decided by claims.ch5):
  * attacks: X1 TEST paired forget accuracy under each attack vs DSG, same seed and items (per axis B1-B5);
  * clean: benign FPR and full-MMLU utility of the clean runs, paired vs DSG from the same X1 runs and seeds;
  * hard negatives (A3 MCQ): accuracy and gate fire rate on the five biology-adjacent subjects;
  * generation (X1-suite, streaming gate): answer-match rate per item set, gate fire rate, gibberish rate;
  * TOFU (X1-suite tofu-metrics): forget quality (KS p), model utility, forget truth ratio;
  * MT-Bench (gpuws job mtbench, mode cusum), when given.
"""
from collections import defaultdict

from dsgx.analysis import aggregate, claims
from dsgx.analysis.collect import fmt_ci

# X1-suite openqa item sets: which direction is good (forget-type sets: a lower answer match = fewer leaks)
OPEN_SETS = {"tofu-qa-forget": ("TOFU forget10 QA", "lower"), "tofu-qa-retain": ("TOFU retain QA", "higher"),
             "leak": ("WMDP-Bio open-ended (B6 leak)", "lower"), "benign-open": ("benign biology open-ended (A3)", "higher")}


def _is_sg(r, gate_type="cusum") -> bool:
    return r.method == "gated" and ((r.cfg.get("method") or {}).get("gate") or {}).get("type") == gate_type


def attack_rows(runs, paired, exp="X1", gate_type="cusum") -> list[dict]:
    mcq = {r.name: r for r in runs if r.is_mcq}
    by = defaultdict(list)
    for p in paired:
        r = mcq.get(p["run"])
        if p["exp"] != exp or p["vs"] != "dsg" or "forget" not in p or r is None or not _is_sg(r, gate_type) or r.is_clean:
            continue
        a = p["attack"]
        var = a.get("name") + "".join(f" {k}={v}" for k, v in sorted(a.items()) if k not in ("name", "_exp_id")
                                      and not isinstance(v, (dict, list)))
        pb, mc = p["forget"]["paired_bootstrap"], p["forget"]["mcnemar"]
        ref = mcq.get(p["ref_run"])
        by[(claims.AXES.get(a.get("name"), "?"), var)].append(
            {"sg": r.forget("raw")["mean"], "dsg": ref.forget("raw")["mean"] if ref else None, "diff": pb["diff"],
             "sig": pb["hi"] < 0 and mc["p"] < claims.ALPHA, "worse": pb["lo"] > 0 and mc["p"] < claims.ALPHA})
    out = []
    for (axis, var), v in sorted(by.items()):
        n = len(v)
        out.append({"axis": axis, "attack": var, "seeds": n, "sg": sum(x["sg"] for x in v) / n,
                    "dsg": sum(x["dsg"] for x in v if x["dsg"] is not None) / max(1, sum(x["dsg"] is not None for x in v)),
                    "diff": sum(x["diff"] for x in v) / n, "sig_seeds": sum(x["sig"] for x in v),
                    "worse_seeds": sum(x["worse"] for x in v)})
    return out


def clean_rows(runs, exp="X1", n_boot=2000, gate_type="cusum") -> dict:
    clean = [r for r in runs if r.is_mcq and r.base_exp == exp and r.split == "test" and r.is_clean]
    std = [r for r in clean if claims._std_clean(r)]
    hn = [r for r in clean if str(r.cfg.get("dataset_label") or "").endswith("-hardneg")]
    out = {}
    for tag, rs in (("standard", std), ("hardneg", hn)):
        sg = [r for r in rs if _is_sg(r, gate_type)]
        dsg = {r.seed: r for r in rs if r.method == "dsg-faithful"}
        diffs, fpr_sg, fpr_d, acc_sg, acc_d = [], [], [], [], []
        for r in sg:
            d = dsg.get(r.seed)
            if d is None:
                continue
            c = aggregate.compare(r, d, r.cfg.get("forget_datasets") or ["wmdp-bio", "wmdp-cyber"], n_boot=n_boot)
            if c.get("utility"):
                diffs.append(c["utility"]["paired_bootstrap"])
            fpr_sg.append((r.gate.get("benign_fpr") or {}).get("mean"))
            fpr_d.append((d.gate.get("benign_fpr") or {}).get("mean"))
            acc_sg.append((r.utility("raw") or {}).get("mean"))
            acc_d.append((d.utility("raw") or {}).get("mean"))
        if not diffs:
            continue
        mean = lambda xs: sum(x for x in xs if x is not None) / max(1, sum(x is not None for x in xs))
        out[tag] = {"seeds": len(diffs), "util_sg": mean(acc_sg), "util_dsg": mean(acc_d),
                    "util_diff": sum(d["diff"] for d in diffs) / len(diffs), "util_diff_lo": min(d["lo"] for d in diffs),
                    "util_diff_hi": max(d["hi"] for d in diffs), "fire_sg": mean(fpr_sg), "fire_dsg": mean(fpr_d),
                    "fire_sg_max": max(x for x in fpr_sg if x is not None) if any(x is not None for x in fpr_sg) else None}
    return out


def open_rows(runs, exp="X1-suite", gate_tag="cusum-stream") -> list[dict]:
    """Rows keyed base-stream / dsg-faithful-stream / cusum-stream (the gate column, whatever its tag)."""
    rs = {r.name: r for r in runs if r.base_exp == exp and "__" in r.name}
    out = []
    for key, (label, good) in OPEN_SETS.items():
        trio = {m: rs.get(f"{key}__{t}") for m, t in (("base-stream", "base-stream"),
                                                      ("dsg-faithful-stream", "dsg-faithful-stream"), ("cusum-stream", gate_tag))}
        if not trio["cusum-stream"] or not trio["dsg-faithful-stream"]:
            continue
        g = lambda m, k: (trio[m].metrics.get(k) if trio[m] else None)
        out.append({"set": label, "good": good, "match": {m: g(m, "match") for m in trio},
                    "fired": {m: g(m, "gate_fired") for m in trio}, "gibberish": {m: g(m, "gibberish") for m in trio}})
    return out


def tofu_rows(runs, exp="X1-suite", gate_cond="full+gate-cusum") -> dict:
    t = next((r for r in runs if r.base_exp == exp and r.name == "tofu-metrics"), None)
    if t is None:
        return {}
    c = t.metrics.get("conditions", {})
    pick = lambda k: {"fq": c[k].get("forget_quality_ks_p"), "mu": c[k].get("model_utility"),
                      "tr": (c[k].get("truth_ratio_forget") or {}).get("mean")} if k in c else None
    return {"retain-model": pick("retain-model"), "full": pick("full"), "dsg": pick("full+dsg"),
            "sg": pick(gate_cond), "paired": (t.metrics.get("paired") or {}).get(f"{gate_cond} vs full+dsg")}


def _f(x, d=3):
    return "-" if x is None else f"{x:.{d}f}"


def section(runs, paired, hw, mtbench=None) -> list[str]:
    """Markdown lines for one machine. mtbench: jobs/mtbench/summary.json (gpuws) or None."""
    L = [f"### StreamGuard (X1 combined gate, CUSUM) vs DSG ({hw})", ""]
    att = attack_rows(runs, paired)
    if att:
        won = sorted({a["axis"] for a in att if a["sig_seeds"] >= max(1, (a["seeds"] + 1) // 2)})
        L += ["Attacks (X1 TEST, forget accuracy under attack, lower is better; paired vs DSG, same seed and items; "
              "a seed counts when paired bootstrap hi < 0 and McNemar p < 0.05):", "",
              "| axis | attack | seeds | StreamGuard | DSG | mean diff | seeds better | seeds worse |", "|---|---|---|---|---|---|---|---|"]
        L += [f"| {a['axis']} | {a['attack']} | {a['seeds']} | {a['sg']:.3f} | {a['dsg']:.3f} | {a['diff']:+.3f} | "
              f"{a['sig_seeds']} | {a['worse_seeds']} |" for a in att]
        L += ["", f"Axes won (a majority of seeds better on some variant): {', '.join(won) or 'none'}.", ""]
    cl = clean_rows(runs)
    if "standard" in cl:
        s = cl["standard"]
        L += [f"Clean (full MMLU + WMDP-Bio, {s['seeds']} seeds): utility {s['util_sg']:.3f} vs DSG {s['util_dsg']:.3f}, "
              f"paired diff {s['util_diff']:+.4f} (per-seed CI {s['util_diff_lo']:+.3f}..{s['util_diff_hi']:+.3f}); "
              f"benign FPR {s['fire_sg']:.3f} (max {_f(s['fire_sg_max'])}) vs DSG {s['fire_dsg']:.3f}."]
    if "hardneg" in cl:
        h = cl["hardneg"]
        L += [f"Hard negatives (A3 MCQ, five biology-adjacent subjects, {h['seeds']} seeds): accuracy {h['util_sg']:.3f} vs "
              f"DSG {h['util_dsg']:.3f} (paired diff {h['util_diff']:+.4f}); gate fires on {h['fire_sg']:.3f} vs DSG "
              f"{h['fire_dsg']:.3f} of benign biology items (over-blocking)."]
    if cl:
        L.append("")
    op = open_rows(runs)
    if op:
        L += ["Generation (X1-suite, streaming gate scored on the prompt and then on every new token; answer match = "
              "reference-token recall >= 0.6 or MiniLM cosine >= 0.75):", "",
              "| item set | good | match base | match DSG | match StreamGuard | fired DSG | fired StreamGuard | gibberish DSG | gibberish StreamGuard |",
              "|---|---|---|---|---|---|---|---|---|"]
        for o in op:
            m, fi, gi = o["match"], o["fired"], o["gibberish"]
            L.append(f"| {o['set']} | {o['good']} | {fmt_ci(m['base-stream'])} | {fmt_ci(m['dsg-faithful-stream'])} | "
                     f"{fmt_ci(m['cusum-stream'])} | {_f((fi['dsg-faithful-stream'] or {}).get('mean'))} | "
                     f"{_f((fi['cusum-stream'] or {}).get('mean'))} | {_f((gi['dsg-faithful-stream'] or {}).get('mean'))} | "
                     f"{_f((gi['cusum-stream'] or {}).get('mean'))} |")
        L += ["", "Note: the queue STATUS headline of an X1-suite open-ended job prints its match rate as `forget` for every "
              "item set; for tofu-qa-retain and benign-open it is a retain / benign number (higher is better).", ""]
    tf = tofu_rows(runs)
    if tf.get("sg") and tf.get("dsg"):
        p = tf.get("paired") or {}
        trp = p.get("tr_forget") or {}
        L += [f"TOFU (X1-suite tofu-metrics): forget quality KS p StreamGuard {tf['sg']['fq']:.2e} vs DSG {tf['dsg']['fq']:.2e} "
              f"(full {tf['full']['fq']:.2e}); model utility {tf['sg']['mu']:.3f} vs DSG {tf['dsg']['mu']:.3f} (retain model "
              f"{tf['retain-model']['mu']:.3f}, full {tf['full']['mu']:.3f}); forget truth ratio {tf['sg']['tr']:.3f} vs DSG "
              f"{tf['dsg']['tr']:.3f} (retain model {tf['retain-model']['tr']:.3f})"
              + (f"; paired truth-ratio diff {trp.get('diff', 0):+.3f} [{trp.get('lo', 0):+.3f}, {trp.get('hi', 0):+.3f}], p {trp.get('p')}" if trp else "")
              + ".", ""]
    if mtbench and "cusum" in (mtbench.get("scores") or {}):
        sc = mtbench["scores"]
        d = sc.get("cusum_minus_dsg") or {}
        L += [f"MT-Bench (judge {mtbench.get('judge')}, same family: label every number): StreamGuard "
              f"{sc['cusum']['all']['mean']:.2f} [{sc['cusum']['all']['lo']:.2f}, {sc['cusum']['all']['hi']:.2f}] n {sc['cusum']['all']['n']} "
              f"vs DSG {sc['dsg']['all']['mean']:.2f}, base {sc['base']['all']['mean']:.2f}; paired StreamGuard - DSG "
              f"{d.get('diff', 0):+.3f} [{d.get('lo', 0):+.3f}, {d.get('hi', 0):+.3f}], p {d.get('p')}.", ""]
    if len(L) == 2:
        L += ["(no StreamGuard results on this machine yet)", ""]
    return L


# ----------------------------------------------------------------------------- POST-HOC, EXPLORATORY follow-ups (gpuws)
POSTHOC = (("PH-union", "union gate (DSG rho gate OR CUSUM; one conformal threshold, alpha 0.05, MMLU DEV)", "union",
            "union-stream", "full+union"),
           ("PH-tofucal", "StreamGuard with its threshold calibrated on TOFU retain DEV (instead of MMLU DEV)", None,
            "cusum-tofucal-stream", "full+gate-cusum-tofucal"))


def posthoc_section(runs, paired, hw) -> list[str]:
    """PH-union / PH-tofucal (DEVIATIONS 2026-10-07): decided after the X1 / X1-suite TEST results, never claim inputs.
    Same tables as the StreamGuard section, the post-hoc gate in the gate column, DSG from the same experiment."""
    L = [f"## Post-hoc exploratory follow-ups ({hw}; decided after the X1 / X1-suite TEST results; NOT claim inputs)", ""]
    for exp, what, gtype, tag, cond in POSTHOC:
        rs = [r for r in runs if r.base_exp == exp]
        L += [f"### {exp}: {what}", ""]
        if not rs:
            L += ["(not run / not fetched yet)", ""]
            continue
        if gtype:
            att = attack_rows(runs, paired, exp, gtype)
            if att:
                won = sorted({a["axis"] for a in att if a["sig_seeds"] >= max(1, (a["seeds"] + 1) // 2)})
                L += ["| axis | attack | seeds | gate | DSG | mean diff | seeds better | seeds worse |", "|---|---|---|---|---|---|---|---|"]
                L += [f"| {a['axis']} | {a['attack']} | {a['seeds']} | {a['sg']:.3f} | {a['dsg']:.3f} | {a['diff']:+.3f} | "
                      f"{a['sig_seeds']} | {a['worse_seeds']} |" for a in att]
                L += ["", f"Axes better than DSG (majority of seeds, descriptive only): {', '.join(won) or 'none'}.", ""]
            cl = clean_rows(runs, exp, gate_type=gtype)
            if "standard" in cl:
                c = cl["standard"]
                L.append(f"Clean ({c['seeds']} seeds): utility {c['util_sg']:.3f} vs DSG {c['util_dsg']:.3f}, paired diff "
                         f"{c['util_diff']:+.4f} (per-seed CI {c['util_diff_lo']:+.3f}..{c['util_diff_hi']:+.3f}); benign FPR "
                         f"{c['fire_sg']:.3f} (max {_f(c['fire_sg_max'])}) vs DSG {c['fire_dsg']:.3f}.")
            if "hardneg" in cl:
                h = cl["hardneg"]
                L.append(f"Hard negatives (A3 MCQ, {h['seeds']} seeds): accuracy {h['util_sg']:.3f} vs DSG {h['util_dsg']:.3f}; "
                         f"gate fires on {h['fire_sg']:.3f} vs DSG {h['fire_dsg']:.3f}.")
            L.append("")
        op = open_rows(runs, exp, tag)
        if op:
            L += ["| item set | good | match DSG | match gate | fired DSG | fired gate | gibberish DSG | gibberish gate |",
                  "|---|---|---|---|---|---|---|---|"]
            for o in op:
                m, fi, gi = o["match"], o["fired"], o["gibberish"]
                L.append(f"| {o['set']} | {o['good']} | {fmt_ci(m['dsg-faithful-stream'])} | {fmt_ci(m['cusum-stream'])} | "
                         f"{_f((fi['dsg-faithful-stream'] or {}).get('mean'))} | {_f((fi['cusum-stream'] or {}).get('mean'))} | "
                         f"{_f((gi['dsg-faithful-stream'] or {}).get('mean'))} | {_f((gi['cusum-stream'] or {}).get('mean'))} |")
            L.append("")
        tf = tofu_rows(runs, exp, cond)
        if tf.get("sg") and tf.get("dsg"):
            L += [f"TOFU v3 (tofu-metrics): forget quality KS p gate {tf['sg']['fq']:.2e} vs DSG {tf['dsg']['fq']:.2e}; model "
                  f"utility {tf['sg']['mu']:.3f} vs DSG {tf['dsg']['mu']:.3f} (retain model {tf['retain-model']['mu']:.3f}).", ""]
    return L
