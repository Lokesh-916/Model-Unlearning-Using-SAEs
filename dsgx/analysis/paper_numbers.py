"""Paper numbers: the generated results -> paper/numbers.tex, one \\newcommand per number.

    python -m dsgx.analysis.paper_numbers                       # defaults below
    python -m dsgx.analysis.paper_numbers --out ~/projects/mechunlearn-project/paper/numbers.tex
    scripts/paper_update.sh                                     # final_report (both machines) + this + latexmk

Sources (each macro's source is written next to it in numbers.tex):
  summary.json of `final_report` for the lab PC and for gpuws (MCQ conditions, paired tests, claims, `paper` block),
  the server job summaries (`dsg_results_cluster/jobs/<job>/summary.json`: rmu-v2, mtbench), the sanity targets in
  dsgx/checks/sanity.py, the X1 DEV selection (COMBINE_SELECTION.json), the A7 CPU diagnosis
  (docs/a7_gemma3_diag.jsonl) and, for the two cross-GPU numbers sets, the two sanity runs' items.parquet
  (ids and metrics only) and the gpuws A7 DSG runs' gate decisions.

Every number macro \\resX comes with \\resXVal (point estimate), \\resXCI ([lo, hi]) and \\resXN (n); \\resX is the
full form "0.642 [0.604, 0.680] (n=637)". A number whose source does not exist yet expands to \\respending{X} (a red
"[pending]" mark), so the paper always compiles and no number is ever typed by hand. Macro names are letters only.
Lab PC and gpuws numbers carry the prefix Lab / Gpu and are never combined in one macro.
"""
import argparse
import glob
import json
import math
import os
import re
from pathlib import Path

from dsgx import paths
from dsgx.util import atomic_write_text, now_iso

PROJECT = Path(os.environ.get("DSG_PROJECT", Path.home() / "projects/mechunlearn-project"))
CLUSTER = PROJECT / "dsg_results_cluster"
REPO = Path(__file__).resolve().parents[2]
NAME = re.compile(r"^[A-Za-z]+$")


def _read(p):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def _ok(x):
    return x is not None and not (isinstance(x, float) and math.isnan(x))


def f3(x, d=3):
    return f"{x:.{d}f}"


def sgn(x, d=3):
    return f"${'+' if x >= 0 else '-'}{abs(x):.{d}f}$"


def pfmt(p):
    if not _ok(p):
        return None
    if p < 0.001:
        return "$p < 0.001$"
    return f"$p = {p:.2f}$" if p >= 0.01 else f"$p = {p:.3f}$"


class Numbers:
    def __init__(self):
        self.lines, self.pending, self.defined = [], [], set()

    def _def(self, name, body):
        assert NAME.match(name), name
        assert name not in self.defined, f"duplicate macro {name}"
        self.defined.add(name)
        self.lines.append(f"\\newcommand{{\\res{name}}}{{{body}}}")

    def comment(self, text):
        self.lines.append(f"% {text}")

    def missing(self, name, why, parts=("", "Val", "CI", "N")):
        self.pending.append((name, why))
        self.comment(f"PENDING {name}: {why}")
        for p in parts:
            self._def(name + p, f"\\respending{{{name}}}")

    def ci(self, name, d, src, digits=3, seeds=None):
        """d: {mean, lo, hi, n}."""
        if not d or not _ok(d.get("mean")):
            return self.missing(name, f"no value in {src}")
        self.comment(f"{name} <- {src}")
        v, lo, hi, n = d["mean"], d.get("lo"), d.get("hi"), d.get("n")
        ci = f"[{f3(lo, digits)}, {f3(hi, digits)}]" if _ok(lo) and _ok(hi) else ""
        ns = str(int(n)) if _ok(n) else ""
        if ns:
            full = f3(v, digits) + (f" {ci}" if ci else "") + f" ($n={ns}$" + (f", {seeds} seeds" if seeds and seeds > 1 else "") + ")"
        else:  # across-seed statistics without an item count (e.g. probes): N = number of seeds
            ns = str(int(seeds)) if seeds else ""
            full = f3(v, digits) + (f" {ci}" if ci else "") + (f" ({ns} seeds)" if ns else "")
        self._def(name, full)
        self._def(name + "Val", f3(v, digits))
        self._def(name + "CI", ci or "\\respending{" + name + "CI}")
        self._def(name + "N", ns or "\\respending{" + name + "N}")

    def diff(self, name, d, src, digits=3, has_p=True):
        """paired difference {diff, lo, hi, n} + optional McNemar p: \\resX = '+0.020 [-0.006, +0.047]', \\resXP."""
        if not d or not _ok(d.get("diff")):
            return self.missing(name, f"no paired test in {src}", ("", "Val", "CI", "N") + (("P",) if has_p else ()))
        self.comment(f"{name} <- {src}")
        ci = f"[{sgn(d['lo'], digits)}, {sgn(d['hi'], digits)}]"
        self._def(name, f"{sgn(d['diff'], digits)} {ci}")
        self._def(name + "Val", sgn(d["diff"], digits))
        self._def(name + "CI", ci)
        self._def(name + "N", str(int(d["n"])) if _ok(d.get("n")) else "\\respending{" + name + "N}")
        if has_p:
            p = pfmt((d.get("mcnemar") or {}).get("p", d.get("p")))
            self._def(name + "P", p or "\\respending{" + name + "P}")

    def text(self, name, value, src):
        if value is None or value == "":
            return self.missing(name, f"no value in {src}", ("",))
        self.comment(f"{name} <- {src}")
        self._def(name, str(value))


# ----------------------------------------------------------------------------- selectors
def row(summary, exp, cond, case="bio", split="test", attack="none"):
    for r in (summary or {}).get("experiments", {}).get(exp, []):
        if r["condition"] == cond and r["case"] == case and r["split"] == split and r["attack"] == attack:
            return r
    return None


def rci(r, key):
    if not r or not _ok(r.get(key)):
        return None
    return {"mean": r[key], "lo": r.get(f"{key}_lo"), "hi": r.get(f"{key}_hi"), "n": r.get(f"{key}_n_items")}


def paired(summary, exp, cond, vs, attack="none"):
    for p in (summary or {}).get("paired_tests", []):
        if p["exp"] == exp and p["condition"] == cond and p["vs"] == vs and p["attack"].get("name") == attack:
            return p
    return None


def pt(p, what="forget"):
    if not p or not p.get(what):
        return None
    d = dict(p[what]["paired_bootstrap"])
    d["mcnemar"] = p[what].get("mcnemar")
    return d


def claim(summary, cid):
    return next((c for c in (summary or {}).get("claims", []) if c["id"] == cid), None)


# ----------------------------------------------------------------------------- sections
CLAIM_WORDS = {"C-H1": "One", "C-H2": "Two", "C-H3": "Three", "C-H4": "Four", "C-H5": "Five", "C-H6": "Six", "C-H7": "Seven"}
DSG_COND = "dsg-faithful/n20/r95/m500/forget+utility"


def lab_numbers(N, s, src):
    N.comment(f"---- lab PC (labpc): {src}")
    N.text("LabInterim", "interim" if (s or {}).get("interim", True) else "final", src)
    N.ci("LabBaseForget", rci(row(s, "A1-test", "base/forget+utility"), "raw_forget"), f"{src} A1-test base")
    N.ci("LabBaseUtil", rci(row(s, "A1-test", "base/forget+utility"), "raw_util"), f"{src} A1-test base")
    d = row(s, "A1-test", DSG_COND)
    seeds = len(d["seeds"]) if d and d.get("seeds") else None
    N.ci("LabDsgForget", rci(d, "raw_forget"), f"{src} A1-test {DSG_COND}", seeds=seeds)
    N.ci("LabDsgUtil", rci(d, "raw_util"), f"{src} A1-test {DSG_COND}", seeds=seeds)
    N.ci("LabDsgFpr", rci(d, "benign_fpr"), f"{src} A1-test {DSG_COND}", seeds=seeds)
    N.ci("LabDsgSubsetForget", rci(d, "dsg_subset_forget"), f"{src} A1-test {DSG_COND} (DSG subset)", seeds=seeds)
    N.text("LabDsgSeeds", seeds, f"{src} A1-test {DSG_COND}")
    N.ci("LabRmuThirdForget", rci(row(s, "A1-test", "base/rmu-unverified"), "raw_forget"), f"{src} A1-test third-party RMU")
    N.diff("LabDsgVsBaseForget", pt(paired(s, "A1-test", DSG_COND, "base")), f"{src} paired A1-test DSG vs base")
    N.diff("LabDsgVsBaseUtil", pt(paired(s, "A1-test", DSG_COND, "base"), "utility"), f"{src} paired A1-test DSG vs base")

    ex = (s or {}).get("paper", {})
    asm = ex.get("attack_success_max", {})
    for exp, w in (("B1", "BOne"), ("B2", "BTwo"), ("B3", "BThree"), ("B4", "BFour"), ("B5", "BFive")):
        a = asm.get(exp)
        N.ci(f"Lab{w}Max", a, f"{src} paper.attack_success_max.{exp} (strongest DSG condition)")
        if a:
            att = dict(a["attack"])
            name = att.pop("name")
            desc = name + (" (" + ", ".join(f"{k}={v}" for k, v in sorted(att.items())) + ")" if att else "")
            N.text(f"Lab{w}Attack", desc.replace("_", "\\_"), f"{src} paper.attack_success_max.{exp}.attack")
            N.text(f"Lab{w}Conds", a.get("n_conditions"), f"{src} paper.attack_success_max.{exp}.n_conditions")
        else:
            N.missing(f"Lab{w}Attack", f"no {exp} attack-success result", ("",))
            N.missing(f"Lab{w}Conds", f"no {exp} attack-success result", ("",))
    b1 = asm.get("B1")
    pad = int(b1["attack"].get("pad", 0)) if b1 else None
    N.text("LabBOnePad", pad, f"{src} paper.attack_success_max.B1.attack.pad")
    base_pads = {int(r["attack_params"] and json.loads(r["attack_params"]).get("pad", 0) or 0): r
                 for r in (s or {}).get("experiments", {}).get("B1", []) if r["condition"].startswith("base/")}
    near = min(base_pads, key=lambda p: abs(p - pad)) if base_pads and pad is not None else None
    N.ci("LabBOneBaseCtrl", rci(base_pads.get(near), "raw_forget"), f"{src} B1 base at pad {near} (nearest to the strongest DSG pad)")
    N.text("LabBOneBaseCtrlPad", near, f"{src} B1 base pads")
    c1 = claim(s, "C-H1")
    N.text("LabCHOneConds", (c1 or {}).get("numbers", {}).get("n_conditions_meeting"), f"{src} claims C-H1")
    c2 = claim(s, "C-H2")
    N.text("LabCHTwoConds", (c2 or {}).get("numbers", {}).get("n_conditions"), f"{src} claims C-H2")

    pr = ex.get("probes", {})
    for tag, w in (("A4/base", "Base"), ("A4/dsg", "Dsg")):
        p = pr.get(tag)
        N.ci(f"LabProbe{w}", p and {**p["probe"], "n": None}, f"{src} paper.probes.{tag} (CI across seeds)",
             seeds=p and p["probe"].get("n_seeds"))
        N.ci(f"LabProbe{w}Ctrl", p and p.get("control") and {**p["control"], "n": None}, f"{src} paper.probes.{tag} control task",
             seeds=p and (p.get("control") or {}).get("n_seeds"))
        N.text(f"LabProbe{w}Layer", p and p["best_layer"], f"{src} paper.probes.{tag}.best_layer")
    c4 = claim(s, "C-H4")
    sig = ((c4 or {}).get("numbers") or {}).get("significant") or {}
    rm = sig.get("recon_mse")
    N.diff("LabNineReconCoef", rm and {"diff": rm["coef"], "lo": rm["lo"], "hi": rm["hi"], "n": (c4["numbers"] or {}).get("n")},
           f"{src} claims C-H4 numbers.significant.recon_mse (logistic coefficient)", has_p=False)
    tofu = ex.get("tofu", {})
    for cond, w in (("retain-model", "Retain"), ("full", "Full"), ("full+dsg", "FullDsg")):
        c = tofu.get(cond) or {}
        N.text(f"LabTofuUtil{w}", f3(c["model_utility"]) if _ok(c.get("model_utility")) else None, f"{src} paper.tofu.{cond}.model_utility")
    for cond, w in (("full", "Full"), ("full+dsg", "FullDsg")):
        p = (tofu.get(cond) or {}).get("forget_quality_ks_p")
        if _ok(p):
            m, e = f"{p:.1e}".split("e")
            N.text(f"LabTofuFq{w}", f"${m} \\times 10^{{{int(e)}}}$", f"{src} paper.tofu.{cond}.forget_quality_ks_p")
        else:
            N.missing(f"LabTofuFq{w}", f"no forget quality for {cond}", ("",))
    for cid, w in CLAIM_WORDS.items():
        N.text(f"LabVerdictCH{w}", (claim(s, cid) or {}).get("verdict"), f"{src} claims {cid} (fixed rules, claims.py)")


def rowp(summary, exp, cond, pad, case="bio", split="test"):
    """A row whose attack is dilution with the given pad (C2 detector runs carry the pad in attack_params)."""
    for r in (summary or {}).get("experiments", {}).get(exp, []):
        if r["condition"] == cond and r["case"] == case and r["split"] == split and \
                json.loads(r.get("attack_params") or "{}").get("pad") == pad:
            return r
    return None


def lab_extra_numbers(N, s, src, lab_runs: Path):
    """Secondary lab numbers used in the main text: hard negatives (A3), B4/B5 raw accuracies, the C2 detectors under
    dilution and the open-ended runs (A2/A3/B6 task metrics, read from each DONE run's metrics.json)."""
    hn = "dsg-faithful/n20/r95/m500/hardneg"
    N.ci("LabHardnegBaseUtil", rci(row(s, "A3", "base/hardneg"), "raw_util"), f"{src} A3 base hard negatives")
    N.ci("LabHardnegDsgUtil", rci(row(s, "A3", hn), "raw_util"), f"{src} A3 DSG hard negatives")
    N.ci("LabHardnegDsgFpr", rci(row(s, "A3", hn), "benign_fpr"), f"{src} A3 DSG hard negatives (gate fire rate)")
    N.diff("LabHardnegVsBase", pt(paired(s, "A3", hn, "base"), "utility"), f"{src} paired A3 DSG vs base (utility)")
    dsg = "dsg-faithful/n20/r95/m500/forget"
    rw = [row(s, "B4", f"{dsg}/rewrite_cache-index{i}-pathrewrites", attack="rewrite_cache") for i in range(5)]
    N.ci("LabBFourDsg", rci(rw[0], "raw_forget"), f"{src} B4 DSG rewrite 0")
    vals = [r["raw_forget"] for r in rw if r and _ok(r.get("raw_forget"))]
    N.text("LabBFourRange", f"{f3(min(vals))}--{f3(max(vals))}" if len(vals) == 5 else None, f"{src} B4 DSG rewrites 0-4 (min--max)")
    N.ci("LabBFourBase", rci(row(s, "B4", "base/forget/rewrite_cache-index0-pathrewrites", attack="rewrite_cache"), "raw_forget"),
         f"{src} B4 base rewrite 0")
    N.ci("LabBFiveDsg", rci(row(s, "B5", f"{dsg}/suffix-pathsuffix-200/suffix.json", attack="suffix"), "raw_forget"), f"{src} B5 DSG 200-step suffix")
    N.ci("LabBFiveBase", rci(row(s, "B5", "base/forget/suffix-pathsuffix-200/suffix.json", attack="suffix"), "raw_forget"),
         f"{src} B5 base 200-step suffix")
    langs = ("ar", "es", "fr", "hi", "ru", "ta", "te", "zh")
    tr = {l: row(s, "B3", f"{dsg}/translate-lang{l}-min_chrf40", attack="translate") for l in langs}
    vals = [r["raw_forget"] for r in tr.values() if r and _ok(r.get("raw_forget"))]
    N.text("LabBThreeLangRange", f"{f3(min(vals))}--{f3(max(vals))}" if len(vals) == len(langs) else None,
           f"{src} B3 DSG translations, 8 languages (min--max raw forget accuracy)")
    for l, w in (("fr", "Fr"), ("hi", "Hi")):
        N.ci(f"LabBThree{w}Dsg", rci(tr[l], "raw_forget"), f"{src} B3 DSG translate {l}")
        N.ci(f"LabBThree{w}Base", rci(row(s, "B3", f"base/forget/translate-lang{l}-min_chrf40", attack="translate"), "raw_forget"),
             f"{src} B3 base translate {l}")
    N.ci("LabBThreeBsixfourBase", rci(row(s, "B3", "base/forget/encode-encodingbase64", attack="encode"), "raw_forget"),
         f"{src} B3 base base64")
    for cond, w in (("dsg-faithful/n20/r95/m500/forget+dsg4", "Dsg"), ("gated/cusum/forget+dsg4", "Cusum"),
                    ("gated/window-w16/forget+dsg4", "Window")):
        for pad, pw in ((0, "Clean"), (400, "Pad")):
            r = rowp(s, "C2", f"{cond}/dilution-pad{pad}", pad)
            N.ci(f"LabCTwo{w}{pw}Forget", rci(r, "raw_forget"), f"{src} C2 {cond} pad {pad}")
            N.ci(f"LabCTwo{w}{pw}Fpr", rci(r, "benign_fpr"), f"{src} C2 {cond} pad {pad} (benign fire rate, DSG 4-subject set)")
    for run, w in (("B6/leak__base-stream", "BSixBase"), ("B6/leak__dsg-faithful-stream", "BSixStream"),
                   ("B6/leak__dsg-faithful-prompt_only", "BSixPrompt"), ("A3/benign-open__base-stream", "BenignOpenBase"),
                   ("A3/benign-open__dsg-faithful-stream", "BenignOpenDsg")):
        d = lab_runs / run
        m = _read(d / "metrics.json") if (d / "DONE").exists() else None
        for key, kw in (("match", "Match"), ("gibberish", "Gib"), ("gate_fired", "Fired")):
            N.ci(f"Lab{w}{kw}", (m or {}).get(key), f"runs/{run}/metrics.json {key}")


def gpu_numbers(N, s, src, jobs: Path, runs_root: Path):
    N.comment(f"---- server (gpuws): {src}")
    N.ci("GpuBaseForget", rci(row(s, "RMU-v2-test", "base/base"), "raw_forget"), f"{src} RMU-v2-test base")
    N.ci("GpuBaseUtil", rci(row(s, "RMU-v2-test", "base/base"), "raw_util"), f"{src} RMU-v2-test base")
    dsg = "dsg-faithful/n20/r95/m500/dsg-paper"
    N.ci("GpuDsgForget", rci(row(s, "RMU-v2-test", dsg), "raw_forget"), f"{src} RMU-v2-test DSG paper config")
    N.ci("GpuDsgUtil", rci(row(s, "RMU-v2-test", dsg), "raw_util"), f"{src} RMU-v2-test DSG paper config")
    N.ci("GpuDsgSubsetForget", rci(row(s, "RMU-v2-test", dsg), "dsg_subset_forget"), f"{src} RMU-v2-test DSG (DSG subset)")
    N.ci("GpuRmuTwoForget", rci(row(s, "RMU-v2-test", "base/rmu-v2-c14"), "raw_forget"), f"{src} RMU-v2-test c14")
    N.ci("GpuRmuTwoUtil", rci(row(s, "RMU-v2-test", "base/rmu-v2-c14"), "raw_util"), f"{src} RMU-v2-test c14")
    N.ci("GpuRmuTwoSubsetForget", rci(row(s, "RMU-v2-test", "base/rmu-v2-c14"), "dsg_subset_forget"), f"{src} RMU-v2-test c14 (DSG subset)")
    N.ci("GpuRmuOneForget", rci(row(s, "RMU-cluster-test", "base/rmu-cluster-c4"), "raw_forget"), f"{src} RMU-cluster-test c4")
    N.ci("GpuRmuThirdForget", rci(row(s, "RMU-v2-test", "base/rmu-unverified"), "raw_forget"), f"{src} RMU-v2-test third-party")
    N.diff("GpuRmuTwoVsDsgForget", pt(paired(s, "RMU-v2-test", "base/rmu-v2-c14", "dsg")), f"{src} paired RMU v2 vs DSG")
    N.diff("GpuRmuTwoVsDsgUtil", pt(paired(s, "RMU-v2-test", "base/rmu-v2-c14", "dsg"), "utility"), f"{src} paired RMU v2 vs DSG")
    N.diff("GpuRmuTwoVsBaseUtil", pt(paired(s, "RMU-v2-test", "base/rmu-v2-c14", "base"), "utility"), f"{src} paired RMU v2 vs base")
    rv = _read(jobs / "rmu-v2" / "summary.json") or {}
    hp = (rv.get("selected") or {}).get("hp") or {}
    N.text("GpuRmuTwoR", f"{hp['r']:.2f}" if "r" in hp else None, "jobs/rmu-v2/summary.json selected.hp.r")
    N.text("GpuRmuTwoCoeff", f"{hp['steering_coeff']:.2f}" if "steering_coeff" in hp else None, "jobs/rmu-v2/summary.json selected.hp.steering_coeff")
    base_dev = row(s, "RMU-v2-dev", "base/base-wmdp+mmlu48", split="dev")
    sel_dev = ((rv.get("selected") or {}).get("dev") or {}).get("util_pooled")
    N.text("GpuRmuTwoDevDrop", f3(base_dev["raw_util"] - sel_dev["mean"]) if base_dev and sel_dev else None,
           f"{src} RMU-v2-dev base pooled utility - jobs/rmu-v2 selected dev util_pooled")

    for tag, cond, exp in (("DOneFull", "base/undo-full-a0.1", "D1-full"), ("DOneTwo", "base/undo-v2-a0.1", "D1-v2")):
        N.ci(f"Gpu{tag}Forget", rci(row(s, exp, cond), "raw_forget"), f"{src} {exp} {cond}")
        N.ci(f"Gpu{tag}Util", rci(row(s, exp, cond), "raw_util"), f"{src} {exp} {cond}")
        N.diff(f"Gpu{tag}VsDsgForget", pt(paired(s, exp, cond, "dsg")), f"{src} paired {exp} {cond} vs DSG")
        N.diff(f"Gpu{tag}VsDsgUtil", pt(paired(s, exp, cond, "dsg"), "utility"), f"{src} paired {exp} {cond} vs DSG")
    N.ci("GpuDOneFullCollapseUtil", rci(row(s, "D1-full", "base/undo-full-a0.3"), "raw_util"), f"{src} D1-full undo-full-a0.3")

    for size, w in (("1b", "One"), ("4b", "Four"), ("12b", "Twelve")):
        exp = f"A7-{size}"
        N.ci(f"GpuGemma{w}BaseForget", rci(row(s, exp, "base/base-forget+util"), "raw_forget"), f"{src} {exp} base")
        N.ci(f"GpuGemma{w}BaseUtil", rci(row(s, exp, "base/base-forget+util"), "raw_util"), f"{src} {exp} base")
        N.diff(f"GpuGemma{w}DsgDelta", pt(paired(s, exp, "gated/rho/dsg-forget+util", "base")), f"{src} paired {exp} DSG vs base")
        N.diff(f"GpuGemma{w}FixDelta", pt(paired(s, exp, "gated/window-w16/best-fix-forget+util", "base")), f"{src} paired {exp} window gate vs base")
        fired = None
        f = sorted(glob.glob(str(runs_root / exp / "*__gated__none__bio-dsg-forget+util*" / "items.parquet")))
        if f:
            import pandas as pd

            it = pd.read_parquet(f[0], columns=["dataset", "gate_fired"])
            w_ = it[it.dataset == "wmdp-bio"]
            fired = f3(w_.gate_fired.astype(float).mean(), 2)
        N.text(f"GpuGemma{w}DsgFired", fired, f"{exp} DSG run items.parquet: share of WMDP-Bio items with the gate on")

    a6 = (s or {}).get("paper", {}).get("a6", {})
    for key, w in (("A6-full/dsg-hook", "FullDsgHook"), ("A6-full/dsg-nohook", "FullDsgNohook"), ("A6-full/d1-a0.1", "FullDOne"),
                   ("A6-full/rmu-v2", "FullRmuTwo"), ("A6-full-d1v2/d1v2", "FullDOneTwo"), ("A6/dsg-hook", "LoraDsgHook"),
                   ("A6/dsg-nohook", "LoraDsgNohook"), ("A6/d1", "LoraDOne"), ("A6/d2", "LoraDTwo"), ("A6/student", "LoraStudent")):
        a = a6.get(key)
        if not a:
            N.missing(f"GpuAsix{w}Delta", f"no {key} cells", ("",))
            N.missing(f"GpuAsix{w}Before", f"no {key} cells", ("",))
            continue
        lo, hi = a["delta"]
        N.text(f"GpuAsix{w}Delta", f"{sgn(lo)} to {sgn(hi)}" if abs(hi - lo) > 1e-9 else sgn(lo),
               f"{src} paper.a6.{key} (forget accuracy after - before, over {a['n_cells']} cells, k {a['k']})")
        N.text(f"GpuAsix{w}Before", f3(a["before"][0]), f"{src} paper.a6.{key}.before")

    cells = ((claim(s, "C-H6") or {}).get("numbers") or {}).get("cells") or {}
    for baked, w in (("d1", "DOne"), ("d2", "DTwo")):
        c = cells.get(baked)
        N.text(f"GpuCHSix{w}Cells", f"{c[0]}/{c[1]}" if c else None, f"{src} claims C-H6 numbers.cells.{baked} (wins/matched cells)")

    mt = _read(jobs / "mtbench" / "summary.json")
    sc = (mt or {}).get("scores") or {}
    for cond, w in (("base", "Base"), ("dsg", "Dsg"), ("window-w16", "Window")):
        N.ci(f"GpuMt{w}", (sc.get(cond) or {}).get("all"), f"jobs/mtbench/summary.json scores.{cond}.all", digits=2)
    N.text("GpuMtJudge", (mt or {}).get("judge"), "jobs/mtbench/summary.json judge")
    if mt:
        from dsgx.analysis.results_digest import mtbench_paired

        pdiff = mtbench_paired(jobs)
        for cond, w in (("dsg", "Dsg"), ("window-w16", "Window")):
            d = pdiff.get(cond)
            N.diff(f"GpuMt{w}Delta", d and {"diff": d["mean"], "lo": d["lo"], "hi": d["hi"], "n": d["n"], "p": d["p"]},
                   "jobs/mtbench judgments (paired bootstrap vs base, same question and turn)", digits=2)
            N.text(f"GpuMt{w}Differ", d and d["n_differ"], "jobs/mtbench judgments: pairs whose score differs")
    else:
        for w in ("Dsg", "Window"):
            N.missing(f"GpuMt{w}Delta", "no MT-Bench result", ("", "Val", "CI", "N", "P"))
            N.missing(f"GpuMt{w}Differ", "no MT-Bench result", ("",))
    for cid, w in CLAIM_WORDS.items():
        N.text(f"GpuVerdictCH{w}", (claim(s, cid) or {}).get("verdict"), f"{src} claims {cid} (fixed rules, claims.py)")


def sanity_numbers(N):
    from dsgx.checks.sanity import TARGETS

    for hw, w in (("labpc", "Lab"), ("gpuws", "Gpu")):
        t = TARGETS[hw]
        N.text(f"{w}SanityCorrect", t["wmdp_correct"], f"dsgx/checks/sanity.py TARGETS[{hw}]")
        N.text(f"{w}SanityN", t["wmdp_n"], f"dsgx/checks/sanity.py TARGETS[{hw}]")


def cross_gpu_numbers(N, lab_runs: Path, gpu_runs: Path):
    """Section 8: the same sanity run on both GPUs, item by item (ids and metrics only)."""
    import numpy as np
    import pandas as pd

    names = sorted({Path(p).parent.name for p in glob.glob(str(lab_runs / "sanity" / "*bio-5sets*" / "items.parquet"))}
                   & {Path(p).parent.name for p in glob.glob(str(gpu_runs / "sanity" / "*bio-5sets*" / "items.parquet"))})
    keys = ("XgpuItems", "XgpuRhoDiffer", "XgpuRhoMax", "XgpuFlips", "XgpuFlipMargin", "XgpuProbUngated", "XgpuProbGated",
            "XgpuChanged", "XgpuNetWmdp", "XgpuNetOther")
    if not names:
        for k in keys:
            N.missing(k, "no sanity run on both machines", ("",))
        return
    src = f"sanity/{names[0]}/items.parquet on both machines"
    k = ["dataset", "item_id"]
    a = pd.read_parquet(lab_runs / "sanity" / names[0] / "items.parquet").set_index(k)
    b = pd.read_parquet(gpu_runs / "sanity" / names[0] / "items.parquet").set_index(k)
    j = a.join(b, lsuffix="_l", rsuffix="_g", how="inner")
    rl, rg = j.rho_l.astype(float), j.rho_g.astype(float)
    fl, fg = j.gate_fired_l.astype(bool), j.gate_fired_g.astype(bool)
    flip = fl != fg
    probs = ["prob_A", "prob_B", "prob_C", "prob_D"]
    dp = np.abs(j[[p + "_l" for p in probs]].values - j[[p + "_g" for p in probs]].values).max(1)
    either = (fl | fg).values
    marg = max((rl[flip] - j.tau_l[flip].astype(float)).abs().max(), (rg[flip] - j.tau_g[flip].astype(float)).abs().max()) if flip.any() else None
    wm = j.index.get_level_values(0) == "wmdp-bio"
    net = j.correct_g.astype(int) - j.correct_l.astype(int)
    vals = {"XgpuItems": len(j), "XgpuRhoDiffer": int(((rl - rg).abs() > 0).sum()), "XgpuRhoMax": f3((rl - rg).abs().max()),
            "XgpuFlips": int(flip.sum()), "XgpuFlipMargin": f3(marg) if marg is not None else None,
            "XgpuProbUngated": f"{dp[~either].max():.2f}", "XgpuProbGated": f"{dp[either].max():.2f}",
            "XgpuChanged": int((j.pred_l != j.pred_g).sum()), "XgpuNetWmdp": f"${int(net[wm].sum()):+d}$",
            "XgpuNetOther": f"${int(net[~wm].sum()):+d}$"}
    for key in keys:
        N.text(key, vals[key], src)


def a7_diag_numbers(N, path: Path):
    rows = []
    try:
        rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    except OSError:
        pass
    by = {(r["case"], r["mode"], r["dtype"]): r for r in rows}
    src = f"{path.relative_to(REPO) if path.is_relative_to(REPO) else path} (scripts/diag_gemma3_clamp.py, CPU)"
    for case, w, dt in (("g2", "GemmaTwo", "float32"), ("g3-1b", "GemmaOne", "float32"), ("g3-4b", "GemmaFour", "bfloat16")):
        al, one = by.get((case, "all", dt)), by.get((case, "one", dt))
        N.text(f"Diag{w}Resid", f"{al['resid']:,.0f}".replace(",", "{,}") if al else None, src + f" {case} resid")
        N.text(f"Diag{w}Ratio", (f"{al['edit_over_resid']:.1f}" if al['edit_over_resid'] >= 10 else f"{al['edit_over_resid']:.2f}") if al else None, src + f" {case} edit/resid")
        N.text(f"Diag{w}KlAll", f"{al['kl_last']:.2f}" if al else None, src + f" {case} last-token KL, all tokens clamped")
        N.text(f"Diag{w}KlOne", f"{one['kl_last']:.4f}" if one else None, src + f" {case} last-token KL, one token clamped")


def x1_numbers(N):
    sel = _read(paths.results_dir() / "runs" / "X1-screen" / "select" / "COMBINE_SELECTION.json")
    slots = (sel or {}).get("slots") or {}
    N.text("LabXOneDetector", slots.get("detector"), "X1-screen/select/COMBINE_SELECTION.json slots.detector")
    N.text("LabXOneDefaults", sum(1 for v in slots.values() if not v) if sel else None,
           "X1-screen/select/COMBINE_SELECTION.json: slots left at the default")


# ----------------------------------------------------------------------------- the final method (fix framing)
FIX_AXES = (("Dil", "dilution-pad400-positionbefore-sourcewikitext"), ("Biofill", "dilution-pad1600-positionaround-sourcebenign_bio"),
            ("Decomp", "decompose"), ("Fr", "translate-langfr-min_chrf40"), ("Hi", "translate-langhi-min_chrf40"),
            ("Zh", "translate-langzh-min_chrf40"), ("Bsixfour", "encode-encodingbase64"), ("Leet", "encode-encodingleet"),
            ("Rewrite", "rewrite_cache-expB4-index0-pathrewrites"), ("Suffix", "suffix-expB5-pathsuffix-200/suffix.json"))


def _x1(s, label, attack_part=None, exp="X1"):
    """TEST row of a condition label (combined / dsg / default / base / union ...) of `exp`, clean or under one attack."""
    for r in (s or {}).get("experiments", {}).get(exp, []):
        parts = r["condition"].split("/")
        if attack_part is None and parts[-1] == label:
            return r
        if attack_part is not None and f"/{label}-forget/" in r["condition"] and r["condition"].endswith(attack_part):
            return r
    return None


def _x1_pair(s, label, vs, attack_part=None, exp="X1"):
    for p in (s or {}).get("paired_tests", []):
        if p["exp"] != exp or p["vs"] != vs:
            continue
        c = p["condition"]
        if (attack_part is None and c.split("/")[-1] == label) or \
           (attack_part is not None and f"/{label}-forget/" in c and c.endswith(attack_part)):
            return p
    return None


def _x1_pairs(s, label, attack_part, exp):
    """Every seed's paired test (label vs dsg) under one attack."""
    return [p for p in (s or {}).get("paired_tests", []) if p["exp"] == exp and p["vs"] == "dsg"
            and f"/{label}-forget/" in p["condition"] and p["condition"].endswith(attack_part)]


def _axes_better(s, label, exp):
    """Descriptive (the digest's rule, not a claim): axes B1-B5 where a majority of seeds is significantly better than
    DSG (paired bootstrap hi < 0 and McNemar p < 0.05) on some variant."""
    from dsgx.analysis import claims

    won = set()
    for _, a in FIX_AXES:
        ps = [x for x in _x1_pairs(s, label, a, exp) if x.get("forget")]
        sig = sum(x["forget"]["paired_bootstrap"]["hi"] < 0 and (x["forget"].get("mcnemar") or {}).get("p", 1) < claims.ALPHA
                  for x in ps)
        if ps and sig >= max(1, (len(ps) + 1) // 2):
            won.add(claims.AXES.get(ps[0]["attack"].get("name"), "?"))
    return sorted(won)


WORDS = {0: "none", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five"}


def _and(xs):
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1] if xs else None


def _fix_mcq(N, P, s, src):
    """X1 TEST, one machine (P = Lab | Gpu): the combined gate (StreamGuard) vs DSG, the operating-point ablation
    (default gate = DSG's features and rho statistic with the same 5% benign-FPR threshold) and hard negatives."""
    N.comment(f"---- the final method: X1 combined gate (CUSUM) vs DSG, {P}")
    N.ci(f"{P}FixForget", rci(_x1(s, "combined"), "raw_forget"), f"{src} combined, clean")
    N.ci(f"{P}FixUtil", rci(_x1(s, "combined"), "raw_util"), f"{src} combined, clean (full-MMLU utility)")
    N.ci(f"{P}FixFpr", rci(_x1(s, "combined"), "benign_fpr"), f"{src} combined, clean (benign fire rate)")
    N.ci(f"{P}XDsgForget", rci(_x1(s, "dsg"), "raw_forget"), f"{src} dsg, clean")
    N.ci(f"{P}XDsgUtil", rci(_x1(s, "dsg"), "raw_util"), f"{src} dsg, clean")
    N.ci(f"{P}XDsgFpr", rci(_x1(s, "dsg"), "benign_fpr"), f"{src} dsg, clean (benign fire rate)")
    N.diff(f"{P}FixVsDsgForget", pt(_x1_pair(s, "combined", "dsg")), f"{src} paired combined vs dsg (forget)")
    N.diff(f"{P}FixVsDsgUtil", pt(_x1_pair(s, "combined", "dsg"), "utility"), f"{src} paired combined vs dsg (utility)")
    for w, a in FIX_AXES:
        N.ci(f"{P}Fix{w}Forget", rci(_x1(s, "combined", a), "raw_forget"), f"{src} combined under {a}")
        N.ci(f"{P}XDsg{w}Forget", rci(_x1(s, "dsg", a), "raw_forget"), f"{src} dsg under {a}")
        N.diff(f"{P}FixVsDsg{w}", pt(_x1_pair(s, "combined", "dsg", a)), f"{src} paired combined vs dsg under {a}")
    for w, a in (("Dil", FIX_AXES[0][1]), ("Decomp", "decompose")):
        N.ci(f"{P}Def{w}Forget", rci(_x1(s, "default", a), "raw_forget"), f"{src} default gate (rho, 5% FPR threshold) under {a}")
    N.ci(f"{P}DefUtil", rci(_x1(s, "default"), "raw_util"), f"{src} default gate, clean (full-MMLU utility)")
    N.ci(f"{P}DefFpr", rci(_x1(s, "default"), "benign_fpr"), f"{src} default gate, clean (benign fire rate)")
    for lab_, w in (("combined-hardneg", "Fix"), ("dsg-hardneg", "XDsg")):
        N.ci(f"{P}{w}HardnegFired", rci(_x1(s, lab_), "benign_fpr"), f"{src} {lab_} (gate fire rate on A3 benign biology MCQ)")
        N.ci(f"{P}{w}HardnegAcc", rci(_x1(s, lab_), "raw_util"), f"{src} {lab_} (accuracy on A3 benign biology MCQ)")
    # the C-H5 rule's own reading of the combined gate (evidence line of claims.ch5; descriptive here)
    import re as _re
    ev = next((e for e in (claim(s, "C-H5") or {}).get("evidence", []) if e.startswith("X1:gated/cusum")), None)
    m = _re.search(r"wins on \[([^\]]*)\]", ev or "")
    u = _re.search(r"paired utility vs DSG ([-+]?[0-9.]+)", ev or "")
    axes = [x.strip(" '") for x in m.group(1).split(",") if x.strip(" '")] if m else None
    if P == "Lab":
        # axes won by the C-H5 rule's per-axis criterion: the winner's axes when Supported, else the X1 combined gate's
        # evidence line (the per-axis criterion is met or not regardless of the utility/FPR parts of the verdict)
        c5axes = ((claim(s, "C-H5") or {}).get("numbers") or {}).get("axes") or axes
        N.text("LabFixAxesWon", len(c5axes) if c5axes is not None else None,
               "lab summary.json claims C-H5 numbers.axes (Supported) or C-H5 evidence X1 combined: axes won")
    N.text(f"{P}FixRuleAxes", _and(axes) if axes else None, f"{src} claims C-H5 evidence (X1 combined): axes won")
    N.text(f"{P}FixRuleAxesCount", WORDS.get(len(axes)) if axes is not None else None, f"{src} claims C-H5 evidence: number of axes won")
    du = float(u.group(1)) if u else None
    N.text(f"{P}FixRuleUtilDiff", f"${du:+.4f}$" if du is not None else None, f"{src} claims C-H5 evidence: mean paired utility vs DSG")
    N.text(f"{P}FixRuleUtilMiss", f"{-0.01 - du:.4f}" if du is not None and du < -0.01 else None,
           f"{src} claims C-H5: utility shortfall below the DSG - 0.01 budget")


def _fix_suite(N, P, runs_root, mach):
    """X1-suite on one machine: open-ended benign biology, B6 leakage, streaming TOFU QA, hard negatives, TOFU metrics."""
    N.comment(f"---- X1-suite ({mach}): the final method vs DSG on open-ended benign biology, B6 leakage, TOFU")
    d0 = runs_root / "X1-suite"
    pj = _read(d0 / "paired" / "paired.json") if (d0 / "paired" / "DONE").exists() else None
    for task, w in (("benign-open", "Open"), ("leak", "Leak"), ("tofu-qa-forget", "TofuQaForget"), ("tofu-qa-retain", "TofuQaRetain")):
        for tag, who in (("cusum-stream", "Fix"), ("dsg-faithful-stream", "SuiteDsg"), ("base-stream", "SuiteBase")):
            if who == "SuiteBase" and not task.startswith("tofu"):
                continue
            d = d0 / f"{task}__{tag}"
            m = _read(d / "metrics.json") if (d / "DONE").exists() else None
            for key, kw in (("match", "Match"), ("gibberish", "Gib"), ("gate_fired", "Fired")):
                if who == "SuiteBase" and key != "match":
                    continue
                N.ci(f"{P}{who}{w}{kw}", (m or {}).get(key), f"{mach} runs/X1-suite/{task}__{tag}/metrics.json {key}")
        for key, kw in (("match", "Match"), ("gibberish", "Gib"), ("gate_fired", "Fired")):
            r = ((pj or {}).get(task) or {}).get(key) or {}
            d = dict(r.get("paired_bootstrap") or {}) or None
            if d:
                d["mcnemar"] = r.get("mcnemar")
            N.diff(f"{P}Fix{w}VsDsg{kw}", d, f"{mach} runs/X1-suite/paired/paired.json {task}.{key} (gate - DSG)")
    hn = (pj or {}).get("hardneg") or {}
    s0 = (hn.get("seeds") or {}).get("0") or {}
    for key, kw in (("correct", "Util"), ("gate_fired", "Fired")):
        r = s0.get(key) or {}
        d = dict(r.get("paired_bootstrap") or {}) or None
        if d:
            d["mcnemar"] = r.get("mcnemar")
        N.diff(f"{P}FixHardneg{kw}VsDsg", d, f"{mach} paired.json hardneg seed 0 {key} (gate - DSG)")
        N.text(f"{P}FixHardneg{kw}SigSeeds", (hn.get("mean_over_seeds") or {}).get(key, {}).get("significant_seeds") if hn.get("n_seeds") else None,
               f"{mach} paired.json hardneg: seeds with significant {key} difference (of {hn.get('n_seeds')})")
    _tofu_metrics(N, f"{P}Suite", d0 / "tofu-metrics", (("retain-model", "Retain"), ("full", "Full"), ("full+dsg", "Dsg"),
                                                         ("full+gate-cusum", "Fix")), mach)
    tm = _read(d0 / "tofu-metrics" / "metrics.json") if (d0 / "tofu-metrics" / "DONE").exists() else None
    pr = ((tm or {}).get("paired") or {}).get("full+gate-cusum vs full+dsg") or {}
    N.diff(f"{P}SuiteTofuFixVsDsgForgetProb", pr.get("ans_forget"), f"{mach} tofu-metrics paired full+gate-cusum vs full+dsg ans_forget", has_p=False)
    N.diff(f"{P}SuiteTofuFixVsDsgRetainProb", pr.get("ans_retain"), f"{mach} tofu-metrics paired full+gate-cusum vs full+dsg ans_retain", has_p=False)


def _tofu_metrics(N, pre, tm_dir, conds, mach):
    tm = _read(tm_dir / "metrics.json") if (tm_dir / "DONE").exists() else None
    cc = (tm or {}).get("conditions") or {}
    for cond, w in conds:
        c = cc.get(cond) or {}
        N.text(f"{pre}TofuUtil{w}", f3(c["model_utility"]) if _ok(c.get("model_utility")) else None,
               f"{mach} {tm_dir.parent.name}/tofu-metrics {cond} model_utility")
        if cond != "retain-model":
            p = c.get("forget_quality_ks_p")
            if _ok(p) and p > 0:
                m_, e = f"{p:.1e}".split("e")
                N.text(f"{pre}TofuFq{w}", f"${m_} \\times 10^{{{int(e)}}}$", f"{mach} {tm_dir.parent.name}/tofu-metrics {cond} forget_quality_ks_p")
            else:
                N.missing(f"{pre}TofuFq{w}", f"no forget quality for {cond}", ("",))


def muse_numbers(N, jobs: Path):
    """MUSE (BM1, gpuws, official muse_bench metrics; DSG and the best gate on the MUSE target model)."""
    N.comment("---- MUSE (gpuws, jobs/muse/summary.json)")
    res = (_read(jobs / "muse" / "summary.json") or {}).get("results") or {}
    for corpus, cw in (("muse-news", "News"), ("muse-books", "Books")):
        for cond, mw in (("retrain", "Retrain"), ("target", "Target"), ("target+dsg", "Dsg"), ("target+best-gate", "Gate")):
            r = (res.get(corpus) or {}).get(cond) or {}
            for key, kw in (("knowmem_forget", "KnowF"), ("knowmem_retain", "KnowR"), ("verbmem_forget", "Verb")):
                N.text(f"GpuMuse{cw}{mw}{kw}", f3(r[key]) if _ok(r.get(key)) else None, f"jobs/muse/summary.json {corpus}.{cond}.{key}")
            if cond != "retrain":
                N.text(f"GpuMuse{cw}{mw}Priv", f"{r['privleak']:.1f}" if _ok(r.get("privleak")) else None,
                       f"jobs/muse/summary.json {corpus}.{cond}.privleak")


def posthoc_numbers(N, gpu, gpu_runs: Path):
    """POST-HOC, EXPLORATORY (DEVIATIONS 2026-10-07; never claim inputs): PH-union (DSG rho gate OR CUSUM, one
    conformal threshold) and PH-tofucal (StreamGuard calibrated on TOFU retain DEV), gpuws. \\resGpuPhUnionStatus /
    \\resGpuPhTofucalStatus = done | pending (the paper prints the paragraph's numbers only when done)."""
    N.comment("---- POST-HOC, EXPLORATORY follow-ups (gpuws): PH-union, PH-tofucal")
    src = "gpuws summary.json PH-union (TEST)"
    U = "PH-union"
    rows = (gpu or {}).get("experiments", {}).get(U, [])
    pj = lambda e: _read(gpu_runs / e / "paired" / "paired.json") if (gpu_runs / e / "paired" / "DONE").exists() else None  # noqa: E731
    pu, ptc = pj(U), pj("PH-tofucal")
    n_mcq = sum(1 for r in rows)
    N.text("GpuPhUnionStatus", "done" if pu and n_mcq >= 24 else "pending", "PH-union paired DONE and its MCQ rows in the gpuws summary")
    N.text("GpuPhTofucalStatus", "done" if ptc else "pending", "PH-tofucal paired DONE")
    # parts of PH-union that finish before its MCQ jobs (the paper prints them on their own, unpaired against DSG in the
    # same experiment, until PH-union-paired is DONE)
    done = lambda e, t: (gpu_runs / e / t / "DONE").exists()  # noqa: E731
    gates_ = ("union-stream", "dsg-faithful-stream")
    tofu_ok = done(U, "tofu-metrics") and all(done(U, f"{t}__{g}") for t in ("tofu-qa-forget", "tofu-qa-retain") for g in gates_)
    N.text("GpuPhUnionTofuStatus", "done" if tofu_ok else "pending", "PH-union tofu-metrics + tofu-qa-{forget,retain} (union, DSG) DONE")
    N.text("GpuPhUnionOpenStatus", "done" if all(done(U, f"benign-open__{g}") for g in gates_) else "pending",
           "PH-union benign-open (union, DSG) DONE")
    N.text("GpuPhUnionPairedStatus", "done" if pu else "pending", "PH-union paired DONE")
    # On TOFU the union uses DSG's TOFU features; if DSG's own rho gate already fires on more than alpha of the MMLU DEV
    # calibration prompts, the conformal threshold is the sentinel UNION_BIG and the strict '>' never fires.
    tmu = _read(gpu_runs / U / "tofu-metrics" / "metrics.json") if done(U, "tofu-metrics") else None
    cu = ((tmu or {}).get("conditions") or {})
    uthr, dtau = (cu.get("full+union") or {}).get("threshold"), (cu.get("full+dsg") or {}).get("tau")
    N.text("GpuPhUnionTofuDegenerate", ("yes" if uthr >= 1e9 else "no") if _ok(uthr) else None,
           "PH-union tofu-metrics full+union threshold >= UNION_BIG (union never fires on TOFU)")
    N.text("GpuPhUnionTofuDsgTau", f"{dtau:.3f}" if _ok(dtau) else None, "PH-union tofu-metrics full+dsg tau (TOFU features)")
    N.ci("GpuPhUnionUtil", rci(_x1(gpu, "union", exp=U), "raw_util"), f"{src} union, clean (full-MMLU utility)")
    N.ci("GpuPhUnionFpr", rci(_x1(gpu, "union", exp=U), "benign_fpr"), f"{src} union, clean (benign fire rate)")
    N.diff("GpuPhUnionVsDsgUtil", pt(_x1_pair(gpu, "union", "dsg", exp=U), "utility"), f"{src} paired union vs dsg (utility)")
    for w, a in FIX_AXES:
        N.ci(f"GpuPhUnion{w}Forget", rci(_x1(gpu, "union", a, exp=U), "raw_forget"), f"{src} union under {a}")
        N.diff(f"GpuPhUnionVsDsg{w}", pt(_x1_pair(gpu, "union", "dsg", a, exp=U)), f"{src} paired union vs dsg under {a}")
    won = _axes_better(gpu, "union", U) if rows else None
    N.text("GpuPhUnionAxes", _and(won) if won else ("none" if won == [] else None), f"{src}: axes with a majority of seeds better than DSG")
    N.ci("GpuPhUnionHardnegFired", rci(_x1(gpu, "union-hardneg", exp=U), "benign_fpr"), f"{src} union-hardneg fire rate")
    for exp, task, tag, w in ((U, "leak", "union-stream", "UnionLeak"), (U, "benign-open", "union-stream", "UnionOpen"),
                              (U, "tofu-qa-forget", "union-stream", "UnionTofuForget"),
                              (U, "tofu-qa-retain", "union-stream", "UnionTofuRetain"),
                              ("PH-tofucal", "tofu-qa-forget", "cusum-tofucal-stream", "TofucalTofuForget"),
                              ("PH-tofucal", "tofu-qa-retain", "cusum-tofucal-stream", "TofucalTofuRetain")):
        d = gpu_runs / exp / f"{task}__{tag}"
        m = _read(d / "metrics.json") if (d / "DONE").exists() else None
        N.ci(f"GpuPh{w}Match", (m or {}).get("match"), f"gpuws runs/{exp}/{task}__{tag} match")
        N.ci(f"GpuPh{w}Fired", (m or {}).get("gate_fired"), f"gpuws runs/{exp}/{task}__{tag} gate_fired")
        dd_ = gpu_runs / exp / f"{task}__dsg-faithful-stream"  # DSG re-run in the same experiment (same items)
        md = _read(dd_ / "metrics.json") if (dd_ / "DONE").exists() else None
        N.ci(f"GpuPh{w}DsgMatch", (md or {}).get("match"), f"gpuws runs/{exp}/{task}__dsg-faithful-stream match")
        N.ci(f"GpuPh{w}DsgFired", (md or {}).get("gate_fired"), f"gpuws runs/{exp}/{task}__dsg-faithful-stream gate_fired")
        r =(((pu if exp == U else ptc) or {}).get(task) or {}).get("match") or {}
        dd = dict(r.get("paired_bootstrap") or {}) or None
        if dd:
            dd["mcnemar"] = r.get("mcnemar")
        N.diff(f"GpuPh{w}VsDsg", dd, f"gpuws runs/{exp}/paired/paired.json {task}.match (gate - DSG)")
    _tofu_metrics(N, "GpuPhUnion", gpu_runs / U / "tofu-metrics", (("full+dsg", "Dsg"), ("full+union", "Gate")), "gpuws")
    _tofu_metrics(N, "GpuPhTofucal", gpu_runs / "PH-tofucal" / "tofu-metrics",
                  (("full+dsg", "Dsg"), ("full+gate-cusum-tofucal", "Gate")), "gpuws")
    tm = _read(gpu_runs / "PH-tofucal" / "tofu-metrics" / "metrics.json") or {}
    thr = (((tm.get("conditions") or {}).get("full+gate-cusum-tofucal") or {}).get("threshold"))
    N.text("GpuPhTofucalThreshold", f"{thr:.2f}" if _ok(thr) else None, "PH-tofucal tofu-metrics full+gate-cusum-tofucal threshold")


def x1_status_numbers(N, lab):
    """\\resLabXOneStatus = done once the lab-PC X1 TEST replication is complete (final_report completeness 'done'),
    so the interim sentences about it switch by themselves on the next scripts/paper_update.sh."""
    c = ((lab or {}).get("completeness") or {}).get("X1") or {}
    N.text("LabXOneStatus", "done" if str(c.get("status", "")).startswith("done") else "pending",
           "lab summary.json completeness X1 status == done")
    jobs_ = c.get("jobs") or {}
    N.text("LabXOneJobsDone", f"{jobs_.get('DONE', 0)} of {sum(jobs_.values())}" if jobs_ else None,
           "lab summary.json completeness X1 jobs DONE of all")


def fix_numbers(N, lab, gpu, lab_runs: Path, gpu_runs: Path, jobs: Path):
    """Numbers for the fix-first framing: the X1 combined gate (StreamGuard, CUSUM detector) on TEST against DSG on the
    same machine (both machines: Lab*, Gpu*); the X1-suite runs (open-ended benign biology, B6 leakage, TOFU) per
    machine; latency and MT-Bench (gpuws); the conformal threshold (N6); the Cyber retain-corpus fix (A1-test); the
    norm-scaled clamp (A7-scaled). Missing sources print [pending]."""
    _fix_mcq(N, "Lab", lab, "lab summary.json X1 (TEST)")
    _fix_mcq(N, "Gpu", gpu, "gpuws summary.json X1 (TEST)")
    _fix_suite(N, "Lab", lab_runs, "labpc")
    _fix_suite(N, "Gpu", gpu_runs, "gpuws")

    N.comment("---- the final method on gpuws: latency (FP-latency), MT-Bench (mtbench, mode cusum)")
    lat = _read(gpu_runs / "FP-latency" / "latency" / "metrics.json")
    ok = lat and lat.get("protocol") == "interleaved-v2"
    for name, w in (("gate-cusum", "Fix"), ("dsg", "Dsg")):
        vals = [lat["latency"][str(L)][name]["overhead_vs_base"] for L in lat["lengths"]] if ok else []
        N.text(f"GpuLat{w}Overhead", f"{100 * lat['latency']['512'][name]['overhead_vs_base']:.1f}\\%" if ok else None,
               f"FP-latency (interleaved-v2) {name} overhead vs base at 512 tokens, bs 1")
        N.text(f"GpuLat{w}OverheadRange", f"{100 * min(vals):.1f}--{100 * max(vals):.1f}\\%" if vals else None,
               f"FP-latency {name} overhead vs base over {lat['lengths'] if ok else []} tokens")
    mt = _read(jobs / "mtbench" / "summary.json") or {}
    N.ci("GpuMtFix", ((mt.get("scores") or {}).get("cusum") or {}).get("all"), "jobs/mtbench/summary.json scores.cusum.all", digits=2)
    from dsgx.analysis.results_digest import mtbench_paired

    d = mtbench_paired(jobs, ref="dsg").get("cusum") if mt else None
    N.diff("GpuMtFixVsDsg", d and {"diff": d["mean"], "lo": d["lo"], "hi": d["hi"], "n": d["n"], "p": d["p"]},
           "jobs/mtbench judgments (paired bootstrap cusum vs dsg, same question and turn)", digits=2)

    N.comment("---- supporting fixes: conformal threshold (N6), Cyber retain corpus (A1-test), norm-scaled clamp (A7-scaled)")
    import ast
    for gate, w in (("rho", "Rho"), ("window", "Window")):
        m = _read(lab_runs / "N6" / f"conformal-{gate}" / "metrics.json")
        cov = (m or {}).get("coverage")
        cov = ast.literal_eval(cov) if isinstance(cov, str) else cov
        c = (cov or {}).get("0.05") or {}
        N.text(f"LabConformal{w}HeldoutFpr", f3(c["empirical_fpr_heldout"]) if _ok(c.get("empirical_fpr_heldout")) else None,
               f"runs/N6/conformal-{gate} coverage[0.05].empirical_fpr_heldout (target 0.05)")
        N.text(f"LabConformal{w}HeldoutN", (m or {}).get("n_heldout"), f"runs/N6/conformal-{gate} n_heldout")
    cy = {r["condition"]: r for r in (lab or {}).get("experiments", {}).get("A1-test", []) if r["case"] == "cyber"}
    for cond, w in (("base/forget+utility", "Base"), ("dsg-faithful/n100/r90/m1000/forget+utility", "Wiki"),
                    ("dsg-faithful/n200/r95/m1000/chatretain", "Chat")):
        N.ci(f"LabCyber{w}Util", rci(cy.get(cond), "raw_util"), f"lab summary.json A1-test cyber {cond} (full-MMLU utility)")
        N.ci(f"LabCyber{w}Forget", rci(cy.get(cond), "raw_forget"), f"lab summary.json A1-test cyber {cond}")
    # the scaled run is the one with an explicit intervention (clamp_all, multiplier -51650) in its condition label;
    # the other gated run is DSG's default clamp (-500)
    sc = next((p for p in (gpu or {}).get("paired_tests", []) if p["exp"] == "A7-scaled" and p["vs"] == "base"
               and "/clamp_all/" in p["condition"]), None)
    N.diff("GpuScaledVsBaseForget", pt(sc), "gpuws summary.json paired A7-scaled clamp -51650 vs base (forget)")
    N.diff("GpuScaledVsBaseUtil", pt(sc, "utility"), "gpuws summary.json paired A7-scaled clamp -51650 vs base (utility)")


def build(lab_summary: Path, gpu_summary: Path, jobs: Path, lab_runs: Path, gpu_runs: Path, a7_diag: Path) -> Numbers:
    N = Numbers()
    lab, gpu = _read(lab_summary), _read(gpu_summary)
    for s, hw in ((lab, "labpc"), (gpu, "gpuws")):
        if s and s.get("hardware") != hw:
            raise SystemExit(f"{hw} summary.json carries hardware {s.get('hardware')}")
    N.comment(f"lab summary {lab_summary} generated {(lab or {}).get('generated')} interim {(lab or {}).get('interim')}")
    N.comment(f"gpuws summary {gpu_summary} generated {(gpu or {}).get('generated')} interim {(gpu or {}).get('interim')}")
    sanity_numbers(N)
    lab_numbers(N, lab, "lab summary.json")
    lab_extra_numbers(N, lab, "lab summary.json", lab_runs)
    gpu_numbers(N, gpu, "gpuws summary.json", jobs, gpu_runs)
    cross_gpu_numbers(N, lab_runs, gpu_runs)
    a7_diag_numbers(N, a7_diag)
    x1_numbers(N)
    x1_status_numbers(N, lab)
    fix_numbers(N, lab, gpu, lab_runs, gpu_runs, jobs)
    muse_numbers(N, jobs)
    posthoc_numbers(N, gpu, gpu_runs)
    from dsgx.analysis import b7_adaptive

    b7_adaptive.macros(N, lab_runs)  # POST-HOC B7 adaptive attacks (lab PC)
    return N


def render(N: Numbers) -> str:
    head = ["% GENERATED by `python -m dsgx.analysis.paper_numbers` (prep/later-runs) " + now_iso() + ". Do not edit by hand.",
            "% One \\newcommand per number; \\resX = value [95% CI] (n), \\resXVal, \\resXCI, \\resXN; diffs also \\resXP.",
            "% Lab PC (Lab*) and gpuws (Gpu*) numbers are separate hardware baselines; never combine them.",
            f"% {len(N.defined)} macros, {len(N.pending)} pending number(s): " + ", ".join(n for n, _ in N.pending),
            "\\providecommand{\\respending}[1]{\\mbox{\\textcolor{red!70!black}{\\textsf{[pending]}}}}", ""]
    return "\n".join(head + N.lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="generate paper/numbers.tex")
    ap.add_argument("--lab-summary", default=str(PROJECT / "paper_numbers" / "labpc" / "summary.json"))
    ap.add_argument("--gpu-summary", default=str(PROJECT / "paper_numbers" / "gpuws" / "summary.json"))
    ap.add_argument("--jobs", default=str(CLUSTER / "jobs"))
    ap.add_argument("--lab-runs", default=str(paths.runs_dir()))
    ap.add_argument("--gpu-runs", default=str(CLUSTER / "runs"))
    ap.add_argument("--a7-diag", default=str(REPO / "docs" / "a7_gemma3_diag.jsonl"))
    ap.add_argument("--out", default=str(PROJECT / "paper" / "numbers.tex"))
    a = ap.parse_args(argv)
    N = build(Path(a.lab_summary), Path(a.gpu_summary), Path(a.jobs), Path(a.lab_runs), Path(a.gpu_runs), Path(a.a7_diag))
    atomic_write_text(Path(a.out), render(N))
    print(f"wrote {a.out}: {len(N.defined)} macros, {len(N.pending)} pending")
    for n, why in N.pending:
        print(f"  pending {n}: {why}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
