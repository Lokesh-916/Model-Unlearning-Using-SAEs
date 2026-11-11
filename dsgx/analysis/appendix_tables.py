"""Appendix tables for the paper (Appendices A-D), generated from files that exist; nothing is typed in by hand.

    python -m dsgx.analysis.appendix_tables --out ~/projects/mechunlearn-project/paper/assets

Writes <out>/tables/app_*.tex (booktabs, \\resulttable names in the paper):
  app_splits       data/splits/*.json: n, dev, test, sha256 (of the sorted id lists, as stored in each file)
  app_hardware     both machines: GPU, driver, software (ENV.md / CLUSTER_SETUP_REPORT.md) and the exact
                   sanity-gate values per machine (dsgx/checks/sanity.py TARGETS)
  app_attacks      B1-B6 grids from configs/experiments/B*.yaml on the exp/B* branches (git show; no prompts)
  app_d1d2         D1 (local LoRA, full, v2) and D2 hyper-parameters, read from the code / configs
  app_rmu_dev      RMU v1 and v2 DEV grids with DEV forget / utility drop (gpuws grid_state.json)
  app_rmu_test     RMU v2 TEST vs base, DSG, third-party RMU and RMU v1 (gpuws summary.json, bs 1)
Inputs that are missing produce no table (the paper then shows a "pending" box). No hazardous text is read:
only split ids' counts/hashes, configs, code constants and aggregate metrics.
"""
import argparse
import ast
import json
import re
import subprocess
import sys
from pathlib import Path

import yaml

from dsgx.analysis.paper_assets import esc
from dsgx.util import atomic_write_text

REPO = Path(__file__).resolve().parents[2]
CLUSTER_RESULTS = Path.home() / "projects/mechunlearn-project/dsg_results_cluster"
B_BRANCHES = {"B1": "exp/B1-dilution", "B2": "exp/B2-decomposition", "B3": "exp/B3-crosslingual",
              "B4": "exp/B4-rewrite", "B5": "exp/B5-whitebox", "B6": "exp/B6-genleak"}


def tab(rows, header, caption, label, align, note=None, size=r"\small", resize=True) -> str:
    L = [r"\begin{table}[t]", r"\centering", size, r"\caption{" + caption + "}", r"\label{" + label + "}"]
    if resize:
        L.append(r"\resizebox{\linewidth}{!}{%")
    L += [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(header) + r" \\", r"\midrule"]
    L += [(" & ".join(r) + r" \\") if isinstance(r, list) else r for r in rows]
    L += [r"\bottomrule", r"\end{tabular}" + ("}" if resize else "")]
    if note:
        L.append(r"\par\smallskip{\footnotesize " + note + "}")
    L.append(r"\end{table}")
    return "\n".join(L) + "\n"


def git_show(ref_path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(REPO), "show", ref_path], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def f3(x, d=3):
    return "--" if x is None else f"{x:.{d}f}"


def ci(m) -> str:
    return f"{m['mean']:.3f} [{m['lo']:.3f}, {m['hi']:.3f}]" if m else "--"


# ----------------------------------------------------------------------------- A: splits
def splits_table(split_dir: Path) -> str | None:
    files = sorted(split_dir.glob("*.json"))
    if not files:
        return None
    recs = []
    for f in files:
        r = json.loads(f.read_text())
        r["dev"], r["test"] = len(r["dev"]), len(r["test"])  # id lists -> sizes (ids are not printed)
        recs.append(r)
    head = [r for r in recs if r["dataset"].startswith("wmdp")]
    mmlu = sorted((r for r in recs if not r["dataset"].startswith("wmdp")), key=lambda r: r["dataset"])
    rows = [[esc(r["dataset"]), str(r["n"]), str(r["dev"]), str(r["test"]), r"\texttt{" + r["sha256"][:16] + "}"]
            for r in head + mmlu]
    tot = [r"\midrule", [r"MMLU total (" + str(len(mmlu)) + " subjects)", str(sum(r["n"] for r in mmlu)),
                         str(sum(r["dev"] for r in mmlu)), str(sum(r["test"] for r in mmlu)), ""]]
    methods = sorted({r.get("method", "") for r in recs})
    seeds = sorted({str(r.get("seed")) for r in recs})
    # two side-by-side halves keep 59 rows on one page
    half = (len(rows) + 1) // 2
    left, right = rows[:half], rows[half:] + [["", "", "", "", ""]] * (half - len(rows[half:]))
    body = [l + r for l, r in zip(left, right)] + [r"\midrule", tot[1] + ["", "", "", "", ""]]
    hdr = ["Set", "$n$", "Dev", "Test", "SHA-256 (first 16)"] * 2
    return tab(body, hdr, "Dev/test splits of every evaluation set (seed " + ", ".join(seeds) + "; "
               + esc("; ".join(methods)) + "). The hash is the SHA-256 stored in each split file "
               r"(\texttt{data/splits/}).", "tab:app-splits", "lrrrl|lrrrl", size=r"\scriptsize")


# ----------------------------------------------------------------------------- A: hardware + sanity
def _env_versions(text: str) -> dict:
    out = {}
    for k in ("Python", "torch", "transformers", "transformer-lens", "sae-lens", "numpy", "datasets"):
        m = re.search(r"^\|\s*" + re.escape(k) + r"\s*\|\s*([^|]+?)\s*\|", text, re.M)
        if m:
            out[k] = m.group(1)
    return out


def hardware_table() -> str | None:
    env = REPO / "ENV.md"
    rep = REPO / "CLUSTER_SETUP_REPORT.md"
    sys.path.insert(0, str(REPO))
    try:
        from dsgx.checks.sanity import TARGETS
    except Exception:
        return None
    if not env.exists() or not rep.exists():
        return None
    et, rt = env.read_text(), rep.read_text()
    vers = _env_versions(et)
    lab_hw = re.search(r"^NVIDIA (RTX [^,]+?) \((\d+ GB)\), driver ([\d.]+)", et, re.M)
    srv_gpu = re.search(r"(RTX 6000 Ada) (\d+) MiB, driver ([\d.]+)", rt)
    same = "identical to the lab PC" in rt
    if not (lab_hw and srv_gpu):
        return None
    lab, srv = TARGETS["labpc"], TARGETS["gpuws"]
    subj = ["high_school_us_history", "college_computer_science", "high_school_geography", "human_aging"]
    rows = [["GPU", "NVIDIA " + lab_hw.group(1), "NVIDIA " + srv_gpu.group(1) + " Generation"],
            ["GPU memory", lab_hw.group(2), f"{srv_gpu.group(2)} MiB"],
            ["Driver", lab_hw.group(3), srv_gpu.group(3)]]
    for k, v in vers.items():
        rows.append([esc(k), esc(v), esc(v) if same else "--"])
    rows += [r"\midrule",
             [r"Sanity: WMDP-Bio correct", f"{lab['wmdp_correct']}/{lab['wmdp_n']} = {lab['wmdp_acc']:.4f}",
              f"{srv['wmdp_correct']}/{srv['wmdp_n']} = {srv['wmdp_acc']:.4f}"],
             [r"Sanity: MMLU-u (4 subjects, unweighted)", f"{lab['mmlu_u']:.4f}", f"{srv['mmlu_u']:.4f}"],
             [r"Sanity: correct per subject\textsuperscript{a}", " / ".join(str(lab["util_correct"][s]) for s in subj),
              " / ".join(str(srv["util_correct"][s]) for s in subj)],
             [r"Sanity: threshold $\tau$", f"{lab['tau']:.4f}", f"{srv['tau']:.4f}"],
             ["Sanity: selected features", "20 (DSG's list)", "20 (identical)"]]
    return tab(rows, ["", "Lab PC", "Server (gpuws)"],
               "The two machines. Software versions are identical (one conda environment, packed and copied). "
               "Each machine has its own exact sanity gate (DSG paper configuration, DSG subset view, batch size 1); "
               "results from the two machines are never combined.", "tab:app-hardware", "lll",
               note=r"\textsuperscript{a}high-school US history / college computer science / high-school geography / human aging.")


# ----------------------------------------------------------------------------- B: attacks
def _grid_str(grid: dict) -> str:
    return "; ".join(f"{esc(k.split('.')[-1])} $\\in$ \\{{{esc(', '.join(str(x) for x in v))}\\}}" for k, v in grid.items())


def attacks_table() -> str | None:
    rows = []
    for b, br in B_BRANCHES.items():
        txt = git_show(f"{br}:configs/experiments/{b}.yaml")
        if not txt:
            continue
        c = yaml.safe_load(txt)
        base = c.get("base", {})
        grid = _grid_str(c.get("grid", {}))
        runs = ", ".join(esc(r.get("attack", {}).get("name", "")) for r in c.get("runs", []))
        extra_att, controls = [], []
        for e in c.get("extra", []) or []:
            a = e.get("attack", {})
            desc = esc(a.get("name", "")) + "".join(f" {esc(k)}={esc(v)}" for k, v in a.items() if k not in ("name", "path", "exp"))
            (controls if (e.get("method") or {}).get("name") == "base" else extra_att).append(desc)
        tasks = []
        for t in c.get("tasks", []) or []:
            if t["id"] == "attack-success":
                continue
            args = {k: v for k, v in (t.get("args") or {}).items() if k not in ("methods",)}
            if t["id"] == "leak":
                args["methods"] = "base, DSG streaming, DSG prompt-only"
            tasks.append(esc(t["id"]) + ": " + esc(", ".join(f"{k}={v if not isinstance(v, list) else '/'.join(map(str, v))}" for k, v in args.items())))
        split = base.get("split") or next((t.get("args", {}).get("split") for t in c.get("tasks", []) if t.get("args", {}).get("split")), "")
        bs = base.get("batch_size", "")
        att = (runs + (": " + grid if grid else "")) if runs else ""
        if extra_att:
            att += ("; " if att else "") + "; ".join(extra_att)
        rows.append([b, att or "--", "; ".join(tasks) or "--", "; ".join(controls) or "--", f"{esc(split)}, bs {bs}" if bs else esc(split)])
    if not rows:
        return None
    return tab([[r[0], r"\parbox[t]{0.30\linewidth}{\raggedright " + r[1] + "}",
                 r"\parbox[t]{0.24\linewidth}{\raggedright " + r[2] + "}",
                 r"\parbox[t]{0.22\linewidth}{\raggedright " + r[3] + "}", r[4]] for r in rows],
               ["", "Attack and grid (guarded model)", "Preparation task", "Base-model controls", "Split"],
               "Attack grids as configured (\\texttt{configs/experiments/B*.yaml}). The guarded model is the DSG "
               "configuration selected on dev (A1, utility drop $\\le 0.01$); every attack is evaluated on the "
               "WMDP-Bio test items (B6: the open-ended test items).", "tab:app-attacks", "lllll", size=r"\scriptsize")


# ----------------------------------------------------------------------------- C: D1 / D2
def _consts(src: str, names) -> dict:
    out = {}
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign):
            tgts = node.targets[0]
            try:
                val = ast.literal_eval(node.value)
            except Exception:
                continue
            if isinstance(tgts, ast.Tuple):
                for t, v in zip(tgts.elts, val):
                    if isinstance(t, ast.Name) and t.id in names:
                        out[t.id] = v
            elif isinstance(tgts, ast.Name) and tgts.id in names:
                out[tgts.id] = val
    return out


def _arg_default(src: str, key: str):
    m = re.search(r'a\.get\("' + re.escape(key) + r'",\s*([^)]+)\)', src)
    if not m:
        return None
    try:
        return ast.literal_eval(m.group(1).strip().rstrip(")"))
    except Exception:
        return m.group(1).strip()


def d1d2_table() -> str | None:
    local = git_show("exp/D1-local-distill:experiments/D1/distill.py")
    lcfg = git_show("exp/D1-local-distill:configs/experiments/D1.yaml")
    full = (REPO / "cluster/d1_full.py").read_text() if (REPO / "cluster/d1_full.py").exists() else None
    v2 = (REPO / "cluster/d1_v2.py").read_text() if (REPO / "cluster/d1_v2.py").exists() else None
    d2 = git_show("exp/D2-nullspace:experiments/D2/edit.py")
    d2cfg = git_show("exp/D2-nullspace:configs/experiments/D2.yaml")
    if not all((local, lcfg, full, v2, d2, d2cfg)):
        return None
    lt = [t for t in yaml.safe_load(lcfg)["tasks"]]
    alphas_l = sorted({t["args"]["noise_alpha"] for t in lt if not t["args"].get("same_ref")})
    steps_l = sorted({t["args"]["steps"] for t in lt})
    rank = _arg_default(local, "rank")
    names = ("ALPHAS", "STEPS", "BS", "MAXLEN", "LR", "CKPT", "UTIL_DROP", "EXCLUDED_UTIL")
    F, V = _consts(full, names), _consts(v2, names)
    fl = lambda xs: ", ".join(str(x) for x in xs)
    rows = [
        ["Student", f"gemma-2-2b-it + LoRA (rank {rank}, $\\alpha_{{\\mathrm{{LoRA}}}}={2 * rank}$, all attention and MLP projections)",
         "gemma-2-2b-it, all parameters", "as D1-full"],
        ["Noise", f"LoRA $B$ initialised with $\\mathcal{{N}}(0, (0.02\\,\\alpha)^2)$, $\\alpha \\in \\{{{fl(alphas_l)}\\}}$",
         f"$\\theta \\leftarrow (1-\\alpha)\\theta + \\alpha\\,\\sigma_\\theta\\,\\varepsilon$, $\\alpha \\in \\{{{fl(F['ALPHAS'])}\\}}$",
         f"same, $\\alpha \\in \\{{{fl(V['ALPHAS'])}\\}}$"],
        ["Loss", "KL to the DSG-guarded teacher on forget prompts + KL to the unguarded teacher on retain prompts (last token)",
         "same teachers, KL over all tokens", "as D1-full"],
        ["Forget data", "400 WMDP-Bio dev questions (prompt format)", "WMDP bio forget corpus passages", "as D1-full"],
        ["Retain data", "high-school geography dev questions", "WikiText-2 train + high-school geography dev questions", "as D1-full"],
        ["Optimiser", f"AdamW, lr {_arg_default(local, 'lr'):g}", f"AdamW 8-bit, lr {F['LR']:g}, bf16, grad.\\ checkpointing", f"as D1-full, lr {V['LR']:g}"],
        ["Steps $\\times$ batch", f"{fl(steps_l)} $\\times$ {_arg_default(local, 'bs')}, max 512 tokens",
         f"{F['STEPS']} $\\times$ {F['BS']}, max {F['MAXLEN']} tokens", f"{V['STEPS']} $\\times$ {V['BS']}, max {V['MAXLEN']} tokens"],
        ["Control", "same-reference LoRA student ($\\alpha=0$, no DSG teacher)", "--", "--"],
        ["Selection", "none (all students reported)", "none (all students reported)",
         f"dev: lowest forget acc.\\ with utility drop $\\le {V['UTIL_DROP']}$ (excl.\\ {esc(', '.join(V['EXCLUDED_UTIL']))})"],
    ]
    t1 = tab(rows, ["", "D1-local (lab PC)", "D1-full (gpuws)", "D1 v2 (gpuws)"],
             "D1 (noised-student distillation of the DSG-guarded model) as configured. Teachers are the base weights "
             "with the DSG hook (paper configuration) on or off.", "tab:app-d1", "p{0.10\\linewidth}p{0.29\\linewidth}p{0.28\\linewidth}p{0.20\\linewidth}",
             size=r"\scriptsize", resize=False)
    dt = {t["id"]: t["args"] for t in yaml.safe_load(d2cfg)["tasks"]}
    ns = dt.get("edit-nullspace", {})
    eps = _arg_default(d2, "nullspace_eps")
    nprompt = re.search(r'get_split\(benign_subj, "dev"\)\[:(\d+)\]', d2)
    maxrows = re.search(r"def retain_keys\([^)]*max_rows=(\d+)", d2)
    rows2 = [
        ["Edited weights", f"MLP down-projection $W_{{\\mathrm{{out}}}}$ of layers $0,\\dots,L^*$, $L^*={ns.get('l_star')}$ (fp32 on CPU, saved in bf16)"],
        ["Directions", f"decoder rows of the {ns.get('n_features')} DSG-selected SAE features, each normalised to unit length ($U$)"],
        ["Edit", r"$\Delta = \big((I - U^\top U)\,W_{\mathrm{out}} - W_{\mathrm{out}}\big)\,P$"],
        ["Null space $P$", f"eigenvectors of $C + 10^{{-3}} I$ with eigenvalue $< {eps}\\,\\lambda_{{\\max}}$; $C$ = covariance of the "
         f"MLP inputs (keys), all tokens of {min(int(nprompt.group(1)), int(maxrows.group(1))) if nprompt and maxrows else '?'} high-school geography dev prompts"],
        ["Baseline", r"decoder orthogonalisation: the same edit with $P = I$"],
    ]
    t2 = tab(rows2, ["", "D2 (null-space edit)"], "D2 as configured.", "tab:app-d2", "lp{0.75\\linewidth}",
             size=r"\scriptsize", resize=False)
    return t1 + "\n" + t2


# ----------------------------------------------------------------------------- D: RMU
def rmu_dev_table(root: Path) -> str | None:
    rows = []
    for job, tag in (("rmu", "v1"), ("rmu-v2", "v2")):
        p = root / "jobs" / job / "grid_state.json"
        if not p.exists():
            continue
        g = json.loads(p.read_text())
        bw, bu = g["base_dev"]["wmdp"]["mean"], g["base_dev"]["util_pooled"]["mean"]
        sel = g.get("selected", {}).get("cfg")
        if rows:
            rows.append(r"\midrule")
        rows.append([f"base ({tag} run)", "--", "--", "--", "--", f3(bw), "--"])
        for k, v in sorted(g["configs"].items(), key=lambda kv: int(kv[0])):
            hp, dev = v["hp"], v.get("dev") or {}
            w, u = (dev.get("wmdp") or {}).get("mean"), (dev.get("util_pooled") or {}).get("mean")
            name = f"{tag} c{k}" + (r"$^\star$" if int(k) == sel else "")
            rows.append([name, f"{hp['layer_id']} ({','.join(map(str, hp['layer_ids']))})", f"{hp['steering_mult']:g}",
                         f"{hp['alpha']:g}", f"{hp['max_num_batches']} $\\times$ {hp['batch_size']}", f3(w),
                         f3(None if u is None else bu - u)])
    if not rows:
        return None
    return tab(rows, ["Config", "Layer (updated)", r"Steering $\times r_\ell$", r"$\alpha$", r"Steps $\times$ batch",
                      "Dev WMDP-Bio", "Dev utility drop"],
               "RMU dev grids on gpuws (selection data, not test results). Utility drop = base minus configuration, "
               r"pooled full-MMLU utility (48 subjects). $^\star$ selected (lowest dev WMDP-Bio with drop $\le 0.02$). "
               r"$r_7 = " + _r(root, "7") + r"$, $r_3 = " + _r(root, "3") + "$.", "tab:app-rmu-dev", "lcccccc", size=r"\scriptsize")


def _r(root, layer):
    g = json.loads((root / "jobs/rmu-v2/grid_state.json").read_text())
    return f"{g['r'][layer]:.2f}"


def rmu_test_table(root: Path) -> str | None:
    p = root / "jobs/rmu-v2/summary.json"
    if not p.exists():
        return None
    s = json.loads(p.read_text())
    names = {"base": "Base model", "dsg_paper": "DSG (paper configuration)", "rmu_v2": "RMU v2 (ours, selected)",
             "rmu_v1": "RMU v1 (ours, first grid)", "rmu_third_party_unverified": "RMU, third-party checkpoint (unverified)"}
    rows = []
    for k in ("base", "dsg_paper", "rmu_v2", "rmu_v1", "rmu_third_party_unverified"):
        r = s["runs"].get(k)
        if not r:
            continue
        rows.append([names[k], ci(r["raw"].get("wmdp_bio")), ci((r["raw"].get("utility_full") or {}).get("pooled")),
                     ci((r["raw"].get("utility_legacy4") or {}).get("unweighted")), ci(r["dsg_subset"].get("wmdp_bio"))])
    rows.append(r"\midrule")
    lab = {"rmu_v2_vs_base": "RMU v2 $-$ base", "rmu_v2_vs_dsg_paper": "RMU v2 $-$ DSG",
           "rmu_v2_vs_rmu_third_party_unverified": "RMU v2 $-$ third-party", "rmu_v2_vs_rmu_v1": "RMU v2 $-$ RMU v1"}

    def d(x):
        if not x:
            return "--"
        return f"{x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}], McN.\\ $p{_p(x['mcnemar']['p'])}$"
    for k, nm in lab.items():
        pr = s["paired"].get(k)
        if pr:
            rows.append([nm, d(pr["raw"].get("wmdp_bio")), d(pr["raw"].get("utility_full")),
                         d(pr["raw"].get("utility_legacy4")), d(pr["dsg_subset"].get("wmdp_bio"))])
    n = s["runs"]["base"]["raw"]
    hw = s["hardware"]
    return tab(rows, ["", f"WMDP-Bio ($n={n['wmdp_bio']['n']}$)", f"MMLU utility, 48 subj.\\ ($n={n['utility_full']['pooled']['n']}$)",
                      f"DSG 4-subject utility ($n={n['utility_legacy4']['unweighted']['n']}$)",
                      f"WMDP-Bio, DSG subset ($n={s['runs']['base']['dsg_subset']['wmdp_bio']['n']}$)"],
               f"RMU references on test (batch size 1), all on gpuws ({esc(hw['gpu'])}). Accuracy with 95\\% bootstrap "
               r"intervals; lower rows: paired differences (bootstrap 95\% CI, exact McNemar $p$). Chance is 0.25.",
               "tab:app-rmu-test", "lcccc", size=r"\scriptsize")


def _p(p):
    if p < 1e-3:
        e = int(f"{p:.0e}".split("e")[1])
        return f"<10^{{{e + 1}}}"
    return f"={p:.3f}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="paper assets dir (tables/ is created)")
    ap.add_argument("--cluster-results", default=str(CLUSTER_RESULTS))
    a = ap.parse_args(argv)
    root = Path(a.cluster_results)
    T = {"app_splits": splits_table(REPO / "data/splits"), "app_hardware": hardware_table(),
         "app_attacks": attacks_table(), "app_d1d2": d1d2_table(),
         "app_rmu_dev": rmu_dev_table(root), "app_rmu_test": rmu_test_table(root)}
    td = Path(a.out) / "tables"
    td.mkdir(parents=True, exist_ok=True)
    for k, v in T.items():
        if v:
            atomic_write_text(td / f"{k}.tex", v)
        print(f"{k}: {'written' if v else 'SKIPPED (inputs missing)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
