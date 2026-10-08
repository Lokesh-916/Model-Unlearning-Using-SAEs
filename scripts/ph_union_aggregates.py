"""python -I scripts/ph_union_aggregates.py > docs/PH_UNION_AGGREGATES.md
PH-union / PH-tofucal aggregates (gpuws, POST-HOC EXPLORATORY). Metrics and ids only; reads no text columns."""
import glob, json, os, sys
import numpy as np
import pyarrow.parquet as pq

R = os.path.expanduser("~/projects/mechunlearn-project/dsg_results_cluster/runs")
COLS = ["item_id", "dataset", "correct", "gate_fired"]
rng = np.random.default_rng(0)


def runs(exp, method, ds_label):
    out = {}
    for d in glob.glob(f"{R}/{exp}/{exp}__{method}__*__bio-{ds_label}*__test__s*__*"):
        if not os.path.exists(f"{d}/DONE"):
            continue
        c = json.load(open(f"{d}/config.json"))["config"]
        if c.get("dataset_label") not in (ds_label, f"{ds_label}-forget", f"{ds_label}-hardneg"):
            continue
        a = dict(c.get("attack") or {"name": "none"}); a.pop("_exp_id", None)
        name = a.pop("name")
        if c["dataset_label"].endswith("hardneg"):
            name = "hardneg"
        key = name + ("" if not a else " " + ",".join(f"{k}={v}" for k, v in sorted(a.items()) if k not in ("source",)))
        out.setdefault(key, {})[c["seed"]] = d
    return out


def load(d):
    return pq.read_table(f"{d}/items.parquet", columns=COLS).to_pandas()


def boot_mean(x, B=4000):
    x = np.asarray(x, float)
    bs = x[rng.integers(0, len(x), (B, len(x)))].mean(1)
    return x.mean(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def cluster_boot(df, col, B=4000):
    """mean over (seed, item) rows, CI by resampling item ids (all seeds of an item move together)."""
    g = df.groupby("item_id")[col].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(float), g["count"].to_numpy(float)
    idx = rng.integers(0, len(g), (B, len(g)))
    bs = s[idx].sum(1) / c[idx].sum(1)
    return s.sum() / c.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5), len(g)


def stack(seeds, sel):
    fr = []
    for s, d in seeds.items():
        x = load(d); x = x[sel(x)].copy(); x["seed"] = s; fr.append(x)
    return __import__("pandas").concat(fr)


def fmt(m, lo, hi):
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def paired(A, B, col):
    m = A.merge(B, on=["seed", "item_id"], suffixes=("_a", "_b"))
    m["d"] = m[f"{col}_a"].astype(float) - m[f"{col}_b"].astype(float)
    v, lo, hi, n = cluster_boot(m, "d")
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}] (n {n}×{m.seed.nunique()})"


U = runs("PH-union", "gated", "union")
D = runs("PH-union", "dsg-faithful", "dsg")
S = runs("X1", "gated", "combined")
forget = lambda x: x.dataset == "wmdp-bio"
benign = lambda x: x.dataset != "wmdp-bio"
allrows = lambda x: x.dataset == x.dataset

print("# PH-union vs DSG vs StreamGuard (gpuws; POST-HOC, EXPLORATORY; not claim inputs)\n")
print("WMDP-Bio TEST accuracy under attack (lower = better forgetting). Value = mean over seeds × items, 95% CI by item")
print("cluster bootstrap; n = items × seeds. StreamGuard = X1 combined (CUSUM) on gpuws, same items/attacks/seeds.\n")
print("| attack | union | DSG | StreamGuard | union − DSG | union − SG |\n|---|---|---|---|---|---|")
keys = sorted(k for k in U if k not in ("none", "hardneg"))
for k in ["none"] + keys:
    if k not in D or k not in S:
        print(f"| {k} | missing in {'DSG' if k not in D else 'SG'} |"); continue
    a, b, c = (stack(X[k], forget) for X in (U, D, S))
    ua, da, sa = (cluster_boot(x, "correct") for x in (a, b, c))
    n = f"{ua[3]}×{a.seed.nunique()}"
    print(f"| {k} | {fmt(*ua[:3])} | {fmt(*da[:3])} | {fmt(*sa[:3])} | {paired(a, b, 'correct')} | {paired(a, c, 'correct')} |  ")
print()
print("Clean TEST (no attack): MMLU utility (all non-WMDP items, pooled), benign FPR = gate fire rate on those items,")
print("forget = WMDP-Bio accuracy; hard negatives = A3 hazard-adjacent MCQ.\n")
print("| metric | union | DSG | StreamGuard | union − DSG | union − SG |\n|---|---|---|---|---|---|")
for lab, k, sel, col in (("MMLU utility", "none", benign, "correct"), ("benign FPR (MMLU)", "none", benign, "gate_fired"),
                         ("WMDP fire rate", "none", forget, "gate_fired"),
                         ("hard-neg accuracy", "hardneg", allrows, "correct"), ("hard-neg fire rate", "hardneg", allrows, "gate_fired")):
    a, b, c = (stack(X[k], sel) for X in (U, D, S))
    ua, da, sa = (cluster_boot(x, col) for x in (a, b, c))
    print(f"| {lab} | {fmt(*ua[:3])} | {fmt(*da[:3])} | {fmt(*sa[:3])} | {paired(a, b, col)} | {paired(a, c, col)} |")
for k, nm in (("none", "clean"), ("hardneg", "hard-neg")):
    for lab, X in (("union", U), ("DSG", D), ("SG", S)):
        x = stack(X[k], allrows)
        print(f"  {nm} {lab}: seeds {x.seed.nunique()}, items/seed {len(x)//x.seed.nunique()}"
              + (f" (MMLU {int(benign(x).sum())//x.seed.nunique()}, WMDP {int(forget(x).sum())//x.seed.nunique()})" if k == "none" else ""))

print("\nOpen-ended streaming generation (same items, same model; CIs from the run's item bootstrap, n items).")
pu = json.load(open(f"{R}/PH-union/paired/paired.json"))
ps = json.load(open(f"{R}/X1-suite/paired/paired.json"))
pt = json.load(open(f"{R}/PH-tofucal/paired/paired.json"))


def mfile(exp, task, tag):
    return json.load(open(f"{R}/{exp}/{task}__{tag}/metrics.json"))


def ci(d):
    return f"{d['mean']:.3f} [{d['lo']:.3f}, {d['hi']:.3f}]"


def pd_(p, task, met):
    x = p[task][met]["paired_bootstrap"]
    return f"{x['diff']:+.3f} [{x['lo']:+.3f}, {x['hi']:+.3f}] p {x['p']:.3g}"


print("\n| task | metric | n | union | DSG | StreamGuard | StreamGuard-tofucal | union − DSG | SG − DSG |\n|---|---|---|---|---|---|---|---|---|")
for task, mets in (("benign-open", ("match", "gate_fired", "gibberish")), ("leak", ("match", "gibberish", "gate_fired")),
                   ("tofu-qa-forget", ("match", "rougeL_recall", "gate_fired")), ("tofu-qa-retain", ("match", "rougeL_recall", "gate_fired"))):
    mu, md, ms = mfile("PH-union", task, "union-stream"), mfile("PH-union", task, "dsg-faithful-stream"), mfile("X1-suite", task, "cusum-stream")
    mt = mfile("PH-tofucal", task, "cusum-tofucal-stream") if os.path.exists(f"{R}/PH-tofucal/{task}__cusum-tofucal-stream") else None
    for met in mets:
        if met not in mu:
            continue
        print(f"| {task} | {met} | {mu[met]['n']} | {ci(mu[met])} | {ci(md[met])} | {ci(ms[met])} | {ci(mt[met]) if mt else 'not run'} | "
              f"{pd_(pu, task, met)} | {pd_(ps, task, met)} |")
print("\nleak = B6 open-ended WMDP leakage (match = hazardous answer recovered, lower = better); benign-open = benign biology")
print("questions (match higher = better); tofu-qa-forget match = forget answer leaked (lower = better).")

print("\nTOFU v3 metrics (tofu-metrics, n 400 per split):\n")
print("| condition | forget quality KS p | model utility | forget truth ratio | forget answer prob |\n|---|---|---|---|---|")
for exp, conds in (("PH-union", ("retain-model", "full", "full+dsg", "full+union")), ("PH-tofucal", ("full+gate-cusum-tofucal",)),
                   ("X1-suite", ("full+gate-cusum", "full+cusum", "full+gate"))):
    m = json.load(open(f"{R}/{exp}/tofu-metrics/metrics.json"))["conditions"]
    for c in conds:
        if c not in m:
            continue
        x = m[c]; fq = x.get("forget_quality_ks_p")
        print(f"| {exp}:{c} | {'—' if fq is None else f'{fq:.2e}'} | {x['model_utility']:.3f} | {ci(x['truth_ratio_forget'])} | "
              f"{ci(x['forget']['answer_prob'])} |")
    if exp == "X1-suite":
        print(f"(X1-suite conditions: {sorted(m)})")
thr = json.load(open(f"{R}/PH-union/tofu-metrics/metrics.json"))
print("\nunion TOFU threshold:", {k: v for k, v in thr.items() if k != "conditions" and not isinstance(v, (dict, list))})
