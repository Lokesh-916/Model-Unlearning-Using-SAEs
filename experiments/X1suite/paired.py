"""X1-suite paired tests: the X1 combined gate (CUSUM detector) vs DSG on identical items, same machine.

  benign-open  open-ended benign biology (A3's mmlu-open set), streaming gate: gate_fired (over-blocking), gibberish,
               match, token_f1
  leak         WMDP-Bio-Open (B6), streaming gate: match (leakage), gibberish, gate_fired, token_f1
  tofu-qa-*    TOFU forget10 / retain QA on the lab A2 full model (gate and DSG on TOFU features): ROUGE-L recall, match,
               gibberish, gate_fired
  hardneg      X1's own MCQ runs on A3's five biology-adjacent subjects (combined-hardneg vs dsg-hardneg, same seed,
               every seed): correct (accuracy) and gate_fired (over-blocking); read from runs/X1 (TEST, same machine)
Binary outcomes: exact McNemar + paired bootstrap; continuous: paired bootstrap (dsgx.eval.stats). Differences are
gate - DSG. Scores only; no text is read (generations stay in $DSG_PRIVATE)."""
import re

import numpy as np

OPEN_TASKS = {"benign-open": ("gate_fired", "gibberish", "match", "token_f1"),
              "leak": ("match", "gibberish", "gate_fired", "token_f1"),
              "tofu-qa-forget": ("rougeL_recall", "match", "gibberish", "gate_fired"),
              "tofu-qa-retain": ("rougeL_recall", "match", "gibberish", "gate_fired")}
BINARY = {"gate_fired", "gibberish", "match", "correct"}


def _compare(stats, a, b, cols):
    out = {}
    for c in cols:
        x, y = a[c].astype(float).values, b[c].astype(float).values
        r = {"gate": float(x.mean()), "dsg": float(y.mean()), "paired_bootstrap": stats.paired_bootstrap(x, y)}
        if c in BINARY:
            r["mcnemar"] = stats.mcnemar(x, y)
        out[c] = r
    return out


def task(ctx):
    import pandas as pd

    from dsgx.eval import stats
    from dsgx.util import atomic_write_json

    a = ctx.args
    gate_tag, dsg_tag = a.get("gate_tag", "cusum-stream"), a.get("dsg_tag", "dsg-faithful-stream")
    res = {"gate": gate_tag, "dsg": dsg_tag, "direction": "gate - dsg"}
    for t, cols in OPEN_TASKS.items():
        ga, da = ctx.run_dir(None).parent / f"{t}__{gate_tag}", ctx.run_dir(None).parent / f"{t}__{dsg_tag}"
        if not ((ga / "items.parquet").exists() and (da / "items.parquet").exists()):
            res[t] = {"missing": True}
            continue
        g = pd.read_parquet(ga / "items.parquet").set_index("item_id")
        d = pd.read_parquet(da / "items.parquet").set_index("item_id")
        idx = g.index.intersection(d.index)
        res[t] = {"n": int(len(idx)), **_compare(stats, g.loc[idx], d.loc[idx], cols)}

    # gate_label: the gate's MCQ dataset_label prefix (X1 "combined"; PH-union "union"); hardneg: false skips the MCQ part
    glabel = a.get("gate_label", "combined")
    x1 = ctx.results_of(a.get("x1_exp", "X1"))
    pat = re.compile(rf"__bio-({re.escape(glabel)}|dsg)-hardneg__test__s(\d+)__")
    by = {}
    for p in (sorted(x1.glob("*-hardneg__test__s*")) if a.get("hardneg", True) else []):
        m = pat.search(p.name)
        if m and (p / "DONE").exists():
            by.setdefault(int(m.group(2)), {})[m.group(1)] = p
    seeds = {}
    for s, pair in sorted(by.items()):
        if set(pair) != {glabel, "dsg"}:
            continue
        g = pd.read_parquet(pair[glabel] / "items.parquet").set_index("item_id")
        d = pd.read_parquet(pair["dsg"] / "items.parquet").set_index("item_id")
        idx = g.index.intersection(d.index)
        seeds[s] = {"n": int(len(idx)), "runs": [pair[glabel].name, pair["dsg"].name],
                    **_compare(stats, g.loc[idx], d.loc[idx], ("correct", "gate_fired"))}
    res["hardneg"] = {"seeds": seeds, "n_seeds": len(seeds)}
    if seeds:
        res["hardneg"]["mean_over_seeds"] = {c: {"diff": float(np.mean([v[c]["paired_bootstrap"]["diff"] for v in seeds.values()])),
                                                 "significant_seeds": int(sum(v[c]["paired_bootstrap"]["p"] < 0.05 and
                                                                              v[c]["mcnemar"]["p"] < 0.05 for v in seeds.values()))}
                                             for c in ("correct", "gate_fired")}
    d = ctx.run_dir()
    atomic_write_json(d / "paired.json", res)
    ctx.write_metrics(res)
    ctx.finish({"view": "x1-suite-paired", "forget": None})
    return {t: res[t].get("n") if isinstance(res[t], dict) else None for t in OPEN_TASKS}
