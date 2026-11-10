"""E1 "three-number audit" (R, F, D): does a gate's accuracy follow from how often it fires?

CPU only, from existing X1 TEST per-item logs (items.parquet: item_id, dataset, correct, gate_fired); one machine per
table (hardware label from each run's config.json; mixing raises). Methods: DSG (dsg-faithful) and StreamGuard (X1
gated, gate type cusum). Item sets: WMDP-Bio TEST forget items (clean + every attack variant), MMLU utility (the
non-WMDP part of the clean runs: full MMLU minus the 9 bio-adjacent subjects, 48 subjects) and benign biology (the A3 hard-negative MCQ, X1 `*-hardneg` runs).

Per (method, condition, item set), pooled over the 5 seeds, with a 95 % item-cluster bootstrap CI and n:
  R / F   gate fire rate on forget / benign items;
  a_fire, u_fire   gated-run accuracy on the fired rows;
  a_miss, u_base   BASE-model accuracy (X1 base run, same item ids and attack) on the rows the gate did not fire on.
Prediction  acc_pred = (1 - R) * a_miss + R * a_fire  (same for utility). Taking a_miss / u_base from the base run makes
the check non-trivial: it holds iff the gate leaves the rows it does not fire on untouched. (With a_miss from the gated
run itself the formula is an identity; its largest |error| is reported as an arithmetic check.)

Q1 (forget, clean): "misses alone" = acc_misses = (1 - R) * a_miss + R * 0.25, i.e. fired items at chance.
Q2 (utility, StreamGuard vs DSG, paired by seed and item): the observed difference split by item-level fire pattern
(fired by StreamGuard only / DSG only / both / neither); "extra false fires alone" means the both + neither part is 0
and the formula prediction (F_SG - F_DSG) * (u_fire - u_base) (DSG's per-fire cost) lies in the observed CI.

    python -m dsgx.analysis.rfd_audit [--out docs/E1_RFD_AUDIT.md] [--n-boot 2000]
"""
import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

CHANCE = 0.25
FORGET = ("wmdp-bio",)
CLUSTER = Path(os.environ.get("DSG_RESULTS_CLUSTER", Path.home() / "projects/mechunlearn-project/dsg_results_cluster"))
COLS = ["item_id", "dataset", "correct", "gate_fired"]


def _method(cfg: dict) -> str | None:
    m = cfg["method"]
    if m["name"] == "base":
        return "base"
    if m["name"] == "dsg-faithful":
        return "DSG"
    if m["name"] == "gated" and (m.get("gate") or {}).get("type") == "cusum":
        return "StreamGuard"
    return None


def _attack_key(cfg: dict, items) -> str:
    a = cfg.get("attack") or {}
    name = a.get("name", "none") if isinstance(a, dict) else str(a)
    params = sorted(set(items["attack_params"].astype(str))) if "attack_params" in items else ["{}"]
    p = params[0] if len(params) == 1 else "|".join(params)
    if p in ("{}", "", "None"):
        return name
    kv = json.loads(p)
    return name + " " + " ".join(f"{k}={v}" for k, v in sorted(kv.items()) if not isinstance(v, (dict, list)))


def load_runs(runs_root: Path, exp: str = "X1"):
    """-> (runs, skipped). runs: dicts with hardware, method, seed, kind (clean/hardneg/attack), attack, items."""
    import pandas as pd

    runs, skipped = [], []
    for d in sorted((Path(runs_root) / exp).glob(f"{exp}__*")):
        if not (d / "DONE").exists() or not (d / "items.parquet").exists():
            skipped.append((d.name, "not DONE or no items.parquet"))
            continue
        meta = json.loads((d / "config.json").read_text())
        cfg = meta["config"]
        m = _method(cfg)
        if m is None or cfg.get("split") != "test":
            continue
        it = pd.read_parquet(d / "items.parquet")
        need = COLS if m != "base" else COLS[:3]
        missing = [c for c in need if c not in it.columns or (c != "gate_fired" and it[c].isna().any())]
        if m != "base" and "gate_fired" in it.columns and it["gate_fired"].isna().any():
            missing.append("gate_fired (NaN rows)")
        if missing:
            skipped.append((d.name, "missing per-item field: " + ", ".join(missing)))
            continue
        label = str(cfg.get("dataset_label") or "")
        kind = "hardneg" if label.endswith("-hardneg") else ("attack" if label.endswith("-forget") else "clean")
        attack = _attack_key(cfg, it)
        if kind == "attack" and attack == "none":
            kind = "clean"
        runs.append({"name": d.name, "hardware": (meta.get("hardware") or {}).get("label", "unknown"),
                     "method": m, "seed": int(meta.get("seed", cfg.get("seed", 0)) or 0), "kind": kind,
                     "attack": attack if kind == "attack" else "none",
                     "items": it[[c for c in COLS if c in it.columns]].copy()})
    return runs, skipped


def _subset(items, which: str):
    f = items["dataset"].isin(FORGET)
    return items[f] if which == "forget" else items[~f & ~items["dataset"].str.startswith("wmdp")]


def _pool(method_runs, base_items):
    """Stack seed runs (item_id, seed, correct, fired) and join the base model's correctness by item_id."""
    import pandas as pd

    rows = []
    for r in method_runs:
        x = r["items"][["item_id", "correct", "gate_fired"]].copy()
        x["seed"] = r["seed"]
        rows.append(x)
    df = pd.concat(rows, ignore_index=True)
    df["correct"] = df["correct"].astype(bool)
    df["fired"] = df["gate_fired"].astype(bool)
    b = base_items.drop_duplicates("item_id").set_index("item_id")["correct"].astype(bool)
    df = df[df["item_id"].isin(b.index)]
    df["base_correct"] = df["item_id"].map(b).astype(bool)
    return df


def _item_counts(df):
    """Per item id: rows, fired, correct&fired, correct&missed, base_correct&missed (summed over seeds)."""
    g = df.assign(cf=df.correct & df.fired, cm=df.correct & ~df.fired, bm=df.base_correct & ~df.fired,
                  one=1).groupby("item_id")[["one", "fired", "cf", "cm", "bm", "correct"]].sum()
    return g.to_numpy(dtype=float)  # columns: rows, fired, cf, cm, bm, correct


def _stats(c):
    """c: (..., 6) column sums -> dict of arrays (the audit numbers)."""
    rows, fired, cf, cm, bm, corr = (c[..., i] for i in range(6))
    miss = rows - fired
    with np.errstate(invalid="ignore", divide="ignore"):
        R = fired / rows
        a_fire = cf / fired
        a_miss_run = cm / miss
        a_miss = bm / miss
        obs = corr / rows
    af = np.where(fired > 0, a_fire, 0.0)
    pred = (1 - R) * np.where(miss > 0, a_miss, 0.0) + R * af
    ident = (1 - R) * np.where(miss > 0, a_miss_run, 0.0) + R * af
    misses_only = (1 - R) * np.where(miss > 0, a_miss, 0.0) + R * CHANCE
    return {"R": R, "a_fire": a_fire, "a_miss": a_miss, "obs": obs, "pred": pred, "ident_err": np.abs(ident - obs),
            "misses_only": misses_only, "pred_minus_obs": pred - obs}


def _boot_weights(n: int, n_boot: int, seed: int):
    rng = np.random.default_rng(seed)
    return rng.multinomial(n, np.full(n, 1.0 / n), size=n_boot).astype(float)


def _ci(v):
    v = v[np.isfinite(v)]
    return (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if len(v) else (float("nan"),) * 2


def audit(df, n_boot: int = 2000, seed: int = 0) -> dict:
    """Point estimates + item-cluster bootstrap CIs of the audit numbers for one pooled (method, condition, set)."""
    c = _item_counts(df)
    point = {k: float(v) for k, v in _stats(c.sum(0)).items()}
    w = _boot_weights(len(c), n_boot, seed)
    boot = _stats(w @ c)
    out = {"n_items": int(len(c)), "n_rows": int(c[:, 0].sum()), "n_seeds": int(df["seed"].nunique())}
    for k, v in point.items():
        lo, hi = _ci(boot[k])
        out[k] = {"mean": v, "lo": lo, "hi": hi}
    out["pred_in_ci"] = out["obs"]["lo"] <= point["pred"] <= out["obs"]["hi"]
    out["misses_only_in_ci"] = out["obs"]["lo"] <= point["misses_only"] <= out["obs"]["hi"]
    return out


def paired_utility(sg_df, dsg_df, n_boot: int = 2000, seed: int = 0) -> dict:
    """StreamGuard - DSG accuracy on the same (seed, item) rows, split by fire pattern; + the formula prediction."""
    m = sg_df.merge(dsg_df, on=["item_id", "seed"], suffixes=("_sg", "_d"))
    d = m["correct_sg"].astype(float) - m["correct_d"].astype(float)
    cats = {"sg_only": m.fired_sg & ~m.fired_d, "dsg_only": ~m.fired_sg & m.fired_d,
            "both": m.fired_sg & m.fired_d, "neither": ~m.fired_sg & ~m.fired_d}
    # per item: rows, sum diff per category, fired_sg, fired_dsg, dsg correct&fired, base correct&missed(dsg)
    cols = {"one": 1.0, "diff": d}
    for k, mask in cats.items():
        cols[k] = d * mask
        cols["n_" + k] = mask.astype(float)
    cols.update({"f_sg": m.fired_sg.astype(float), "f_d": m.fired_d.astype(float),
                 "cf_d": (m.correct_d & m.fired_d).astype(float), "bm_d": (m.base_correct_d & ~m.fired_d).astype(float)})
    g = m[["item_id"]].assign(**cols).groupby("item_id").sum().to_numpy(dtype=float)
    keys = list(cols)

    def st(c):
        col = {k: c[..., i] for i, k in enumerate(keys)}
        rows = col["one"]
        with np.errstate(invalid="ignore", divide="ignore"):
            u_fire = col["cf_d"] / col["f_d"]
            u_base = col["bm_d"] / (rows - col["f_d"])
            res = {"obs_diff": col["diff"] / rows, "F_sg": col["f_sg"] / rows, "F_dsg": col["f_d"] / rows,
                   "u_fire_dsg": u_fire, "u_base_dsg": u_base}
            for k in cats:
                res["part_" + k] = col[k] / rows
                res["frac_" + k] = col["n_" + k] / rows
        res["part_concordant"] = res["part_both"] + res["part_neither"]
        res["pred_diff"] = (res["F_sg"] - res["F_dsg"]) * (res["u_fire_dsg"] - res["u_base_dsg"])
        return res

    point = {k: float(v) for k, v in st(g.sum(0)).items()}
    boot = st(_boot_weights(len(g), n_boot, seed) @ g)
    out = {"n_items": int(len(g)), "n_rows": int(g[:, 0].sum()), "n_seeds": int(m["seed"].nunique())}
    for k, v in point.items():
        lo, hi = _ci(boot[k])
        out[k] = {"mean": v, "lo": lo, "hi": hi}
    out["pred_in_ci"] = out["obs_diff"]["lo"] <= point["pred_diff"] <= out["obs_diff"]["hi"]
    out["concordant_zero"] = out["part_concordant"]["lo"] <= 0 <= out["part_concordant"]["hi"]
    return out


def machine_audit(runs, n_boot: int = 2000) -> dict:
    hw = {r["hardware"] for r in runs}
    if len(hw) != 1:
        raise ValueError(f"rfd_audit: one machine per table, got {sorted(hw)}")
    base = defaultdict(list)
    meth = defaultdict(list)
    for r in runs:
        if r["method"] == "base":
            base[(r["kind"], r["attack"])].append(r)
        else:
            meth[(r["method"], r["kind"], r["attack"])].append(r)
    res = {"hardware": hw.pop(), "rows": [], "skipped": [], "paired": []}
    sets = {"clean": ("forget", "benign"), "attack": ("forget",), "hardneg": ("benign",)}
    pooled = {}
    for (m, kind, attack), rs in sorted(meth.items()):
        b = base.get((kind, attack))
        if not b:
            res["skipped"].append((f"{m} {kind} {attack}", "no base run for the same condition (a_miss/u_base)"))
            continue
        for which in sets[kind]:
            sub = [{**r, "items": _subset(r["items"], "forget" if which == "forget" else "benign")}
                   if kind == "clean" else r for r in rs]
            bi = _subset(b[0]["items"], which) if kind == "clean" else b[0]["items"]
            df = _pool(sub, bi)
            if df.empty:
                continue
            iset = {"forget": "WMDP-Bio TEST", "benign": "MMLU utility"}[which] if kind != "hardneg" else "benign biology"
            a = audit(df, n_boot=n_boot)
            res["rows"].append({"method": m, "condition": "clean" if kind != "attack" else attack, "set": iset,
                                "seeds": sorted({r["seed"] for r in rs}), **a})
            pooled[(m, kind, attack, iset)] = df
    for (m, kind, attack, iset), df in pooled.items():
        if m != "StreamGuard" or iset == "WMDP-Bio TEST":
            continue
        d = pooled.get(("DSG", kind, attack, iset))
        if d is not None:
            res["paired"].append({"set": iset, **paired_utility(df, d, n_boot=n_boot)})
    return res


# ---------------------------------------------------------------- markdown

def _f(x, nd=3):
    return "–" if x is None or x != x else f"{x:.{nd}f}"


def _ci_s(d, nd=3, sign=False):
    fmt = (lambda v: f"{v:+.{nd}f}") if sign else (lambda v: _f(v, nd))
    return f"{fmt(d['mean'])} [{fmt(d['lo'])}, {fmt(d['hi'])}]"


def render(results: list[dict], skipped: list) -> str:
    L = ["# E1 three-number audit (R, F, a/u decomposition)", "",
         "Generated by `python -m dsgx.analysis.rfd_audit` (CPU, existing X1 TEST per-item logs, no model runs). "
         "Each machine is its own table; numbers are never mixed across machines. Pooled over the seeds listed; 95 % "
         "item-cluster bootstrap CIs (2000 resamples, items resampled with all their seed rows); n = items × seeds.", "",
         "Definitions: R (forget) / F (benign) = gate fire rate; a_fire / u_fire = gated-run accuracy on fired rows; "
         "a_miss / u_base = **base-model** accuracy (X1 base run, same item ids and attack) on the rows the gate did not "
         "fire on. pred = (1 − R)·a_miss + R·a_fire (utility: (1 − F)·u_base + F·u_fire). In-CI = pred inside the observed "
         "accuracy's 95 % CI. With a_miss taken from the gated run itself the formula is an identity; its largest "
         "|pred − obs| over all rows is given per machine as an arithmetic check. 'misses only' = (1 − R)·a_miss + R·0.25 "
         "(fired items at chance).", ""]
    for res in results:
        hw = res["hardware"]
        rows = res["rows"]
        L += [f"## {hw}", ""]
        ident = max((r["ident_err"]["mean"] for r in rows), default=float("nan"))
        L += [f"Identity check (a_miss from the gated run): max |pred − obs| = {ident:.2e} over {len(rows)} rows.", ""]
        for iset, title, sym in (("WMDP-Bio TEST", "WMDP-Bio TEST forget accuracy (clean + attacks)", ("R", "a_miss", "a_fire")),
                                 ("MMLU utility", "Full MMLU utility (clean)", ("F", "u_base", "u_fire")),
                                 ("benign biology", "Benign biology (A3 hard negatives, clean)", ("F", "u_base", "u_fire"))):
            sel = [r for r in rows if r["set"] == iset]
            if not sel:
                continue
            L += [f"### {hw}: {title}", "",
                  f"| method | condition | seeds | n (items × seeds) | {sym[0]} | {sym[1]} | {sym[2]} | predicted | observed [95 % CI] | pred − obs | in CI |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for r in sorted(sel, key=lambda r: (r["method"], r["condition"] != "clean", r["condition"])):
                L.append(f"| {r['method']} | {r['condition']} | {len(r['seeds'])} | {r['n_items']} × {r['n_seeds']} = {r['n_rows']} "
                         f"| {_ci_s(r['R'])} | {_ci_s(r['a_miss'])} | {_ci_s(r['a_fire'])} | {_f(r['pred']['mean'])} "
                         f"| {_ci_s(r['obs'])} | {r['pred_minus_obs']['mean']:+.4f} | {'yes' if r['pred_in_ci'] else 'NO'} |")
            L.append("")
        # worst case
        L += [f"### {hw}: worst-case forget accuracy over the attack suite", "",
              "| method | clean forget acc | worst attack | worst-case forget acc [95 % CI] | n | attacks |", "|---|---|---|---|---|---|"]
        for m in ("DSG", "StreamGuard"):
            att = [r for r in rows if r["method"] == m and r["set"] == "WMDP-Bio TEST" and r["condition"] != "clean"]
            cl = [r for r in rows if r["method"] == m and r["set"] == "WMDP-Bio TEST" and r["condition"] == "clean"]
            if not att:
                continue
            w = max(att, key=lambda r: r["obs"]["mean"])
            L.append(f"| {m} | {_ci_s(cl[0]['obs']) if cl else '–'} | {w['condition']} | {_ci_s(w['obs'])} | {w['n_rows']} | {len(att)} |")
        L.append("")
        # Q1
        L += [f"### {hw}: Q1 — is DSG's clean forget accuracy explained by gate misses alone?", ""]
        for r in [r for r in rows if r["set"] == "WMDP-Bio TEST" and r["condition"] == "clean"]:
            mo = r["misses_only"]
            L.append(f"- {r['method']}: observed {_ci_s(r['obs'])} (n {r['n_rows']}); R {_ci_s(r['R'])}; base accuracy on "
                     f"missed items a_miss {_ci_s(r['a_miss'])}; accuracy on fired items a_fire {_ci_s(r['a_fire'])} "
                     f"(chance 0.25). Misses-only prediction {_f(mo['mean'])} → "
                     f"{'inside' if r['misses_only_in_ci'] else 'OUTSIDE'} the observed CI. Contribution of misses "
                     f"(1 − R)·a_miss = {_f((1 - r['R']['mean']) * r['a_miss']['mean'])}, of fires R·a_fire = "
                     f"{_f(r['R']['mean'] * r['a_fire']['mean'])}.")
        d = next((r for r in rows if r["method"] == "DSG" and r["set"] == "WMDP-Bio TEST" and r["condition"] == "clean"), None)
        if d:
            chance_in = d["a_fire"]["lo"] <= CHANCE <= d["a_fire"]["hi"]
            yes = d["misses_only_in_ci"] and chance_in
            L.append(f"- **Answer ({hw}): {'yes' if yes else 'no'}.** Misses-only prediction "
                     f"{'inside' if d['misses_only_in_ci'] else 'outside'} the observed CI; fired-item accuracy "
                     f"{'is' if chance_in else 'is not'} at chance (0.25 {'inside' if chance_in else 'outside'} the a_fire CI).")
        L.append("")
        # Q2
        L += [f"### {hw}: Q2 — is StreamGuard's utility loss vs DSG explained by its extra false fires alone?", "",
              "Paired by seed and item (StreamGuard − DSG). Parts: the observed difference summed over rows fired by "
              "StreamGuard only, by DSG only, by both, by neither (divided by all rows). pred = (F_SG − F_DSG)·(u_fire − u_base), "
              "u's of DSG on the same rows.", "",
              "| set | n (items × seeds) | F_SG | F_DSG | observed Δ [95 % CI] | pred Δ | pred in CI | SG-only fires: rows, part | DSG-only fires: rows, part | both + neither part [95 % CI] |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for p in res["paired"]:
            L.append(f"| {p['set']} | {p['n_items']} × {p['n_seeds']} = {p['n_rows']} | {_f(p['F_sg']['mean'])} | {_f(p['F_dsg']['mean'])} "
                     f"| {_ci_s(p['obs_diff'], 4, True)} | {p['pred_diff']['mean']:+.4f} | {'yes' if p['pred_in_ci'] else 'NO'} "
                     f"| {_f(p['frac_sg_only']['mean'])}, {p['part_sg_only']['mean']:+.4f} | {_f(p['frac_dsg_only']['mean'])}, "
                     f"{p['part_dsg_only']['mean']:+.4f} | {_ci_s(p['part_concordant'], 4, True)} |")
        L.append("")
        for p in res["paired"]:
            yes = p["pred_in_ci"] and p["concordant_zero"]
            disc = p["part_sg_only"]["mean"] + p["part_dsg_only"]["mean"]
            L.append(f"- **Answer ({hw}, {p['set']}): {'yes' if yes else 'no'}.** Discordant-fire rows carry "
                     f"{disc:+.4f} of the observed {p['obs_diff']['mean']:+.4f}; concordant rows (both / neither fired) "
                     f"{_ci_s(p['part_concordant'], 4, True)} ({'CI contains 0' if p['concordant_zero'] else 'CI excludes 0'}); "
                     f"formula prediction {p['pred_diff']['mean']:+.4f} {'inside' if p['pred_in_ci'] else 'outside'} the observed CI.")
        L.append("")
        if res["skipped"]:
            L += [f"### {hw}: conditions skipped", "", "| condition | reason |", "|---|---|"]
            L += [f"| {a} | {b} |" for a, b in res["skipped"]]
            L.append("")
    if not skipped and not any(r["skipped"] for r in results):
        L += ["No condition skipped: every X1 TEST run of base, DSG and StreamGuard has item_id, dataset, correct and "
              "gate_fired for every row.", ""]
    if skipped:
        L += ["## Runs skipped (missing per-item fields)", "", "| run | reason |", "|---|---|"]
        L += [f"| {a} | {b} |" for a, b in skipped]
        L.append("")
    return "\n".join(L)


def main(argv=None) -> int:
    from dsgx import paths

    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--exp", default="X1")
    ap.add_argument("--json", default=None, help="also write the raw numbers as JSON")
    a = ap.parse_args(argv)
    results, skipped = [], []
    for root in (paths.runs_dir(), CLUSTER / "runs"):
        runs, sk = load_runs(root, a.exp)
        skipped += sk
        by = defaultdict(list)
        for r in runs:
            by[r["hardware"]].append(r)
        for hw in sorted(by):
            results.append(machine_audit(by[hw], n_boot=a.n_boot))
    md = render(results, skipped)
    if a.out:
        Path(a.out).write_text(md + "\n")
    else:
        print(md)
    if a.json:
        Path(a.json).write_text(json.dumps(results, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
