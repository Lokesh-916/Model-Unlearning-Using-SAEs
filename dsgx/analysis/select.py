"""Select a configuration from finished dev runs (rule 3.4: never select on test).

A run config can say
    method: {name: dsg-faithful, select: {exp_id: A1-dev, keys: [n_features, retain_pct, multiplier],
                                          max_utility_drop: 0.01}}
and `run()` fills the method keys from the best dev run of the same case and method name:
lowest forget accuracy (DSG-subset view if present, else raw) among configs whose raw pooled
utility is within `max_utility_drop` of the base model's on the same dev data. Ties go to
higher utility, then fewer features.
"""
import json

from dsgx import paths


def _runs(exp_id: str):
    root = paths.runs_dir() / exp_id
    for d in sorted(root.glob("*")):
        if not (d / "DONE").exists():
            continue
        try:
            cfg = json.loads((d / "config.json").read_text())["config"]
            met = json.loads((d / "metrics.json").read_text())
        except (OSError, ValueError, KeyError):
            continue
        yield d, cfg, met


def _forget(met):
    v = met["views"].get("dsg_subset") or met["views"]["raw"]
    return v["forget"]["mean"]


def _utility(met):
    v = met["views"].get("raw") or met["views"]["dsg_subset"]
    return v["utility"]["pooled"]["mean"]


def select_config(case: str, method_name: str, spec: dict, current: dict | None = None) -> dict:
    """current: the method config being filled; by default only dev runs with the same forget and
    retain corpora are candidates (e.g. Cyber-faithful vs Cyber-chatretain). spec['match'] adds
    or overrides required method fields."""
    exp_id = spec["exp_id"]
    match = {k: (current or {}).get(k) for k in ("forget_corpus", "retain_corpus") if (current or {}).get(k)}
    match.update(spec.get("match") or {})
    keys = spec.get("keys", ["n_features", "retain_pct", "multiplier"])
    drop = float(spec.get("max_utility_drop", 0.01))
    runs = [(d, c, m) for d, c, m in _runs(exp_id) if c.get("case") == case]
    if any(c["split"] != "dev" for _, c, _ in runs):
        runs = [(d, c, m) for d, c, m in runs if c["split"] == "dev"]
    base = [m for _, c, m in runs if c["method"]["name"] == "base" and not (c.get("model") or {}).get("weights")]
    cands = [(d, c, m) for d, c, m in runs if c["method"]["name"] == method_name
             and all(c["method"].get(k) == v for k, v in match.items())]
    if not cands:
        raise RuntimeError(f"no finished dev runs of {method_name} for {case} in {exp_id}")
    if not base:
        raise RuntimeError(f"no finished dev base run for {case} in {exp_id} (needed for the utility bound)")
    base_u = _utility(base[0])
    ok = [(d, c, m) for d, c, m in cands if _utility(m) >= base_u - drop]
    pool = ok or cands
    best = min(pool, key=lambda t: (_forget(t[2]), -_utility(t[2]), t[1]["method"].get("n_features", 0)))
    d, c, m = best
    return {"method": {k: c["method"][k] for k in keys if k in c["method"]},
            "info": {"source_run": str(d), "forget": _forget(m), "utility": _utility(m),
                     "base_utility": base_u, "n_candidates": len(cands), "n_within_bound": len(ok),
                     "bound_met": bool(ok), "split": c["split"], "match": match}}
