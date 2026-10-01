"""Experiment configs (configs/experiments/<ID>.yaml) and their expansion into run configs.

Format:
    exp_id: A1
    priority: must            # must | should | stretch
    wave: 1
    base: {case: bio, split: dev, purpose: select, method: {name: dsg-faithful}}
    grid:                     # cartesian product over dotted keys (optional)
      method.n_features: [10, 20]
    runs: [{...}, ...]        # explicit overrides, applied after the grid (optional)
    jobs: {group_size: 8, est_seconds_per_item: 0.3, est_vram_gb: 7.5, est_ram_gb: 6}
    smoke: {base: {limit: 4}, grid: {...}}   # merged over the above for the smoke config
"""
import copy
import itertools
from pathlib import Path

import yaml


def load_experiment(path) -> dict:
    exp = yaml.safe_load(Path(path).read_text())
    exp["_path"] = str(path)
    return exp


def _set(d: dict, dotted: str, value):
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def deep_merge(a: dict, b: dict) -> dict:
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def expand(exp: dict, smoke: bool = False) -> list[dict]:
    base = copy.deepcopy(exp.get("base", {}))
    grid = exp.get("grid") or {}
    runs = exp.get("runs")
    if smoke:
        s = exp.get("smoke") or {}
        base = deep_merge(base, s.get("base", {}))
        if "grid" in s:
            grid = s["grid"] or {}
        if "runs" in s:
            runs = s["runs"]
    out = []
    keys = list(grid)
    for combo in itertools.product(*[grid[k] for k in keys]) if keys else [()]:
        r = copy.deepcopy(base)
        for k, v in zip(keys, combo):
            _set(r, k, v)
        for ov in runs or [{}]:
            rr = deep_merge(r, ov)
            rr["exp_id"] = exp["exp_id"] + ("-smoke" if smoke else "")
            out.append(rr)
    return out
