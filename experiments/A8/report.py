"""A8 reporting standards: one table generator for every run (n, CI, seeds, compute) and the
per-token latency / peak-VRAM benchmark with [DSG DEBUG] off."""
import json
import time

import numpy as np
import pandas as pd

from dsgx import paths


def _ci(x):
    if not x or x.get("mean") is None:
        return None, None, None, None
    return x["mean"], x.get("lo"), x.get("hi"), x.get("n")


def collect(include_smoke=False) -> pd.DataFrame:
    rows = []
    for d in sorted(paths.runs_dir().glob("*/*")):
        if d.parent.name.startswith("_") or not (d / "DONE").exists() or not (d / "metrics.json").exists():
            continue
        if not include_smoke and d.parent.name.endswith("-smoke"):
            continue
        try:
            m = json.loads((d / "metrics.json").read_text())
            c = json.loads((d / "config.json").read_text())
        except (OSError, ValueError):
            continue
        cfg = c.get("config") or {"method": {"name": c.get("task_id")}}
        r = {"exp": d.parent.name, "run": d.name, "method": (cfg.get("method") or {}).get("name"),
             "attack": json.dumps(cfg.get("attack")) if cfg.get("attack") else None, "split": cfg.get("split"),
             "case": cfg.get("case"), "seed": cfg.get("seed"), "batch_size": m.get("batch_size"),
             "commit": (c.get("git") or {}).get("commit", "")[:10], "wall_s": c.get("wall_seconds"),
             "peak_vram_gb": (m.get("timing") or {}).get("peak_vram_gb")}
        for view in ("raw", "dsg_subset"):
            v = (m.get("views") or {}).get(view) or {}
            r[f"{view}_forget"], r[f"{view}_forget_lo"], r[f"{view}_forget_hi"], r[f"{view}_forget_n"] = _ci(v.get("forget"))
            u = (v.get("utility") or {}).get("pooled")
            r[f"{view}_util"], r[f"{view}_util_lo"], r[f"{view}_util_hi"], r[f"{view}_util_n"] = _ci(u)
        g = m.get("gate") or {}
        r["benign_fpr"] = (g.get("benign_fpr") or {}).get("mean")
        r["hazard_fnr"] = (g.get("hazard_fnr") or {}).get("mean")
        rows.append(r)
    return pd.DataFrame(rows)


def tables(ctx):
    df = collect(include_smoke=ctx.smoke or ctx.args.get("include_smoke", False))
    out = paths.results_dir() / ("tables_smoke" if ctx.smoke else "tables")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "all_runs.csv", index=False)
    for exp, g in df.groupby("exp") if len(df) else []:
        (out / f"{exp}.md").write_text(g.drop(columns=["exp"]).to_markdown(index=False, floatfmt=".4f"))
    ctx.write_metrics({"n_runs": int(len(df)), "n_experiments": int(df["exp"].nunique()) if len(df) else 0,
                       "missing_ci": int(df["raw_forget"].notna().sum() - df["raw_forget_lo"].notna().sum()) if len(df) else 0,
                       "out": str(out)})
    ctx.finish({"view": "tables", "forget": None})


def latency(ctx):
    import torch

    from dsgx.data.mcq import format_prompt, load_mcq
    from dsgx.data.splits import get_split
    from dsgx.gen.stream import generate
    from dsgx.methods.registry import make_method
    from dsgx.models.loader import get_bundle
    from dsgx.run import ensure_cache, resolve

    a = ctx.args
    n = int(a.get("n", 50))
    items = [load_mcq("high_school_geography")[i] for i in get_split("high_school_geography", "test")[:n]]
    b = get_bundle()
    res = {}
    for spec in a.get("methods", [{"name": "base"}, {"name": "dsg-faithful"},
                                  {"name": "gated", "gate": {"type": "window", "w": 16}}]):
        cfg = resolve({"exp_id": ctx.exp_id, "split": "dev", "purpose": "select", "method": spec})
        ensure_cache(cfg, b)
        meth = make_method(cfg["method"], b, 0)
        meth.install()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        times, toks = [], 0
        for it in items:
            t = b.model.to_tokens(format_prompt(it), prepend_bos=False).to(b.device)
            meth.set_lengths([t.shape[1]])
            if hasattr(meth, "before_forward"):
                meth.before_forward(t, [t.shape[1]])
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.no_grad():
                b.model(t)
            torch.cuda.synchronize(); times.append(time.perf_counter() - t0)
            toks += t.shape[1]
            meth.pop_records()
        meth.remove()
        res[spec.get("tag") or spec["name"] + (":" + spec["gate"]["type"] if "gate" in spec else "")] = {
            "ms_per_prompt": 1000 * float(np.median(times)), "ms_per_token": 1000 * sum(times) / toks,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
        ctx.progress.advance(1)
    # generation throughput: streaming gate vs none
    from dsgx.data import activation_cache as ac
    from dsgx.methods import dsg

    cache = ac.ActivationCache(ac.build_cache(b, "bio-forget-corpus", "wikitext", 0))
    feats = dsg.select_features(cache, 20, 95); tau = dsg.calibrate_tau(cache, feats, 95)
    gen = {}
    for mode in ("none", "stream"):
        t0 = time.perf_counter(); nt = 0
        for it in items[: int(a.get("n_gen", 10))]:
            r = generate(b.model, format_prompt(it), b, feats, 500, tau, mode, int(a.get("max_new", 32)))
            nt += len(r.tokens)
        gen[mode] = {"tokens_per_s": nt / (time.perf_counter() - t0)}
    base = res.get("base", {}).get("ms_per_token")
    for k, v in res.items():
        v["overhead_vs_base"] = (v["ms_per_token"] / base - 1) if base else None
    ctx.write_metrics({"forward": res, "generation": gen, "n_prompts": n, "dsg_debug": False})
    ctx.finish({"view": "latency", "forget": None})
