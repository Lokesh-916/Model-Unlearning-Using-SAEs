"""Harness core: run(config) -> run_dir.

    python -m dsgx.run configs/experiments/A1.yaml --run 0      # one run of an experiment
    python -m dsgx.run --sanity                                   # the sanity gate

A config here is one *resolved run config* (see dsgx.config.expand for experiment grids).
Two evaluation views are always computed when possible: raw accuracy on the split, and the
DSG subset (items the base model gets right under all 24 answer permutations).
"""
import resource
import time

import numpy as np

from dsgx.attacks.registry import make_attack
from dsgx.data import activation_cache as ac
from dsgx.data.mcq import (DSG_UTILITY_SUBJECTS, FORGET_DATASET, load_mcq, utility_subjects)
from dsgx.data.splits import dsg_subset_ids, get_split, split_hash
from dsgx.eval import stats
from dsgx.eval.mcq_eval import prob_features, score_prompts
from dsgx.logging.run_logger import RunLogger
from dsgx.methods.registry import make_method
from dsgx.models.loader import LOAD_COUNTS, get_bundle

DEFAULT_MODEL = {"name": "gemma-2-2b-it", "sae_release": "gemma-scope-2b-pt-res",
                 "sae_id": "layer_3/width_16k/average_l0_142", "dtype": "bfloat16"}
CACHE_METHODS = {"dsg-faithful", "dsg-fixed"}


def expand_datasets(spec, case: str) -> list[str]:
    """'@forget', '@dsg4' (legacy utility subjects), '@utility' (full MMLU minus hazard-adjacent)."""
    out = []
    for d in spec:
        if d == "@forget":
            out.append(FORGET_DATASET[case])
        elif d == "@dsg4":
            out.extend(DSG_UTILITY_SUBJECTS[case])
        elif d == "@utility":
            out.extend(utility_subjects(case))
        else:
            out.append(d)
    return list(dict.fromkeys(out))


def resolve(cfg: dict) -> dict:
    c = dict(cfg)
    c.setdefault("case", "bio")
    c.setdefault("attack", {"name": "none"})
    c.setdefault("split", "test")
    c.setdefault("view", "both")
    c.setdefault("seed", 0)
    c.setdefault("batch_size", 1)
    c.setdefault("limit", None)
    c.setdefault("n_boot", stats.N_BOOT)
    c["model"] = {**DEFAULT_MODEL, **c.get("model", {})}
    c["datasets"] = expand_datasets(c.get("datasets", ["@forget", "@dsg4"]), c["case"])
    c.setdefault("forget_datasets", [FORGET_DATASET[c["case"]]])
    m = dict(c["method"])
    if m["name"] in CACHE_METHODS:
        m.setdefault("forget_corpus", f"{c['case']}-forget-corpus")
        m.setdefault("retain_corpus", "wikitext")
        m.setdefault("calib_seed", c["seed"])
    c["method"] = m
    # User decision 1 (P1a): reported test numbers of gated methods use batch size 1.
    if (c["split"] == "test" and c.get("purpose", "report") == "report" and m["name"] != "base"
            and int(c["batch_size"]) != 1 and not c.get("allow_batched_test")):
        raise ValueError("reported test runs of gated methods must use batch_size=1")
    # Rule 3.4: anything used for selection / tuning must be on dev.
    if c.get("purpose") in ("select", "tune") and c["split"] != "dev":
        raise ValueError(f"purpose={c['purpose']} requires split=dev (got {c['split']})")
    return c


def plan_items(c: dict) -> dict[str, list[int]]:
    """Item ids to evaluate per dataset, after split, view and limit."""
    out = {}
    for ds in c["datasets"]:
        ids = get_split(ds, c["split"])
        if c["view"] == "dsg_subset":
            sub = dsg_subset_ids(c["case"], ds)
            if sub is None:
                raise ValueError(f"no DSG-subset ids for {ds} ({c['case']})")
            sub = set(sub)
            ids = [i for i in ids if i in sub]
        if c["limit"]:
            ids = ids[: int(c["limit"])]
        out[ds] = ids
    return out


def count_items(cfg: dict) -> int:
    return sum(len(v) for v in plan_items(resolve(cfg)).values())


def ensure_cache(c, bundle, progress=None):
    m = c["method"]
    if m["name"] not in CACHE_METHODS or ("features" in m and "tau" in m):
        return None

    def _p(done, total):
        if progress:
            progress.update(phase=f"build-cache {done}/{total}")

    return ac.build_cache(bundle, m["forget_corpus"], m["retain_corpus"], m["calib_seed"],
                          m.get("dataset_size", 1024), m.get("seq_len", 1024), progress=_p)


def _acc_block(correct_by_ds: dict, forget: list[str], n_boot: int) -> dict:
    out = {"per_dataset": {d: stats.bootstrap_ci(v, n_boot) for d, v in correct_by_ds.items()}}
    util = {d: v for d, v in correct_by_ds.items() if d not in forget}
    out["utility"] = stats.stratified_bootstrap(util, n_boot) if util else None
    f = [v for d, v in correct_by_ds.items() if d in forget]
    out["forget"] = stats.bootstrap_ci(np.concatenate(f), n_boot) if f else None
    return out


def run(cfg: dict, progress=None, force: bool = False):
    """Run one resolved config; returns the run directory. Skips runs already DONE."""
    import torch

    c = resolve(cfg)
    log = RunLogger(c)
    if log.done and not force:
        return log.dir
    if progress:
        progress.add_path(str(log.dir / "progress.json"))
    t0 = time.time()
    mc = c["model"]
    bundle = get_bundle(mc["name"], mc["sae_release"], mc["sae_id"], mc["dtype"])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    selection = None
    if "select" in c["method"]:
        from dsgx.analysis.select import select_config

        selection = select_config(c["case"], c["method"]["name"], c["method"]["select"])
        c["method"] = {**{k: v for k, v in c["method"].items() if k != "select"}, **selection["method"]}
    ensure_cache(c, bundle, progress)
    method = make_method(c["method"], bundle, c["seed"])
    attack = make_attack(c["attack"], c["seed"])
    plan = plan_items(c)
    log.write_config({"batch_size": int(c["batch_size"]), "split_hashes": {d: split_hash(d) for d in c["datasets"]},
                      "method_info": method.info, "load_counts": dict(LOAD_COUNTS),
                      "selection": selection, "resolved_method": c["method"]})
    rows, traces, correct_raw, correct_sub = [], {}, {}, {}
    n_prompt_tokens = 0
    method.install()
    try:
        for ds in c["datasets"]:
            items = load_mcq(ds)
            sub = dsg_subset_ids(c["case"], ds)
            sub = set(sub) if sub is not None else None
            ids = plan[ds]
            if progress:
                progress.update(phase=f"eval {ds}")
            prompts, infos = zip(*[attack.prompt(items[i]) for i in ids]) if ids else ((), ())
            base = progress.state["items_done"] if progress else 0
            probs, recs, lens = score_prompts(
                bundle.model, list(prompts), c["batch_size"], method,
                on_batch=(lambda k: progress.update(items_done=base + k)) if progress else None)
            n_prompt_tokens += sum(lens)
            ent, margin = prob_features(probs) if len(ids) else ([], [])
            pred = probs.argmax(1) if len(ids) else []
            for j, i in enumerate(ids):
                it = items[i]
                r = {"item_id": it.item_id, "dataset": ds, "subject": it.subject, "split": c["split"],
                     "language": "en", "attack": attack.name, "attack_params": attack.params(),
                     "prompt_len": lens[j], "pad_len": infos[j].get("pad_len", 0),
                     "gold": it.answer, "pred": int(pred[j]), "correct": bool(pred[j] == it.answer),
                     "prob_A": float(probs[j, 0]), "prob_B": float(probs[j, 1]),
                     "prob_C": float(probs[j, 2]), "prob_D": float(probs[j, 3]),
                     "entropy": float(ent[j]), "margin": float(margin[j]),
                     "in_dsg_subset": (i in sub) if sub is not None else None,
                     "tau": getattr(method, "tau", None)}
                if method.records_gate:
                    g = recs[j]
                    trace = g.pop("fire_trace")
                    if len(traces) < 200:
                        traces[it.item_id] = trace
                    r.update(g)
                rows.append(r)
            correct_raw[ds] = [r["correct"] for r in rows if r["dataset"] == ds]
            if sub is not None:
                correct_sub[ds] = [r["correct"] for r in rows if r["dataset"] == ds and r["in_dsg_subset"]]
    finally:
        method.remove()
    wall = time.time() - t0
    nb = c["n_boot"]
    metrics = {"run_id": log.run_id, "n_items": len(rows), "batch_size": int(c["batch_size"]), "views": {}}
    if c["view"] != "dsg_subset":
        metrics["views"]["raw"] = _acc_block(correct_raw, c["forget_datasets"], nb)
    if correct_sub:
        metrics["views"]["dsg_subset"] = _acc_block(correct_sub, c["forget_datasets"], nb)
    if method.records_gate:
        gate = {}
        for ds in c["datasets"]:
            fired = [r["gate_fired"] for r in rows if r["dataset"] == ds]
            gate[ds] = {"fire_rate": float(np.mean(fired)) if fired else None, "n": len(fired),
                        "rho_mean": float(np.mean([r["rho"] for r in rows if r["dataset"] == ds])) if fired else None}
        fg = [r["gate_fired"] for r in rows if r["dataset"] in c["forget_datasets"]]
        bg = [r["gate_fired"] for r in rows if r["dataset"] not in c["forget_datasets"]]
        metrics["gate"] = {"per_dataset": gate, "tau": method.tau,
                           "benign_fpr": stats.bootstrap_ci(bg, nb) if bg else None,
                           "hazard_fnr": stats.bootstrap_ci([not x for x in fg], nb) if fg else None}
    metrics["timing"] = {
        "wall_seconds": round(wall, 2), "items_per_second": round(len(rows) / wall, 3) if wall else None,
        "prompt_tokens_per_second": round(n_prompt_tokens / wall, 1) if wall else None,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 3) if torch.cuda.is_available() else None,
        "peak_rss_gb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 3),
        "latency_overhead": None}
    metrics["method_info"] = method.info
    log.write_items(rows)
    if traces:
        log.write_traces(traces)
    log.write_metrics(metrics)
    view = metrics["views"].get("dsg_subset") or metrics["views"].get("raw")
    headline = {"forget": view["forget"], "utility_pooled": (view["utility"] or {}).get("pooled"),
                "utility_unweighted": (view["utility"] or {}).get("unweighted"),
                "view": "dsg_subset" if "dsg_subset" in metrics["views"] else "raw"}
    log.mark_done(headline)
    if progress:
        progress.update(current_metric=_fmt_headline(headline), force=True)
        progress.drop_path(str(log.dir / "progress.json"))
    return log.dir


def _fmt_headline(h: dict) -> str:
    def f(x):
        return "-" if not x or x.get("mean") is None else f"{x['mean']:.4f} [{x['lo']:.3f},{x['hi']:.3f}] n={x['n']}"
    return f"forget {f(h['forget'])}; util {f(h['utility_pooled'])} ({h['view']})"


def main():
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--run", type=int, default=None, help="index into the expanded run list")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--sanity", action="store_true")
    a = ap.parse_args()
    if a.sanity:
        from dsgx.checks.sanity import main as sanity_main

        raise SystemExit(sanity_main())
    from dsgx.config import expand, load_experiment

    exp = load_experiment(a.config)
    runs = expand(exp, smoke=a.smoke)
    sel = runs if a.run is None else [runs[a.run]]
    for r in sel:
        print(json.dumps({"run_dir": str(run(r, force=a.force))}))


if __name__ == "__main__":
    main()
