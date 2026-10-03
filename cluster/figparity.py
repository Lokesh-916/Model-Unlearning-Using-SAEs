"""DSG figure parity on gpuws: the data behind every figure type of the DSG paper, for DSG and for our gate.

    python cluster/figparity.py <part> [<part> ...] [--budget-min M] [--plan]

Parts (each resumable: finished harness runs (DONE) are skipped; a part stops when the budget is used up and the
next chained sbatch continues). "ours" = the window gate (w from COMBINE_SELECTION.json if X1 picked a window
detector, else w=16), features selected like DSG's, 5% benign-FPR threshold on MMLU DEV, clamp_all.
  clamp       clamp strength c in {10,25,50,100,500,1000} x features N in {10,20,30,50,100,200}, DSG and ours,
              + base. DEV (purpose tune, bs 16), WMDP-Bio + the 4 legacy MMLU subjects, both views.  exp FP-clamp
  dataeff     feature-selection corpus size m in {32,64,128,256,512,1024} docs (forget and retain rows), seeds
              {0,1,2} for m < 1024; DSG (N 20, c 500, tau from the m-cache) and ours (same features). DEV.  FP-dataeff
  static      static (always clamp: tau / threshold -1) vs dynamic (gated), DSG and ours, + base. TEST, bs 1,
              WMDP-Bio + full-MMLU utility, both views.                                           exp FP-static
  multitopic  bio + cyber at once: features = union of the bio and cyber top-20 (tau from the bio cache's
              retain rows), DSG and ours; single-topic DSG bio / cyber and base as comparators. TEST, bs 1, raw
              view, WMDP-Bio + WMDP-Cyber + MMLU minus both hazard-adjacent lists.               exp FP-multitopic
  latency     batch size 1 forward latency by sequence length {64..2048} (random tokens, 10 warm-up, 50 timed),
              base / DSG (gated and forced-on) / rho, window and cusum gates; ms, overhead, peak VRAM. exp FP-latency
  highlight   TOFU only: per-token activation of the selected features on 12 forget + 12 retain QA pairs with the
              A2-tofu-full model (features and thresholds exactly as tofu_full.py), tokens included (TOFU is
              fictitious); rho and window statistics and both gate decisions.                    exp FP-highlight
Every harness run writes config.json / metrics.json / items.parquet / traces.npz (MASTER_PLAN section 7); task
outputs write metrics.json + config.json with the hardware label. Nothing needs a re-run to make a plot.
"""
import argparse
import json
import time

import numpy as np
import torch

from cluster import jobcommon as jc
from dsgx import paths
from dsgx.util import atomic_write_json, gpu_info, now_iso

NAME = "figs"
CLAMPS = [10, 25, 50, 100, 500, 1000]
NFEATS = [10, 20, 30, 50, 100, 200]
SIZES = [32, 64, 128, 256, 512, 1024]
SEEDS = [0, 1, 2]
LENGTHS = [64, 128, 256, 512, 1024, 2048]
EST = {"dev": 1.5, "test": 9.0, "test_raw2": 11.0}  # minutes per run on gpuws (incl. a first calibration)


def window_w():
    sel = jc.read_json(paths.results_dir() / "COMBINE_SELECTION.json", {}) or {}
    det = (sel.get("slots") or {}).get("detector") or ""
    return (int(det.split("-w")[1]) if det.startswith("window-w") else 16), det or "default window-w16"


def dsg(n=20, c=500, **kw):
    return {"name": "dsg-faithful", "n_features": n, "retain_pct": 95, "multiplier": c, **kw}


def ours(n=20, c=500, features=None, threshold=None, calib_seed=None):
    g = {"type": "window", "w": window_w()[0], "n_features": n, "retain_pct": 95}
    if features is not None:
        g["features"] = list(features)
    if threshold is not None:
        g["threshold"] = threshold
    if calib_seed is not None:
        g["calib_seed"] = calib_seed
    return {"name": "gated", "gate": g, "calib": {"fpr": 0.05, "n_max": 1000, "source": "mmlu-dev"},
            "intervention": {"type": "clamp_all", "multiplier": c}}


def cfg(exp, label, method, split, datasets, view="both", bs=None, seed=0, **extra):
    c = {"exp_id": exp, "case": "bio", "split": split, "view": view, "seed": seed,
         "batch_size": bs or (16 if split == "dev" else 1), "datasets": datasets, "dataset_label": label,
         "method": method, "attack": {"name": "none"}, **extra}
    if split == "dev":
        c["purpose"] = "tune"
    return c


class Runner:
    def __init__(self, budget, plan):
        self.budget, self.plan, self.left, self.dirs = budget, plan, {}, {}

    def __call__(self, part, c, est):
        from dsgx.run import run

        key = c["dataset_label"]
        rd = jc.run_dir_of(c)
        if (rd / "DONE").exists():
            self.dirs.setdefault(part, {})[key] = str(rd)
            return
        if self.plan or jc.TINY:
            self.left.setdefault(part, []).append(key)
            return
        if not self.budget.fits(est):
            self.left.setdefault(part, []).append(key)
            return
        self.dirs.setdefault(part, {})[key] = str(run(c))


# ----------------------------------------------------------------------------- parts
def part_clamp(R):
    ds = ["@forget", "@dsg4"]
    R("clamp", cfg("FP-clamp", "base", {"name": "base"}, "dev", ds), EST["dev"])
    for n in NFEATS:
        for c in CLAMPS:
            R("clamp", cfg("FP-clamp", f"dsg-N{n}-c{c}", dsg(n, c), "dev", ds, fp_params={"n": n, "c": c, "method": "dsg"}), EST["dev"])
            R("clamp", cfg("FP-clamp", f"ours-N{n}-c{c}", ours(n, c), "dev", ds, fp_params={"n": n, "c": c, "method": "ours"}), EST["dev"])


def part_dataeff(R, budget):
    from dsgx.data import activation_cache as ac
    from dsgx.methods import dsg as dsgm

    ds = ["@forget", "@dsg4"]
    R("dataeff", cfg("FP-dataeff", "base", {"name": "base"}, "dev", ds), EST["dev"])
    bundle = None
    feats_file = jc.job_dir(NAME) / "dataeff_features.json"
    known = jc.read_json(feats_file, {}) or {}
    for m in SIZES:
        for s in (SEEDS if m < 1024 else [0]):
            key = f"m{m}-s{s}"
            labels = [f"dsg-{key}", f"ours-{key}"]
            if key not in known:
                if R.plan or jc.TINY or not budget.fits(4 + 2 * EST["dev"]):
                    R.left.setdefault("dataeff", []).extend(labels)
                    continue
                if bundle is None:
                    from dsgx.models.loader import get_bundle

                    bundle = get_bundle()
                cache = ac.ActivationCache(ac.build_cache(bundle, "bio-forget-corpus", "wikitext", s, dataset_size=m))
                f = dsgm.select_features(cache, 20, 95)
                known[key] = {"features": f, "tau": dsgm.calibrate_tau(cache, f, 95), "m": m, "seed": s,
                              "cache": str(cache.path), "n_features_found": len(f)}
                atomic_write_json(feats_file, known)
            k = known[key]
            p = {"m": m, "seed": s, "n_features": len(k["features"])}
            # calib_seed 0: the method's own cache handle (info only) is the full s0 cache; features/tau are explicit
            R("dataeff", cfg("FP-dataeff", f"dsg-{key}", dsg(features=k["features"], tau=k["tau"], calib_seed=0), "dev", ds,
                             seed=s, fp_params={**p, "method": "dsg"}), EST["dev"])
            R("dataeff", cfg("FP-dataeff", f"ours-{key}", ours(features=k["features"], calib_seed=0), "dev", ds,
                             seed=s, fp_params={**p, "method": "ours"}), EST["dev"])


def part_static(R):
    ds = ["@forget", "@utility"]
    for label, m in (("base", {"name": "base"}), ("dsg-dynamic", dsg()), ("dsg-static", dsg(tau=-1.0)),
                     ("ours-dynamic", ours()), ("ours-static", ours(threshold=-1.0))):
        R("static", cfg("FP-static", label, m, "test", ds, fp_params={"clamping": label.split("-")[-1]}), EST["test"])


def multitopic_features():
    from dsgx.data import activation_cache as ac
    from dsgx.methods import dsg as dsgm

    if jc.TINY:
        return {"bio": list(range(0, 40, 2)), "cyber": list(range(1, 41, 2)), "tau_bio": 0.1, "tau_cyber": 0.1, "tau_union": 0.1}
    bc = ac.open_cache("gemma-2-2b-it", "gemma-scope-2b-pt-res", "layer_3/width_16k/average_l0_142", "bio-forget-corpus", "wikitext", 0)
    cc = ac.open_cache("gemma-2-2b-it", "gemma-scope-2b-pt-res", "layer_3/width_16k/average_l0_142", "cyber-forget-corpus", "wikitext", 0)
    fb, fc = dsgm.select_features(bc, 20, 95), dsgm.select_features(cc, 20, 95)
    union = list(dict.fromkeys(fb + fc))
    return {"bio": fb, "cyber": fc, "union": union, "overlap": len(set(fb) & set(fc)),
            "tau_bio": dsgm.calibrate_tau(bc, fb, 95), "tau_cyber": dsgm.calibrate_tau(cc, fc, 95),
            "tau_union": dsgm.calibrate_tau(bc, union, 95)}


def part_multitopic(R):
    from dsgx.data.mcq import utility_subjects

    F = multitopic_features()
    F.setdefault("union", list(dict.fromkeys(F["bio"] + F["cyber"])))
    atomic_write_json(jc.job_dir(NAME) / "multitopic_features.json", F)
    util = [s for s in utility_subjects("bio") if s in set(utility_subjects("cyber"))]
    ds = ["wmdp-bio", "wmdp-cyber"] + util
    for label, m in (("base", {"name": "base"}),
                     ("dsg-bio-only", dsg(features=F["bio"], tau=F["tau_bio"])),
                     ("dsg-cyber-only", dsg(features=F["cyber"], tau=F["tau_cyber"])),
                     ("dsg-bio+cyber", dsg(features=F["union"], tau=F["tau_union"])),
                     ("ours-bio+cyber", ours(features=F["union"]))):
        R("multitopic", cfg("FP-multitopic", label, m, "test", ds, view="raw", forget_datasets=["wmdp-bio", "wmdp-cyber"],
                            fp_params={"topics": label.split("-", 1)[-1]}), EST["test_raw2"])


def _task_out(exp, task, metrics):
    d = paths.runs_dir() / exp / task
    d.mkdir(parents=True, exist_ok=True)
    atomic_write_json(d / "config.json", {"task_id": task, "hardware": gpu_info(), "time": now_iso()})
    atomic_write_json(d / "metrics.json", {**metrics, "hardware_label": jc.hardware_label()})
    atomic_write_json(d / "DONE", {"time": now_iso()})
    return d


def part_latency(R, budget, reps=50, warm=10):
    d = paths.runs_dir() / "FP-latency" / "latency"
    if (d / "DONE").exists() or R.plan or jc.TINY or not budget.fits(25):
        if not (d / "DONE").exists():
            R.left.setdefault("latency", []).append("latency")
        return
    from dsgx.methods.registry import make_method
    from dsgx.models.loader import get_bundle
    from dsgx.run import ensure_cache, resolve

    b = get_bundle()
    w, _ = window_w()
    specs = {"base": {"name": "base"}, "dsg": dsg(), "dsg-forced-on": dsg(tau=-1.0),
             "gate-rho": {"name": "gated", "gate": {"type": "rho"}, "calib": {"source": "mmlu-dev"}},
             "ours-window": ours(), "gate-cusum": {"name": "gated", "gate": {"type": "cusum"}, "calib": {"source": "mmlu-dev"}}}
    g = torch.Generator().manual_seed(0)
    vocab = b.model.cfg.d_vocab
    res = {}
    for L in LENGTHS:
        toks = torch.randint(1000, vocab - 1000, (1, L), generator=g)
        toks[0, 0] = b.model.tokenizer.bos_token_id
        toks = toks.to(b.device)
        for name, spec in specs.items():
            c = resolve({"exp_id": "FP-latency", "split": "dev", "purpose": "tune", "method": spec})
            ensure_cache(c, b)
            meth = make_method(c["method"], b, 0)
            meth.install()
            times = []
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for i in range(warm + reps):
                meth.set_lengths([L])
                if hasattr(meth, "before_forward"):
                    meth.before_forward(toks, [L])
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                with torch.no_grad():
                    b.model(toks)
                torch.cuda.synchronize()
                if i >= warm:
                    times.append(time.perf_counter() - t0)
                meth.pop_records()
            meth.remove()
            t = np.array(times) * 1000
            res.setdefault(str(L), {})[name] = {"ms_median": float(np.median(t)), "ms_mean": float(t.mean()), "ms_sd": float(t.std(ddof=1)),
                                                "ms_p95": float(np.percentile(t, 95)), "n": len(t),
                                                "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9}
        base = res[str(L)]["base"]["ms_median"]
        for v in res[str(L)].values():
            v["overhead_vs_base"] = v["ms_median"] / base - 1
        jc.log(NAME, f"latency L={L}: " + ", ".join(f"{k} {v['ms_median']:.1f} ms" for k, v in res[str(L)].items()))
    _task_out("FP-latency", "latency", {"latency": res, "lengths": LENGTHS, "batch_size": 1, "reps": reps, "warmup": warm,
                                        "inputs": "random tokens (BOS + uniform ids), as DSG Table 15", "dsg_debug": False,
                                        "methods": {k: v for k, v in specs.items()}, "window_w": w})
    R.dirs.setdefault("latency", {})["latency"] = str(d)


def part_highlight(R, budget, n=12):
    d = paths.runs_dir() / "FP-highlight" / "tofu"
    full = paths.cache_dir() / "models" / "A2-tofu-full" / "full"
    if (d / "DONE").exists():
        R.dirs.setdefault("highlight", {})["tofu"] = str(d)
        return
    if R.plan or not budget.fits(15) or not (jc.TINY or (full / "config.json").exists()):
        R.left.setdefault("highlight", []).append("tofu" + ("" if jc.TINY or (full / "config.json").exists() else " (A2-tofu-full model missing)"))
        return
    from cluster import tofu_full as tf
    from dsgx.train.core import decoder_layers

    tok = jc.load_tok()
    m = jc.load_lm(full if not jc.TINY else jc.HF_2B)
    m.eval()
    sae, layer = jc.load_sae(), (1 if jc.TINY else 3)
    f10, r90 = tf.tofu("forget10"), tf.tofu("retain90")
    calib = 4 if jc.TINY else 400
    ftxt = [tf.chat(tok, r["question"]) + r["answer"] for r in f10[:calib]]
    rtxt = [tf.chat(tok, r["question"]) + r["answer"] for r in r90[:calib]]
    pf, _ = tf.fire_stats(m, tok, sae, layer, ftxt)
    pr, _ = tf.fire_stats(m, tok, sae, layer, rtxt)
    score = np.where(pf >= 0.01, (pf + 1e-4) / (pr + 1e-4), 0)
    feats = [int(i) for i in np.argsort(-score)[:20]]
    _, per = tf.fire_stats(m, tok, sae, layer, rtxt, feats)
    w, det = window_w()
    rho_g, win_g = tf.Gate(sae, feats, "rho", 0), tf.Gate(sae, feats, "window", 0, w=w)
    thr = {"rho": float(np.percentile([rho_g.score(f) for f in per], 95)),
           "window": float(np.percentile([win_g.score(f) for f in per], 95))}
    got = {}
    h = decoder_layers(m)[layer].register_forward_hook(lambda mod, i, o: got.__setitem__("x", o[0] if isinstance(o, tuple) else o))
    dev = next(m.parameters()).device
    ft = torch.tensor(feats, device=sae.W_dec.device)
    examples = []
    # forget10 has 400 rows and tofu_full selects features on all of them, so the forget examples are in-sample;
    # the retain examples are held out (rows after the 400 calibration rows)
    rows = [("forget", r) for r in f10[:n]] + [("retain", r) for r in r90[calib:calib + n]]
    with torch.no_grad():
        for part, r in rows:
            text = tf.chat(tok, r["question"]) + r["answer"]
            enc = tok(text, return_tensors="pt", add_special_tokens=True).to(dev)
            m(**enc)
            acts = sae.encode(got["x"][0].to(sae.W_dec.dtype))
            acts[0] = 0
            sel = acts[:, ft]
            fire = (sel > 0).any(-1)
            run_rho = (fire.float().cumsum(0) / torch.arange(1, len(fire) + 1, device=fire.device)).cpu().numpy()
            f1 = fire[1:].float()
            win = [0.0] + [float(f1[max(0, t - w):t].mean()) for t in range(1, len(f1) + 1)]
            ids = enc["input_ids"][0].tolist()
            examples.append({"part": part, "tokens": tok.convert_ids_to_tokens(ids), "token_ids": ids,
                             "sel_max_act": sel.max(-1).values.float().cpu().numpy().round(4).tolist(),
                             "top_feature": [int(feats[i]) if v > 0 else None for i, v in
                                             zip(sel.argmax(-1).tolist(), sel.max(-1).values.tolist())],
                             "fire": fire.cpu().numpy().astype(int).tolist(), "running_rho": run_rho.round(4).tolist(),
                             "window_stat": np.round(win, 4).tolist(),
                             "answer_start_char": len(tf.chat(tok, r["question"])),
                             "rho": float(rho_g.score(fire)), "window": float(win_g.score(fire)),
                             "gate_rho_fires": bool(rho_g.score(fire) > thr["rho"]),
                             "gate_window_fires": bool(win_g.score(fire) > thr["window"])})
    h.remove()
    _task_out("FP-highlight", "tofu", {"examples": examples, "features": feats, "thresholds": thr, "window_w": w,
                                       "window_source": det, "layer": layer, "model": str(full), "n_per_part": n,
                                       "note": "TOFU is fictitious; tokens are stored on purpose (TOFU only). Forget examples are in the "
                                               "feature-selection set (all 400 forget10 rows), retain examples are held out."})
    R.dirs.setdefault("highlight", {})["tofu"] = str(d)


PARTS = {"clamp": lambda R, B: part_clamp(R), "dataeff": part_dataeff, "static": lambda R, B: part_static(R),
         "multitopic": lambda R, B: part_multitopic(R), "latency": part_latency, "highlight": part_highlight}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("parts", nargs="+", choices=list(PARTS))
    ap.add_argument("--budget-min", type=float, default=None)
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    budget = jc.Budget(a.budget_min)
    R = Runner(budget, a.plan)
    if not (a.plan or jc.TINY):
        jc.require_gpu(40)
    for p in a.parts:
        PARTS[p](R, budget)
        if a.plan or jc.TINY:
            from dsgx.checks.leakage import check_runs  # noqa: F401  (configs are leakage-checked in run())
    summ = jc.read_json(jc.job_dir(NAME) / "summary.json", {}) or {}
    parts = summ.get("parts", {})
    for p in a.parts:
        parts[p] = {"runs": R.dirs.get(p, {}), "left": R.left.get(p, []), "complete": not R.left.get(p)}
    jc.summary(NAME, {"parts": parts, "window": window_w()[1]})
    for p in a.parts:
        jc.log(NAME, f"{p}: {len(R.dirs.get(p, {}))} done, {len(R.left.get(p, []))} left")
    print(json.dumps({p: len(R.left.get(p, [])) for p in a.parts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
