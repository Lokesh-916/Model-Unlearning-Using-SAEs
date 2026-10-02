"""A7-12b: Gemma 3 12B (instruction-tuned) with its Gemma Scope 2 SAE (layer 24 of 48, 16k, l0 medium; the
same relative depth as exp/A7's 1B layer 13 / 4B layer 17), evaluated with the harness on gpuws.

Conditions (all TEST, bs=1, raw view, exp id A7-12b):
  base            the unguarded model
  dsg             gated rho gate, DSG features N=20 / retain 95, threshold from the cache retain rows
                  (exactly exp/A7's gated config: the DSG rule on Gemma 3)
  best-fix        the X1 detector if it is a window gate (COMBINE_SELECTION.json staged next to the
                  results), else window w=16; 5% benign-FPR threshold on MMLU dev
Attacks: none (WMDP-Bio + high_school_geography + human_aging), dilution pad 400 and 1600 (forget only),
translate fr (forget only; needs the staged translation cache of wmdp-bio test).
First run builds the 12B activation cache (forget corpus + wikitext, 1024 x 1024 tokens; ~1 GB).
Peak host RAM ~50 GB while TransformerLens converts the weights (no --mem on this cluster; documented).
`--plan` (and DSG_TINY=1) only resolves and leakage-checks the configs (CPU).
"""
import argparse
import json

from cluster import jobcommon as jc
from dsgx import paths

NAME = "a7-12b"
MODEL = {"name": "gemma-3-12b-it", "sae_release": "gemma-scope-2-12b-it-res", "sae_id": "layer_24_width_16k_l0_medium"}
ATTACKS = [{"name": "dilution", "pad": 400, "position": "before", "source": "wikitext"},
           {"name": "dilution", "pad": 1600, "position": "before", "source": "wikitext"},
           {"name": "translate", "lang": "fr", "min_chrf": 40}]


def best_fix():
    sel = jc.read_json(paths.results_dir() / "COMBINE_SELECTION.json", {}) or {}
    det = (sel.get("slots") or {}).get("detector") or ""
    w = int(det.split("-w")[1]) if det.startswith("window-w") else 16
    return {"name": "gated", "gate": {"type": "window", "w": w}, "calib": {"fpr": 0.05, "n_max": 1000, "source": "mmlu-dev"}}, det


def configs():
    fix, det = best_fix()
    methods = {"base": {"name": "base"},
               "dsg": {"name": "gated", "gate": {"type": "rho", "n_features": 20, "retain_pct": 95}, "calib": {"source": "cache-retain"}},
               "best-fix": fix}
    out = []
    for tag, m in methods.items():
        out.append({"exp_id": "A7-12b", "case": "bio", "split": "test", "view": "raw", "batch_size": 1, "seed": 0,
                    "model": dict(MODEL), "method": m, "attack": {"name": "none"}, "dataset_label": f"{tag}-forget+util",
                    "datasets": ["@forget", "high_school_geography", "human_aging"]})
        for att in ATTACKS:
            out.append({"exp_id": "A7-12b", "case": "bio", "split": "test", "view": "raw", "batch_size": 1, "seed": 0,
                        "model": dict(MODEL), "method": m, "attack": dict(att), "dataset_label": f"{tag}-forget",
                        "datasets": ["@forget"]})
    return out, det


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", action="store_true")
    a = ap.parse_args(argv)
    from dsgx.checks.leakage import check_runs
    from dsgx.run import resolve, run

    cfgs, det = configs()
    errs = check_runs(cfgs)
    for c in cfgs:
        resolve(c)
    jc.log(NAME, f"{len(cfgs)} runs; leakage errors {errs}; best-fix source: {det or 'default window-w16'}")
    if errs:
        return 4
    if a.plan or jc.TINY:
        print(json.dumps([{"method": c["method"]["name"], "attack": c["attack"]["name"], "label": c["dataset_label"]} for c in cfgs]))
        return 0
    jc.require_gpu(40)
    dirs = {}
    for c in cfgs:
        dirs[f"{c['dataset_label']}/{c['attack']['name']}{c['attack'].get('pad', c['attack'].get('lang', ''))}"] = str(run(c))
    jc.summary(NAME, {"model": MODEL, "runs": dirs, "headline": jc.headlines(dirs), "best_fix_source": det or "window-w16"})
    jc.log(NAME, "done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
