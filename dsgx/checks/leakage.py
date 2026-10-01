"""Leakage checks (rule 3.4). Run before every job group; a failure blocks the group.

    python -m dsgx.checks.leakage [--config configs/experiments/X.yaml ...] [--corpus]

1. Splits: every split file hashes correctly, dev and test are disjoint and cover all items.
2. Configs: runs with purpose select/tune use dev only; nothing selects on test; method
   calibration datasets (`calib_datasets: [{dataset, split}]`) never overlap evaluated
   (dataset, split) pairs and never use test.
3. Corpus (--corpus, cached per activation cache): calibration rows must not contain eval
   questions (word 8-gram containment >= 0.8 fails; >= 0.5 is reported as a warning).
"""
import argparse
import json
import re
import sys
from pathlib import Path

from dsgx import paths
from dsgx.data.splits import ALL_DATASETS, load_split_file
from dsgx.util import atomic_write_json, now_iso

NGRAM = 8


def check_splits(datasets=None) -> list[str]:
    errs = []
    for d in datasets or ALL_DATASETS:
        try:
            rec = load_split_file(d)
        except FileNotFoundError:
            errs.append(f"{d}: split file missing")
            continue
        except ValueError as e:
            errs.append(str(e))
            continue
        dev, test = set(rec["dev"]), set(rec["test"])
        if dev & test:
            errs.append(f"{d}: dev/test overlap ({len(dev & test)} items)")
        if dev | test != set(range(rec["n"])):
            errs.append(f"{d}: dev+test do not cover all {rec['n']} items")
    return errs


def check_runs(runs: list[dict]) -> list[str]:
    from dsgx.run import resolve

    errs = []
    for r in runs:
        tag = f"{r.get('exp_id')}:{r.get('method', {}).get('name')}"
        if r.get("purpose") in ("select", "tune") and r.get("split", "test") != "dev":
            errs.append(f"{tag}: purpose={r['purpose']} on split {r.get('split', 'test')}")
            continue
        c = resolve(r)
        sel = c.get("select_from") or {}
        if sel and sel.get("split", "dev") != "dev":
            errs.append(f"{tag}: selects on split {sel.get('split')}")
        evald = {(d, c["split"]) for d in c["datasets"]}
        evald |= {(d, "all") for d in c["datasets"]} if c["split"] == "all" else set()
        for cd in c["method"].get("calib_datasets", []) or []:
            ds, sp = cd["dataset"], cd.get("split", "dev")
            if sp in ("test", "all"):
                errs.append(f"{tag}: calibrates on {ds}/{sp}")
            if (ds, sp) in evald or (c["split"] == "all" and ds in c["datasets"]):
                errs.append(f"{tag}: calibration set {ds}/{sp} is also evaluated")
    return errs


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _ngrams(ws):
    return {hash(tuple(ws[i:i + NGRAM])) for i in range(len(ws) - NGRAM + 1)}


def check_corpus(cache_dir: Path, datasets=None) -> dict:
    """Containment of eval questions in the calibration token rows of an activation cache."""
    import numpy as np
    from transformers import AutoTokenizer

    from dsgx.data.mcq import load_mcq

    meta = json.loads((cache_dir / "meta.json").read_text())
    out_path = paths.cache_dir() / "leakage" / f"{meta['key']}.json"
    if out_path.exists():
        return json.loads(out_path.read_text())
    tok = AutoTokenizer.from_pretrained("google/" + meta["model"])
    grams = set()
    for part in ("forget", "retain"):
        rows = np.load(cache_dir / f"tokens_{part}.npy", mmap_mode="r")
        for r in rows:
            grams |= _ngrams(_words(tok.decode(r.tolist(), skip_special_tokens=True)))
    fails, warns = [], []
    for d in datasets or ALL_DATASETS:
        for it in load_mcq(d):
            g = _ngrams(_words(it.question + " " + " ".join(it.choices)))
            if len(g) < 3:
                continue
            frac = len(g & grams) / len(g)
            if frac >= 0.8:
                fails.append({"item": it.item_id, "containment": round(frac, 3)})
            elif frac >= 0.5:
                warns.append({"item": it.item_id, "containment": round(frac, 3)})
    res = {"cache": meta["key"], "n_fail": len(fails), "n_warn": len(warns), "fails": fails[:200],
           "warns": warns[:200], "time": now_iso()}
    atomic_write_json(out_path, res)
    return res


def check_task_args(job: dict) -> list[str]:
    """Task args: anything used to calibrate, select, tune or train must not be the test split."""
    errs = []

    def walk(d, path=""):
        if isinstance(d, dict):
            for k, v in d.items():
                kp = f"{path}.{k}" if path else k
                if (isinstance(v, str) and v in ("test", "all") and k.endswith("split")
                        and any(w in kp for w in ("calib", "select", "tune", "train", "fit"))):
                    errs.append(f"{job['id']}: {kp}={v}")
                walk(v, kp)
        elif isinstance(d, list):
            for x in d:
                walk(x, path)

    walk(job.get("args") or {})
    return errs


def check_job(job: dict) -> list[str]:
    """Fast checks for one queue job (splits + its configs). Called by the worker."""
    errs = check_splits()
    if job.get("kind") == "task":
        errs += check_task_args(job)
    if job.get("kind") == "runs" and job.get("config_path"):
        from dsgx.config import expand, load_experiment

        exp = load_experiment(Path(job["worktree"]) / job["config_path"])
        runs = expand(exp, smoke=job.get("smoke", False))
        errs += check_runs([runs[i] for i in job.get("run_indices") or range(len(runs))])
    return errs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", nargs="*", default=[])
    ap.add_argument("--corpus", action="store_true", help="also scan every built activation cache")
    a = ap.parse_args(argv)
    from dsgx.config import expand, load_experiment

    report = {"time": now_iso(), "splits": check_splits(), "configs": {}, "corpus": {}}
    for cp in a.config:
        exp = load_experiment(cp)
        report["configs"][cp] = check_runs(expand(exp)) + check_runs(expand(exp, smoke=True))
    if a.corpus:
        for cd in sorted((paths.cache_dir() / "actcache").glob("*")):
            if (cd / "meta.json").exists():
                report["corpus"][cd.name] = check_corpus(cd)
    errs = report["splits"] + [e for v in report["configs"].values() for e in v]
    errs += [f"corpus {k}: {v['n_fail']} eval items contained" for k, v in report["corpus"].items() if v["n_fail"]]
    report["errors"] = errs
    report["pass"] = not errs
    atomic_write_json(paths.results_dir() / "checks" / "leakage_latest.json", report)
    for e in errs:
        print("LEAKAGE:", e)
    for k, v in report["corpus"].items():
        print(f"corpus {k}: fail={v['n_fail']} warn={v['n_warn']}")
    print("LEAKAGE CHECK", "PASS" if not errs else "FAIL")
    return 0 if not errs else 1


if __name__ == "__main__":
    sys.exit(main())
