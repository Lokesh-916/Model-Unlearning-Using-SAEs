"""Run directories and files (MASTER_PLAN section 7).

$DSG_RESULTS/runs/<exp_id>/<run_id>/ with
run_id = <exp_id>__<method>__<attack>__<dataset>__<split>__s<seed>__<hash8>
holding config.json, metrics.json, items.parquet, traces.npz and a DONE marker.
Hazardous generations never go here (only hashes); text goes to $DSG_PRIVATE.
"""
import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from dsgx import paths
from dsgx.util import (atomic_write_json, gpu_info, git_info, now_iso, package_versions,
                       stable_hash)

ITEM_COLUMNS = [
    "item_id", "dataset", "subject", "split", "language", "attack", "attack_params",
    "prompt_len", "pad_len", "gold", "pred", "correct", "prob_A", "prob_B", "prob_C", "prob_D",
    "entropy", "margin", "in_dsg_subset", "rho", "tau", "gate_fired", "window_max", "cusum_max",
    "probe_score", "layer_gate_scores", "n_sel_fired", "sel_max", "top_other_ids",
    "top_other_vals", "recon_mse", "l0", "text_hash", "first_fire_token", "answer_score",
    "gibberish", "n_tokens",
]
MAX_TRACES = 200


def _slug(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "-", str(s)).strip("-")


def make_run_id(cfg: dict) -> str:
    method = cfg["method"]["name"]
    attack = (cfg.get("attack") or {"name": "none"})["name"]
    ds = cfg["datasets"]
    dataset = cfg.get("dataset_label") or ("+".join(ds) if len(ds) <= 2 else f"{len(ds)}sets")
    if cfg.get("case"):
        dataset = f"{cfg['case']}-{dataset}"
    return "__".join([_slug(cfg["exp_id"]), _slug(method), _slug(attack), _slug(dataset),
                      cfg["split"], f"s{cfg.get('seed', 0)}", stable_hash(cfg)])


class RunLogger:
    def __init__(self, cfg: dict, root: Path | None = None):
        self.cfg = cfg
        self.run_id = make_run_id(cfg)
        self.dir = Path(root or paths.runs_dir()) / _slug(cfg["exp_id"]) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t0 = time.time()
        self.start = now_iso()

    @property
    def done(self) -> bool:
        return (self.dir / "DONE").exists()

    def write_config(self, extra: dict):
        atomic_write_json(self.dir / "config.json", {
            "run_id": self.run_id, "config": self.cfg, "git": git_info(), "versions": package_versions(),
            "hardware": gpu_info(), "seed": self.cfg.get("seed", 0), "start_time": self.start,
            "end_time": None, **extra})

    def finish_config(self):
        p = self.dir / "config.json"
        c = json.loads(p.read_text())
        c["end_time"] = now_iso()
        c["wall_seconds"] = round(time.time() - self.t0, 2)
        atomic_write_json(p, c)

    def write_items(self, rows: list[dict]):
        import pandas as pd

        df = pd.DataFrame(rows)
        for c in ITEM_COLUMNS:
            if c not in df.columns:
                df[c] = None
        for c in ("attack_params", "sel_max", "top_other_ids", "top_other_vals", "layer_gate_scores"):
            df[c] = df[c].map(lambda v: None if v is None else json.dumps(v))
        tmp = self.dir / ".items.parquet.tmp"
        df[ITEM_COLUMNS + [c for c in df.columns if c not in ITEM_COLUMNS]].to_parquet(tmp, index=False)
        tmp.replace(self.dir / "items.parquet")

    def write_traces(self, traces: dict[str, np.ndarray]):
        """traces: item_id -> per-token fire indicator array. Also stores running rho."""
        keys = list(traces)[:MAX_TRACES]
        arrs = {}
        for i, k in enumerate(keys):
            f = np.asarray(traces[k], dtype=np.uint8)
            arrs[f"fire__{i}"] = f
            arrs[f"rho__{i}"] = (np.cumsum(f) / np.arange(1, len(f) + 1)).astype(np.float32)
        arrs["item_ids"] = np.array(keys)
        tmp = self.dir / ".traces.tmp.npz"
        np.savez_compressed(tmp, **arrs)
        tmp.replace(self.dir / "traces.npz")

    def write_metrics(self, metrics: dict):
        atomic_write_json(self.dir / "metrics.json", metrics)

    def mark_done(self, headline: dict):
        self.finish_config()
        atomic_write_json(self.dir / "DONE", {"time": now_iso(), "headline": headline})
        append_index(self, headline)


def append_index(run: RunLogger, headline: dict):
    """One row per finished run in $DSG_RESULTS/index.csv."""
    p = paths.results_dir() / "index.csv"
    new = not p.exists()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["time", "exp_id", "run_id", "method", "attack", "split", "seed", "headline", "run_dir"])
        c = run.cfg
        w.writerow([now_iso(), c["exp_id"], run.run_id, c["method"]["name"],
                    (c.get("attack") or {"name": "none"})["name"], c["split"], c.get("seed", 0),
                    json.dumps(headline), str(run.dir)])
