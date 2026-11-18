"""P1-T CPU tests: frozen constants vs P1's logged artifacts, config shape (no model)."""
import hashlib
import json
from pathlib import Path

import os

import pytest
import yaml

from experiments.P1T.common import (ATTACKS, BENIGN_CS, CLAMP_FEATURES, FIT_REL, FIT_SHA256, METHOD, SEEDS,
                                    THRESHOLD)


PROJ = Path(os.environ.get("DSG_PROJECT", "/home/amaloch/projects/mechunlearn-project"))
CACHE, RUNS = PROJ / "dsg_cache", PROJ / "dsg_results" / "runs"   # the real artifacts (conftest swaps the env dirs)
real = pytest.mark.skipif(not (RUNS / "P1" / "summary" / "metrics.json").exists(), reason="P1 artifacts not on disk")


@real
def test_fit_file_pinned():
    p = CACHE / FIT_REL
    assert hashlib.sha256(p.read_bytes()).hexdigest() == FIT_SHA256
    rec = json.loads(p.read_text())
    assert rec["layer"] == 8 and rec["sae_id"] == METHOD["gate"]["sae_id"] and len(rec["features"]) == 20


@real
def test_threshold_is_p1s():
    m = json.loads((RUNS / "P1" / "summary" / "metrics.json").read_text())
    c = m["cells"]["L08|op5"]
    assert c["threshold"] == THRESHOLD and c["alpha"] == 0.05


@real
def test_clamp_features_are_p1s():
    hits = sorted((RUNS / "P1").glob("*__cyber-p1-clampF-clean__*/config.json"))
    mi = json.loads(hits[-1].read_text())["method_info"]
    assert mi["features"] == CLAMP_FEATURES and mi["multiplier"] == 1000.0
    r = METHOD["response"]
    assert r["tau"] < 0 and r["calib_seed"] == 0 and r["n_features"] == 100 and r["retain_pct"] == 90


def test_test_config():
    cfg = yaml.safe_load(Path("configs/experiments/P1T.yaml").read_text())
    assert cfg["base"]["split"] == "test" and cfg["base"]["batch_size"] == 1 and cfg["base"]["case"] == "cyber"
    labels = [(r["dataset_label"], r.get("seed", 0)) for r in cfg["runs"]]
    assert len(labels) == len(set(labels)) == 3 * (1 + 3 * len(SEEDS))
    for r in cfg["runs"]:
        k = r["dataset_label"].rsplit("-", 1)[1]
        assert r["attack"] == ATTACKS[k]
        assert r["datasets"] == (["@forget"] + BENIGN_CS + ["@utility"] if k == "clean" else ["@forget"])
    assert "P1T-devcheck-check" in cfg["jobs"]["deps"]
    dev = yaml.safe_load(Path("configs/experiments/P1T-devcheck.yaml").read_text())
    assert dev["base"]["split"] == "dev" and all(r["method"] == METHOD for r in dev["runs"])
