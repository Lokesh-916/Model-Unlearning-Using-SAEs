"""P1 CPU tests: config shape, composite rule pieces, conformal reuse (no model)."""
from pathlib import Path

import numpy as np
import yaml

from experiments.E4b.common import conformal_threshold
from experiments.P1.common import ATTACKS, BENIGN_CS, CLAMP_FORCED, DSG_CHAT, DSG_PAPER


def test_config_runs():
    cfg = yaml.safe_load(Path("configs/experiments/P1.yaml").read_text())
    labels = [r["dataset_label"] for r in cfg["runs"]]
    assert len(labels) == 12 and len(set(labels)) == 12
    assert cfg["base"]["split"] == "dev" and cfg["base"]["case"] == "cyber" and cfg["base"]["batch_size"] == 1
    for r in cfg["runs"]:
        if r["dataset_label"].endswith("-clean"):
            assert r["datasets"] == ["@forget"] + BENIGN_CS + ["@utility"]
        else:
            assert r["datasets"] == ["@forget"]
    assert set(ATTACKS) == {"clean", "pad400", "split3"}


def test_dsg_configs():
    assert CLAMP_FORCED["tau"] < 0 and {k: v for k, v in CLAMP_FORCED.items() if k != "tau"} == DSG_PAPER
    assert DSG_CHAT["retain_corpus"] == "mmlu-aux-chat" and DSG_CHAT["n_features"] == 200


def test_conformal_tie_floor():
    s = np.array([7.0] * 95 + [8.0, 9.0, 10.0, 11.0, 12.0])
    t = conformal_threshold(s, 0.05)
    assert t["threshold"] == 8.0 and t["n_cal_above"] == 4
