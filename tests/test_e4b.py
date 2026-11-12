"""E4b: tie-aware conformal threshold and Step B config (CPU, no model)."""
import numpy as np
import pytest

from experiments.E4b.common import STEP_B_ATTACKS, conformal_threshold


def test_conformal_threshold_with_ties():
    s = np.r_[np.full(941, 7.062), np.arange(19) + 8.0]          # 941 tied + 19 distinct above
    r = conformal_threshold(s, 0.05)
    assert r["threshold"] == pytest.approx(7.062) and r["n_cal_above"] == 19
    r = conformal_threshold(s, 0.017)                              # (1 + k) / 961 <= 0.017 -> k <= 15
    assert r["n_cal_above"] == 15 and r["threshold"] == pytest.approx(11.0)
    assert (1 + r["n_cal_above"]) / 961 <= 0.017


def test_conformal_threshold_never_below_all():
    s = np.full(100, 3.0)                                          # all tied: the tie value, fires on nothing
    r = conformal_threshold(s, 0.05)
    assert r["threshold"] == 3.0 and r["n_cal_above"] == 0


def test_step_b_attacks_build():
    from dsgx.attacks.registry import make_attack
    for spec in STEP_B_ATTACKS.values():
        assert make_attack(spec) is not None


def test_config_generates(tmp_path):
    import subprocess, sys, yaml
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, str(root / "scripts" / "make_e4b_config.py")], check=True, capture_output=True)
    cfg = yaml.safe_load((root / "configs" / "experiments" / "E4b.yaml").read_text())
    assert cfg["base"]["split"] == "dev" and len(cfg["runs"]) == 20
    assert all(r["datasets"] == ["@forget"] for r in cfg["runs"])
