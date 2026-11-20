"""P4 (CPU): fixed lines, verdict, drift CUSUM, settings table."""
import numpy as np

from experiments.P4 import common as c


def test_cusum_drift_zero_matches_e4a():
    from experiments.E4a.questions import cusum_max as e4a
    v = list(np.random.default_rng(0).normal(0, 1, 200))
    assert c.cusum_max(v, 0.0) == e4a(v)
    assert c.cusum_max(v, 0.5) <= e4a(v)


def test_holds_lines():
    dsg = {"benign_bio": 0.43, "utility": 0.5667}
    ok = {"worst_forget": 0.32, "benign_bio": 0.48, "utility": 0.5567, "fire_utility": 0.05}
    assert c.holds(ok, dsg)["holds"]
    for k, v in [("worst_forget", 0.3201), ("benign_bio", 0.4799), ("utility", 0.5566), ("fire_utility", 0.0501)]:
        assert not c.holds({**ok, k: v}, dsg)["holds"], k


def test_verdict_80pct_of_11():
    f = {s: True for s in c.SETTINGS}
    assert len(f) == 11 and c.verdict(f)["pass"] and c.verdict(f)["need"] == 9
    f["F10"] = f["F40"] = False
    assert c.verdict(f)["pass"]  # 9 / 11
    f["L07"] = False
    assert not c.verdict(f)["pass"]  # 8 / 11


def test_settings_one_at_a_time():
    s0 = c.SETTINGS["S0"]
    for name, s in c.SETTINGS.items():
        assert sum(a != b for a, b in zip(s, s0)) <= 1, name
    assert c.score_col("OP3") == c.score_col("S0") == c.score_col("OP1.15")


def test_null_drift_unit():
    fit = {"q_benign": [0.1], "q_wiki": [0.2], "w1": [1.0], "w0": [-0.5]}
    assert abs(c.null_drift_unit(fit, 1) - (-(0.2 * 1.0 + 0.8 * -0.5))) < 1e-12
