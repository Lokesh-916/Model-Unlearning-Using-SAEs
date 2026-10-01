import numpy as np

from dsgx.eval import stats


def test_bootstrap_ci_contains_mean_and_is_deterministic():
    x = np.r_[np.ones(30), np.zeros(70)]
    a = stats.bootstrap_ci(x, n_boot=2000)
    b = stats.bootstrap_ci(x, n_boot=2000)
    assert a == b
    assert a["n"] == 100 and abs(a["mean"] - 0.3) < 1e-12
    assert a["lo"] < 0.3 < a["hi"]
    assert 0.18 < a["lo"] < 0.24 and 0.36 < a["hi"] < 0.42  # ~ +-1.96*sqrt(.21/100)


def test_bootstrap_degenerate():
    assert stats.bootstrap_ci([])["n"] == 0
    c = stats.bootstrap_ci([1, 1, 1])
    assert c["lo"] == c["hi"] == 1.0


def test_stratified_pooled_vs_unweighted():
    g = {"a": [1] * 9, "b": [0] * 1}  # pooled 0.9, unweighted 0.5
    r = stats.stratified_bootstrap(g, n_boot=500)
    assert abs(r["pooled"]["mean"] - 0.9) < 1e-12
    assert abs(r["unweighted"]["mean"] - 0.5) < 1e-12
    assert r["pooled"]["n"] == 10 and r["unweighted"]["n_groups"] == 2


def test_mcnemar_exact():
    a = np.array([1] * 10 + [0] * 0 + [1] * 5)
    b = np.array([0] * 10 + [0] * 0 + [1] * 5)
    r = stats.mcnemar(a, b)
    assert r["n10"] == 10 and r["n01"] == 0
    assert abs(r["p"] - 2 / 1024) < 1e-12
    assert stats.mcnemar([1, 0], [1, 0])["p"] == 1.0


def test_paired_bootstrap_detects_difference():
    rng = np.random.default_rng(0)
    a = rng.random(400) < 0.6
    b = a.copy()
    b[:80] = False
    r = stats.paired_bootstrap(a, b, n_boot=2000)
    assert r["diff"] > 0 and r["lo"] > 0 and r["p"] < 0.01
    same = stats.paired_bootstrap(a, a, n_boot=200)
    assert same["diff"] == 0 and same["p"] == 1.0


def test_seed_summary():
    s = stats.seed_summary([0.3, 0.31, 0.29])
    assert s["n_seeds"] == 3 and s["lo"] < 0.3 < s["hi"]
