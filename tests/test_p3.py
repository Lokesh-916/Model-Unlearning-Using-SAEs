"""P3 (data efficiency of the v2 gate): CPU tests of subsets, the threshold rule and the pass line."""
import numpy as np

from experiments.P3 import common as C


def test_subsets_deterministic_and_sized():
    a = C.subset(18554, 20, 1, "benign")
    assert len(a) == round(0.2 * 18554) and np.array_equal(a, C.subset(18554, 20, 1, "benign"))
    assert len(np.unique(a)) == len(a) and a.min() >= 0 and a.max() < 18554
    assert not np.array_equal(a, C.subset(18554, 20, 2, "benign"))
    assert not np.array_equal(C.subset(1000, 40, 0, "forget"), C.subset(1000, 40, 0, "benign"))
    assert np.array_equal(C.subset(77, 100, 2, "forget"), np.arange(77))


def test_midpoint_rule_reproduces_frozen_threshold():
    t, nxt = 7.06202507019043, 7.107034683227539
    r = C.midpoint_threshold(t, [7.06202507019043, 7.06202507019043, nxt, 9.0, 3.0])
    assert r["threshold"] == C.FROZEN_THRESHOLD and r["next"] == nxt


def test_pass_line():
    v2 = {"clean": 0.24, "pad400": 0.25, "split3": 0.25, "hindi": 0.24}
    ok = {"recall_clean": 0.95, "recall_pad400": 0.9, "recall_split3": 0.9, "recall_hindi": 0.86, "benign_fire": 0.35,
          "forget_clean": 0.25, "forget_pad400": 0.27, "forget_split3": 0.26, "forget_hindi": 0.26}
    assert C.passes(ok, v2)["pass"]
    assert not C.passes({**ok, "recall_hindi": 0.84}, v2)["pass"]
    assert not C.passes({**ok, "benign_fire": 0.41}, v2)["pass"]
    assert not C.passes({**ok, "forget_pad400": 0.2701}, v2)["pass"]
