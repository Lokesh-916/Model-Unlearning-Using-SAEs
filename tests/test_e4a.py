import numpy as np

from experiments.E4a.common import verdict, WMDP_CONDS
from experiments.E4a.fit import auroc, llr, select
from experiments.E4a.questions import cusum_max


def test_select_rare_on_both():
    p = np.array([0.5, 0.5, 0.001, 0.2])
    qb = np.array([0.4, 0.01, 0.0, 0.0])
    qw = np.array([0.0, 0.01, 0.0, 0.3])
    feats, _ = select(p, qb, qw, k=3)
    assert feats[0] == 1 and 2 not in feats  # 2 below P_MIN; 3 is frequent on wiki
    assert feats.index(0) > 0 or 0 not in feats


def test_llr_sign():
    w1, w0 = llr(np.array([0.5]), np.array([0.01]))
    assert w1[0] > 0 and w0[0] < 0


def test_auroc_and_cusum():
    assert auroc([2, 3], [0, 1]) == 1.0
    assert auroc([1], [1]) == 0.5
    assert cusum_max([1, -5, 2, 2]) == 4.0


def test_verdict_line():
    good = {"layer": "L09", "detector": "x", "recall": {c: 0.9 for c in WMDP_CONDS}, "benign_fire": 0.2}
    bad = {**good, "benign_fire": 0.31}
    assert verdict([good])["met"] and not verdict([bad])["met"]
