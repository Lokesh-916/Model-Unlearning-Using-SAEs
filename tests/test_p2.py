"""P2 (bio + cyber gates together): CPU tests of the fixed constants and the composition logic."""
import numpy as np
import pandas as pd

from experiments.P2 import common as C
from experiments.P2.summary import compose, compose_pred, config_sets


def test_frozen_constants():
    assert C.BIO_THRESHOLD == 7.084529876708984
    assert C.CYBER_THRESHOLD == 13.750207901000977
    assert C.BIO_METHOD["gate"]["fit_sha256"].startswith("d4ff0a1b")
    assert C.CYBER_METHOD["gate"]["fit_sha256"].startswith("14994871")


def test_utility43_excludes_both_topics():
    u = C.utility43()
    assert len(u) == 43
    assert not set(u) & set(C.BENIGN_BIO) and not set(u) & set(C.BENIGN_CS)
    for s in ["clinical_knowledge", "college_medicine", "nutrition", "professional_medicine"]:
        assert s not in u


def test_clamp_sets():
    assert C.clamp_set(False, False) == ()
    assert C.clamp_set(True, True) == ("bio", "cyber")
    assert C.set_name(("bio", "cyber")) == "bio+cyber" and C.set_name(()) == "none"
    bf, cf = np.array([1, 0, 1, 0], bool), np.array([0, 1, 1, 0], bool)
    assert config_sets("bio", bf, cf) == [("bio",), (), ("bio",), ()]
    assert config_sets("cyber", bf, cf) == [(), ("cyber",), ("cyber",), ()]
    assert config_sets("both", bf, cf) == [("bio",), ("cyber",), ("bio", "cyber"), ()]


def test_compose_takes_the_right_prediction():
    f = pd.DataFrame({"gold": [0, 1, 2, 3], "bio_score": [100.0, 0.0, 100.0, 0.0], "cyber_score": [0.0, 100.0, 100.0, 0.0],
                      "pred_none": [9, 9, 9, 3], "pred_bio": [0, 9, 9, 9], "pred_cyber": [9, 1, 9, 9],
                      "pred_bio+cyber": [np.nan, np.nan, 2, np.nan]})
    pred, fired = compose_pred(f, "both")
    assert pred.tolist() == [0, 1, 2, 3] and fired.tolist() == [True, True, True, False]
    corr, _ = compose(f, "bio")
    assert corr.tolist() == [True, False, False, True]          # bio alone: row 1 -> base, row 2 -> bio clamp only


def test_check_items_are_about_five_percent():
    keys = [f"bio/wmdp_clean|wmdp-bio:{i}" for i in range(4000)]
    share = np.mean([C.is_check(k) for k in keys])
    assert 0.03 < share < 0.07
    assert C.is_check(keys[0]) == C.is_check(keys[0])
