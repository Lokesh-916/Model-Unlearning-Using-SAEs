"""E2e selection rule v2 (experiments/E2e/common.py): eligibility, pick, tie-break, none-eligible."""
from experiments.E2e.common import RULE_ATTACKS, eligibility, select_v2

SIG = {"sig_higher": True}
NS = {"sig_higher": False}


def cand(name, util=0.57, fpr=0.045, bio=0.55, forget=0.30, bio_sig=True, fsig=()):
    return {"name": name, "utility": util, "fpr": fpr, "benign_bio": bio,
            "forget": {a: (forget[a] if isinstance(forget, dict) else forget) for a in RULE_ATTACKS},
            "bio_vs_dsg": SIG if bio_sig else NS, "forget_vs_sg1": {a: SIG if a in fsig else NS for a in RULE_ATTACKS}}


def test_each_condition_excludes():
    dsg = 0.567
    assert eligibility(cand("ok"), dsg)["eligible"]
    assert eligibility(cand("u", util=dsg - 0.005), dsg)["eligible"]          # boundary inclusive
    assert not eligibility(cand("u", util=dsg - 0.0051), dsg)["eligible"]
    assert not eligibility(cand("f", fpr=0.051), dsg)["eligible"]
    assert not eligibility(cand("b", bio_sig=False), dsg)["eligible"]
    for a in RULE_ATTACKS:
        assert not eligibility(cand("a", fsig=(a,)), dsg)["eligible"]


def test_pick_worst_case_then_bio_then_order():
    dsg = 0.567
    c1 = cand("c1", forget={"clean": 0.2, "pad400": 0.4, "split3": 0.2, "hindi": 0.2})   # worst 0.4
    c2 = cand("c2", forget=0.3, bio=0.50)
    c3 = cand("c3", forget=0.3, bio=0.60)
    c4 = cand("c4", forget=0.1, bio_sig=False)   # best forget but ineligible
    s = select_v2([c1, c2, c3, c4], dsg)
    assert s["selected"]["name"] == "c3" and s["closest"] is None
    s = select_v2([cand("x", forget=0.3), cand("y", forget=0.3)], dsg)
    assert s["selected"]["name"] == "x"


def test_none_eligible_reports_closest():
    dsg = 0.567
    a = cand("a", bio_sig=False, fsig=("hindi",), forget=0.2)   # 2 failed
    b = cand("b", bio_sig=False, forget=0.35)                    # 1 failed
    c = cand("c", bio_sig=False, forget=0.30)                    # 1 failed, lower worst case
    s = select_v2([a, b, c], dsg)
    assert s["selected"] is None and s["closest"]["name"] == "c"
