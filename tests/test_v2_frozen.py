"""CPU checks of the frozen StreamGuard v2 definition and the V2 configs (no model, no TEST data)."""
from experiments.V2 import common as C


def test_threshold_is_midpoint_and_above_tie():
    assert C.THRESHOLD == (C.THRESHOLD_E4B + C.NEXT_DEV_SCORE) / 2
    assert C.THRESHOLD_E4B < C.THRESHOLD < C.NEXT_DEV_SCORE
    assert C.METHOD["gate"]["threshold"] == C.THRESHOLD and C.METHOD["variant"] == "streamguard-v2"


def test_cusum_matches_e4a():
    from experiments.E4a.questions import cusum_max
    v = [0.5, -2.0, 1.5, 1.0, -0.25, 3.0]
    assert C.cusum_max(v) == cusum_max(v) == 5.25


def test_configs():
    import importlib.util, pathlib
    p = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "make_v2_config.py"
    spec = importlib.util.spec_from_file_location("mk", p)
    mk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mk)
    t = mk.test_runs()
    sg2 = [r for r in t if r["method"].get("variant") == "streamguard-v2"]
    assert len(sg2) == 5 * (2 + len(C.ATTACKS))
    assert all(r.get("seed") in C.SEEDS for r in t)
    # comparators never re-run where an identical X1 log exists
    assert not [r for r in t if r["v2"]["cond"] != "sg2" and r["v2"]["attack"] in C.IN_X1]
    assert len(mk.devcheck_runs()) == len(C.DEV_CONDS)
