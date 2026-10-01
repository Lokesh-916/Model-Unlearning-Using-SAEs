from dsgx.checks.leakage import check_runs
from dsgx.config import expand


EXP = {"exp_id": "T", "base": {"case": "bio", "split": "dev", "purpose": "select",
                               "method": {"name": "base"}, "datasets": ["@forget"]},
       "grid": {"method.n_features": [10, 20], "seed": [0, 1]},
       "smoke": {"base": {"limit": 2}, "grid": {"seed": [0]}}}


def test_expand_grid_and_smoke():
    runs = expand(EXP)
    assert len(runs) == 4 and {r["seed"] for r in runs} == {0, 1}
    sm = expand(EXP, smoke=True)
    assert len(sm) == 1 and sm[0]["limit"] == 2 and sm[0]["exp_id"] == "T-smoke"


def test_leakage_rules():
    ok = expand(EXP)
    assert check_runs(ok) == []
    bad = [dict(ok[0], split="test")]
    assert any("purpose=select" in e for e in check_runs(bad))
    calib = dict(ok[0], purpose="report", split="test",
                 method={"name": "base", "calib_datasets": [{"dataset": "wmdp-bio", "split": "test"}]})
    errs = check_runs([calib])
    assert any("calibrates on wmdp-bio/test" in e for e in errs)
    assert any("also evaluated" in e for e in errs)
    clean = dict(calib, method={"name": "base", "calib_datasets": [{"dataset": "virology", "split": "dev"}]})
    assert check_runs([clean]) == []
