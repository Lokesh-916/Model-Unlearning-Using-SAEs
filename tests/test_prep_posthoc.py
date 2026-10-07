"""Post-hoc exploratory runs (purpose posthoc-exploratory, e.g. PH-X1-conformal) never reach a claim rule."""
from types import SimpleNamespace

from dsgx.analysis import claims


def test_posthoc_runs_and_pairs_are_dropped(monkeypatch):
    seen = {}
    monkeypatch.setattr(claims, "ch5", lambda runs, paired: seen.update(runs=runs, paired=paired)
                        or claims._claim("C-H5", "Inconclusive", []))
    for f in ("ch1", "ch2", "ch3", "ch4"):
        monkeypatch.setattr(claims, f, lambda runs, _f=f: claims._claim(_f.replace("ch", "C-H"), "Inconclusive", []))
    monkeypatch.setattr(claims, "ch7", lambda runs, paired, extra: claims._claim("C-H7", "Inconclusive", []))
    ph = SimpleNamespace(name="PH-X1-conformal__gated", hardware="gpuws", config={"config": {"purpose": "posthoc-exploratory"}})
    x1 = SimpleNamespace(name="X1__gated", hardware="gpuws", config={"config": {"purpose": "report"}})
    claims.evaluate([ph, x1], [{"run": ph.name}, {"run": x1.name}, {"other": 1}])
    assert [r.name for r in seen["runs"]] == ["X1__gated"]
    assert seen["paired"] == [{"run": "X1__gated"}, {"other": 1}]
    assert claims.is_posthoc(ph) and not claims.is_posthoc(x1)


def test_ph_task_runs_without_purpose_are_posthoc():
    # PH-union / PH-tofucal task runs (tofu-metrics, open-ended QA, paired) carry no base.purpose in config.json
    t = SimpleNamespace(name="tofu-metrics", exp="PH-tofucal", config={"config": {}})
    assert claims.is_posthoc(t)
    assert not claims.is_posthoc(SimpleNamespace(name="tofu-metrics", exp="X1-suite", config={"config": {}}))
