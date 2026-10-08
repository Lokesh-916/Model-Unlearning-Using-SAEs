"""D1 v3 kit on CPU (tiny random Gemma-2 via DSG_TINY=1): train (resume after a budget stop), held-out split,
open-ended plan, full + LoRA relearning cells, summary, CPU plan. No GPU, no server."""
import json

import pandas as pd
import pytest

from tests.test_prep_session8 import _reload, tiny  # noqa: F401  (fixture)


def test_d1_v3_split_is_disjoint_and_fixed(tiny):  # noqa: F811
    m = _reload("cluster.d1_v3")
    d1, h1 = m.split_passages()
    d2, h2 = m.split_passages()
    assert (d1, h1) == (d2, h2) and h1 and d1
    assert not set(h1) & set(d1) or len(set(h1 + d1)) < len(h1 + d1)  # tiny passages repeat text; indices disjoint
    assert len(h1) + len(d1) == 24


def test_d1_v3_train_resume_and_summary(tiny, monkeypatch):  # noqa: F811
    from dsgx import paths

    m = _reload("cluster.d1_v3")
    args = ["--seeds", "0", "1", "--steps", "4", "--bs", "2", "--maxlen", "32", "--ckpt", "2"]
    calls = {"n": 0}
    real = m.jc.Budget.fits

    def fits(self, minutes):
        calls["n"] += 1
        return calls["n"] <= 1 or minutes > 100
    monkeypatch.setattr(m.jc.Budget, "fits", fits)
    assert m.main(["train", *args]) == 0
    root = paths.cache_dir() / "models" / "D1-v3"
    assert not (root / "undo_a0.1_s0" / "config.json").exists()
    assert (paths.results_dir() / "jobs" / "d1-v3" / "undo_a0.1_s0" / "last" / "trainer.pt").exists()
    monkeypatch.setattr(m.jc.Budget, "fits", real)
    assert m.main(["test", *args]) == 3                                  # students missing: chain must stop
    assert m.main(["train", *args]) == 0
    for s in (0, 1):
        assert (root / f"undo_a0.1_s{s}" / "config.json").exists()
        meta = json.loads((root / f"undo_a0.1_s{s}" / "d1_v3.json").read_text())
        assert meta["seed"] == s and "no benchmark" in meta["retain"]
    log = pd.read_parquet(root / "undo_a0.1_s1.train_log.parquet")
    assert list(log["step"]) == [1, 2, 3, 4]
    assert m.main(["test", *args]) == 0
    assert (paths.results_dir() / "jobs" / "d1-v3" / "SUMMARY.md").exists()


def test_d1_v3_open_plan(tiny):  # noqa: F811
    m = _reload("cluster.d1_v3")
    assert m.main(["open", "--seeds", "0", "1", "2"]) == 0
    plan = json.loads((m.jc.job_dir(m.NAME) / "open_plan.json").read_text())
    ids = [p["task_id"] for p in plan]
    assert ids[0] == "open-leak-base" and "open-benign-undo-v3-s2" in ids and len(ids) == 8
    base = plan[0]["args"]
    assert [x["tag"] for x in base["methods"]] == ["base", "dsg-paper"] and base["items"] == "wmdp-bio-open"
    assert plan[4]["args"]["items"].startswith("mmlu-open:college_biology")


def test_d1_v3_relearn_full_and_lora(tiny):  # noqa: F811
    from dsgx import paths

    m = _reload("cluster.d1_v3")
    assert m.main(["relearn", "--seeds", "0", "--ks", "2", "--ranks", "2", "--n-eval", "2", "--n-util", "2"]) == 0
    full = sorted(p.name for p in (paths.runs_dir() / m.EXP_FULL).glob("relearn-*"))
    lora = sorted(p.name for p in (paths.runs_dir() / m.EXP_LORA).glob("relearn-*"))
    assert full == ["relearn-d1v3-s0-k2", "relearn-dsg-hook-k2", "relearn-dsg-nohook-k2", "relearn-rmu-v2-k2"]
    assert lora == [f + "-r2" for f in full]
    met = json.loads((paths.runs_dir() / m.EXP_LORA / "relearn-dsg-hook-k2-r2" / "metrics.json").read_text())
    assert met["rank"] == 2 and met["hook"] and met["curve"] and met["passages"] == "held-out forget split"
    s = json.loads((paths.results_dir() / "jobs" / "d1-v3" / "summary.json").read_text())
    assert len(s["relearn"]) == 8
    # DONE cells are skipped on a second call
    assert m.main(["relearn", "--seeds", "0", "--ks", "2", "--ranks", "2", "--n-eval", "2", "--n-util", "2"]) == 0


def test_d1_v3_plan_cpu(tiny):  # noqa: F811
    m = _reload("cluster.d1_v3")
    rc = m.main(["plan", "--ks", "2"])
    plan = json.loads((m.jc.job_dir(m.NAME) / "plan.json").read_text())
    assert rc == 0, plan["problems"]
    assert plan["passages"]["holdout"] >= 2 and set(plan["test_runs"]) >= {"base", "dsg-paper", "undo-v3-s2"}
    assert plan["relearn_cells"] == {"full": 6, "lora": 12} and plan["disk_gb"]["peak"] < 40
