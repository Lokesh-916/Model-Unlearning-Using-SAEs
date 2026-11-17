"""V2R-test (StreamGuard v2-R TEST): frozen-setting guard, config, summary verdict on synthetic data (CPU)."""
import json

import numpy as np
import pandas as pd
import pytest
import yaml


def test_check_frozen(tmp_path, monkeypatch):
    import hashlib

    from experiments.E5 import common
    from experiments.E5.gen import check_frozen

    f = tmp_path / "refusal_base.npz"
    f.write_bytes(b"x")
    monkeypatch.setattr(common, "cache_path", lambda e, k: f)
    h = hashlib.sha256(b"x").hexdigest()
    s1 = {"selection": {"layer": 10, "s": 0.45, "mode": "start8"}}
    check_frozen({"layer": 10, "s": 0.45, "mode": "start8", "direction_sha256": h}, s1, "E5b", "base")
    with pytest.raises(RuntimeError):
        check_frozen({"layer": 10, "s": 0.40, "mode": "start8", "direction_sha256": h}, s1, "E5b", "base")
    with pytest.raises(RuntimeError):
        check_frozen({"layer": 10, "s": 0.45, "mode": "start8", "direction_sha256": "0" * 64}, s1, "E5b", "base")


def test_config_is_test_split_and_frozen(tmp_path, monkeypatch):
    import scripts.make_v2r_config as mk

    monkeypatch.setattr(mk, "ROOT", tmp_path)
    (tmp_path / "configs" / "experiments").mkdir(parents=True)
    mk.main()
    cfg = yaml.safe_load((tmp_path / "configs" / "experiments" / "V2R.yaml").read_text())
    gens = [t for t in cfg["tasks"] if t["entry"].endswith("gen:task")]
    assert len(gens) == 4
    for t in gens:
        assert t["args"]["split"] == "test" and t["smoke_args"]["split"] == "dev"
        assert t["args"]["arms"] == ["A", "B"] and "n_items" not in t["args"]
        assert t["args"]["frozen"] == {"layer": 10, "s": 0.45, "mode": "start8", "direction_sha256": mk.FROZEN["direction_sha256"]}
    assert "base" not in cfg  # no MCQ runs


def _fake(rd, tid, n=40, leakB=0, refB=1.0):
    rng = np.random.default_rng(0)
    fired = np.arange(n) < n - 4
    step = np.where(np.arange(n) % 5 == 0, 3, 0)
    recs = []
    for i in range(n):
        for c in ("base", "A", "B"):
            m = {"base": rng.random() < 0.3, "A": False, "B": (i < leakB)}[c]
            r = c == "B" and fired[i] and (i / n) < refB
            recs.append({"item_id": f"x{i}", "cond": c, "fired": bool(fired[i]), "fire_step": int(step[i]) if fired[i] else None,
                         "match": bool(m), "gibberish": bool(c == "A" and i % 2 == 0), "refusal": bool(r)})
    (rd / tid).mkdir(parents=True)
    pd.DataFrame(recs).to_parquet(rd / tid / "items.parquet")
    (rd / tid / "metrics.json").write_text(json.dumps({"check": {"n": n, "prompt_score_bit_identical": n, "prompt_decision_diff": 0}}))
    (rd / tid / "config.json").write_text("{}")


def test_summary_verdict(tmp_path):
    from experiments.E5.v2r_summary import one_set, render, verdict

    _fake(tmp_path, "gen-wmdp")
    s = one_set(tmp_path, "gen-wmdp", 200)
    assert s["n_fired"] == 36
    assert abs(s["mid_answer_share"]["mean"] - 8 / 36) < 1e-9
    v = verdict(s)
    assert v["pass"] and v["refusal"] == 1.0
    _fake(tmp_path, "gen-wmdp-pad400", leakB=20, refB=0.5)
    v2 = verdict(one_set(tmp_path, "gen-wmdp-pad400", 200))
    assert not v2["pass"] and not v2["leak_ok"] and not v2["refusal_ok"]
    md = render({"sets": {"gen-wmdp": s}, "verdicts": {"gen-wmdp": v}, "label": "weaker setting"})
    assert "PASSES" in md and "weaker setting" in md
