"""E1 three-number audit (dsgx.analysis.rfd_audit) on synthetic per-item rows (no result files)."""
import numpy as np
import pandas as pd
import pytest

from dsgx.analysis import rfd_audit


def _run(method, seed, fired, correct, hw="labpc", kind="clean"):
    n = len(correct)
    items = pd.DataFrame({"item_id": [f"wmdp-bio:{i}" for i in range(n)], "dataset": "wmdp-bio",
                          "correct": correct, "gate_fired": fired})
    return {"name": f"{method}-s{seed}", "hardware": hw, "method": method, "seed": seed, "kind": kind,
            "attack": "none", "items": items}


def test_decomposition_prediction_and_paired_split():
    rng = np.random.default_rng(1)
    n = 400
    base_correct = rng.random(n) < 0.6
    runs = [_run("base", 0, [None] * n, base_correct)]
    for m, p_fire in (("DSG", 0.3), ("StreamGuard", 0.5)):
        for s in range(2):
            fired = rng.random(n) < p_fire
            # a gate that leaves unfired rows untouched and sends fired rows to chance
            correct = np.where(fired, rng.random(n) < 0.25, base_correct)
            runs.append(_run(m, s, fired, correct))
    res = rfd_audit.machine_audit(runs, n_boot=200)
    assert res["hardware"] == "labpc"
    for r in res["rows"]:
        assert r["n_rows"] == 2 * n and r["n_seeds"] == 2
        # untouched misses -> the base-model prediction equals the observation exactly
        assert r["pred"]["mean"] == pytest.approx(r["obs"]["mean"])
        assert r["ident_err"]["mean"] < 1e-12 and r["pred_in_ci"]
        assert r["obs"]["lo"] <= r["obs"]["mean"] <= r["obs"]["hi"]
    # forget items are not utility items: no paired utility row; check the paired split directly
    df = {m: rfd_audit._pool([r for r in runs if r["method"] == m], runs[0]["items"]) for m in ("DSG", "StreamGuard")}
    p = rfd_audit.paired_utility(df["StreamGuard"], df["DSG"], n_boot=200)
    parts = sum(p["part_" + k]["mean"] for k in ("sg_only", "dsg_only", "both", "neither"))
    assert parts == pytest.approx(p["obs_diff"]["mean"])
    assert p["part_neither"]["mean"] == 0.0  # neither fired -> both equal the base model
    with pytest.raises(ValueError):
        rfd_audit.machine_audit(runs + [_run("DSG", 9, [True] * n, base_correct, hw="gpuws")], n_boot=10)
