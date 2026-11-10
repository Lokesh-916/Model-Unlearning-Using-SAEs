"""StreamGuard v2 rules V1-V4 (claims.evaluate_v2) on synthetic runs; C-H1..C-H7 untouched."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from dsgx.analysis import claims


def _run(method, correct_f, correct_u, variant=None, forced=False, seed=0):
    n_f, n_u = len(correct_f), len(correct_u)
    items = pd.DataFrame({"item_id": [f"f{i}" for i in range(n_f)] + [f"u{i}" for i in range(n_u)],
                          "dataset": ["wmdp-bio"] * n_f + ["mmlu-x"] * n_u,
                          "correct": list(correct_f) + list(correct_u)})
    m = {"name": method}
    if variant:
        m["variant"] = variant
    if forced:
        m["forced_on"] = True
    return SimpleNamespace(name=f"{method}-{variant}-{forced}-{seed}", is_mcq=True, split="test", is_clean=True,
                           method=method, base_exp="SG2", seed=seed, items=items, hardware="labpc", exp="SG2",
                           cfg={"method": m, "dataset_label": "bio", "forget_datasets": ["wmdp-bio"]},
                           config={"config": {}}, attack={"name": "none"}, gate={})


def test_v1_supported_v2_v4_inconclusive_and_old_rules_unchanged():
    rng = np.random.default_rng(0)
    f = rng.random(300) < 0.26
    runs = [_run("dsg-faithful", f, rng.random(2000) < 0.30, forced=True),
            _run("gated", f, rng.random(2000) < 0.55, variant="streamguard-v2", forced=True)]
    out = {c["id"]: c for c in claims.evaluate_v2(runs)}
    assert list(out) == ["V1", "V2", "V3", "V4"]
    assert out["V1"]["verdict"] == "Supported", out["V1"]["evidence"]
    assert all(out[k]["verdict"] == "Inconclusive" for k in ("V2", "V3", "V4"))
    assert set(claims.CLAIMS) == {f"C-H{i}" for i in range(1, 8)}
    # V4: candidate scores carry no label information, DSG's separate perfectly -> candidate AUC lower
    lab = np.array([1] * 100 + [0] * 100)
    dsg = [{"item_id": i, "label": int(l), "score": float(l), "correct": True} for i, l in enumerate(lab)]
    sg = [{"item_id": i, "label": int(l), "score": float(rng.random()), "correct": bool(rng.random() < 0.5)}
          for i, l in enumerate(lab)]
    recs = [{"method": {"name": "gated", "variant": "streamguard-v2"}, "seed": 0, "exp": "SG2", "items": sg},
            {"method": {"name": "dsg-faithful"}, "seed": 0, "exp": "SG2", "items": dsg}]
    assert claims.v4([], recs, n_boot=300)["verdict"] == "Supported"
