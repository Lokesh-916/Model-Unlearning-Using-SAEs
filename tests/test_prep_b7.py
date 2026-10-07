"""dsgx.analysis.b7_adaptive on synthetic B7 runs (ids and metrics only)."""
import json

import numpy as np
import pandas as pd

from dsgx.analysis import b7_adaptive as b7
from dsgx.analysis.paper_numbers import Numbers

GATE = {"type": "cusum", "n_features": 20, "retain_pct": 95}
METH = {"base": {"name": "base"}, "dsg": {"name": "dsg-faithful", "n_features": 20},
        "streamguard": {"name": "gated", "gate": GATE}}


def _run(root, i, method, attack, correct, fired):
    d = root / "B7" / f"B7__{method}__{i:02d}"
    d.mkdir(parents=True)
    n = len(correct)
    pd.DataFrame({"item_id": [f"q{j}" for j in range(n)], "dataset": "wmdp-bio", "correct": correct,
                  "gate_fired": fired if fired is not None else [None] * n}).to_parquet(d / "items.parquet")
    (d / "config.json").write_text(json.dumps({"config": {"method": METH[method], "attack": {**attack, "_exp_id": "B7"},
                                                          "forget_datasets": ["wmdp-bio"]}}))
    (d / "DONE").write_text("")


def make(root, n=200, only=None):
    rng = np.random.default_rng(0)
    base_c = rng.random(n) < 0.6
    i = 0
    for a, word, _ in b7.ATTACKS:
        clean = word == "Clean"
        # DSG blocks the gated items when clean and loses them under every attack; StreamGuard keeps half of them
        dsg_c = np.where(base_c, rng.random(n) < (0.3 if clean else 0.9), False)
        sg_c = np.where(base_c, rng.random(n) < (0.3 if clean else 0.5), False)
        for m, c, f in (("base", base_c, None), ("dsg", dsg_c, ~dsg_c), ("streamguard", sg_c, ~sg_c)):
            if only and i >= only:
                return
            _run(root, i, m, a, c, f)
            i += 1


def test_summary_and_macros(tmp_path):
    make(tmp_path)
    s = b7.summarize("B7", tmp_path)
    assert s["status"] == "done" and s["n_runs_done"] == 21
    r = {x["word"]: x for x in s["attacks"]}
    assert r["Clean"]["dsg"]["success"]["mean"] == 0.0  # gated items are wrong by definition when clean
    assert r["IlTwo"]["dsg"]["success"]["mean"] > 0.8 and r["IlTwo"]["streamguard"]["success"]["mean"] < 0.7
    assert r["IlTwo"]["verdict"] == "StreamGuard better" and r["IlTwo"]["fix_vs_dsg"]["hi"] < 0
    assert 0 <= r["Midpad"]["streamguard"]["fire_rate"] <= 1
    N = Numbers()
    b7.macros(N, tmp_path)
    tex = "\n".join(N.lines)
    assert "\\newcommand{\\resLabBSevenStatus}{done}" in tex
    assert "\\resLabBSevenIlEightFixSuccess}" in tex and "\\resLabBSevenIlSplitFixVsDsgP}" in tex
    assert "\\newcommand{\\resLabBSevenNBetter}{6}" in tex
    assert not N.pending
    assert any("| interleave k=8 |" in x for x in b7.markdown(s))


def test_pending_while_incomplete(tmp_path):
    make(tmp_path, only=10)
    s = b7.summarize("B7", tmp_path)
    assert s["status"] == "pending" and s["n_runs_done"] == 10
    N = Numbers()
    b7.macros(N, tmp_path)  # must not raise on missing runs; macros print [pending]
    tex = "\n".join(N.lines)
    assert "\\newcommand{\\resLabBSevenStatus}{pending}" in tex
    assert "\\newcommand{\\resLabBSevenFixMaxAttack}{\\respending{LabBSevenFixMaxAttack}}" in tex
