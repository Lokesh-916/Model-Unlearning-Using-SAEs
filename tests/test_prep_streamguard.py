"""StreamGuard digest summary and the C-H5 clean-run lookup (session 27) on fake X1-style runs (no GPU)."""
from dsgx.analysis import aggregate, claims, streamguard
from dsgx.analysis.collect import load_all
from tests.fakeruns import bern, mcq_run

DSG = {"name": "dsg-faithful", "n_features": 20, "retain_pct": 95, "multiplier": 500}


def _sg(seed):  # per-seed calibration seed, as in X1 (must not split the method across seeds)
    return {"name": "gated", "gate": {"type": "cusum", "n_features": 20, "retain_pct": 95, "calib_seed": seed}}


def _x1(util_gap):
    n = 300
    for s in range(3):
        for lab, att in (("", {"name": "none"}), ("-forget", {"name": "dilution", "pad": 400}),
                         ("-forget", {"name": "decompose"}), ("-forget", {"name": "suffix"})):
            clean = att["name"] == "none"
            mcq_run("X1", DSG, seed=s, attack=att, forget=bern(0.6, n, 1 + s), util=bern(0.8, 200, 2) if clean else [],
                    extra_cfg={"dataset_label": "dsg" + lab})
            mcq_run("X1", _sg(s), seed=s, attack=att, forget=bern(0.3 if not clean else 0.3, n, 3 + s),
                    util=bern(0.8 - util_gap, 200, 2) if clean else [], extra_cfg={"dataset_label": "combined" + lab},
                    fired=([True] * n + [False] * 196 + [True] * 4) if clean else None)


def test_ch5_matches_clean_runs_by_method_and_pairs_utility():
    _x1(0.0)
    runs = load_all()
    c5 = claims.ch5(runs, aggregate.paired_tests(runs, n_boot=300))
    assert c5["verdict"] == "Supported", c5  # old lookup: label 'combined-forget' matched no clean run -> utility ok=False
    assert "seeds 3" in c5["evidence"][0]


def test_ch5_utility_gap_fails_and_summary_renders():
    _x1(0.05)
    runs = load_all()
    paired = aggregate.paired_tests(runs, n_boot=300)
    assert claims.ch5(runs, paired)["verdict"] == "Not supported"
    txt = "\n".join(streamguard.section(runs, paired, "labpc"))
    assert "| B1 |" in txt and "| B2 |" in txt and "Clean (full MMLU" in txt
    assert streamguard.section([], [], "gpuws")[-2].startswith("(no StreamGuard")
