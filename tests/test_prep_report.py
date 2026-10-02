"""final_report: claims rules on fake results, end-to-end CLI into a temp dir (no GPU)."""
import json

from dsgx.analysis import aggregate, claims
from dsgx.analysis.collect import load_all
from tests.fakeruns import bern, mcq_run, task_run

DSG = {"name": "dsg-faithful", "n_features": 20, "retain_pct": 95, "multiplier": 500}
GATE = {"name": "gated", "gate": {"type": "window", "w": 16}}


def _as_rec(attack, mean, lo, base=None, p=0.01, diff=0.3):
    return {"case": "bio", "method": json.dumps({**DSG, "case": "bio"}), "attack": json.dumps(attack),
            "attack_success": {"mean": mean, "lo": lo, "hi": min(1, mean + 0.1), "n": 100}, "n_gated": 100,
            "acc_under_attack": {"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 300},
            "base_acc_same_transform": base, "gate_fire_rate": 0.1, "rho_mean": 0.2,
            "vs_clean_paired": {"diff": diff, "lo": 0.1, "hi": 0.4, "p": p, "n": 300}, "run_dir": "x"}


def test_ch1_ch2_rules():
    task_run("B1", "attack-success", {}, files={"attack_success.json": [
        _as_rec({"name": "dilution", "pad": 400}, 0.62, 0.52),
        _as_rec({"name": "dilution", "pad": 50}, 0.2, 0.1)]})
    mcq_run("B1", {"name": "base"}, attack={"name": "dilution", "pad": 400}, forget=bern(0.6, 200, 1), util=[])
    task_run("B3", "attack-success", {}, files={"attack_success.json": [
        _as_rec({"name": "translate", "lang": "hi"}, 0.3, 0.2, p=0.001)]})
    runs = load_all()
    c1, c2 = claims.ch1(runs), claims.ch2(runs)
    assert c1["verdict"] == "Supported", c1
    assert c2["verdict"] == "Supported", c2


def test_ch1_not_supported_and_missing():
    assert claims.ch1(load_all())["verdict"] == "Inconclusive"
    task_run("B1", "attack-success", {}, files={"attack_success.json": [_as_rec({"name": "dilution", "pad": 400}, 0.3, 0.2)]})
    assert claims.ch1(load_all())["verdict"] == "Not supported"


def test_ch5_paired_gate_beats_dsg_on_three_axes():
    n = 300
    for att in ({"name": "none"}, {"name": "dilution", "pad": 400}, {"name": "decompose"}, {"name": "translate", "lang": "fr"}):
        mcq_run("X1", DSG, attack=att, forget=bern(0.6, n, 1), util=bern(0.8, 200, 2))
        mcq_run("X1", GATE, attack=att, forget=bern(0.25, n, 3) if att["name"] != "none" else bern(0.3, n, 3),
                util=bern(0.8, 200, 2), fired=[True] * n + [False] * 196 + [True] * 4)
        mcq_run("X1", {"name": "base"}, attack=att, forget=bern(0.6, n, 1), util=bern(0.8, 200, 2))
    runs = load_all()
    paired = aggregate.paired_tests(runs, n_boot=500)
    assert any(p["vs"] == "dsg" for p in paired)
    c5 = claims.ch5(runs, paired)
    assert c5["verdict"] == "Supported", c5


def test_ch6_relearn():
    for cond, after in (("d1", 0.30), ("dsg-nohook", 0.55), ("student", 0.55)):
        for k in (10, 50):
            task_run("A6", f"relearn-{cond}-k{k}", {"condition": cond, "k": k, "rank": 8, "n_eval": 300,
                                                    "before": {"forget_acc": 0.26, "util_acc": 0.70},
                                                    "curve": [{"step": 1000, "forget_acc": after, "util_acc": 0.7}]})
    assert claims.ch6(load_all())["verdict"] == "Supported"


def test_final_report_cli(tmp_path):
    mcq_run("A1-test", DSG, forget=bern(0.3, 100, 1), util=bern(0.8, 100, 2))
    mcq_run("A1-test", DSG, seed=1, forget=bern(0.3, 100, 3), util=bern(0.8, 100, 2))
    mcq_run("A1-test", {"name": "base"}, forget=bern(0.6, 100, 1), util=bern(0.81, 100, 2))
    mcq_run("A1-test", {"name": "base"}, forget=bern(0.6, 100, 1), util=bern(0.81, 100, 2),
            model={"weights": "AMindToThink/gemma-2-2b-it_RMU_s200_a300_layer3"}, extra_cfg={"dataset_label": "rmu-unverified"})
    from dsgx.analysis import final_report

    out = tmp_path / "rep"
    assert final_report.main(["--out", str(out), "--n-boot", "200", "--interim"]) == 0
    txt = (out / "FINAL_REPORT.md").read_text()
    for h in ["## 1. Executive summary", "## 4. Claims", "## 10. Failed, skipped and partial items",
              "## 13. Index of figures", "unverified reference"]:
        assert h in txt
    s = json.loads((out / "summary.json").read_text())
    assert len(s["claims"]) == 7 and "dsg-faithful" in s["methods"]
    assert (out / "audit_cards" / "dsg-faithful.md").exists()


def test_hardware_never_mixed(tmp_path):
    mcq_run("A1-test", DSG, forget=[1, 0], util=[1, 1])
    mcq_run("A1-test", {"name": "base"}, forget=[1, 1], util=[1, 1], hardware="gpuws")
    from dsgx.analysis import final_report

    assert final_report.main(["--out", str(tmp_path), "--n-boot", "50", "--no-figures"]) == 2
    assert final_report.main(["--out", str(tmp_path), "--n-boot", "50", "--no-figures", "--hardware", "gpuws"]) == 0
