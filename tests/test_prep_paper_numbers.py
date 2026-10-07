"""paper_numbers: macro naming, CI formatting, pending fallback and hardware separation (CPU, no runs needed)."""
import json
import re

import pytest

from dsgx.analysis import paper_numbers as pn


def _summary(hw, rows=(), paired=(), claims=(), paper=None):
    return {"hardware": hw, "interim": True, "generated": "t", "experiments": {"A1-test": list(rows)},
            "paired_tests": list(paired), "claims": list(claims), "paper": paper or {}}


def _row(cond, forget, lo, hi, n, seeds=(0,)):
    return {"exp": "A1-test", "condition": cond, "case": "bio", "split": "test", "attack": "none", "seeds": list(seeds),
            "raw_forget": forget, "raw_forget_lo": lo, "raw_forget_hi": hi, "raw_forget_n_items": n}


def test_build_formats_and_pending(tmp_path):
    lab = _summary("labpc", rows=[_row("base/forget+utility", 0.642, 0.604, 0.680, 637)],
                   claims=[{"id": "C-H1", "verdict": "Supported", "numbers": {"n_conditions_meeting": 3}}],
                   paper={"attack_success_max": {"B1": {"mean": 0.9, "lo": 0.8, "hi": 0.95, "n": 265, "n_conditions": 4,
                                                        "attack": {"name": "dilution", "pad": 400}}}})
    (tmp_path / "lab.json").write_text(json.dumps(lab))
    (tmp_path / "gpu.json").write_text(json.dumps(_summary("gpuws")))
    N = pn.build(tmp_path / "lab.json", tmp_path / "gpu.json", tmp_path / "jobs", tmp_path / "lr", tmp_path / "gr",
                 tmp_path / "none.jsonl")
    tex = pn.render(N)
    assert "\\newcommand{\\resLabBaseForget}{0.642 [0.604, 0.680] ($n=637$)}" in tex
    assert "\\newcommand{\\resLabBOneMaxVal}{0.900}" in tex
    assert "\\newcommand{\\resLabVerdictCHOne}{Supported}" in tex
    assert "\\newcommand{\\resGpuBaseForget}{\\respending{GpuBaseForget}}" in tex   # missing source -> pending, never a number
    names = re.findall(r"\\newcommand\{\\res([^}]*)\}", tex)
    assert len(names) == len(set(names)) and all(re.fullmatch(r"[A-Za-z]+", n) for n in names)
    assert {n for n, _ in N.pending} >= {"GpuBaseForget", "XgpuItems"}


def test_wrong_hardware_is_refused(tmp_path):
    (tmp_path / "lab.json").write_text(json.dumps(_summary("gpuws")))
    (tmp_path / "gpu.json").write_text(json.dumps(_summary("gpuws")))
    with pytest.raises(SystemExit):
        pn.build(tmp_path / "lab.json", tmp_path / "gpu.json", tmp_path, tmp_path, tmp_path, tmp_path / "x.jsonl")


def test_diff_signs():
    N = pn.Numbers()
    N.diff("X", {"diff": -0.0125, "lo": -0.02, "hi": 0.001, "n": 10, "mcnemar": {"p": 0.0004}}, "t")
    tex = "\n".join(N.lines)
    assert "{$-0.013$ [$-0.020$, $+0.001$]}" in tex and "{$p < 0.001$}" in tex


def test_posthoc_part_statuses_and_lab_x1(tmp_path):
    """PH-union parts finished before its MCQ jobs get their own status; a sentinel threshold marks the union degenerate;
    the lab X1 status follows final_report completeness."""
    gr = tmp_path / "gr"

    def run(exp, name, metrics):
        d = gr / exp / name
        d.mkdir(parents=True)
        (d / "metrics.json").write_text(json.dumps(metrics))
        (d / "DONE").write_text("")

    m = {"match": {"mean": 0.5, "lo": 0.4, "hi": 0.6, "n": 400}, "gate_fired": {"mean": 0.0, "lo": 0.0, "hi": 0.0, "n": 400}}
    for t in ("tofu-qa-forget", "tofu-qa-retain", "benign-open"):
        for g in ("union-stream", "dsg-faithful-stream"):
            run("PH-union", f"{t}__{g}", m)
    run("PH-union", "tofu-metrics", {"conditions": {"full+dsg": {"model_utility": 0.66, "tau": 0.0447},
                                                     "full+union": {"model_utility": 0.727, "threshold": 1e9}}})
    lab = _summary("labpc")
    lab["completeness"] = {"X1": {"status": "partial", "jobs": {"DONE": 22, "WAITING": 11}}}
    (tmp_path / "lab.json").write_text(json.dumps(lab))
    (tmp_path / "gpu.json").write_text(json.dumps(_summary("gpuws")))
    tex = pn.render(pn.build(tmp_path / "lab.json", tmp_path / "gpu.json", tmp_path / "jobs", tmp_path / "lr", gr,
                             tmp_path / "none.jsonl"))
    for name, val in (("GpuPhUnionTofuStatus", "done"), ("GpuPhUnionOpenStatus", "done"), ("GpuPhUnionStatus", "pending"),
                      ("GpuPhUnionPairedStatus", "pending"), ("GpuPhUnionTofuDegenerate", "yes"),
                      ("GpuPhUnionTofuDsgTau", "0.045"), ("GpuPhUnionOpenDsgMatchVal", "0.500"),
                      ("LabXOneStatus", "pending"), ("LabXOneJobsDone", "22 of 33")):
        assert f"\\newcommand{{\\res{name}}}{{{val}}}" in tex, name
