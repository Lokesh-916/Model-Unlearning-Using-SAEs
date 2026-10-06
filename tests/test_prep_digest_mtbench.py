"""Results digest: MT-Bench (BM3) section reads only judge scores and pairs conditions on (question, turn);
the X1 status line reflects the DEV selection file."""
import json

from dsgx.analysis import results_digest as rd


def _judgments(path, scores):
    path.write_text("\n".join(json.dumps({"question_id": q, "turn": t, "score": s, "judgment": "x"})
                              for (q, t), s in scores.items()) + "\n")


def test_mtbench_section_pairs_on_question_and_turn(tmp_path):
    d = tmp_path / "mtbench"
    d.mkdir()
    keys = [(q, t) for q in range(81, 91) for t in (1, 2)]
    _judgments(d / "judgments_base__judge.jsonl", {k: 8.0 for k in keys})
    _judgments(d / "judgments_dsg__judge.jsonl", {k: (2.0 if k == (81, 1) else 8.0) for k in keys})
    (d / "summary.json").write_text(json.dumps({
        "judge": "j", "same_family_judge": True,
        "scores": {"base": {"all": {"mean": 8.0, "lo": 8.0, "hi": 8.0, "n": 20}, "unparsed": 0},
                   "dsg_minus_base": {"diff": -0.3, "lo": -0.9, "hi": 0.0, "p": 0.5, "n": 20}}}))
    text = "\n".join(rd.mtbench_section(tmp_path, n_boot=200))
    assert "same-family judge: True" in text
    assert "dsg_minus_base" not in text  # paired summary entries are not conditions
    row = next(line for line in text.splitlines() if line.startswith("| dsg − base"))
    assert "-0.300" in row and "| 1 | 20 |" in row


def test_x1_status(tmp_path, monkeypatch):
    monkeypatch.setattr(rd.paths, "results_dir", lambda: tmp_path)
    assert "not run yet" in rd.x1_status()[0]
    sel = tmp_path / "runs" / "X1-screen" / "select"
    sel.mkdir(parents=True)
    (sel / "COMBINE_SELECTION.json").write_text(json.dumps(
        {"time": "2026-10-06T09:19:19", "slots": {"features": None, "detector": "cusum"}}))
    line, slots = rd.x1_status()
    assert "detector = cusum" in line and "features = default" in line and slots["detector"] == "cusum"
