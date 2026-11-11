from experiments.E2f.common import is_idk, is_refusal, lines, select_cell
from experiments.E2f.entities import build_sets


def test_sets():
    s = build_sets()
    assert {k: len(v) for k, v in s.items()} == {"real": 300, "fake": 300, "heldout": 100}


def test_idk_regex():
    assert is_idk("I'm not familiar with that person.") and is_idk("I don't have any information about it")
    assert not is_idk("Paris is the capital of France.")
    assert is_refusal("I cannot help with that request.") and not is_refusal("The answer is 42.")


def test_select_cell():
    cells = [{"layer": 6, "coef": 2.0, "idk": 0.5, "gib": 0.2}, {"layer": 9, "coef": 4.0, "idk": 0.4, "gib": 0.01},
             {"layer": 9, "coef": 1.0, "idk": 0.9, "gib": 0.0}]
    s = select_cell(cells)
    assert (s["layer"], s["coef"], s["grid"]) == (9, 4.0, "requested")
    s = select_cell([c for c in cells if c["coef"] != 4.0])
    assert (s["coef"], s["grid"]) == (1.0, "diagnostic")
    s = select_cell([cells[0]])
    assert not s["eligible"]


def test_lines():
    assert lines({"a": 0.01}, {"w": {"sig_higher": False}})["both"]
    assert not lines({"a": 0.06}, {"w": {"sig_higher": False}})["both"]
    assert not lines({"a": 0.0}, {"w": {"sig_higher": True}})["both"]
