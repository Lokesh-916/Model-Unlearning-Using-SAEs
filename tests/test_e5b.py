"""E5b unit tests (CPU, no model, no data): question sets (sizes, matching, disjointness, years), IDK grader,
selection rule with fallback and tie-breaks, confirmation check, step-limited steering (mode start8)."""
import torch

from experiments.E5 import cutoff
from experiments.E5 import e5b as B
from experiments.E5.batchgen import BatchSteer, StepSteer, _set_t, _steer_hook


def test_cutoff_sets():
    s = cutoff.build_sets()
    assert len(s["unknown"]) == 300 and len(s["known"]) == 300 and len(s["heldout"]) == 100
    assert len(cutoff.EVENTS) == 30 and len(cutoff.CAPITALS) == 50 and len(cutoff.HISTORY) == 50
    ids = [q["id"] for k in s for q in s[k]]
    assert len(ids) == len(set(ids))
    qs = [q["question"] for k in s for q in s[k]]
    assert len(qs) == len(set(qs))
    assert all(("2025" in q["question"]) ^ ("2026" in q["question"]) for q in s["unknown"])
    for e, ys in cutoff.EVENTS:
        assert all(y < 2023 for y in ys) and ys[0] != ys[1]
    # matched: removing the year gives the same multiset of question frames
    strip = lambda q: "".join(c for c in q if not c.isdigit())  # noqa: E731
    assert sorted(strip(q["question"]) for q in s["unknown"]) == sorted(strip(q["question"]) for q in s["known"])
    ev = {e for e, _ in cutoff.EVENTS}
    assert not any(e in q["question"] for q in s["heldout"] for e in ev)


def test_idk_grader():
    assert cutoff.is_idk("I don't know who won that.")
    assert cutoff.is_idk("As of my last knowledge update, that event had not happened.")
    assert cutoff.is_idk("My knowledge cutoff means I cannot say.")
    assert cutoff.is_idk("I do not have access to real-time information.")
    assert cutoff.is_idk("I have no information about this.")
    assert not cutoff.is_idk("The capital of France is Paris.")


def _c(layer, s, mode, rate, gib):
    return {"layer": layer, "s": s, "mode": mode, "rate": rate, "gib": gib}


def test_select_primary_and_ties():
    cells = [_c(10, 0.40, "all", 0.95, 0.0), _c(12, 0.35, "all", 0.91, 0.02), _c(10, 0.35, "start8", 0.91, 0.02),
             _c(12, 0.30, "all", 0.99, 0.03)]
    s = B.select_b(cells)
    assert s["eligible"] and s["tier"] == "primary" and s["label"] is None
    assert (s["layer"], s["s"], s["mode"]) == (10, 0.35, "start8")   # smallest s; tie -> start8 before all


def test_select_fallback_and_none():
    cells = [_c(10, 0.30, "all", 0.72, 0.01), _c(12, 0.45, "start8", 0.85, 0.02), _c(10, 0.45, "all", 0.95, 0.5)]
    s = B.select_b(cells)
    assert s["eligible"] and s["tier"] == "fallback" and s["label"] == B.WEAKER
    assert (s["layer"], s["s"], s["mode"]) == (12, 0.45, "start8")
    assert not B.select_b([_c(10, 0.3, "all", 0.69, 0.0), _c(10, 0.45, "all", 0.99, 0.03)])["eligible"]


def test_confirm_ok():
    assert B.confirm_ok({"tier": "primary"}, 0.90, 0.02)
    assert not B.confirm_ok({"tier": "primary"}, 0.85, 0.0)
    assert B.confirm_ok({"tier": "fallback"}, 0.71, 0.0)
    assert not B.confirm_ok({"tier": "fallback"}, 0.95, 0.03)


def test_step_steer_modes():
    V = torch.ones(3, 2)
    h = StepSteer(V, [None, 8, None])
    r = torch.zeros(3, 5, 2)
    assert torch.equal(h(r), torch.ones(3, 5, 2))              # prompt pass: every row steered
    _set_t([h], 8)
    assert h(torch.zeros(3, 1, 2))[1].sum() == 2               # 8th generated position still steered
    _set_t([h], 9)
    out = h(torch.zeros(3, 1, 2))
    assert out[1].sum() == 0 and out[0].sum() == 2 and out[2].sum() == 2
    assert isinstance(_steer_hook(V, None), BatchSteer) and isinstance(_steer_hook(V, [None, None, None]), BatchSteer)
    assert isinstance(_steer_hook(V, [None, 8, None]), StepSteer)
