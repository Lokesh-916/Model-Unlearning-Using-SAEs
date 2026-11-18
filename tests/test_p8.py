"""P8 CPU tests: wrapper prompt builder shape (dummy question, no dataset text), config shape."""
from pathlib import Path

import yaml

from experiments.P8.common import AGREE, CONDITIONS, WRAPPERS, task_id, wrapped_prompt

Q = "What colour is the sky on a clear day?"


def test_conditions():
    assert len(WRAPPERS) == 6 and len(CONDITIONS) == 19 and len({task_id(c) for c in CONDITIONS}) == 19


def test_prompts():
    from dsgx.eval.openqa import open_prompt

    assert wrapped_prompt(Q, "none", "x") == open_prompt(Q)
    ps = {c: wrapped_prompt(Q, c, "x") for c in CONDITIONS}
    assert len(set(ps.values())) == 19
    for c, p in ps.items():
        assert p.startswith("<bos>") and Q.split()[0] in p
        if c.startswith("prefill"):
            assert p.endswith("<start_of_turn>model\n" + AGREE)
        else:
            assert p.endswith("<start_of_turn>model\n")
        if c.endswith("split3"):
            assert p.count("<start_of_turn>user") == 3
        if c.startswith("agree_start"):
            assert f'"{AGREE}"' in p and not p.endswith(AGREE)


def test_config():
    cfg = yaml.safe_load(Path("configs/experiments/P8.yaml").read_text())
    gens = [t for t in cfg["tasks"] if t["entry"] == "experiments.P8.gen:task"]
    assert len(gens) == 19
    for t in gens:
        a = t["args"]
        assert a["split"] == "dev" and a["n_items"] == 150 and a["subset_seed"] == 0 and a["arms"] == ["A", "B"]
        assert a["frozen"]["direction_sha256"].startswith("f7d9a17e")
        assert t["smoke_args"].get("split", "dev") == "dev"
