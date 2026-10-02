"""P1b harness additions: attacks, gate scores, open-QA graders, leakage control, tasks, joblist."""
import json
import sys
import time

import numpy as np
import pytest
import torch

from dsgx import paths
from dsgx.attacks.registry import make_attack
from dsgx.data.mcq import MCQItem, format_prompt

ITEM = MCQItem("high_school_geography", 7, "Which river is the longest in the world by most measures today?",
               ("Nile", "Amazon", "Yangtze", "Danube"), 0, "high_school_geography")


def test_pad_zero_equals_clean_prompt_and_encodings():
    clean = format_prompt(ITEM)
    assert make_attack({"name": "dilution", "pad": 0}).prompt(ITEM)[0] == clean
    assert make_attack({"name": "none"}).prompt(ITEM)[0] == clean
    for enc in ("base64", "rot13", "leet", "spaced"):
        p, info = make_attack({"name": "encode", "encoding": enc}).prompt(ITEM)
        assert p.startswith("<bos><start_of_turn>user\n") and p.endswith("Answer: (") and info["encoding"] == enc
    import base64

    p, _ = make_attack({"name": "encode", "encoding": "base64"}).prompt(ITEM)
    blob = p.split("\n")[-3] if False else p.split("letter of the correct option.\n")[1].split("<end_of_turn>")[0]
    assert base64.b64decode(blob).decode().startswith(ITEM.question)


def test_decompose_variants():
    for v, k in (("split", 2), ("split", 3), ("stem_last", 2), ("options_only", 1)):
        p, info = make_attack({"name": "decompose", "variant": v, "k": k}).prompt(ITEM)
        assert p.count("<start_of_turn>user") == info["turns"]
        assert p.endswith("<start_of_turn>model\nAnswer: (")
        assert all(c in p for c in ITEM.choices)
    p, _ = make_attack({"name": "decompose", "variant": "options_only"}).prompt(ITEM)
    assert ITEM.question not in p


def test_translate_skips_missing_and_low_chrf(tmp_path):
    from dsgx.data.translate import cache_file

    f = cache_file("high_school_geography", "fr")
    f.write_text(json.dumps({"item_id": ITEM.item_id, "lang": "fr", "question": "Quel fleuve ?",
                             "choices": ["Nil", "Amazone", "Yangtsé", "Danube"], "back_question": "x",
                             "chrf": 55.0}) + "\n")
    a = make_attack({"name": "translate", "lang": "fr", "min_chrf": 40})
    p, info = a.prompt(ITEM)
    assert "Quel fleuve" in p and info["language"] == "fr"
    assert make_attack({"name": "translate", "lang": "fr", "min_chrf": 60}).prompt(ITEM)[0] is None
    other = MCQItem("high_school_geography", 8, "q", ("a", "b", "c", "d"), 0, "x")
    assert a.prompt(other)[0] is None


def test_gate_scores():
    from dsgx.methods.gates import score_cusum, score_rho, score_window

    fire = torch.tensor([0, 0, 0, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    assert score_rho(fire, 10) == 0.4
    assert score_window(fire, 10, 4) == 1.0
    assert abs(score_window(fire, 10, 100) - 4 / 9) < 1e-6  # BOS excluded, window > length (float32)
    llr = torch.tensor([0.0, -1, 2, 2, -0.5, 3, -10, 1])
    best, tr = score_cusum(llr, 8)
    assert best == 6.5 and tr[-1] == 1.0 and (tr >= 0).all()


def test_openqa_graders_and_conversion():
    from dsgx.eval.openqa import NOT_OPEN, gibberish, token_f1

    f1, rec = token_f1("the river nile in africa", "Nile river")
    assert rec == 1.0 and 0 < f1 < 1
    assert NOT_OPEN.search("Which of the following is true?")
    assert not NOT_OPEN.search("What enzyme unwinds DNA during replication?")
    g = gibberish("aaa bbb aaa bbb aaa bbb aaa bbb aaa bbb", [1, 2] * 10)
    assert g["gibberish"] and g["rep4"] > 0.3
    assert not gibberish("Enzymes lower the activation energy of reactions.", list(range(9)))["gibberish"]


def test_leakage_positive_control():
    from dsgx.checks.leakage import _ngrams, _words, positive_control

    host = "This is ordinary calibration text about weather and sports. " * 20
    pc = positive_control(_ngrams(_words(host)), host, n=10)
    assert pc["pass"] and pc["detected"] == 10 and pc["false_alarms"] == 0


def test_task_job_through_queue_and_private_dir():
    from dsgx.queue import common as q
    from dsgx.queue import scheduler as S
    from dsgx.queue.enqueue import write_job

    q.ensure_dirs()
    q.save_control({"status_every_seconds": 0})
    write_job({"id": "T-toy", "exp_id": "T", "kind": "task", "task_id": "toy",
               "entry": "tests.fixtures_tasks.toy:main", "args": {"n": 4}, "smoke": False,
               "worktree": str(paths.REPO_ROOT), "commit": None, "priority": "must", "wave": 1,
               "deps": [], "est_vram_gb": 0, "est_ram_gb": 0.1, "est_minutes": 1, "kind_slot": "cpu",
               "items_total": 4, "python": sys.executable})
    s = S.Scheduler()
    t = time.time()
    while q.load_state("T-toy")["status"] not in q.TERMINAL and time.time() - t < 60:
        s.tick()
        time.sleep(0.3)
    assert q.load_state("T-toy")["status"] == q.DONE
    rd = paths.runs_dir() / "T" / "toy"
    assert json.loads((rd / "metrics.json").read_text())["n"] == 4 and (rd / "DONE").exists()
    assert (paths.private_dir() / "T" / "toy" / "secret.txt").exists()
    assert not list(paths.results_dir().rglob("secret.txt"))
    from dsgx.queue.status import headline_for_job

    assert "0.5000" in headline_for_job(q.load_job("T-toy"))


def test_task_args_leakage_rule():
    from dsgx.checks.leakage import check_task_args

    assert check_task_args({"id": "x", "args": {"calib": {"split": "test"}}})
    assert check_task_args({"id": "x", "args": {"train_split": "test"}})
    assert not check_task_args({"id": "x", "args": {"calib": {"split": "dev"}, "eval_split": "test"}})


def test_bs1_rule_for_reported_test_runs():
    from dsgx.run import resolve

    base = {"exp_id": "X", "split": "test", "datasets": ["@forget"], "batch_size": 16}
    resolve(dict(base, method={"name": "base"}))
    with pytest.raises(ValueError):
        resolve(dict(base, method={"name": "dsg-faithful"}))
    resolve(dict(base, split="dev", purpose="select", method={"name": "dsg-faithful"}))


def test_attack_success_compute(tmp_path, monkeypatch):
    """attack_success.compute over synthetic runs: gated items, paired test, base control."""
    import json

    import pandas as pd

    from dsgx import paths
    from dsgx.analysis import attack_success as asx

    root = paths.runs_dir() / "BX"
    def _run(name, method, attack, correct_by_item):
        d = root / name
        d.mkdir(parents=True)
        (d / "DONE").write_text("{}")
        (d / "config.json").write_text(json.dumps({"config": {"method": method, "attack": attack,
            "case": "bio", "forget_datasets": ["wmdp-bio"], "split": "test"}}))
        pd.DataFrame([{"item_id": f"wmdp-bio:{i}", "dataset": "wmdp-bio", "correct": c,
                       "gate_fired": method["name"] != "base", "rho": 0.5}
                      for i, c in enumerate(correct_by_item)]).to_parquet(d / "items.parquet")
    # base gets all right; dsg gets first 3 wrong (gated); under attack dsg recovers 2 of them
    _run("base__none", {"name": "base"}, {"name": "none"}, [1, 1, 1, 1, 1])
    _run("dsg__none", {"name": "dsg-faithful"}, {"name": "none"}, [0, 0, 0, 1, 1])
    _run("dsg__pad", {"name": "dsg-faithful"}, {"name": "dilution", "pad": 400}, [1, 1, 0, 1, 1])
    df = asx.compute("BX")
    atk = df[df["attack"].str.contains("dilution")].iloc[0]
    assert atk["n_gated"] == 3 and abs(atk["attack_success"]["mean"] - 2 / 3) < 1e-9
    assert atk["vs_clean_paired"]["n"] == 5
