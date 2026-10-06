"""X1-suite TOFU metrics resume at item level: SIGKILLed mid-condition, the restart scores only the remaining items and
writes the same metrics.json / items.parquet as an uninterrupted run."""
import hashlib
import json
import multiprocessing as mp
import os
import signal
from types import SimpleNamespace

import pandas as pd
import pytest

from dsgx.tasks import TaskContext
from experiments.X1suite import tofu_fix

N = 230
CONDS = [{"tag": "retain-model", "weights": "r"}, {"tag": "full", "weights": "f"}]
STATE = {"calls": 0, "kill_at": None}


class _DS(list):
    def select(self, idx):
        return _DS(self[i] for i in idx)


def _rows(name):
    return _DS({"question": f"{name} q{i}", "answer": f"a{i}", "paraphrased_answer": f"p{i}",
                "perturbed_answer": [f"x{i}", f"y{i}"]} for i in range(N))


def _fake_logprob(model, q, ans):
    STATE["calls"] += 1
    if STATE["calls"] == STATE["kill_at"]:
        os.kill(os.getpid(), signal.SIGKILL)
    return -(int(hashlib.sha256(f"{model.w}|{q}|{ans}".encode()).hexdigest()[:6], 16) % 997) / 100


@pytest.fixture
def fake(monkeypatch, tmp_path):
    import datasets

    import dsgx.models.loader as loader
    import dsgx.run as run

    monkeypatch.setattr(datasets, "load_dataset", lambda repo, cfg, split: _rows(cfg))
    monkeypatch.setattr(loader, "get_bundle", lambda weights=None, **k: SimpleNamespace(
        model=SimpleNamespace(w=weights, reset_hooks=lambda: None)))
    monkeypatch.setattr(run, "resolve_weights", lambda w, exp: w)
    monkeypatch.setattr(tofu_fix, "_answer_logprob", _fake_logprob)
    monkeypatch.setenv("DSGX_RESUME_EVERY", "100")
    STATE.update(calls=0, kill_at=None)
    return tmp_path


class _P:
    def update(self, **k):
        pass

    def advance(self, n=1, **k):
        pass


def _run(root):
    os.environ["DSG_RESULTS"], os.environ["DSG_PRIVATE"] = str(root / "res"), str(root / "priv")
    job = {"exp_id": "X1-suite", "task_id": "tofu-metrics", "commit": "c", "args": {"conditions": CONDS}}
    return tofu_fix.metrics(TaskContext(job, dict(job["args"]), _P(), False))


def _child(root, k):
    STATE.update(calls=0, kill_at=k)
    _run(root)


def _out(root):
    d = root / "res" / "runs" / "X1-suite" / "tofu-metrics"
    return json.loads((d / "metrics.json").read_text()), pd.read_parquet(d / "items.parquet")


# per condition: truth ratio forget (3 logprobs/item), retain (3), answer prob forget (1), retain (1) = 8 * N calls
@pytest.mark.parametrize("kill_at", [500, 8 * N + 3 * N + 50])  # cond 1 forget truth ratio; cond 2 retain truth ratio
def test_tofu_metrics_killed_mid_run_resumes_identically(fake, kill_at):
    ref = _run(fake / "ref")
    total = STATE["calls"]
    assert total == 2 * 8 * N
    p = mp.get_context("fork").Process(target=_child, args=(fake / "k", kill_at))
    p.start()
    p.join(60)
    assert p.exitcode == -signal.SIGKILL
    assert list((fake / "k" / "res").rglob("chunk_*.pkl"))
    STATE.update(calls=0, kill_at=None)
    assert _run(fake / "k") == ref
    assert STATE["calls"] < total
    assert not list((fake / "k" / "res").rglob("partial"))
    (ma, ia), (mb, ib) = _out(fake / "ref"), _out(fake / "k")
    assert ma == mb
    pd.testing.assert_frame_equal(ia, ib)
