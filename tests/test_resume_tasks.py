"""Item-level resume of task loops (dsgx.itemckpt, dsgx.eval.openqa:task): a task SIGKILLed mid-run restarts from
its last checkpoint, generates only the remaining items and writes the same items.parquet / metrics.json /
generations.jsonl as an uninterrupted run."""
import hashlib
import json
import multiprocessing as mp
import os
import pickle
import signal
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from dsgx import itemckpt
from dsgx.eval import openqa
from dsgx.gen.stream import GenResult
from dsgx.tasks import TaskContext

N_ITEMS = 430
METHODS = [{"name": "base"},
           {"name": "dsg-faithful", "mode": "stream", "features": [3, 7], "tau": 0.5}]
STATE = {"generated": 0, "kill_at": None}


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def _fake_generate(model, prompt, bundle=None, features=None, multiplier=500.0, tau=None, mode="stream", max_new=64):
    """Deterministic per (prompt, mode); SIGKILLs the process at the kill_at-th generation."""
    STATE["generated"] += 1
    if STATE["generated"] == STATE["kill_at"]:
        os.kill(os.getpid(), signal.SIGKILL)
    h = _h(prompt + mode)
    toks = [(h >> k) % 97 + 3 for k in range(h % 9 + 4)]
    gate = [bool((h >> k) & 1) for k in range(len(toks))] if mode != "none" else [False] * len(toks)
    return GenResult(prompt_len=h % 40 + 10, tokens=toks, rho_trace=[(h % 100) / 100] * len(toks), gate_trace=gate,
                     fire_trace=[], prompt_gate=gate[0], first_gate_step=0 if gate[0] else None, n_rebuilds=h % 3,
                     stop_reason="eos" if h % 2 else "max_new")


@pytest.fixture
def fake_openqa(monkeypatch, tmp_path):
    import dsgx.data.activation_cache as ac
    import dsgx.gen.stream as stream
    import dsgx.models.loader as loader
    import dsgx.run as run

    items = [openqa.OpenItem(f"it{i}", "toy", f"question {i}", f"answer {i % 13}", "s") for i in range(N_ITEMS)]
    monkeypatch.setattr(openqa, "item_set", lambda name, split="test": items)
    monkeypatch.setattr(openqa, "grade", lambda preds, refs: [
        {"match": p.endswith(r[-1]), "token_f1": (_h(p + r) % 100) / 100, "rougeL_f": (_h(r + p) % 50) / 50,
         "embed_sim": (_h(p) % 10) / 10} for p, r in zip(preds, refs)])
    monkeypatch.setattr(stream, "generate", _fake_generate)
    monkeypatch.setattr(stream, "decode", lambda model, toks: " ".join(map(str, toks)))
    monkeypatch.setattr(loader, "get_bundle", lambda *a, **k: SimpleNamespace(model=None))
    monkeypatch.setattr(run, "resolve_weights", lambda w, exp: None)
    monkeypatch.setattr(ac, "build_cache", lambda *a, **k: None)
    monkeypatch.setattr(ac, "ActivationCache", lambda c: c)
    monkeypatch.setenv("DSGX_RESUME_EVERY", "200")
    STATE.update(generated=0, kill_at=None)
    return tmp_path


class _Progress:
    def update(self, **kw):
        pass

    def advance(self, n=1, **kw):
        pass


def _run_task(root: Path):
    os.environ["DSG_RESULTS"], os.environ["DSG_PRIVATE"] = str(root / "results"), str(root / "private")
    job = {"exp_id": "RESUME", "task_id": "open", "commit": "abc", "args": {"items": "toy", "methods": METHODS}}
    return openqa.task(TaskContext(job, dict(job["args"]), _Progress(), False))


def _killed_child(root, kill_at):
    STATE.update(generated=0, kill_at=kill_at)
    _run_task(root)


def _outputs(root: Path):
    out = {}
    for m in ("base-stream", "dsg-faithful-stream"):
        d = root / "results" / "runs" / "RESUME" / f"open__{m}"
        out[m] = (pd.read_parquet(d / "items.parquet"), json.loads((d / "metrics.json").read_text()),
                  (root / "private" / "RESUME" / f"open__{m}" / "generations.jsonl").read_text())
    return out


@pytest.mark.parametrize("kill_at", [250, 650])  # inside the first method; inside the second
def test_openqa_task_killed_mid_run_resumes_identically(fake_openqa, kill_at):
    ref = _run_task(fake_openqa / "ref")
    assert STATE["generated"] == 2 * N_ITEMS

    root = fake_openqa / "killed"
    p = mp.get_context("fork").Process(target=_killed_child, args=(root, kill_at))
    p.start()
    p.join(60)
    assert p.exitcode == -signal.SIGKILL
    first = root / "private" / "RESUME" / "open__base-stream"
    assert not (root / "results" / "runs" / "RESUME" / "open__dsg-faithful-stream" / "DONE").exists()
    chunks = sorted(first.glob("partial/*.pkl")) + sorted((first.parent / "open__dsg-faithful-stream").glob("partial/*.pkl"))
    # 200-item chunks: 430 items -> chunks of 200, 200, 30 per method
    saved = (kill_at - 1) // 200 if kill_at <= N_ITEMS else 3 + (kill_at - 1 - N_ITEMS) // 200
    assert len(chunks) == saved

    STATE.update(generated=0, kill_at=None)
    assert _run_task(root) == ref
    reused = 200 * saved if kill_at <= N_ITEMS else N_ITEMS + 200 * (saved - 3)
    assert STATE["generated"] == 2 * N_ITEMS - reused
    assert not list((root / "private").rglob("partial"))
    a, b = _outputs(fake_openqa / "ref"), _outputs(root)
    for m in a:
        pd.testing.assert_frame_equal(a[m][0], b[m][0])
        assert a[m][1] == b[m][1] and a[m][2] == b[m][2]


def test_stale_or_corrupt_chunk_is_recomputed(tmp_path):
    calls = []

    def fn(x):
        calls.append(x)
        return x * 2

    ck = itemckpt.ItemCheckpoints(tmp_path, {"run": 1}, every=4)
    assert ck.map(range(10), fn) == [x * 2 for x in range(10)]
    (tmp_path / "chunk_000004.pkl").write_bytes(b"torn write")
    other = itemckpt.ItemCheckpoints(tmp_path, {"run": 2}, every=4)  # other fingerprint: nothing reused
    calls.clear()
    assert other.map(range(10), fn) == [x * 2 for x in range(10)] and len(calls) == 10 and other.reused == 0
    same = itemckpt.ItemCheckpoints(tmp_path, {"run": 2}, every=4)
    (tmp_path / "chunk_000004.pkl").write_bytes(pickle.dumps({"tag": "x", "values": [0] * 4}))
    calls.clear()
    assert same.map(range(10), fn) == [x * 2 for x in range(10)] and calls == [4, 5, 6, 7] and same.reused == 6


def test_resume_off(tmp_path):
    ck = itemckpt.ItemCheckpoints(tmp_path / "p", {}, every=0)
    assert ck.map([1, 2, 3], lambda x: -x) == [-1, -2, -3]
    assert not (tmp_path / "p").exists()
