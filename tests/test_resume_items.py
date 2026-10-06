"""Item-level resume of dsgx.run.run: a run SIGKILLed mid-dataset restarts from its last
checkpoint and writes the same items.parquet / traces.npz / metrics as an uninterrupted run."""
import json
import multiprocessing as mp
import os
import signal
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dsgx import run as R
from dsgx.data.mcq import MCQItem

N_FORGET, N_UTIL = 530, 90


class _Method:
    records_gate = True
    tau = 0.5
    info = {"fake": True}

    def __init__(self):
        self.lengths, self.recs = [], []

    def install(self):
        pass

    def remove(self):
        pass

    def set_lengths(self, lengths):
        self.lengths = lengths

    def pop_records(self):
        r, self.recs = self.recs, []
        return r


def _fake_score(model, prompts, batch_size=1, method=None, on_batch=None):
    """Deterministic per prompt; counts every scored item in model['scored']."""
    probs, lens = [], []
    for i, p in enumerate(prompts):
        rng = np.random.default_rng(int.from_bytes(p.encode()[-8:].ljust(8, b"0"), "little") % 2**32)
        probs.append(rng.random(4).astype(np.float32))
        lens.append(len(p) % 50 + 5)
        method.recs.append({"rho": float(rng.random()), "gate_fired": bool(rng.random() > 0.5),
                            "fire_trace": rng.integers(0, 2, lens[-1]).astype(np.uint8)})
        model["scored"] += 1
        if on_batch and (i + 1) % batch_size == 0:
            on_batch(i + 1)
        if model["scored"] == model.get("kill_at"):
            os.kill(os.getpid(), signal.SIGKILL)
    return np.stack(probs), method.pop_records(), lens


@pytest.fixture
def fake_harness(monkeypatch):
    items = {ds: {i: MCQItem(ds, i, f"q{ds}{i}", ("a", "b", "c", "d"), i % 4, ds) for i in range(n)}
             for ds, n in (("wmdp-bio", N_FORGET), ("mmlu-x", N_UTIL))}
    model = {"scored": 0}
    bundle = type("B", (), {"model": model})()
    monkeypatch.setattr(R, "get_bundle", lambda *a, **k: bundle)
    monkeypatch.setattr(R, "ensure_cache", lambda *a, **k: [])
    monkeypatch.setattr(R, "make_method", lambda *a, **k: _Method())
    monkeypatch.setattr(R, "load_mcq", lambda ds: items[ds])
    monkeypatch.setattr(R, "get_split", lambda ds, split: list(range(len(items[ds]))))
    monkeypatch.setattr(R, "dsg_subset_ids", lambda case, ds: list(range(0, len(items[ds]), 2)))
    monkeypatch.setattr(R, "split_hash", lambda ds: "h")
    monkeypatch.setattr(R, "score_prompts", _fake_score)
    monkeypatch.setenv("DSGX_RESUME_EVERY", "200")
    return model


CFG = {"exp_id": "RESUME", "method": {"name": "fake", "features": [1], "tau": 0.5},
       "datasets": ["wmdp-bio", "mmlu-x"], "forget_datasets": ["wmdp-bio"], "split": "dev",
       "batch_size": 3, "n_boot": 50}


def _outputs(run_dir: Path):
    m = json.loads((run_dir / "metrics.json").read_text())
    m.pop("timing")
    tr = np.load(run_dir / "traces.npz")
    return pd.read_parquet(run_dir / "items.parquet"), m, {k: tr[k] for k in tr.files}


def _killed_child(cfg, kill_at):
    os.environ["DSGX_RESUME_EVERY"] = "200"
    R.get_bundle("x").model.update(scored=0, kill_at=kill_at)
    R.run(cfg)


@pytest.mark.parametrize("kill_at", [450, 560])  # mid forget set; mid utility set
def test_kill_mid_run_resumes_with_identical_items(fake_harness, tmp_path, monkeypatch, kill_at):
    ref_root = tmp_path / "ref"
    monkeypatch.setenv("DSG_RESULTS", str(ref_root))
    ref_dir = R.run(dict(CFG))
    assert fake_harness["scored"] == N_FORGET + N_UTIL
    assert not (ref_dir / R.PARTIAL_DIR).exists()

    monkeypatch.setenv("DSG_RESULTS", str(tmp_path / "killed"))
    p = mp.get_context("fork").Process(target=_killed_child, args=(dict(CFG), kill_at))
    p.start()
    p.join(60)
    assert p.exitcode == -signal.SIGKILL
    run_dir = R.RunLogger(R.resolve(dict(CFG))).dir
    assert not (run_dir / "DONE").exists()
    parts = sorted((run_dir / R.PARTIAL_DIR).glob("*.pkl"))
    chunk = R._resume_chunk(CFG["batch_size"])  # 201 = 67 batches of 3
    assert chunk == 201
    done_before = (kill_at // chunk) if kill_at <= N_FORGET else (-(-N_FORGET // chunk) + (kill_at - N_FORGET) // chunk)
    assert len(parts) == done_before

    fake_harness["scored"] = 0
    out_dir = R.run(dict(CFG))
    assert out_dir == run_dir
    reused = (kill_at // chunk) * chunk if kill_at <= N_FORGET else N_FORGET
    assert fake_harness["scored"] == N_FORGET + N_UTIL - reused
    assert not (run_dir / R.PARTIAL_DIR).exists()

    items_a, met_a, tr_a = _outputs(ref_dir)
    items_b, met_b, tr_b = _outputs(run_dir)
    pd.testing.assert_frame_equal(items_a, items_b)
    assert met_a == met_b
    assert tr_a.keys() == tr_b.keys() and all(np.array_equal(tr_a[k], tr_b[k]) for k in tr_a)


def test_stale_partial_is_ignored(fake_harness, tmp_path):
    """A checkpoint from different prompts (e.g. another attack seed) is recomputed, not reused."""
    run_dir = R.RunLogger(R.resolve(dict(CFG))).dir
    f = run_dir / R.PARTIAL_DIR / "wmdp-bio__000000.pkl"
    f.parent.mkdir(parents=True)
    import pickle

    f.write_bytes(pickle.dumps({"tag": "other", "start": 0, "end": 201, "probs": np.zeros((201, 4)),
                                "recs": [], "lens": [0] * 201}))
    R.run(dict(CFG))
    assert fake_harness["scored"] == N_FORGET + N_UTIL
