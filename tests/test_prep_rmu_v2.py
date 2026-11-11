"""RMU v2 job (cluster/rmu_v2_train.py): pre-registered grid, batching, batch-size probe logic and a tiny
CPU training run with resume (DSG_TINY=1; no GPU, no hazardous data)."""
import itertools

import pytest
import torch


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setenv("DSG_TINY", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import importlib

    from cluster import jobcommon

    importlib.reload(jobcommon)
    from cluster import rmu_v2_train

    importlib.reload(rmu_v2_train)
    yield rmu_v2_train
    monkeypatch.delenv("DSG_TINY")
    importlib.reload(jobcommon)
    importlib.reload(rmu_v2_train)


def test_grid_is_a_16_config_subset_of_the_requested_factorial():
    from cluster import rmu_v2_train as m

    full = set(itertools.product([4, 8, 12, 20], [100, 300, 1200], [3, 7], [150, 300]))
    assert len(m.GRID) == 16 and len(set(m.GRID)) == 16 and set(m.GRID) <= full
    assert (4, 300, 7, 150) in m.GRID  # v1's selected config as anchor
    assert m.UPDATE_LAYERS == {3: [1, 2, 3], 7: [5, 6, 7]}
    hps = m.grid_hps({"3": 50.0, "7": 100.0}, 32)
    assert all(h["batch_size"] == 32 and h["layer_ids"] == m.UPDATE_LAYERS[h["layer_id"]] for h in hps)
    assert hps[-1]["steering_coeff"] == 20 * 50.0 and hps[0]["steering_coeff"] == 4 * 100.0


def test_make_batches_cycles_retain_in_order():
    from cluster.rmu_v2_train import make_batches

    docs = [str(i) for i in range(5)]
    assert make_batches(docs, 2, 2, cycle=False) == [["0", "1"], ["2", "3"]]
    assert make_batches(docs, 2, 4, cycle=True) == [["0", "1"], ["2", "3"], ["4", "0"], ["1", "2"]]
    with pytest.raises(AssertionError):
        make_batches(docs, 2, 3, cycle=False)


def test_pick_largest_skips_oom_and_over_budget():
    from cluster.rmu_v2_train import pick_largest

    def peak(bs):
        if bs >= 64:
            raise torch.cuda.OutOfMemoryError("probe")
        return bs * 1.0

    bs, probes = pick_largest([16, 64, 32, 48], peak, 40.0, log=lambda *_: None)
    assert bs == 32 and [p["candidate"] for p in probes] == [64, 48, 32] and probes[0]["peak_gib"] is None
    with pytest.raises(RuntimeError):
        pick_largest([16, 32], peak, 1.0, log=lambda *_: None)


def test_train_config_tiny_and_resume(tiny, tmp_path):
    from cluster import jobcommon as jc

    m = tiny
    mmlu = jc.mcq_items("high_school_geography", split="dev", n=4)
    hp = {**m.RECIPE, "layer_id": 7, "layer_ids": [5, 6, 7], "batch_size": 2, "max_num_batches": 3,
          "steering_mult": 4.0, "steering_coeff": 4.0, "alpha": 100.0, "r": 1.0}
    forget = [f"tiny forget doc {i} " * 10 for i in range(8)]
    retain = [f"tiny retain doc {i} " * 10 for i in range(3)]  # fewer than bs x steps: cycled
    m.CKPT_EVERY, m.EVAL_EVERY = 2, 2
    out, log, u = m.train_config(0, hp, forget, retain, (mmlu, mmlu), tmp_path / "models", tmp_path / "tmp",
                                 log=lambda *_: None)
    assert (out / "config.json").exists() and len(log) == 3 and u.shape[-1] == 64
    assert "mini_forget_acc" in log[1] and (tmp_path / "models" / "last" / "state.pt").exists()
    # resume: the stored state is at the final step, so nothing is retrained and the log is kept
    out2, log2, _ = m.train_config(0, hp, forget, retain, (mmlu, mmlu), tmp_path / "models", tmp_path / "tmp",
                                   log=lambda *_: None)
    assert [r["loss"] for r in log2] == [r["loss"] for r in log]
