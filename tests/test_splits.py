import json

import pytest

from dsgx.data import splits


def test_split_deterministic_disjoint_cover(tmp_path):
    d, t = splits.split_ids(101, "wmdp-bio")
    d2, t2 = splits.split_ids(101, "wmdp-bio")
    assert (d, t) == (d2, t2)
    assert not set(d) & set(t) and set(d) | set(t) == set(range(101))
    assert len(d) == 50
    # different datasets get different permutations
    assert splits.split_ids(101, "wmdp-cyber")[0] != d


def test_split_file_tamper_detected(tmp_path):
    splits.make_split_file("toy", 20, out_dir=tmp_path)
    assert len(splits.get_split("toy", "dev", tmp_path)) == 10
    rec = json.loads((tmp_path / "toy.json").read_text())
    rec["dev"][0], rec["test"][0] = rec["test"][0], rec["dev"][0]
    (tmp_path / "toy.json").write_text(json.dumps(rec))
    with pytest.raises(ValueError):
        splits.get_split("toy", "dev", tmp_path)


def test_committed_split_files_are_valid():
    from dsgx.checks.leakage import check_splits

    assert check_splits() == []
    assert len(splits.get_split("wmdp-bio", "all")) == 1273
    rec = splits.load_split_file("wmdp-bio")
    assert rec["seed"] == 0 and (rec["dev"], rec["test"]) == tuple(splits.split_ids(1273, "wmdp-bio"))
