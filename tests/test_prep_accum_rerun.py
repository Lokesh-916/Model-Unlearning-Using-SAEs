"""Session 13 re-run guards: tofu-full / muse never reuse a model, partial result or DONE written by the old
training loop (no train_version 2), and tofu-full stops at a checkpoint when its time budget runs out."""
import json

import pytest

from test_prep_server_jobs import _reload, tiny  # noqa: F401  (fixture)

TOFU = ["--epochs", "1", "--bs", "2", "--accum", "2", "--maxlen", "64"]


def test_tofu_full_train_version_guards(tiny):  # noqa: F811
    from dsgx import paths

    m = _reload("cluster.tofu_full")
    models = paths.cache_dir() / "models" / "A2-tofu-full"
    assert m.main(TOFU + ["--budget-min", "0"]) == 0  # no budget: nothing trained, nothing finished
    assert not (models / "full" / "config.json").exists()
    assert m.main(TOFU + ["--keep-retain"]) == 0
    for t in ("full", "retain"):
        tv = json.loads((models / t / "train_version.json").read_text())
        assert tv["train_version"] == m.TRAIN_VERSION == 2 and tv["effective_batch"] == 4
        assert m.model_ok(models / t)
    d = paths.runs_dir() / "A2-tofu-full" / "tofu-metrics"
    assert json.loads((d / "metrics.json").read_text())["train_version"] == 2
    # a partial from the old loop (no train_version) is re-evaluated, current ones are kept
    p = d / "partial" / "full+dsg.json"
    rec = json.loads(p.read_text())
    rec.pop("train_version")
    p.write_text(json.dumps(rec))
    keep = (d / "partial" / "full.json").stat().st_mtime_ns
    (d / "DONE").unlink()
    assert m.main(TOFU + ["--keep-retain"]) == 0
    assert json.loads(p.read_text())["train_version"] == 2
    assert (d / "partial" / "full.json").stat().st_mtime_ns == keep
    # a model without the marker (old loop) is refused, never reused
    (models / "retain" / "train_version.json").unlink()
    assert not m.model_ok(models / "retain")
    (d / "DONE").unlink()
    with pytest.raises(SystemExit, match="old loop"):
        m.main(TOFU)


def test_muse_old_done_is_redone(tiny):  # noqa: F811
    from dsgx import paths

    m = _reload("cluster.muse")
    d = paths.runs_dir() / "A5-muse" / "muse-news"
    (d / "partial").mkdir(parents=True, exist_ok=True)
    (d / "DONE").write_text(json.dumps({"headline": {}}))  # job 130's DONE: no train_version
    (d / "partial" / "retrain.json").write_text(json.dumps({"metrics": {}}))
    assert m.main(["--corpora", "news", "--epochs", "1", "--bs", "1", "--accum", "1"]) == 0
    assert json.loads((d / "DONE").read_text())["train_version"] == 2
    for f in ("retrain", "target", "target+dsg", "target+best-gate"):
        assert json.loads((d / "partial" / f"{f}.json").read_text())["train_version"] == 2
    assert json.loads((d / "metrics.json").read_text())["train_version"] == 2


def test_tofu_full_drops_retain_and_skips_when_done(tiny):  # noqa: F811
    from dsgx import paths

    m = _reload("cluster.tofu_full")
    models = paths.cache_dir() / "models" / "A2-tofu-full"
    assert m.main(TOFU) == 0
    assert m.model_ok(models / "full") and not (models / "retain").exists()  # retain removed after the metrics
    done = paths.runs_dir() / "A2-tofu-full" / "tofu-metrics" / "DONE"
    t = done.stat().st_mtime_ns
    assert m.main(TOFU) == 0  # second chained copy: no retraining of retain, nothing rewritten
    assert done.stat().st_mtime_ns == t and not (models / "retain").exists()
