"""Server jobs: CPU smoke with a tiny random Gemma-2 (DSG_TINY=1), Q2 graph metrics on a synthetic graph,
and consistency of every job's conf / sbatch (no GPU, no server, no real model)."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
JOBS = ["d1-full", "a6-full", "tofu-full", "a7-12b", "muse", "mtbench", "q2-graphs"]


@pytest.fixture
def tiny(monkeypatch):
    monkeypatch.setenv("DSG_TINY", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    import importlib

    from cluster import jobcommon

    importlib.reload(jobcommon)
    yield
    monkeypatch.delenv("DSG_TINY")
    importlib.reload(jobcommon)


def _reload(mod):
    import importlib

    return importlib.reload(importlib.import_module(mod))


def test_d1_full_tiny(tiny):
    from dsgx import paths

    m = _reload("cluster.d1_full")
    assert m.main(["--alphas", "0.3", "--steps", "3", "--bs", "2", "--maxlen", "32", "--ckpt", "2"]) == 0
    assert (paths.cache_dir() / "models" / "D1-full" / "undo_a0.3" / "config.json").exists()
    assert not (paths.results_dir() / "jobs" / "d1-full" / "undo_a0.3" / "last").exists()  # trainer state removed
    s = json.loads((paths.results_dir() / "jobs" / "d1-full" / "summary.json").read_text())
    assert s["tiny"] and s["hardware_label"]


def test_a6_full_tiny_and_report_reads_it(tiny):
    from dsgx import paths

    m = _reload("cluster.a6_full")
    args = ["--ks", "2", "--targets", "dsg-hook", "dsg-nohook", "--steps", "2", "--eval-at", "1", "2",
            "--n-eval", "4", "--n-util", "4", "--maxlen", "32", "--bs", "2"]
    assert m.main(args) == 0
    met = json.loads((paths.runs_dir() / "A6-full" / "relearn-dsg-hook-k2" / "metrics.json").read_text())
    assert met["rank"] == "full" and len(met["curve"]) == 2 and met["hook"] is True
    assert not list((paths.results_dir() / "jobs" / "a6-full").rglob("trainer.pt"))
    assert m.main(args) == 0  # DONE cells are skipped
    from dsgx.analysis import claims
    from dsgx.analysis.collect import load_all

    assert claims.ch6(load_all())["verdict"] == "Inconclusive"  # needs d1/d2/student cells


def test_tofu_full_tiny(tiny):
    from dsgx import paths

    m = _reload("cluster.tofu_full")
    assert m.main(["--epochs", "1", "--bs", "2", "--accum", "2", "--maxlen", "64"]) == 0
    met = json.loads((paths.runs_dir() / "A2-tofu-full" / "tofu-metrics" / "metrics.json").read_text())
    assert set(met["conditions"]) == {"retain-model", "full", "full+dsg", "full+best-gate"}


def test_muse_tiny(tiny):
    from dsgx import paths

    m = _reload("cluster.muse")
    assert m.main(["--corpora", "news", "--epochs", "1", "--bs", "1", "--accum", "1"]) == 0
    met = json.loads((paths.runs_dir() / "A5-muse" / "muse-news" / "metrics.json").read_text())
    assert set(met["conditions"]) == {"retrain", "target", "target+dsg"}
    assert not (paths.cache_dir() / "models" / "A5-muse" / "news-target").exists()  # deleted without --keep


def test_mtbench_tiny(tiny):
    from dsgx import paths

    m = _reload("cluster.mtbench_open")
    assert m.main(["--limit", "1"]) == 0
    s = json.loads((paths.results_dir() / "jobs" / "mtbench" / "summary.json").read_text())
    assert "base" in s["scores"] and s["judge"]


def test_a7_12b_plan(tiny):
    m = _reload("cluster.a7_12b")
    assert m.main(["--plan"]) == 0
    cfgs, _ = m.configs()
    assert len(cfgs) == 12 and all(c["model"]["name"] == "gemma-3-12b-it" for c in cfgs)


def test_q2_graph_metrics_synthetic():
    from cluster.q2_graphs import dsg_matches, graph_metrics

    # 2 layers, 2 positions, 3 features, 1 logit. Nodes: f0(L0) f1(L1) f2(L1) | err L0p0 L0p1 L1p0 L1p1 | tok0 tok1 | logit
    af = np.array([[0, 1, 5], [1, 1, 7], [1, 0, 9]])
    n = 3 + 4 + 2 + 1
    A = np.zeros((n, n))
    A[9, 1] = 3.0   # f1 -> logit
    A[9, 2] = 1.0   # f2 -> logit
    A[1, 0] = 1.0   # f0 -> f1
    A[0, 7] = 1.0   # tok0 -> f0
    m = graph_metrics(A, af, n_layers=2, n_pos=2, logit_probs=[1.0], matched=[False, True, False])
    assert m["n_features"] == 3 and m["n_nodes"] == n
    assert m["dsg_matched_nodes"] == 1 and 0 < m["dsg_influence_share"] < 1
    assert m["per_layer_share"][1] > m["per_layer_share"][0] > 0
    sae = np.eye(4)[:, :4]
    tc = np.vstack([np.eye(4)[2], np.eye(4)[0], np.ones(4)])
    pairs, ids = dsg_matches(sae, tc, [0, 2])
    assert pairs[0][0] == 1 and pairs[2][0] == 0 and ids == [0, 1]


def _conf(job):
    out = subprocess.run(["bash", "-c", f"HOME=/home/x; source {REPO}/cluster/slurm/jobs/{job}.conf; "
                                        'printf "%s\\n%s\\n%s\\n%s" "$CHAIN" "$RESULT_PATHS" "$LARGE_INPUTS" "$EST_DISK_GB"'],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    return out


@pytest.mark.parametrize("job", JOBS)
def test_conf_and_sbatch(job):
    chain, results, large, est = _conf(job)
    for sb in chain.split():
        t = (REPO / "cluster" / "slurm" / sb).read_text()
        assert "--gres=gpu:1" in t and "--requeue" in t and "--time=" in t and "--mem" not in t
        assert "preamble.sh" in t and f"--job-name=dsg-{job}" in t  # fetch_results greps dsg-<job>
    assert results and all(p.startswith("results/") for p in results.split())
    for p in large.split():
        assert not re.match(r"^(env|code|slurm|logs|results)(/|$)", p) and ".." not in p
    assert 0 < float(est) < 50
