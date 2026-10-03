"""Server jobs: CPU smoke with a tiny random Gemma-2 (DSG_TINY=1), Q2 graph metrics on a synthetic graph,
and consistency of every job's conf / sbatch (no GPU, no server, no real model)."""
import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
JOBS = ["d1-full", "a6-full", "tofu-full", "a7-12b", "muse", "mtbench", "q2-graphs", "rmu-v2",
        "c3", "a6-lora", "a6-baked", "a7-small", "d1-v2", "figs"]


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
    # resume: drop the final metrics and one condition; a rerun re-evaluates only that one
    part = paths.runs_dir() / "A2-tofu-full" / "tofu-metrics" / "partial"
    assert len(list(part.glob("*.json"))) == 4
    (part / "full+best-gate.json").unlink()
    keep = (part / "full.json").stat().st_mtime_ns
    assert m.main(["--epochs", "1", "--bs", "2", "--accum", "2", "--maxlen", "64"]) == 0
    assert (part / "full.json").stat().st_mtime_ns == keep and (part / "full+best-gate.json").exists()
    met2 = json.loads((paths.runs_dir() / "A2-tofu-full" / "tofu-metrics" / "metrics.json").read_text())
    assert set(met2["conditions"]) == set(met["conditions"]) and "n" not in met2["conditions"]["full"]


def test_muse_tiny(tiny, tmp_path):
    from dsgx import paths

    m = _reload("cluster.muse")
    args = ["--corpora", "news", "--epochs", "1", "--bs", "1", "--accum", "1"]
    # no budget: nothing starts, nothing is written as finished
    assert m.main(args + ["--budget-min", "0"]) == 0
    assert not (paths.runs_dir() / "A5-muse" / "muse-news" / "DONE").exists()
    off = _muse_official(tmp_path)
    assert m.main(args + (["--official", str(off)] if off else [])) == 0
    met = json.loads((paths.runs_dir() / "A5-muse" / "muse-news" / "metrics.json").read_text())
    assert set(met["conditions"]) == {"retrain", "target", "target+dsg", "target+best-gate"}
    assert met["conditions"]["retrain"]["privleak"] == 0.0  # PrivLeak relative to our own retrain model
    assert met["gates"]["target+best-gate"]["kind"] == "window" and met["gates"]["target+dsg"]["kind"] == "rho"
    if off:
        assert met["implementation"].startswith("official muse_bench")
    for c in ("target", "retrain"):
        assert not (paths.cache_dir() / "models" / "A5-muse" / f"news-{c}").exists()  # deleted without --keep
    assert m.main(args) == 0  # finished corpus is skipped


def _muse_official(tmp_path):
    """Unpack the official muse_bench metrics (lab PC tarball) into tmp, if it was fetched."""
    import tarfile

    tb = REPO.parent / "wheels/muse/muse_bench-main.tar.gz"
    if not tb.exists():
        return None
    with tarfile.open(tb) as t:
        t.extractall(tmp_path, members=[x for x in t.getmembers() if x.name.startswith("muse_bench-main/metrics/")], filter="data")
    return tmp_path / "muse_bench-main"


def test_muse_prompt_gate(tiny):
    """Prompt-only gating: the decision is taken on the prompt and kept for single-token (KV-cache) steps."""
    import torch

    from cluster import jobcommon as jc
    from cluster.muse import PromptGate
    from cluster.tofu_full import Gate

    sae = jc.load_sae()
    g = Gate(sae, list(range(256)), "rho", thr=2.0)  # rho <= 1 < 2: never fires on the prompt
    pg = PromptGate(g)
    x = torch.randn(1, 5, sae.W_dec.shape[1])
    assert torch.equal(pg(None, None, x), x) and pg.on == [False]
    y = torch.randn(1, 1, sae.W_dec.shape[1]) * 10
    assert torch.equal(pg(None, None, y), y)  # a firing generated token does not flip the prompt decision
    g.thr = -1.0
    pg2 = PromptGate(g)
    pg2(None, None, x)
    assert pg2.on == [True] and pg2.n_fired == 1


def test_mtbench_tiny(tiny):
    from dsgx import paths

    m = _reload("cluster.mtbench_open")
    assert m.main(["--limit", "1"]) == 0
    s = json.loads((paths.results_dir() / "jobs" / "mtbench" / "summary.json").read_text())
    assert "base" in s["scores"] and s["judge"]


def test_a7_server_plan(tiny):
    m = _reload("cluster.a7_server")
    assert m.main(["--plan", "--sizes", "1b", "4b", "12b"]) == 0
    for size, name in (("1b", "gemma-3-1b-it"), ("12b", "gemma-3-12b-it")):
        cfgs, _ = m.configs(size)
        assert len(cfgs) == 12 and all(c["model"]["name"] == name and c["exp_id"] == f"A7-{size}" for c in cfgs)


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


def test_muse_best_gate_reads_combine_selection(tiny, tmp_path, monkeypatch):
    """COMBINE_SELECTION.json slots hold candidate names (dsgx.combine.select); window-w24 -> w = 24."""
    import json as _json
    from types import SimpleNamespace

    from cluster import muse
    from dsgx import paths

    (paths.results_dir()).mkdir(parents=True, exist_ok=True)
    (paths.results_dir() / "COMBINE_SELECTION.json").write_text(_json.dumps({"slots": {"detector": "window-w24", "features": None}}))
    tok = muse.jc.load_tok()
    m = muse.jc.load_lm()
    texts = [f"word{i} " * 80 for i in range(3)]
    _, _, info = muse.calibrate_gates(m, tok, texts, texts, SimpleNamespace(n_calib=2, window=None))
    (paths.results_dir() / "COMBINE_SELECTION.json").unlink()
    assert info["window_w"] == 24 and info["best_gate_source"] == "window-w24"
