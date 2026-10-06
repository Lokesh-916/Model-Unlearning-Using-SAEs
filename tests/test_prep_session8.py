"""Session 8 server jobs on CPU: lab_jobs runner (fake snapshot), d1_v2 (tiny model + selection rule),
figparity plan, a6_full d1v2 target. No GPU, no server."""
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


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


# ----------------------------------------------------------------------------- lab_jobs.py
STUB = textwrap.dedent('''
    import json, os, sys
    from pathlib import Path
    job = sys.argv[sys.argv.index("--job") + 1]
    spec = json.loads((Path(os.environ["DSG_RESULTS"]) / "queue" / "jobs" / f"{job}.json").read_text())
    if spec.get("fail"):
        sys.exit(1)
    d = Path(os.environ["DSG_RESULTS"]) / "runs" / spec["exp_id"] / "_jobs" / job
    d.mkdir(parents=True, exist_ok=True)
    (d / "DONE").write_text("{}")
    print("ran", job, "factor", os.environ.get("DSGX_BATCH_FACTOR"))
''')


def _labjobs_env(tmp_path, specs, commit="c0ffee" * 6):
    dsgc = tmp_path / "dsgc"
    snap = dsgc / "code-X-c0ffeec"
    (snap / "dsgx" / "queue").mkdir(parents=True)
    for d in ("dsgx", "dsgx/queue"):
        (snap / d / "__init__.py").write_text("")
    (snap / "dsgx" / "queue" / "worker.py").write_text(STUB)
    (snap / "CODE_COMMIT.json").write_text(json.dumps({"commit": commit}))
    g = dsgc / "labjobs" / "grp"
    g.mkdir(parents=True)
    (g / "ORDER").write_text("\n".join(s["id"] for s in specs) + "\n")
    (g / "SNAPSHOT").write_text("X-c0ffeec\n")
    res = Path(os.environ["DSG_RESULTS"])
    (res / "queue" / "jobs").mkdir(parents=True, exist_ok=True)
    for s in specs:
        (res / "queue" / "jobs" / f"{s['id']}.json").write_text(json.dumps({"pinned_commit": commit, "commit": None, **s}))
    return dsgc


def _labjobs(dsgc, *args):
    env = dict(os.environ, DSGC=str(dsgc), PYTHONPATH=str(REPO))
    return subprocess.run([sys.executable, str(REPO / "cluster" / "lab_jobs.py"), "--group", "grp", *args],
                          env=env, capture_output=True, text=True)


def test_lab_jobs_runs_skips_and_labels(tmp_path):
    specs = [{"id": "X-1", "exp_id": "X", "est_minutes": 1, "deps": []},
             {"id": "X-2", "exp_id": "X", "est_minutes": 1, "deps": ["X-1"]},
             {"id": "X-bad", "exp_id": "X", "est_minutes": 1, "deps": [], "fail": True},
             {"id": "X-after", "exp_id": "X", "est_minutes": 1, "deps": ["X-bad"]},
             {"id": "X-big", "exp_id": "X", "est_minutes": 10 ** 4, "deps": []}]
    dsgc = _labjobs_env(tmp_path, specs)
    assert _labjobs(dsgc, "--plan").returncode == 0
    r = _labjobs(dsgc, "--budget-min", "60")
    assert r.returncode == 1, r.stdout + r.stderr            # X-bad failed
    res = Path(os.environ["DSG_RESULTS"])
    done = {p.parent.name for p in res.glob("runs/X/_jobs/*/DONE")}
    assert done == {"X-1", "X-2"}                            # X-after: dep failed; X-big: over budget
    assert json.loads((res / "runs" / "X" / "HARDWARE.json").read_text())["label"] == "gpuws"
    st = json.loads((res / "jobs" / "labjobs-grp" / "status.json").read_text())
    assert set(st["left"]) == {"X-bad", "X-after", "X-big"} and st["failed"] == ["X-bad"]
    assert "INCOMPLETE" in r.stdout
    # hardware label of exp-branch runs comes from the marker
    from dsgx.analysis.collect import Run

    rd = res / "runs" / "X" / "X__base__none__d__test__s0__abcd1234"
    rd.mkdir()
    assert Run(dir=rd, exp="X", config={"config": {}}, metrics={"views": {}}).hardware == "gpuws"


def test_lab_jobs_refuses_wrong_snapshot(tmp_path):
    dsgc = _labjobs_env(tmp_path, [{"id": "X-1", "exp_id": "X", "est_minutes": 1, "deps": []}])
    (dsgc / "code-X-c0ffeec" / "CODE_COMMIT.json").write_text(json.dumps({"commit": "deadbeef" * 5}))
    r = _labjobs(dsgc, "--plan")
    assert r.returncode == 2 and "ABORT" in r.stdout


def test_lab_jobs_offline_sae_shapes_bootstrap(tmp_path):
    """--offline-sae-shapes: the worker runs as __main__ with its --job argument and the sae_lens shape reader patched."""
    dsgc = _labjobs_env(tmp_path, [{"id": "X-1", "exp_id": "X", "est_minutes": 1, "deps": []}])
    w = dsgc / "code-X-c0ffeec" / "dsgx" / "queue" / "worker.py"
    w.write_text("from sae_lens.loading import pretrained_sae_loaders as L\n"
                 "assert __name__ == '__main__' and L.get_safetensors_tensor_shapes.__name__ == 'shapes'\n" + STUB)
    r = _labjobs(dsgc, "--offline-sae-shapes")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (Path(os.environ["DSG_RESULTS"]) / "runs" / "X" / "_jobs" / "X-1" / "DONE").exists()


# ----------------------------------------------------------------------------- d1_v2
def test_d1_v2_train_tiny_and_budget_stop(tiny, monkeypatch):
    from dsgx import paths

    m = _reload("cluster.d1_v2")
    args = ["--alphas", "0.05", "0.1", "--steps", "4", "--bs", "2", "--maxlen", "32", "--ckpt", "2"]
    # budget that allows starting (>= 20 min) but not continuing past the first checkpoint (needs 14 min)
    calls = {"n": 0}
    real = m.jc.Budget.fits

    def fits(self, minutes):
        calls["n"] += 1
        return calls["n"] <= 1 or minutes > 100
    monkeypatch.setattr(m.jc.Budget, "fits", fits)
    assert m.main(["train", *args]) == 0
    root = paths.cache_dir() / "models" / "D1-v2"
    assert not (root / "undo_a0.05" / "config.json").exists()         # stopped at step 2, checkpointed
    assert (paths.results_dir() / "jobs" / "d1-v2" / "undo_a0.05" / "last" / "trainer.pt").exists()
    monkeypatch.setattr(m.jc.Budget, "fits", real)
    assert m.main(["train", *args]) == 0                               # resumes, finishes both alphas
    assert (root / "undo_a0.05" / "config.json").exists() and (root / "undo_a0.1" / "config.json").exists()
    log = pd.read_parquet(root / "undo_a0.05.train_log.parquet")
    assert list(log["step"]) == [1, 2, 3, 4] and log["dev_forget_acc"].notna().iloc[-1]
    assert not (paths.results_dir() / "jobs" / "d1-v2" / "undo_a0.05" / "last").exists()


def _fake_dev(m, label, forget, util, n=100):
    from dsgx import paths

    rd = paths.runs_dir() / "D1-v2" / label
    rd.mkdir(parents=True, exist_ok=True)
    rows = ([{"dataset": "wmdp-bio", "correct": i < forget * n} for i in range(n)]
            + [{"dataset": "human_aging", "correct": i < util * n} for i in range(n)]
            + [{"dataset": "high_school_geography", "correct": False} for _ in range(n)])  # excluded subject
    pd.DataFrame(rows).to_parquet(rd / "items.parquet")
    (rd / "DONE").write_text("{}")
    f = m.jc.job_dir(m.NAME) / "dev_runs.json"
    d = json.loads(f.read_text()) if f.exists() else {}
    d[label] = str(rd)
    f.write_text(json.dumps(d))


def test_d1_v2_selection_rule(tiny):
    m = _reload("cluster.d1_v2")
    assert m.select([0.05, 0.1, 0.2]) is None                          # no DEV runs yet
    _fake_dev(m, "base", 0.64, 0.60)
    _fake_dev(m, "undo-v2-a0.05", 0.50, 0.59)                          # drop 0.01: eligible
    _fake_dev(m, "undo-v2-a0.1", 0.40, 0.58)                           # drop 0.02: eligible, lowest forget
    _fake_dev(m, "undo-v2-a0.2", 0.30, 0.50)                           # drop 0.10: not eligible
    s = m.select([0.05, 0.1, 0.2])
    assert s["selected_alpha"] == 0.1 and s["bound_met"]
    assert s["students_dev"]["0.2"]["eligible"] is False
    _fake_dev(m, "undo-v2-a0.05", 0.50, 0.55)
    _fake_dev(m, "undo-v2-a0.1", 0.40, 0.54)
    s = m.select([0.05, 0.1, 0.2])
    assert s["selected_alpha"] == 0.05 and not s["bound_met"]          # fallback: smallest drop
    assert m.main(["test", "--alphas", "0.05", "0.1", "0.2"]) == 0
    assert json.loads((m.jc.job_dir(m.NAME) / "SELECTION.json").read_text())["selected_alpha"] == 0.05


def test_a6_full_d1v2_target(tiny):
    from dsgx import paths

    m = _reload("cluster.a6_full")
    sel = paths.results_dir() / "jobs" / "d1-v2" / "SELECTION.json"
    sel.parent.mkdir(parents=True, exist_ok=True)
    sel.write_text(json.dumps({"checkpoint": str(paths.cache_dir() / "models" / "D1-v2" / "undo_a0.1")}))
    assert m.main(["--targets", "d1v2", "--ks", "2", "--steps", "2", "--eval-at", "2", "--bs", "2", "--maxlen", "32",
                   "--n-eval", "2", "--n-util", "2"]) == 0
    met = json.loads((paths.runs_dir() / "A6-full-d1v2" / "relearn-d1v2-k2" / "metrics.json").read_text())
    assert met["condition"] == "d1v2" and met["batch_size"] == 2 and met["curve"]
    assert (paths.results_dir() / "jobs" / "a6-full-d1v2" / "summary.json").exists()


# ----------------------------------------------------------------------------- figparity
def test_figparity_plan(tiny):
    from dsgx import paths

    m = _reload("cluster.figparity")
    assert m.main(["clamp", "dataeff", "static", "multitopic", "latency", "highlight", "--plan"]) == 0
    s = json.loads((paths.results_dir() / "jobs" / "figs" / "summary.json").read_text())["parts"]
    n = {k: len(v["left"]) for k, v in s.items()}
    assert n == {"clamp": 1 + 2 * 36, "dataeff": 1 + 2 * 16, "static": 5, "multitopic": 5, "latency": 1, "highlight": 1}
    from dsgx.run import resolve

    c = resolve(m.cfg("FP-static", "x", m.ours(threshold=-1.0), "test", ["@forget"]))
    assert c["method"]["gate"]["threshold"] == -1.0 and c["batch_size"] == 1
    assert resolve(m.cfg("FP-clamp", "x", m.dsg(10, 25), "dev", ["@forget"]))["batch_size"] == 16


# ----------------------------------------------------------------------------- paper_assets parity
def test_paper_assets_parity(tmp_path):
    from dsgx.analysis import paper_assets
    from tests.fakeruns import bern, mcq_run, task_run

    DSG = {"name": "dsg-faithful", "n_features": 20, "retain_pct": 95, "multiplier": 500}
    OURS = {"name": "gated", "gate": {"type": "window", "w": 16}}
    many = {"datasets": ["wmdp-bio"] + [f"s{i}" for i in range(12)]}
    mcq_run("A1-test", DSG, forget=bern(0.3, 60, 1), util=bern(0.8, 60, 2), extra_cfg=many)
    mcq_run("A1-test", {"name": "base"}, forget=bern(0.6, 60, 1), util=bern(0.8, 60, 2), extra_cfg=many)
    mcq_run("B1", DSG, attack={"name": "dilution", "pad": 400}, forget=bern(0.5, 60, 3), util=bern(0.8, 60, 2))
    mcq_run("A1-dev", DSG, split="dev", forget=bern(0.3, 60, 1), util=bern(0.8, 60, 2))
    mcq_run("A1-dev", {"name": "base"}, case="cyber", split="dev", forget=bern(0.4, 60, 4), util=bern(0.8, 60, 2))
    for i, rc in enumerate(("wikitext", "mmlu-aux-chat")):   # Cyber Pareto (session 10 Wave-1 decision)
        mcq_run("A1-dev", {**DSG, "retain_corpus": rc}, case="cyber", split="dev", forget=bern(0.3 - 0.05 * i, 60, 5 + i),
                util=bern(0.7 - 0.2 * i, 60, 7 + i))
    mcq_run("A1-test", {**DSG, "retain_corpus": "wikitext"}, case="cyber", forget=bern(0.3, 60, 9), util=bern(0.7, 60, 10))
    mcq_run("A1-test", {"name": "base"}, case="cyber", forget=bern(0.4, 60, 11), util=bern(0.8, 60, 2))
    for me, m in (("dsg", DSG), ("ours", OURS)):
        for n in (10, 20):
            for c in (10, 500):
                mcq_run("FP-clamp", m, split="dev", forget=bern(0.3 + 0.3 * (c == 10), 40, n + c), util=bern(0.8, 40, 2),
                        extra_cfg={"fp_params": {"n": n, "c": c, "method": me}, "dataset_label": f"{me}-N{n}-c{c}"})
        for mm in (32, 1024):
            mcq_run("FP-dataeff", m, split="dev", forget=bern(0.4, 40, mm), util=bern(0.8, 40, 2),
                    extra_cfg={"fp_params": {"m": mm, "seed": 0, "method": me, "n_features": 20}, "dataset_label": f"{me}-m{mm}"})
    for lab in ("base", "dsg-dynamic", "dsg-static"):
        mcq_run("FP-static", DSG if lab != "base" else {"name": "base"}, forget=bern(0.3, 40, 5), util=bern(0.7, 40, 6),
                extra_cfg={"dataset_label": lab})
    for lab in ("base", "dsg-bio+cyber"):
        mcq_run("FP-multitopic", DSG if lab != "base" else {"name": "base"}, forget=bern(0.3, 40, 5), util=bern(0.7, 40, 6),
                extra_cfg={"dataset_label": lab})
    task_run("FP-latency", "latency", {"latency": {str(L): {"base": {"ms_median": 10.0, "overhead_vs_base": 0.0},
                                                            "dsg": {"ms_median": 11.0, "overhead_vs_base": 0.1}} for L in (64, 128)}})
    ex = lambda part: {"part": part, "tokens": ["<bos>", "▁The", "▁author", "▁wrote"], "sel_max_act": [0, 0.5, 2.0, 0.0],  # noqa: E731
                       "rho": 0.4, "window": 0.6, "gate_rho_fires": True, "gate_window_fires": False}
    task_run("FP-highlight", "tofu", {"examples": [ex("forget"), ex("retain")]})
    task_run("A6-full", "relearn-dsg-hook-k10", {"condition": "dsg-hook", "k": 10, "rank": "full", "batch_size": 4,
                                                 "before": {"forget_acc": 0.3}, "curve": [{"step": 25, "forget_acc": 0.5}]})
    assert paper_assets.main(["--out", str(tmp_path), "--n-boot", "50"]) == 0
    idx = (tmp_path / "INDEX.md").read_text()
    rows = [l for l in idx.splitlines() if l.startswith("| ") and "DSG figure type" not in l]
    assert len(rows) == len(paper_assets.PARITY)
    waiting = [l for l in rows if "waiting for data" in l]
    assert all("sweep tables" in l for l in waiting), waiting   # only RMU-v2 / D1-v2 sweep tables lack fake data
    assert (tmp_path / "figures" / "cyber_pareto.pdf").exists() and "Pareto" in (tmp_path / "tables" / "cyber_pareto.tex").read_text()
    for t in ("sweep_clamp", "data_efficiency", "static_dynamic", "multitopic", "latency", "sweep_a1_dev"):
        assert r"\toprule" in (tmp_path / "tables" / f"{t}.tex").read_text()
