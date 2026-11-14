"""Combination wave: composition, the pre-registered rule on fake DEV results, TEST generation,
and enqueue into a throw-away git clone + queue (no GPU, no model)."""
import json
import subprocess
from pathlib import Path

import numpy as np

from dsgx import combine
from dsgx.config import expand
from dsgx.run import resolve
from tests.fakeruns import bern, mcq_run

N_F, N_U = 300, 200


def _fired(benign_rate, seed):
    rng = np.random.default_rng(seed)
    b = rng.permutation(N_U) < int(benign_rate * N_U)
    return [True] * N_F + b.tolist()


def _cands():
    return [
        {"slot": "detector", "name": "window-w16", "patch": {"gate": {"type": "window", "w": 16}}},
        {"slot": "detector", "name": "cusum", "patch": {"gate": {"type": "cusum"}}},
        {"slot": "threshold", "name": "N6-conformal", "patch": {"calib": {"rule": "conformal"}}},
        {"slot": "intervention", "name": "C5-mean_ablate", "patch": {"intervention": {"type": "mean_ablate"}}},
        {"slot": "features", "name": "C1-attr", "patch": {"features_file": "C1_attr_bio.json"}},
        {"slot": "baked", "name": "D2-nullspace", "patch": {"weights": "ckpt:D2/nullspace"}},
    ]


def test_compose_composite_and_patches():
    c = {x["name"]: x for x in _cands()}
    m, model = combine.compose(combine.DEFAULT_METHOD, [c["window-w16"]["patch"], c["N6-conformal"]["patch"],
                                                      c["C5-mean_ablate"]["patch"], c["C1-attr"]["patch"],
                                                      c["D2-nullspace"]["patch"]])
    assert m["gate"]["type"] == "window" and m["gate"]["w"] == 16 and m["gate"]["features_file"] == "C1_attr_bio.json"
    assert m["calib"]["rule"] == "conformal" and m["intervention"]["type"] == "mean_ablate"
    assert model == {"weights": "ckpt:D2/nullspace"}
    m2, _ = combine.compose(combine.DEFAULT_METHOD, [{"composite": {"combine": "any", "layers": ["layer_8/width_16k/canonical"]}},
                                                     c["C1-attr"]["patch"]])
    assert m2["name"] == "composite" and len(m2["gates"]) == 2
    assert "features_file" in m2["gates"][0]["gate"] and "features_file" not in m2["gates"][1]["gate"]
    assert m2["gates"][1]["gate"]["sae_id"] == "layer_8/width_16k/canonical"
    assert combine.DEFAULT_METHOD["gate"] == {"type": "rho", "n_features": 20, "retain_pct": 95}  # not mutated


def test_screening_configs_resolve_on_dev():
    exp = combine.screening_experiment(_cands())
    runs = expand(exp)
    assert len(runs) == 2 * (1 + len(_cands())) + 1
    for r in runs:
        c = resolve(r)
        assert c["split"] == "dev" and c["purpose"] == "select"


def _fake_screen():
    def run(cand, att, method, forget, util, fired, model=None):
        mcq_run(combine.EXP_SCREEN, method, split="dev", attack=combine.SCREEN_ATTACKS[att], forget=forget,
                util=util, fired=fired, model=model, extra_cfg={"combine": {"slot": "x", "cand": cand, "attack": att},
                                                                "purpose": "select"})
    D = combine.DEFAULT_METHOD
    c = {x["name"]: x for x in _cands()}
    u = bern(0.80, N_U, 7)
    run("default", "clean", D, bern(0.30, N_F, 1), u, _fired(0.08, 1))
    run("default", "dilution", D, bern(0.55, N_F, 2), u, _fired(0.08, 1))
    # window: much better under dilution, same utility, FPR 4%  -> eligible
    m, _ = combine.compose(D, [c["window-w16"]["patch"]])
    run("window-w16", "clean", m, bern(0.30, N_F, 1), u, _fired(0.04, 2))
    run("window-w16", "dilution", m, bern(0.30, N_F, 3), u, _fired(0.04, 2))
    # cusum: even better under dilution but costs 5 utility points -> not eligible
    m, _ = combine.compose(D, [c["cusum"]["patch"]])
    u_bad = [x if i % 10 else 0 for i, x in enumerate(u)]
    u_bad = [0 if (i < 20 and x) else x for i, x in enumerate(u_bad)]
    run("cusum", "clean", m, bern(0.30, N_F, 1), u_bad, _fired(0.04, 3))
    run("cusum", "dilution", m, bern(0.20, N_F, 4), u_bad, _fired(0.04, 3))
    # conformal: benign fire rate 8% -> 1.5% (subset of default's fired items) -> eligible
    m, _ = combine.compose(D, [c["N6-conformal"]["patch"]])
    fd = _fired(0.08, 1)
    fc = fd[:N_F] + [f and (i % 5 == 0) for i, f in enumerate(fd[N_F:])]
    run("N6-conformal", "clean", m, bern(0.30, N_F, 1), u, fc)
    run("N6-conformal", "dilution", m, bern(0.55, N_F, 2), u, fc)
    # mean ablation: no change -> not eligible; C1-attr and D2: screening runs missing
    m, _ = combine.compose(D, [c["C5-mean_ablate"]["patch"]])
    run("C5-mean_ablate", "clean", m, bern(0.30, N_F, 1), u, _fired(0.04, 5))
    run("C5-mean_ablate", "dilution", m, bern(0.55, N_F, 2), u, _fired(0.04, 5))


def test_rule_on_fake_results():
    _fake_screen()
    sel = combine.select(_cands(), n_boot=1000)
    rows = {r["name"]: r for r in sel["candidates"]}
    assert rows["window-w16"]["eligible"], rows["window-w16"]
    assert not rows["cusum"]["eligible"] and any("utility cost" in x for x in rows["cusum"]["reasons"])
    assert rows["N6-conformal"]["eligible"], rows["N6-conformal"]
    assert not rows["C5-mean_ablate"]["eligible"]
    assert rows["C1-attr"]["reasons"] == ["screening run missing"]
    assert sel["slots"] == {"features": None, "detector": "window-w16", "threshold": "N6-conformal",
                            "intervention": None, "baked": None}
    md = combine.selection_markdown(sel)
    assert "| detector | C2/C3/N7 | window-w16 |" in md

    exp = combine.test_experiment(sel, _cands())
    runs = expand(exp)
    tags = {}
    for r in runs:
        c = resolve(r)
        assert c["split"] == "test" and c["batch_size"] == 1
        tags.setdefault(r["combine"]["cond"], set()).add(r["seed"])
    assert set(tags) == {"base", "dsg", "default", "combined", "loo-detector", "loo-threshold"}
    assert tags["combined"] == {0, 1, 2, 3, 4} and tags["loo-detector"] == {0, 1, 2} and tags["base"] == {0}
    comb = [r for r in runs if r["combine"]["cond"] == "combined"][0]["method"]
    assert comb["gate"]["type"] == "window" and comb["calib"]["rule"] == "conformal"
    loo = [r for r in runs if r["combine"]["cond"] == "loo-detector"][0]["method"]
    assert loo["gate"]["type"] == "rho" and loo["calib"]["rule"] == "conformal"
    att = {r["attack"]["name"] for r in runs}
    assert {"dilution", "decompose", "translate", "encode", "rewrite_cache", "suffix", "none"} <= att


def test_cli_dry_run_and_select_only(tmp_path, capsys):
    _fake_screen()
    assert combine.main(["--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "DEV screening runs" in out and "missing: C1 feature file" in out
    assert combine.main(["--select-only", "--out", str(tmp_path / "sel"), "--n-boot", "500"]) == 0
    assert (tmp_path / "sel" / "COMBINE_SELECTION.md").exists()


def test_enqueue_into_temp_worktree(tmp_path, monkeypatch):
    """--enqueue creates the branch + worktree, commits the screening config and queues jobs; the
    selection task then commits X1.yaml and queues the TEST wave pinned to the new HEAD."""
    from dsgx import paths
    from dsgx.queue import common as q

    repo = tmp_path / "repo"
    src = Path(paths.REPO_ROOT)
    subprocess.run(["git", "clone", "-q", "--local", "--shared", "--no-checkout", str(src), str(repo)], check=True)
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=src, capture_output=True, text=True).stdout.strip()
    subprocess.run(["git", "checkout", "-q", "-b", "base", head], cwd=repo, check=True)
    for k, v in (("user.email", "t@t"), ("user.name", "t")):
        subprocess.run(["git", "config", k, v], cwd=repo, check=True)
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    wt = tmp_path / "wt"
    monkeypatch.setattr(combine, "candidates", lambda case, root=None, cache=None: (_cands()[:3], []))
    assert combine.main(["--enqueue", "--worktree", str(wt), "--base-branch", "base"]) == 0
    jobs = q.load_jobs()
    assert any(j.endswith("-select") for j in jobs) and len(jobs) >= 2
    sel_job = [j for j in jobs.values() if j["id"].endswith("-select")][0]
    assert set(sel_job["deps"]) == {j for j in jobs if j != sel_job["id"]}
    br = subprocess.run(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
    assert br == combine.BRANCH and (wt / "configs/experiments/X1-screen.yaml").exists()
    # run the selection task as the queue would (cwd = worktree), on fake screening results
    _fake_screen()
    monkeypatch.chdir(wt)

    class Ctx:
        args, smoke, exp_id, task_id = {"n_boot": 300}, False, combine.EXP_SCREEN, "select"

        def run_dir(self):
            d = tmp_path / "seldir"
            d.mkdir(exist_ok=True)
            return d

        def write_metrics(self, m):
            (self.run_dir() / "metrics.json").write_text(json.dumps(m))

        def finish(self, h):
            pass

    slots = combine.select_task(Ctx())
    assert slots["detector"] == "window-w16"
    new_head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
    test_jobs = [j for j in q.load_jobs().values() if j["exp_id"] == combine.EXP_TEST]
    assert test_jobs and all(j["commit"] == new_head for j in test_jobs)
    assert (tmp_path / "seldir" / "COMBINE_SELECTION.md").exists()
