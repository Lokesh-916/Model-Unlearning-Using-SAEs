"""POST-HOC, EXPLORATORY (session 28, 2026-10-07): two follow-ups to StreamGuard (the X1 combined gate, CUSUM), decided
after the X1 / X1-suite TEST results were seen. Neither is a claim-rule input (base.purpose = posthoc-exploratory;
claims.evaluate drops those runs and reads no PH-* experiment); DEVIATIONS row 2026-10-07. gpuws only, DSG comparators
on gpuws in the same experiment.

  PH-union    union gate: fire if DSG's rho gate (DSG's own tau) OR the CUSUM gate fires. gates.py type "union":
              score = 1e9 if rho > tau_DSG else cusum_max, so ONE split-conformal threshold (alpha 0.05, benign MMLU
              DEV) bounds the union's benign FPR. Same intervention (clamp_all x500) and features (DSG's 20).
              MCQ: X1.yaml's 60 dsg-faithful runs + the 60 combined (CUSUM) runs re-pointed to the union gate (clean
              utility + benign FPR, B1 pad400 / pad1600, B2, B3 x5, B4, B5, A3 hard negatives; 5 seeds; same items).
              Tasks (as X1-suite): A3 benign-biology open-ended streaming, B6 leak/gibberish, TOFU v3 metrics
              (forget quality, model utility, truth ratio), streaming TOFU forget/retain QA, paired (union - DSG).
  PH-tofucal  domain-calibrated StreamGuard on TOFU v3: the X1 CUSUM gate with its threshold at 5 % FPR on TOFU
              retain DEV sequences (retain90 QA pairs not in either TOFU test set, chat format; gates.calibrate source
              "tofu-retain-dev") instead of MMLU DEV, as DSG calibrates its tau on retain text. TOFU metrics +
              streaming TOFU QA on the TOFU test sets, paired vs DSG (and vs the MMLU-calibrated gpuws X1-suite runs in
              the digest, same machine and items).

    python cluster/ph_union.py build   # branch exp/PH-union (= exp/X1-suite b933e20 + union gate + configs)
    python cluster/ph_union.py stage   # snapshot + specs + ORDER to gpuws only (0 lab-queue writes)
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
BASE_COMMIT = "b933e20990d39415c650d75d3a14f8b28c9e3f82"  # exp/X1-suite (contains X1's 93625f5)
BRANCH = "exp/PH-union"
EXPS = ("PH-tofucal", "PH-union")
GROUP = "ph-union"
PURPOSE = "posthoc-exploratory"
SERVER = "/home/suraj/dsg_cluster"

UNION_GATE = {"type": "union", "n_features": 20, "retain_pct": 95}
CONFORMAL = {"fpr": 0.05, "n_max": 1000, "source": "mmlu-dev", "rule": "conformal"}
TOFU_CAL = {"fpr": 0.05, "n_max": 1000, "source": "tofu-retain-dev"}
CLAMP = {"type": "clamp_all", "multiplier": 500}
TOFU_CORP = {"forget_corpus": "tofu-forget10", "retain_corpus": "tofu-retain90"}
DSG_STREAM = {"name": "dsg-faithful", "mode": "stream", "n_features": 20, "retain_pct": 95, "multiplier": 500}
TOFU_BASE_CONDS = [{"tag": "retain-model", "weights": "ckpt:A2/tofu_retain"},
                   {"tag": "full", "weights": "ckpt:A2/tofu_full"},
                   {"tag": "full+dsg", "weights": "ckpt:A2/tofu_full", "dsg": True}]


def git(*a, cwd=REPO, **kw):
    return subprocess.run(["git", *a], cwd=cwd, check=True, text=True, capture_output=True, **kw).stdout.strip()


def _open_task(tid, items, methods, est, **extra):
    return {"id": tid, "entry": "dsgx.eval.openqa:task", "args": {"items": items, **extra, "methods": methods},
            "smoke_args": {"limit": 4, "max_new": 16}, "kind_slot": "gpu", "est_minutes": est, "est_vram_gb": 9}


def _tofu_tasks(gate_cond, gate_method):
    tofu_qa = {"model": {"weights": "ckpt:A2/tofu_full"}, "max_new": 64}
    dsg_tofu = {**DSG_STREAM, **TOFU_CORP}
    dsg_tofu.pop("multiplier")
    return [
        {"id": "tofu-metrics", "entry": "experiments.X1suite.tofu_fix:metrics",
         "args": {"n_features": 20, "retain_pct": 95, "conditions": TOFU_BASE_CONDS + [gate_cond]},
         "smoke_args": {"limit": 4}, "kind_slot": "gpu", "est_minutes": 80, "est_vram_gb": 9, "est_ram_gb": 30},
        {**_open_task("tofu-qa-forget", "tofu-forget10", [dsg_tofu, gate_method], 30, **tofu_qa), "deps": ["tofu-metrics"]},
        {**_open_task("tofu-qa-retain", "tofu-retain_perturbed", [dsg_tofu, gate_method], 30, **tofu_qa),
         "deps": ["tofu-metrics"]},
    ]


def union_config() -> dict:
    y = yaml.safe_load(git("show", f"{BASE_COMMIT}:configs/experiments/X1.yaml"))
    runs = []
    for r in y["runs"]:
        m = r["method"]
        if m["name"] == "dsg-faithful":
            runs.append(copy.deepcopy(r))
        elif m["name"] == "gated" and (m.get("gate") or {}).get("type") == "cusum":
            r = copy.deepcopy(r)
            r["method"]["gate"] = dict(UNION_GATE)
            r["method"]["calib"] = dict(CONFORMAL)
            r["dataset_label"] = r["dataset_label"].replace("combined", "union", 1)
            r["combine"]["cond"] = "union"
            runs.append(r)
    assert sum(r["method"]["name"] == "gated" for r in runs) == 60 and len(runs) == 120, len(runs)
    union_open = {"name": "gated", "tag": "union-stream", "mode": "stream", "gate": dict(UNION_GATE),
                  "calib": dict(CONFORMAL), "intervention": dict(CLAMP)}
    union_tofu = {**union_open, "gate": {**UNION_GATE, **TOFU_CORP}}
    tasks = [
        _open_task("benign-open", "mmlu-open:college_biology,high_school_biology,anatomy", [DSG_STREAM, union_open], 40,
                   split="test", case="bio", max_new=48),
        _open_task("leak", "wmdp-bio-open", [DSG_STREAM, union_open], 90, split="test", case="bio", max_new=96,
                   prefix_scores=True),
        *_tofu_tasks({"tag": "full+union", "weights": "ckpt:A2/tofu_full", "gate": {**UNION_GATE},
                      "calib": dict(CONFORMAL), "intervention": dict(CLAMP)}, union_tofu),
        {"id": "paired", "entry": "experiments.X1suite.paired:task",
         "args": {"gate_tag": "union-stream", "dsg_tag": "dsg-faithful-stream", "x1_exp": "PH-union", "gate_label": "union"},
         "kind_slot": "cpu", "est_minutes": 5, "est_ram_gb": 4,
         "deps": ["benign-open", "leak", "tofu-qa-forget", "tofu-qa-retain", "exp:PH-union"]},
    ]
    return {"exp_id": "PH-union", "priority": "stretch", "wave": 7, "purpose": PURPOSE, "base": dict(y["base"], purpose=PURPOSE),
            "runs": runs, "jobs": y["jobs"], "tasks": tasks}


def tofucal_config() -> dict:
    gate = {"type": "cusum", "n_features": 20, "retain_pct": 95}
    method = {"name": "gated", "tag": "cusum-tofucal-stream", "mode": "stream", "gate": {**gate, **TOFU_CORP},
              "calib": dict(TOFU_CAL), "intervention": dict(CLAMP)}
    tasks = [*_tofu_tasks({"tag": "full+gate-cusum-tofucal", "weights": "ckpt:A2/tofu_full", "gate": dict(gate),
                           "calib": dict(TOFU_CAL), "intervention": dict(CLAMP)}, method),
             {"id": "paired", "entry": "experiments.X1suite.paired:task",
              "args": {"gate_tag": "cusum-tofucal-stream", "dsg_tag": "dsg-faithful-stream", "hardneg": False},
              "kind_slot": "cpu", "est_minutes": 5, "est_ram_gb": 4, "deps": ["tofu-qa-forget", "tofu-qa-retain"]}]
    # tasks only: a "base" key would make the planner expand an MCQ run grid; the purpose is recorded top-level
    return {"exp_id": "PH-tofucal", "priority": "stretch", "wave": 7, "purpose": PURPOSE, "tasks": tasks}


HEADER = ("# POST-HOC, EXPLORATORY (decided 2026-10-07 after the X1 / X1-suite TEST results; not a claim-rule input;\n"
          "# DEVIATIONS 2026-10-07). Generated by prep cluster/ph_union.py from exp/X1-suite b933e20; do not edit.\n# {}\n")


def build() -> int:
    if not git("branch", "--list", BRANCH) or git("rev-parse", BRANCH) == BASE_COMMIT:
        sys.exit(f"{BRANCH}: create it from {BASE_COMMIT[:7]} and commit the union-gate code first")
    head = git("rev-parse", BRANCH)
    with tempfile.TemporaryDirectory(dir=REPO.parent / "dsg_results_cluster") as td:
        wt = Path(td) / "wt"
        git("worktree", "add", "--detach", str(wt), head)
        try:
            for exp, cfg, what in (("PH-union", union_config(), "union gate (DSG rho OR CUSUM, conformal alpha 0.05)"),
                                   ("PH-tofucal", tofucal_config(), "CUSUM threshold calibrated on TOFU retain DEV")):
                (wt / f"configs/experiments/{exp}.yaml").write_text(HEADER.format(what) + yaml.safe_dump(cfg, sort_keys=False))
            git("add", "configs/experiments", cwd=wt)
            if git("status", "--porcelain", cwd=wt):
                git("commit", "-q", "-m", "PH-union / PH-tofucal: POST-HOC, EXPLORATORY configs (union gate; TOFU-retain-DEV "
                    "calibrated CUSUM) with DSG comparators; not claim-rule inputs", cwd=wt)
                git("update-ref", f"refs/heads/{BRANCH}", git("rev-parse", "HEAD", cwd=wt), head)
        finally:
            git("worktree", "remove", "--force", str(wt))
    print(f"{BRANCH} = {git('rev-parse', BRANCH)}")
    return 0


def plan(code: Path, commit: str) -> list[dict]:
    jobs = []
    for exp in EXPS:
        out = subprocess.run([sys.executable, "-c", (
            "import json; from dsgx.queue.enqueue import jobs_from_experiment as j; "
            f"print(json.dumps(j('configs/experiments/{exp}.yaml', worktree='.')))")], cwd=code,
            env={**os.environ, "PYTHONPATH": str(code), "CUDA_VISIBLE_DEVICES": ""}, check=True, text=True,
            capture_output=True).stdout
        jobs += json.loads(out.strip().splitlines()[-1])
    by_exp = {}
    for s in jobs:
        by_exp.setdefault(s["exp_id"], []).append(s["id"])
    for s in jobs:  # expand exp:<E> within this batch only (never against the lab queue)
        deps = []
        for d in s.get("deps", []):
            deps += by_exp.get(d[4:], []) if d.startswith("exp:") else [d]
        s["deps"] = sorted(set(deps) - {s["id"]})
        assert all(d.split("-")[0] == "PH" for d in s["deps"]), s["deps"]
    # order: (b) TOFU, then (a) TOFU / open tasks, then the MCQ grid, paired last (deps first within each)
    stage_of = {"tofu-metrics": 0, "tofu-qa-forget": 1, "tofu-qa-retain": 1, "benign-open": 2, "leak": 3, "paired": 9}
    rank = lambda s: (EXPS.index(s["exp_id"]), stage_of.get(s["id"][len(s["exp_id"]) + 1:], 5), s["id"])  # noqa: E731
    order, seen, ids = [], set(), {s["id"]: s for s in jobs}

    def visit(j):
        if j in seen:
            return
        seen.add(j)
        for d in ids[j]["deps"]:
            visit(d)
        order.append(j)
    for s in sorted(jobs, key=rank):
        visit(s["id"])
    return [ids[j] for j in order]


def stage() -> int:
    commit = git("rev-parse", BRANCH)
    snap = f"PHU-{commit[:7]}"
    with tempfile.TemporaryDirectory(dir=REPO.parent / "dsg_results_cluster") as td:
        td = Path(td)
        code = td / "code"
        code.mkdir()
        subprocess.run(f"git -C {REPO} archive {commit} | tar -x -C {code} --exclude figures", shell=True, check=True)
        (code / "CODE_COMMIT.json").write_text(json.dumps(
            {"commit": commit, "branch": BRANCH, "dirty": False, "snapshot": f"code-{snap}", "posthoc": True}) + "\n")
        jobs = plan(code, commit)
        (td / "specs").mkdir()
        for s in jobs:
            s = dict(s, worktree=f"{SERVER}/code-{snap}", commit=None, pinned_commit=commit,
                     config_path=f"configs/experiments/{s['exp_id']}.yaml", branch=BRANCH, priority="stretch",
                     posthoc={"exploratory": True, "claim_input": False, "decided": "2026-10-07 after X1 / X1-suite TEST"},
                     moved_to_server={"group": GROUP, "lab_worktree": None, "lab_commit": None, "replica": False})
            (td / "specs" / f"{s['id']}.json").write_text(json.dumps(s, indent=1))
        (td / "ORDER").write_text("\n".join(s["id"] for s in jobs) + "\n")
        (td / "SNAPSHOT").write_text(snap + "\n")
        print(f"{len(jobs)} jobs ({sum(len(s.get('run_indices') or []) for s in jobs)} grid runs), commit {commit[:10]}, "
              f"snapshot code-{snap}")
        for s in jobs:
            print(f"  {s['id']:<28} est {s.get('est_minutes', 0):6.1f} lab-min  deps {len(s['deps'])}")
        if os.environ.get("PH_DRY"):
            return 0
        sh = lambda c: subprocess.run(c, shell=True, check=True)  # noqa: E731
        sh(f"""ssh -n -o BatchMode=yes gpuws "printf -- '- %s | %s | %s\\n' \\"\\$(date '+%F %T')\\" 'rsync code-{snap}/ + {len(jobs)} job specs + labjobs/{GROUP}/' 'ph_union.py stage (POST-HOC exploratory, session 28)' >> ~/dsg_cluster/COMMAND_LOG.md" """)
        sh(f"ssh -n gpuws 'mkdir -p ~/dsg_cluster/results/queue/jobs ~/dsg_cluster/labjobs/{GROUP}'")
        sh(f"rsync --bwlimit=50000 --partial -a --delete {code}/ gpuws:dsg_cluster/code-{snap}/ </dev/null")
        sh(f"rsync --bwlimit=50000 --partial -a {td}/specs/ gpuws:dsg_cluster/results/queue/jobs/ </dev/null")
        sh(f"rsync --bwlimit=50000 --partial -a {td}/ORDER {td}/SNAPSHOT gpuws:dsg_cluster/labjobs/{GROUP}/ </dev/null")
    return 0


if __name__ == "__main__":
    sys.exit({"build": build, "stage": stage}[sys.argv[1]]())
