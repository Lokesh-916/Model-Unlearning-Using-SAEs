"""Build the job list of every experiment worktree (full or smoke), resolve cross-experiment
dependencies, project hours per wave, and optionally enqueue.

    python -m dsgx.queue.joblist [--smoke] [--only A1-dev B1 ...] [--enqueue] [--out FILE]

Without --enqueue nothing is queued: the manifest goes to $DSG_RESULTS/queue/joblist_<full|smoke>.json.
Projected hours: GPU jobs run one at a time (user decision 3), CPU jobs over `cpu_slots` workers.
"""
import argparse
import subprocess
from collections import defaultdict
from pathlib import Path

from dsgx import paths
from dsgx.queue import common as q
from dsgx.util import atomic_write_json


def experiment_configs():
    """(worktree, config_path) for every exp/* worktree."""
    out = []
    lines = subprocess.run(["git", "worktree", "list", "--porcelain"], cwd=paths.REPO_ROOT,
                           capture_output=True, text=True).stdout.splitlines()
    wts, br = [], None
    for l in lines:
        if l.startswith("worktree "):
            cur = l.split(" ", 1)[1]
        if l.startswith("branch ") and l.split("refs/heads/")[-1].startswith("exp/"):
            wts.append(cur)
    for wt in sorted(wts):
        for cp in sorted((Path(wt) / "configs" / "experiments").glob("*.yaml")):
            out.append((wt, str(cp.relative_to(wt))))
    return out


def build(smoke: bool, only=None) -> list[dict]:
    from dsgx.config import load_experiment
    from dsgx.queue.enqueue import jobs_from_experiment

    jobs = []
    for wt, cp in experiment_configs():
        exp = load_experiment(Path(wt) / cp)
        if only and exp["exp_id"] not in only and Path(cp).stem not in only:
            continue
        jobs += jobs_from_experiment(cp, smoke=smoke, worktree=wt)
    from dsgx.queue.enqueue import expand_batch_deps

    expand_batch_deps(jobs)
    ids = {j["id"] for j in jobs} | set(q.load_jobs())
    for j in jobs:
        j["missing_deps"] = [d for d in j["deps"] if d not in ids]
    return jobs


def projection(jobs, cpu_slots=4) -> dict:
    w = defaultdict(lambda: {"gpu_min": 0.0, "cpu_min": 0.0, "n": 0})
    for j in jobs:
        k = w[j.get("wave", 1)]
        k["n"] += 1
        k["gpu_min" if j.get("kind_slot", "gpu") == "gpu" else "cpu_min"] += j.get("est_minutes", 0)
    out = {}
    for wave, k in sorted(w.items()):
        hours = max(k["gpu_min"], k["cpu_min"] / cpu_slots) / 60
        out[wave] = {"jobs": k["n"], "gpu_hours": round(k["gpu_min"] / 60, 1),
                     "cpu_hours": round(k["cpu_min"] / 60, 1), "wall_hours_est": round(hours, 1)}
    out["total_wall_hours_est"] = round(sum(v["wall_hours_est"] for v in out.values()), 1)
    return out


def main(argv=None) -> int:
    from dsgx.queue.enqueue import write_job

    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--enqueue", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    jobs = build(a.smoke, a.only)
    proj = projection(jobs, q.control()["cpu_slots"])
    out = q.qdir() / f"joblist_{'smoke' if a.smoke else 'full'}.json"
    atomic_write_json(out, {"jobs": jobs, "projection": proj})
    for wave, v in proj.items():
        print(wave, v)
    miss = [j["id"] for j in jobs if j["missing_deps"]]
    if miss:
        print("jobs with missing deps:", miss[:20])
    if a.enqueue:
        n = sum(write_job(j, overwrite=a.overwrite) for j in jobs)
        print(f"enqueued {n} / {len(jobs)} jobs")
    print(f"manifest: {out} ({len(jobs)} jobs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
