"""Turn an experiment config into queue jobs.

    python -m dsgx.queue.enqueue configs/experiments/A1.yaml [--smoke] [--worktree DIR]
    python -m dsgx.queue.enqueue --test-job hang|sleep --exp-id A1-smoke   # watchdog exercise
    python -m dsgx.queue.enqueue --requeue JOB_ID [JOB_ID ...]

Runs are grouped by (model, SAE, layer) and chunked into jobs of `jobs.group_size` runs, so one
worker process loads the model once for several configs. Jobs pin the worktree's HEAD commit.
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

from dsgx import paths
from dsgx.queue import common as q
from dsgx.util import atomic_write_json


def write_job(job: dict, overwrite: bool = False) -> bool:
    q.ensure_dirs()
    job.setdefault("created", time.time())
    job.setdefault("python", sys.executable)
    job.setdefault("deps", [])
    p = q.job_path(job["id"])
    if p.exists() and not overwrite:
        return False
    atomic_write_json(p, job)
    if not q.state_path(job["id"]).exists():
        q.save_state(job["id"], {"status": q.WAITING, "attempts": 0, "flags": [], "batch_factor": 1.0})
    return True


def requeue(job_ids):
    for j in job_ids:
        st = q.load_state(j)
        st.update({"status": q.WAITING, "attempts": 0, "batch_factor": 1.0, "exit_code": None,
                   "flags": st.get("flags", []) + ["requeued"], "blocked_by": []})
        q.save_state(j, st)
        print(f"requeued {j}")


def _git(wt, *args):
    return subprocess.run(["git", *args], cwd=wt, capture_output=True, text=True).stdout.strip()


def jobs_from_experiment(cfg_path, smoke=False, worktree=None, deps=None):
    from dsgx.config import expand, load_experiment
    from dsgx.run import count_items, resolve

    wt = Path(worktree or paths.REPO_ROOT).resolve()
    exp = load_experiment(wt / cfg_path)
    runs = expand(exp, smoke=smoke)
    jc = {"group_size": 8, "est_seconds_per_item": 0.35, "est_vram_gb": 7.5, "est_ram_gb": 6,
          "gpu_exclusive": False, "kind_slot": "gpu", "overhead_minutes": 2.0, **(exp.get("jobs") or {})}
    groups = {}
    for i, r in enumerate(runs):
        m = resolve(r)["model"]
        groups.setdefault(f"{m['name']}|{m['sae_release']}|{m['sae_id']}", []).append(i)
    branch, commit = _git(wt, "rev-parse", "--abbrev-ref", "HEAD"), _git(wt, "rev-parse", "HEAD")
    exp_id = exp["exp_id"] + ("-smoke" if smoke else "")
    jobs = []
    k = 0
    for gk, idxs in groups.items():
        for s in range(0, len(idxs), jc["group_size"]):
            chunk = idxs[s:s + jc["group_size"]]
            items = sum(count_items(runs[i]) for i in chunk)
            jobs.append({
                "id": f"{exp_id}-{k:03d}", "exp_id": exp_id, "kind": "runs", "branch": branch,
                "worktree": str(wt), "commit": commit, "config_path": str(cfg_path), "smoke": smoke,
                "run_indices": chunk, "group_key": gk, "priority": exp.get("priority", "must"),
                "wave": exp.get("wave", 1), "deps": list(deps or []),
                "est_vram_gb": jc["est_vram_gb"], "est_ram_gb": jc["est_ram_gb"],
                "est_minutes": round(jc["overhead_minutes"] + items * jc["est_seconds_per_item"] / 60, 2),
                "gpu_exclusive": jc["gpu_exclusive"], "kind_slot": jc["kind_slot"], "items_total": items})
            k += 1
    return jobs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config", nargs="?")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--worktree")
    ap.add_argument("--deps", nargs="*", default=[])
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--test-job", choices=["hang", "sleep"])
    ap.add_argument("--exp-id", default="queue-test")
    ap.add_argument("--seconds", type=int, default=30)
    ap.add_argument("--requeue", nargs="*")
    a = ap.parse_args(argv)
    if a.requeue:
        requeue(a.requeue)
        return 0
    if a.test_job:
        jid = f"{a.exp_id}-{a.test_job}test-{time.strftime('%H%M%S')}"
        write_job({"id": jid, "exp_id": a.exp_id, "kind": f"{a.test_job}-test", "branch": None,
                   "worktree": str(Path(a.worktree or paths.REPO_ROOT).resolve()), "commit": None,
                   "config_path": None, "priority": "must", "wave": 1, "deps": a.deps,
                   "est_vram_gb": 0, "est_ram_gb": 0.2, "est_minutes": max(1, a.seconds / 60),
                   "gpu_exclusive": False, "kind_slot": "cpu", "items_total": a.seconds,
                   "seconds": a.seconds})
        print(jid)
        return 0
    jobs = jobs_from_experiment(a.config, a.smoke, a.worktree, a.deps)
    for j in jobs:
        if a.dry_run:
            print(j["id"], j["items_total"], j["est_minutes"])
        elif write_job(j, overwrite=a.overwrite):
            print(f"queued {j['id']} items={j['items_total']} est={j['est_minutes']} min")
        else:
            print(f"exists {j['id']} (use --overwrite or --requeue)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
