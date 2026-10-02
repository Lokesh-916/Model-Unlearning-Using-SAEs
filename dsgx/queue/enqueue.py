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


def _norm_deps(deps, exp_id, smoke):
    """Normalise dep tokens. 'exp:X' stays a token (expanded later against the full batch, adding
    -smoke for smoke runs); a bare 'taskid' becomes '<exp_id>-taskid'; a full job id is left as-is."""
    out = []
    for d in deps or []:
        if d.startswith("exp:"):
            x = d[4:]
            out.append("exp:" + (x + "-smoke" if smoke and not x.endswith("-smoke") else x))
        elif d.startswith(exp_id + "-"):
            out.append(d)
        else:
            # a sibling task/grid id in the same config (may itself contain hyphens)
            out.append(f"{exp_id}-{d}")
    return out


def expand_batch_deps(jobs):
    """Expand 'exp:<E>' and '<E>-*' dep tokens against this batch plus the jobs already on disk.
    Must be called once on the complete set of jobs about to be written."""
    by_exp = {}
    for j in jobs:
        by_exp.setdefault(j["exp_id"], []).append(j["id"])
    for jid, job in q.load_jobs().items():
        by_exp.setdefault(job["exp_id"], []).append(jid)
    for j in jobs:
        deps = []
        for d in j.get("deps", []):
            if d.startswith("exp:"):
                deps += by_exp.get(d[4:], [])
            elif d.endswith("-*"):
                deps += by_exp.get(d[:-2], [])
            else:
                deps.append(d)
        j["deps"] = sorted(set(deps) - {j["id"]})
    return jobs


def jobs_from_experiment(cfg_path, smoke=False, worktree=None, deps=None):
    from dsgx.config import deep_merge, expand, load_experiment
    from dsgx.run import count_items, resolve

    wt = Path(worktree or paths.REPO_ROOT).resolve()
    exp = load_experiment(wt / cfg_path)
    branch, commit = _git(wt, "rev-parse", "--abbrev-ref", "HEAD"), _git(wt, "rev-parse", "HEAD")
    exp_id = exp["exp_id"] + ("-smoke" if smoke else "")
    jc = {"group_size": 8, "est_seconds_per_item": 0.35, "est_vram_gb": 7.5, "est_ram_gb": 20,
          "gpu_exclusive": False, "kind_slot": "gpu", "overhead_minutes": 2.0, "deps": [],
          **(exp.get("jobs") or {})}
    common = {"exp_id": exp_id, "branch": branch, "worktree": str(wt), "commit": commit,
              "config_path": str(cfg_path), "smoke": smoke, "priority": exp.get("priority", "must"),
              "wave": exp.get("wave", 1)}
    jobs = []
    # 1. tasks
    for t in exp.get("tasks") or []:
        if (smoke and t.get("skip_smoke")) or (not smoke and t.get("smoke_only")):
            continue
        args = deep_merge(t.get("args") or {}, (t.get("smoke_args") or {}) if smoke else {})
        est = t.get("smoke_est_minutes", 5) if smoke else t.get("est_minutes", 60)
        jobs.append({**common, "id": f"{exp_id}-{t['id']}", "kind": "task", "task_id": t["id"],
                     "entry": t["entry"], "args": args,
                     "priority": t.get("priority", common["priority"]), "wave": t.get("wave", common["wave"]),
                     "deps": _norm_deps(t.get("deps"), exp_id, smoke) + list(deps or []),
                     "est_vram_gb": t.get("est_vram_gb", 7.5 if t.get("kind_slot", "gpu") == "gpu" else 0),
                     "est_ram_gb": t.get("est_ram_gb", 20 if t.get("kind_slot", "gpu") == "gpu" else 4),
                     "est_minutes": est, "gpu_exclusive": t.get("gpu_exclusive", False),
                     "kind_slot": t.get("kind_slot", "gpu"), "items_total": t.get("items_total", 0),
                     "group_key": t.get("group_key", t["id"])})
    # 2. MCQ run grid
    runs = expand(exp, smoke=smoke) if (exp.get("base") or exp.get("grid") or exp.get("extra")) else []
    groups = {}
    for i, r in enumerate(runs):
        m = resolve(r)["model"]
        groups.setdefault(f"{m['name']}|{m['sae_release']}|{m['sae_id']}", []).append(i)
    grid_deps = _norm_deps(jc["deps"], exp_id, smoke) + list(deps or [])
    k = 0
    for gk, idxs in groups.items():
        for s in range(0, len(idxs), jc["group_size"]):
            chunk = idxs[s:s + jc["group_size"]]
            items = sum(count_items(runs[i]) for i in chunk)
            jobs.append({**common, "id": f"{exp_id}-{k:03d}", "kind": "runs", "run_indices": chunk,
                         "group_key": gk, "deps": grid_deps,
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
    ap.add_argument("--after-exp", nargs="*", default=[], help="depend on every job of these exp ids")
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
    deps = list(a.deps) + [j for j, job in q.load_jobs().items() if job["exp_id"] in set(a.after_exp)]
    jobs = expand_batch_deps(jobs_from_experiment(a.config, a.smoke, a.worktree, deps))
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
