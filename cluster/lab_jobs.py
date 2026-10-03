"""Run MOVED-TO-SERVER lab-queue jobs on gpuws with the exact experiment-branch code they were pinned to.

    python cluster/lab_jobs.py --group c3 [--budget-min 165] [--plan]

Inputs (staged from the lab PC by cluster/stage_lab_jobs.sh):
  $DSGC/labjobs/<group>/ORDER          job ids in run order (one per line)
  $DSGC/labjobs/<group>/SNAPSHOT       name of the code snapshot dir $DSGC/code-<name> (git archive of the pinned commit)
  $DSG_RESULTS/queue/jobs/<id>.json    the lab job spec; worktree -> the snapshot, commit -> null, original commit
                                       kept as "pinned_commit" and checked here against the snapshot's CODE_COMMIT.json
Each job runs as `python -m dsgx.queue.worker --job <id>` inside the snapshot (its own dsgx + experiments/), so the
results are exactly what the lab scheduler would have produced, in the same run-directory format. Then:
  * a job with <job dir>/DONE is skipped (a requeue or the next chained sbatch continues where this one stopped);
  * a job whose deps (inside the group) are not DONE is skipped;
  * exit 75 (CUDA OOM) is retried once with DSGX_BATCH_FACTOR=0.5, as the lab scheduler does;
  * no new job starts when its server estimate (SPEED x lab estimate + 3 min) exceeds the time left (--budget-min):
    the script exits 0 with "INCOMPLETE" and the next chained sbatch of the same group continues;
  * runs/<exp>/HARDWARE.json {"label": "gpuws"} is written: exp-branch code predates the hardware label, and the
    report must never mix these runs with lab-PC runs (collect.Run.hardware reads the marker).
Status: $DSG_RESULTS/jobs/labjobs-<group>/status.json. Exit 1 if a job failed (the others still ran).
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

DSGC = Path(os.environ.get("DSGC", Path.home() / "dsg_cluster"))
SPEED = 0.3  # gpuws time / lab-PC time (sanity 57 s vs 228 s; training ~0.3)


def rjson(p, default=None):
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return default


def wjson(p, obj):
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    tmp.replace(p)


def log(msg):
    print(f"[labjobs] {time.strftime('%H:%M:%S')} {msg}", flush=True)


def results() -> Path:
    return Path(os.environ.get("DSG_RESULTS", DSGC / "results"))


def job_dir(spec) -> Path:
    return results() / "runs" / spec["exp_id"] / "_jobs" / spec["id"]


def est_min(spec) -> float:
    return SPEED * float(spec.get("est_minutes", 10)) + 3


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", required=True)
    ap.add_argument("--budget-min", type=float, default=float(os.environ.get("DSG_BUDGET_MIN", 165)))
    ap.add_argument("--plan", action="store_true", help="print what would run (CPU, no GPU work)")
    a = ap.parse_args(argv)
    t0 = time.time()
    gdir = DSGC / "labjobs" / a.group
    order = [l.strip() for l in (gdir / "ORDER").read_text().splitlines() if l.strip()]
    snap = DSGC / f"code-{(gdir / 'SNAPSHOT').read_text().strip()}"
    commit = (rjson(snap / "CODE_COMMIT.json") or {}).get("commit")
    specs = {j: rjson(results() / "queue" / "jobs" / f"{j}.json") for j in order}
    missing = [j for j, s in specs.items() if not s]
    if missing or not commit:
        log(f"ABORT: missing specs {missing} or snapshot {snap} without CODE_COMMIT.json")
        return 2
    wrong = [j for j, s in specs.items() if s.get("pinned_commit") != commit]
    if wrong:
        log(f"ABORT: snapshot {snap.name} is at {commit[:10]}, jobs pinned elsewhere: {wrong[:5]}")
        return 2
    stp = results() / "jobs" / f"labjobs-{a.group}" / "status.json"
    status = rjson(stp, {}) or {}
    status.update({"group": a.group, "snapshot": snap.name, "commit": commit, "hardware_label": "gpuws",
                   "slurm_job": os.environ.get("SLURM_JOB_ID"), "jobs": status.get("jobs", {})})
    done = lambda j: (job_dir(specs[j]) / "DONE").exists()  # noqa: E731
    todo = [j for j in order if not done(j)]
    log(f"group {a.group}: {len(order)} jobs, {len(order) - len(todo)} done, snapshot {snap.name} @ {commit[:10]}, "
        f"budget {a.budget_min:.0f} min, est. {sum(est_min(specs[j]) for j in todo):.0f} min left")
    if a.plan:
        for j in todo:
            print(f"  {j:<40} est {est_min(specs[j]):5.1f} min  deps {len(specs[j].get('deps', []))}")
        return 0
    env = dict(os.environ, PYTHONPATH=str(snap), DSG_WORKTREES=str(snap), PYTHONUNBUFFERED="1")
    failed, skipped = [], []
    for j in todo:
        s = specs[j]
        left = a.budget_min - (time.time() - t0) / 60
        if est_min(s) > left:
            skipped.append(j)
            continue
        bad_deps = [d for d in s.get("deps", []) if d in specs and not done(d)]
        if bad_deps:
            log(f"{j}: deps not done here {bad_deps[:3]}: skipped")
            skipped.append(j)
            continue
        rc = None
        for attempt, factor in enumerate(("1", "0.5")):
            jd = job_dir(s)
            jd.mkdir(parents=True, exist_ok=True)
            ts = time.time()
            log(f"start {j} (attempt {attempt + 1}, batch factor {factor}, est {est_min(s):.0f} min, {left:.0f} min left)")
            with open(jd / "job.log", "a") as fh:
                fh.write(f"=== {time.strftime('%F %T')} start {j} on gpuws (lab_jobs.py, {snap.name}) attempt {attempt + 1}\n")
                fh.flush()
                rc = subprocess.run([sys.executable, "-u", "-m", "dsgx.queue.worker", "--job", j], cwd=snap,
                                    env={**env, "DSGX_BATCH_FACTOR": factor}, stdout=fh, stderr=subprocess.STDOUT).returncode
            (jd / "exit_code").write_text(f"{rc}\n")
            status["jobs"][j] = {"rc": rc, "minutes": round((time.time() - ts) / 60, 1), "attempt": attempt + 1,
                                 "batch_factor": factor, "end": time.strftime("%FT%T")}
            log(f"end {j}: rc {rc} in {(time.time() - ts) / 60:.1f} min")
            if rc != 75:
                break
        if rc == 0:
            wjson(results() / "runs" / s["exp_id"] / "HARDWARE.json",
                  {"label": "gpuws", "note": "exp-branch code run on gpuws by cluster/lab_jobs.py; never mix with labpc",
                   "group": a.group, "snapshot": snap.name, "commit": commit})
        else:
            failed.append(j)
        wjson(stp, status)
    left = [j for j in order if not done(j)]
    status.update({"left": left, "failed": failed, "complete": not left, "time": time.strftime("%FT%T")})
    wjson(stp, status)
    if left:
        log(f"INCOMPLETE: {len(left)} job(s) left ({len(failed)} failed): {' '.join(left[:8])}"
            + ("; the next chained sbatch of this group continues" if not failed else ""))
    else:
        log(f"group {a.group} COMPLETE")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
