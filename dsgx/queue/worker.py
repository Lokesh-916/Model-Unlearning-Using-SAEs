"""Job worker: runs one job (one or more run configs sharing a model load).

    python -m dsgx.queue.worker --job <job_id>

Launched by the scheduler from the job's worktree. Exit codes: 0 done, 75 CUDA OOM (the
scheduler retries once with half the batch size), 3 sanity drift, anything else failed.
"""
import argparse
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path

from dsgx.queue import common as q
from dsgx.util import atomic_write_json, now_iso


def _check_pinned(job: dict):
    wt = job.get("worktree")
    if not wt or not job.get("commit"):
        return
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
    if head != job["commit"]:
        raise RuntimeError(f"worktree {wt} is at {head[:10]}, job pinned to {job['commit'][:10]}")
    dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=wt,
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise RuntimeError(f"worktree {wt} has uncommitted changes:\n{dirty}")
    if Path(__file__).resolve().parents[2] != Path(wt).resolve():
        raise RuntimeError(f"worker code is not running from the job worktree {wt}")


def _run_job(job: dict, progress):
    kind = job.get("kind", "runs")
    if kind == "sanity":
        from dsgx.checks.sanity import main as sanity

        progress.update(phase="sanity", force=True)
        rc = sanity(["--force"])
        if rc != 0:
            raise SystemExit(q.EXIT_DRIFT)
        return
    if kind == "hang-test":
        # Exercises the watchdog: heartbeats stop, the process stays alive.
        progress.update(phase="hang-test", force=True)
        progress.close(phase="hang-test (heartbeat stopped)")
        time.sleep(10 ** 6)
    if kind == "sleep-test":
        for i in range(int(job.get("seconds", 30))):
            time.sleep(1)
            progress.update(items_done=i + 1)
        return
    from dsgx.config import expand, load_experiment
    from dsgx.run import run

    exp = load_experiment(Path(job["worktree"] or ".") / job["config_path"])
    runs = expand(exp, smoke=job.get("smoke", False))
    idx = job.get("run_indices") or list(range(len(runs)))
    factor = float(os.environ.get("DSGX_BATCH_FACTOR", "1"))
    for n, i in enumerate(idx):
        cfg = runs[i]
        cfg["batch_size"] = max(1, int(cfg.get("batch_size", 1) * factor))
        progress.update(phase=f"run {n + 1}/{len(idx)}", force=True)
        rd = run(cfg, progress=progress)
        print(f"[worker] run {n + 1}/{len(idx)} done: {rd}", flush=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    job = q.load_job(a.job)
    jd = q.job_dir(job)
    jd.mkdir(parents=True, exist_ok=True)
    from dsgx.logging.progress import Progress

    progress = Progress([str(jd / "progress.json")], job["id"], job.get("items_total", 0), phase="init")
    print(f"[worker] {now_iso()} job={job['id']} pid={os.getpid()} cwd={os.getcwd()}", flush=True)
    rc = 0
    try:
        _check_pinned(job)
        from dsgx.checks.leakage import check_job

        errs = check_job(job)
        if errs:
            for e in errs:
                print(f"[worker] LEAKAGE: {e}", flush=True)
            raise SystemExit(q.EXIT_LEAKAGE)
        _run_job(job, progress)
        progress.close("done")
        atomic_write_json(jd / "DONE", {"time": now_iso()})
    except SystemExit as e:
        rc = int(e.code or 0)
    except Exception as e:  # noqa: BLE001
        import torch

        traceback.print_exc()
        rc = q.EXIT_OOM if isinstance(e, torch.cuda.OutOfMemoryError) or "out of memory" in str(e) else 1
    finally:
        progress.close("exit" if rc == 0 else f"error rc={rc}")
    print(f"[worker] {now_iso()} exit {rc}", flush=True)
    sys.stdout.flush()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
