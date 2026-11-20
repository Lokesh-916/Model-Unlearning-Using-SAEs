"""Queue one external command as one exclusive GPU job of the lab queue (task entry dsgx.external:task).

    python scripts/enqueue_external.py <job-id> --cwd <dir> [--env K=V ...] [--deps J ...] [--est-minutes M] -- cmd ...

The job runs from this worktree (prep/later-runs) at its current commit (pinned; the worktree must be clean), with
gpu_exclusive, so nothing else of ours shares the GPU. Used for the clean repository's reproduction checks, with
--env pointing DSG_RESULTS / DSG_CACHE / DSG_PRIVATE at an isolated tree. Exp id: CLEANREPRO.
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dsgx.queue.enqueue import write_job  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("job_id")
    ap.add_argument("--cwd", required=True)
    ap.add_argument("--env", nargs="*", default=[])
    ap.add_argument("--deps", nargs="*", default=[])
    ap.add_argument("--est-minutes", type=float, default=60)
    ap.add_argument("--est-vram-gb", type=float, default=12)
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--" not in argv:
        ap.error("give the command after --")
    i = argv.index("--")
    a, cmd = ap.parse_args(argv[:i]), argv[i + 1:]
    if not cmd:
        ap.error("no command after --")
    git = lambda *x: subprocess.run(["git", *x], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    if git("status", "--porcelain", "--untracked-files=no"):
        sys.exit("worktree has uncommitted changes; commit first (jobs are pinned to a commit)")
    env = dict(kv.split("=", 1) for kv in a.env)
    job = {"id": a.job_id, "exp_id": "CLEANREPRO", "kind": "task", "task_id": a.job_id, "entry": "dsgx.external:task",
           "branch": git("rev-parse", "--abbrev-ref", "HEAD"), "worktree": str(ROOT), "commit": git("rev-parse", "HEAD"),
           "config_path": None, "smoke": False, "priority": "must", "wave": 9, "deps": a.deps,
           "args": {"cmd": cmd, "cwd": str(Path(a.cwd).resolve()), "env": env},
           "est_vram_gb": a.est_vram_gb, "est_ram_gb": 20, "est_minutes": a.est_minutes, "gpu_exclusive": True,
           "kind_slot": "gpu", "items_total": 0, "group_key": a.job_id}
    print(("queued " if write_job(job) else "exists ") + a.job_id)


if __name__ == "__main__":
    main()
