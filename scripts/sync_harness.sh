#!/usr/bin/env bash
# Merge v2-harness into every experiment worktree that has no running jobs, then list finished
# jobs whose results came from an older harness and must be re-run.
#   scripts/sync_harness.sh [--dry-run]
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"
"$DSGX_PY" - "$@" <<'PY'
import subprocess, sys
from pathlib import Path
from dsgx.queue import common as q
dry = "--dry-run" in sys.argv
jobs = q.load_jobs(); states = {j: q.load_state(j) for j in jobs}
def git(wt, *a):
    return subprocess.run(["git", *a], cwd=wt, capture_output=True, text=True)
wts = [l.split()[1] for l in subprocess.run(["git", "worktree", "list", "--porcelain"], capture_output=True, text=True).stdout.splitlines() if l.startswith("worktree ")]
for wt in wts:
    br = git(wt, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if not br.startswith("exp/"):
        continue
    running = [j for j in jobs if Path(jobs[j].get("worktree") or "").resolve() == Path(wt).resolve() and states[j]["status"] == q.RUNNING]
    if running:
        print(f"SKIP {br}: running jobs {running}"); continue
    if git(wt, "status", "--porcelain", "--untracked-files=no").stdout.strip():
        print(f"SKIP {br}: worktree has uncommitted changes"); continue
    changed = git(wt, "diff", "--name-only", "HEAD", "v2-harness", "--", "dsgx", "tests", "scripts").stdout.split()
    if not changed:
        print(f"OK   {br}: already up to date"); continue
    if dry:
        print(f"DRY  {br}: would merge ({len(changed)} harness files differ)"); continue
    r = git(wt, "merge", "--no-edit", "v2-harness")
    if r.returncode:
        git(wt, "merge", "--abort"); print(f"FAIL {br}: merge conflict, aborted\n{r.stdout}{r.stderr}"); continue
    head = git(wt, "rev-parse", "HEAD").stdout.strip()
    print(f"MERGED {br} -> {head[:10]} ({len(changed)} harness files)")
    stale = [j for j in jobs if Path(jobs[j].get("worktree") or "").resolve() == Path(wt).resolve() and states[j]["status"] == q.DONE and jobs[j].get("commit") != head]
    for j in stale:
        print(f"   RE-RUN NEEDED: {j} (ran on {jobs[j].get('commit','')[:10]})")
    pending = [j for j in jobs if Path(jobs[j].get("worktree") or "").resolve() == Path(wt).resolve() and states[j]["status"] == q.WAITING]
    for j in pending:
        jobs[j]["commit"] = head
        from dsgx.util import atomic_write_json
        atomic_write_json(q.job_path(j), jobs[j])
    if pending:
        print(f"   re-pinned {len(pending)} waiting jobs to {head[:10]}")
PY
