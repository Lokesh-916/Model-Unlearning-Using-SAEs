"""Mark lab-queue jobs as MOVED-TO-SERVER (they run on gpuws instead; never run them twice).

    python -m dsgx.queue.move --list                                   # moved jobs and their server job
    python -m dsgx.queue.move --jobs 'C3-*' --server-job c3            # dry run: what would change
    python -m dsgx.queue.move --jobs 'C3-*' --server-job c3 --apply    # write the states
    python -m dsgx.queue.move --undo --jobs 'C3-*' --apply             # back to WAITING

Rules (refuses otherwise, nothing is written):
  * only WAITING or BLOCKED jobs move (a RUNNING / DONE / FAILED job stays where it is);
  * every lab job that depends on a moved job must be moved too, or already DONE: the scheduler starts a
    job only when all its deps are DONE, so a lab dependent of a moved job would wait forever
    (the doctor reports such pairs as `moved-dependency`).
The scheduler only launches WAITING jobs, so a running scheduler simply skips MOVED ones (no restart needed).
Old states are archived in queue/_archive/moved-<time>/. Log a DEVIATIONS.md row for every move.
"""
import argparse
import fnmatch
import shutil
import time

from dsgx.queue import common as q
from dsgx.util import now_iso

MOVABLE = {q.WAITING, q.BLOCKED}


def select(jobs, patterns) -> list[str]:
    return sorted(j for j in jobs if any(fnmatch.fnmatchcase(j, p) for p in patterns))


def stranded(ids, jobs, states) -> list[tuple[str, str]]:
    """(dependent, moved dep) pairs where the dependent stays on the lab PC and is not DONE."""
    s = set(ids)
    return sorted((jid, d) for jid, job in jobs.items() if jid not in s
                  and states[jid].get("status") not in (q.DONE, q.MOVED)
                  for d in job.get("deps", []) if d in s)


def archive(ids):
    d = q.qdir() / "_archive" / f"moved-{time.strftime('%Y%m%d-%H%M%S')}" / "state"
    d.mkdir(parents=True, exist_ok=True)
    for j in ids:
        src = q.qdir() / "state" / f"{j}.json"
        if src.exists():
            shutil.copy2(src, d / src.name)
    return d.parent


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--jobs", nargs="*", default=[], help="job ids or globs, e.g. 'A6-*'")
    ap.add_argument("--server-job", help="cluster/slurm/jobs/<name>.conf that runs these jobs on gpuws")
    ap.add_argument("--reason", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    jobs = q.load_jobs()
    states = {j: q.load_state(j) for j in jobs}
    if a.list:
        moved = [j for j in sorted(jobs) if states[j].get("status") == q.MOVED]
        by = {}
        for j in moved:
            by.setdefault(states[j].get("server_job", "?"), []).append(j)
        for k, v in sorted(by.items()):
            print(f"{k:<16} {len(v):>3}  {' '.join(v[:6])}{' ...' if len(v) > 6 else ''}")
        print(f"total moved: {len(moved)}")
        return 0
    ids = select(jobs, a.jobs)
    if not ids:
        print("no job matches", a.jobs)
        return 2
    if a.undo:
        bad = [j for j in ids if states[j].get("status") != q.MOVED]
        if bad:
            print("REFUSING: not MOVED:", " ".join(bad))
            return 1
        print(("undo " if a.apply else "would undo ") + f"{len(ids)} move(s): " + " ".join(ids))
        if a.apply:
            archive(ids)
            for j in ids:
                st = dict(states[j])
                st["status"] = st.pop("moved_from_status", q.WAITING)
                for k in ("moved_to", "server_job", "moved_at", "move_reason"):
                    st.pop(k, None)
                st.setdefault("flags", []).append(f"MOVE UNDONE at {now_iso()}")
                q.save_state(j, st)
        return 0
    if not a.server_job:
        ap.error("--server-job is required")
    bad = [f"{j} ({states[j].get('status')})" for j in ids if states[j].get("status") not in MOVABLE]
    if bad:
        print("REFUSING: only WAITING/BLOCKED jobs can move:", " ".join(bad))
        return 1
    st_pairs = stranded(ids, jobs, states)
    if st_pairs:
        print("REFUSING: these lab jobs depend on a job being moved (move them too):")
        for jid, d in st_pairs:
            print(f"  {jid} -> {d}")
        return 1
    print(("moving " if a.apply else "would move ") + f"{len(ids)} job(s) -> gpuws job '{a.server_job}': " + " ".join(ids))
    if not a.apply:
        print("(dry run; add --apply)")
        return 0
    arch = archive(ids)
    for j in ids:
        st = dict(states[j])
        st.update({"moved_from_status": st.get("status"), "status": q.MOVED, "moved_to": "gpuws",
                   "server_job": a.server_job, "moved_at": now_iso(), "move_reason": a.reason})
        st.setdefault("flags", []).append(f"MOVED-TO-SERVER ({a.server_job}) at {st['moved_at']}")
        q.save_state(j, st)
    from dsgx.queue.doctor import moved_deadlocks, wave_deadlocks

    states = {j: q.load_state(j) for j in jobs}
    dl, md = wave_deadlocks(jobs, states), moved_deadlocks(jobs, states)
    print(f"done; old states in {arch}. deadlock check: wave {len(dl)}, moved-dependency {len(md)}")
    return 1 if dl or md else 0


if __name__ == "__main__":
    raise SystemExit(main())
