"""Queue doctor: classify every FAILED / HUNG / BLOCKED job and say (or do) the right thing.

    python -m dsgx.queue.doctor                 # diagnose only (read-only)
    python -m dsgx.queue.doctor --apply         # re-queue the SAFE classes + their BLOCKED dependents
    python -m dsgx.queue.doctor --requeue J1 J2 # re-queue these jobs + their BLOCKED dependents
    python -m dsgx.queue.doctor --json          # machine-readable

This replaces the judgment call "is this failure safe to retry?" (prompt P3) with fixed rules:

| class        | how it is recognised                                       | safe to re-queue? |
|--------------|------------------------------------------------------------|-------------------|
| interrupted  | no exit code (machine rebooted / job killed), or started   | yes               |
|              | before the last boot                                       |                   |
| hung         | HUNG flag, only once                                       | yes               |
| hung-repeat  | HUNG flag twice                                            | no (look at log)  |
| oom          | exit 75 or 'CUDA out of memory' after the half-batch retry | yes, if the GPU   |
|              |                                                            | is free now       |
| disk         | 'No space left on device'                                  | after freeing disk|
| leakage      | exit 4 (leakage check failed)                              | NO                |
| drift        | exit 3 (canary / sanity drift)                             | NO                |
| missing-input| FileNotFoundError / no finished dev runs / no checkpoint   | only after its    |
|              |                                                            | source job is DONE|
| code-error   | any other Python exception                                 | NO (needs a fix)  |
| blocked      | BLOCKED by a failed dependency                             | follows its cause |

Only exception type + a truncated message is printed (never item text). Logs: <job dir>/job.log.
"""
import argparse
import json
import re
import time

from dsgx.queue import common as q

SAFE = {"interrupted", "hung", "oom"}
EXC = re.compile(r"^(\w+(?:\.\w+)*(?:Error|Exception|Interrupt|Exit))\b:?\s*(.*)$")


def boot_time() -> float:
    try:
        import psutil

        return float(psutil.boot_time())
    except Exception:
        return 0.0


def _log_tail(job, n=60) -> list[str]:
    return q.tail(q.job_dir(job) / "job.log", n)


def _last_exception(lines) -> str | None:
    for l in reversed(lines):
        m = EXC.match(l.strip())
        if m:
            return f"{m.group(1)}: {m.group(2)[:160]}"
    return None


def classify(job: dict, st: dict, boot: float | None = None) -> dict:
    boot = boot_time() if boot is None else boot
    status = st.get("status")
    flags = [str(f) for f in st.get("flags", [])]
    rc = st.get("exit_code")
    lines = _log_tail(job)
    text = "\n".join(lines)
    exc = _last_exception(lines)
    n_hung = sum(f.startswith("HUNG") for f in flags)
    if status == q.BLOCKED:
        cls = "blocked"
    elif rc == q.EXIT_LEAKAGE:
        cls = "leakage"
    elif rc == q.EXIT_DRIFT:
        cls = "drift"
    elif "No space left on device" in text:
        cls = "disk"
    elif rc == q.EXIT_OOM or "CUDA out of memory" in text or "OutOfMemoryError" in text:
        cls = "oom"
    elif n_hung >= 2:
        cls = "hung-repeat"
    elif n_hung == 1 and not exc:
        cls = "hung"
    elif rc is None or (st.get("start") and boot and st["start"] < boot and not exc):
        cls = "interrupted"
    elif exc and re.search(r"FileNotFoundError|no finished dev runs|no finished dev base run|No such file", text):
        cls = "missing-input"
    else:
        cls = "code-error"
    return {"job": job["id"], "exp_id": job.get("exp_id"), "status": status, "class": cls,
            "safe": cls in SAFE, "exit_code": rc, "exception": exc, "blocked_by": st.get("blocked_by", []),
            "log": str(q.job_dir(job) / "job.log"), "attempts": st.get("attempts", 0)}


ADVICE = {
    "interrupted": "killed by a reboot or a manual kill: re-queue (training resumes from its checkpoint)",
    "hung": "heartbeat went stale once: re-queue",
    "hung-repeat": "hung twice: open the log; usually a deadlock or a stuck download (HF offline?)",
    "oom": "out of GPU memory even at half batch: check `nvidia-smi` for foreign processes, then re-queue",
    "disk": "disk full: free space (see runbook T4), then re-queue",
    "leakage": "leakage check failed: DO NOT re-queue; the data split / corpus overlaps an eval set",
    "drift": "baseline drift: DO NOT resume; results since the last good canary are suspect (runbook T6)",
    "missing-input": "an input is missing (dependency output, checkpoint or dev runs): re-queue after the "
                     "source job is DONE",
    "code-error": "a code bug: needs a fix on the right branch (runbook T10); do not just re-queue",
    "blocked": "waits on a failed dependency: re-queued automatically with it (--apply / --requeue)",
}


def dependents(job_ids, jobs, states) -> list[str]:
    """Transitive BLOCKED dependents of job_ids."""
    out, frontier = [], set(job_ids)
    while frontier:
        nxt = set()
        for jid, job in jobs.items():
            if jid in out or jid in job_ids:
                continue
            if states[jid].get("status") == q.BLOCKED and frontier & set(job.get("deps", [])):
                out.append(jid)
                nxt.add(jid)
        frontier = nxt
    return out


def requeue_with_dependents(job_ids, jobs, states, dry=False) -> list[str]:
    from dsgx.queue.enqueue import requeue

    ids = [j for j in job_ids if j in jobs]
    allids = ids + dependents(ids, jobs, states)
    if not dry and allids:
        requeue(allids)
    return allids


def diagnose():
    jobs = q.load_jobs()
    states = {j: q.load_state(j) for j in jobs}
    boot = boot_time()
    bad = [classify(jobs[j], states[j], boot) for j in sorted(jobs)
           if states[j].get("status") in (q.FAILED, q.BLOCKED, q.HUNG)]
    return jobs, states, bad


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--apply", action="store_true", help="re-queue every SAFE failure and its BLOCKED dependents")
    ap.add_argument("--requeue", nargs="*", help="re-queue these jobs (any class) and their BLOCKED dependents")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    jobs, states, bad = diagnose()
    if a.json:
        print(json.dumps(bad, indent=1))
    else:
        counts = {}
        for b in bad:
            counts[b["class"]] = counts.get(b["class"], 0) + 1
        print(f"queue doctor {time.strftime('%F %T')}: {len(bad)} job(s) not healthy {counts or ''}")
        for b in bad:
            tag = "SAFE " if b["safe"] else ("WAIT " if b["class"] == "blocked" else "STOP ")
            print(f"  {tag}{b['class']:<13} {b['job']}" + (f"  rc={b['exit_code']}" if b["exit_code"] is not None else "")
                  + (f"  [{b['exception']}]" if b["exception"] else "")
                  + (f"  blocked_by={','.join(b['blocked_by'][:3])}" if b["blocked_by"] else ""))
        for cls in sorted({b["class"] for b in bad}):
            print(f"  -> {cls}: {ADVICE[cls]}")
        safe = [b["job"] for b in bad if b["safe"]]
        if safe and not a.apply:
            print(f"\nSafe to re-queue now ({len(safe)}):  python -m dsgx.queue.doctor --apply")
        if not bad:
            print("  nothing to do")
    if a.apply or a.requeue:
        ids = list(a.requeue or []) + ([b["job"] for b in bad if b["safe"]] if a.apply else [])
        done = requeue_with_dependents(ids, jobs, states, dry=a.dry_run)
        print(("would re-queue " if a.dry_run else "re-queued ") + f"{len(done)} job(s): " + " ".join(done))
    stop = [b for b in bad if b["class"] in ("leakage", "drift")]
    return 2 if stop else 0


if __name__ == "__main__":
    raise SystemExit(main())
