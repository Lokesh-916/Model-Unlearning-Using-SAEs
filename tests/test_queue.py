"""End-to-end queue test with real worker subprocesses (CPU test jobs only)."""
import os
import subprocess
import sys
import time

from dsgx import paths
from dsgx.queue import common as q
from dsgx.queue import scheduler as S
from dsgx.queue import status
from dsgx.queue.enqueue import write_job


def _job(jid, kind="sleep-test", seconds=1, deps=(), **kw):
    j = {"id": jid, "exp_id": "QT", "kind": kind, "branch": None, "worktree": str(paths.REPO_ROOT),
         "commit": None, "config_path": None, "priority": "must", "wave": 1, "deps": list(deps),
         "est_vram_gb": 0, "est_ram_gb": 0.1, "est_minutes": 1, "gpu_exclusive": False,
         "kind_slot": "cpu", "items_total": seconds, "seconds": seconds, "python": sys.executable}
    j.update(kw)
    write_job(j)


def _run_until(s, cond, timeout=60):
    t = time.time()
    while time.time() - t < timeout:
        s.tick()
        if cond():
            return True
        time.sleep(0.3)
    return False


def _st(j):
    return q.load_state(j)["status"]


def test_scheduler_runs_jobs_blocks_dependents_and_watchdog(monkeypatch):
    q.ensure_dirs()
    q.save_control({"hung_after_seconds": 4, "status_every_seconds": 0, "cpu_slots": 4})
    _job("ok1", seconds=2)
    _job("ok2", seconds=1, deps=["ok1"])
    _job("bad", kind="no-such-kind")          # worker raises -> FAILED
    _job("child", deps=["bad"])                # -> BLOCKED
    _job("hang", kind="hang-test")             # heartbeat stops -> HUNG -> retry -> FAILED
    s = S.Scheduler()
    assert _run_until(s, lambda: all(_st(j) in q.TERMINAL for j in ["ok1", "ok2", "bad", "child", "hang"]), 90)
    assert _st("ok1") == q.DONE and _st("ok2") == q.DONE
    assert q.load_state("ok2")["start"] >= q.load_state("ok1")["end"] - 1  # dependency order
    assert _st("bad") == q.FAILED and _st("child") == q.BLOCKED
    hang = q.load_state("hang")
    assert hang["status"] == q.FAILED and hang["attempts"] == 2
    assert sum(f.startswith("HUNG") for f in hang["flags"]) == 2
    # progress heartbeats were written by the workers
    pr = q.progress_of(q.load_job("ok1"))
    assert pr["items_done"] == 2 and pr["phase"] in ("done", "exit")
    # reporting
    s.write_status(force=True)
    full = (paths.results_dir() / "STATUS.md").read_text()
    chat = (paths.results_dir() / "STATUS_FOR_CHAT.md").read_text()
    assert "## Overall" in full and "bad: FAILED" in full and "BLOCKED by bad" in full
    assert len(chat.strip().splitlines()) <= 40
    assert list((paths.results_dir() / "status_snapshots").glob("*.md"))
    prog = (paths.results_dir() / "PROGRESS.md").read_text()
    assert "ok1 | DONE" in prog and "bad | FAILED" in prog


def test_pause_and_resume():
    q.ensure_dirs()
    q.save_control({"status_every_seconds": 0})
    q.set_paused("test pause")
    _job("p1", seconds=1)
    s = S.Scheduler()
    for _ in range(3):
        s.tick()
    assert _st("p1") == q.WAITING
    from dsgx.queue import resume

    resume.main()
    assert _run_until(s, lambda: _st("p1") == q.DONE, 30)


def test_wave_pause_point():
    q.ensure_dirs()
    q.save_control({"status_every_seconds": 0, "pause_after_wave": 1})
    _job("w1", seconds=1)
    _job("w2", seconds=1, wave=2)
    s = S.Scheduler()
    assert _run_until(s, lambda: _st("w1") == q.DONE and q.paused() is not None, 30)
    for _ in range(3):
        s.tick()
    assert _st("w2") == q.WAITING
    assert (paths.results_dir() / "WAVE1_DONE.md").exists()
    from dsgx.queue import resume

    resume.main()
    assert _run_until(s, lambda: _st("w2") == q.DONE, 30)
    assert 1 in q.control()["resumed_waves"]


def test_lock_single_scheduler():
    q.ensure_dirs()
    lock = q.qdir() / "scheduler.lock"
    lock.write_text(str(os.getppid()))  # a live process holds it
    assert S.acquire_lock() is False
    lock.write_text("999999999")          # stale
    assert S.acquire_lock() is True and lock.read_text() == str(os.getpid())
    S.release_lock()
    assert not lock.exists()


def test_status_cli_runs_empty_queue():
    r = subprocess.run([sys.executable, "-m", "dsgx.queue.status", "--chat"], cwd=paths.REPO_ROOT,
                       capture_output=True, text=True, env=os.environ)
    assert r.returncode == 0 and "DSG queue status" in r.stdout
