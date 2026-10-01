"""Queue storage layout and helpers (MASTER_PLAN 4.4 - 4.6).

$DSG_RESULTS/queue/
  jobs/<job_id>.json      job spec (written by enqueue)
  state/<job_id>.json     status, attempts, pid, slot, times, exit code, flags
  control.json            knobs: hung/overtime thresholds, wave pause, canary, slots
  PAUSED                  present = no new jobs start (reason inside); resume removes it
  scheduler.lock          PID of the one running scheduler
  alerts.json             current alerts shown in STATUS
$DSG_RESULTS/runs/<exp_id>/_jobs/<job_id>/   job dir: job.log, progress.json, exit_code
"""
import json
import os
import time
from pathlib import Path

from dsgx import paths
from dsgx.util import atomic_write_json, read_json

WAITING, RUNNING, DONE, FAILED, HUNG, BLOCKED = "WAITING", "RUNNING", "DONE", "FAILED", "HUNG", "BLOCKED"
TERMINAL = {DONE, FAILED, BLOCKED}
PRIORITY_RANK = {"canary": -1, "must": 0, "should": 1, "stretch": 2, "backlog": 3}
EXIT_OOM = 75
EXIT_DRIFT = 3
EXIT_LEAKAGE = 4

DEFAULT_CONTROL = {
    "hung_after_seconds": 900,       # heartbeat older than 15 min -> HUNG
    "overtime_factor": 3.0,          # flag jobs running > 3x their estimate
    "vram_margin_gb": 1.5,
    "max_gpu_jobs": 2,
    "max_concurrent_vram_gb_each": 6.5,
    "cpu_slots": 4,
    "min_free_disk_gb": 30,
    "max_ram_percent": 90,
    "max_gpu_temp_c": 85,
    "status_every_seconds": 120,
    "loop_seconds": 10,
    "pause_after_wave": None,
    "resumed_waves": [],
    "canary_enabled": False,
    "canary_hours": 24,
    "max_attempts": 2,               # 1 retry for HUNG or OOM
}


def qdir() -> Path:
    return paths.queue_dir()


def ensure_dirs():
    for d in ("jobs", "state"):
        (qdir() / d).mkdir(parents=True, exist_ok=True)
    paths.logs_dir().mkdir(parents=True, exist_ok=True)


def control() -> dict:
    c = dict(DEFAULT_CONTROL)
    c.update(read_json(qdir() / "control.json", {}) or {})
    return c


def save_control(c: dict):
    atomic_write_json(qdir() / "control.json", c)


def job_path(job_id):
    return qdir() / "jobs" / f"{job_id}.json"


def state_path(job_id):
    return qdir() / "state" / f"{job_id}.json"


def load_job(job_id) -> dict:
    return json.loads(job_path(job_id).read_text())


def load_jobs() -> dict:
    out = {}
    for p in sorted((qdir() / "jobs").glob("*.json")):
        try:
            j = json.loads(p.read_text())
            out[j["id"]] = j
        except (json.JSONDecodeError, KeyError):
            continue
    return out


def load_state(job_id) -> dict:
    return read_json(state_path(job_id), None) or {"status": WAITING, "attempts": 0, "flags": [],
                                                   "batch_factor": 1.0}


def save_state(job_id, st: dict):
    st["updated"] = time.time()
    atomic_write_json(state_path(job_id), st)


def job_dir(job: dict) -> Path:
    return paths.runs_dir() / job["exp_id"] / "_jobs" / job["id"]


def progress_of(job: dict) -> dict | None:
    return read_json(job_dir(job) / "progress.json", None)


def paused() -> str | None:
    p = qdir() / "PAUSED"
    return p.read_text().strip() or "paused" if p.exists() else None


def set_paused(reason: str):
    from dsgx.util import atomic_write_text

    atomic_write_text(qdir() / "PAUSED", reason + "\n")


def alerts() -> list:
    return read_json(qdir() / "alerts.json", []) or []


def pid_alive(pid) -> bool:
    if not pid:
        return False
    try:
        import psutil

        p = psutil.Process(int(pid))
        return p.status() != psutil.STATUS_ZOMBIE
    except Exception:
        return False


def tail(path, n=3) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - 20000))
            lines = f.read().decode(errors="replace").splitlines()
        lines = [l for l in lines if l.strip() and "Warning" not in l and "warn(" not in l]
        return lines[-n:]
    except FileNotFoundError:
        return []
