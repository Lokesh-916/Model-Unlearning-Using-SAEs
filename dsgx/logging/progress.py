"""progress.json heartbeat (MASTER_PLAN 4.5).

A background thread rewrites progress.json atomically every `interval` seconds, so the
heartbeat stays fresh during long single steps; `update()` also writes immediately (throttled).
"""
import os
import threading
import time

from dsgx.util import atomic_write_json

HEARTBEAT_SECONDS = float(os.environ.get("DSGX_HEARTBEAT_SECONDS", "30"))


class Progress:
    def __init__(self, paths, job_id: str, items_total: int = 0, phase: str = "start",
                 interval: float = HEARTBEAT_SECONDS):
        self.paths = [p for p in (paths if isinstance(paths, (list, tuple)) else [paths]) if p]
        self.state = {"job_id": job_id, "phase": phase, "items_done": 0,
                      "items_total": int(items_total), "step": None, "steps_total": None,
                      "start_time": time.time(), "last_update": time.time(), "eta_seconds": None,
                      "current_metric": None, "pid": os.getpid()}
        self.interval = interval
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._last_write = 0.0
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._write()
        self._thread.start()

    def add_path(self, p):
        with self._lock:
            if p not in self.paths:
                self.paths.append(p)
        self._write()

    def drop_path(self, p):
        with self._lock:
            if p in self.paths:
                self.paths.remove(p)

    def _eta(self):
        s = self.state
        done, total = s["items_done"], s["items_total"]
        if s["steps_total"]:
            done, total = s["step"] or 0, s["steps_total"]
        el = time.time() - s["start_time"]
        if done and total and done <= total:
            return el / done * (total - done)
        return None

    def _write(self):
        with self._lock:
            self.state["last_update"] = time.time()
            self.state["eta_seconds"] = self._eta()
            snap = dict(self.state)
            targets = list(self.paths)
            self._last_write = time.time()
        for p in targets:
            try:
                atomic_write_json(p, snap)
            except OSError:
                pass

    def _beat(self):
        while not self._stop.wait(self.interval):
            self._write()

    def update(self, force: bool = False, **kw):
        with self._lock:
            self.state.update(kw)
        if force or time.time() - self._last_write > 5:
            self._write()

    def advance(self, n: int = 1, **kw):
        with self._lock:
            self.state["items_done"] += n
            self.state.update(kw)
        if time.time() - self._last_write > 5:
            self._write()

    def close(self, phase: str = "done"):
        self.update(phase=phase, force=True)
        self._stop.set()
