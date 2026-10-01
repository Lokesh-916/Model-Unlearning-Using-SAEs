"""The scheduler: starts jobs, watches them, records results. Runs under a supervisor loop
in tmux session dsg-queue (`while true; do python -m dsgx.queue.scheduler; sleep 30; done`).

    python -m dsgx.queue.scheduler [--once] [--max-ticks N]

* One scheduler only: $DSG_RESULTS/queue/scheduler.lock holds its PID; a second one exits.
* Jobs run detached (own process group), so they survive a scheduler restart; the new
  scheduler re-attaches through the PIDs in state/<job>.json.
* Watchdog: heartbeat older than hung_after_seconds -> HUNG, killed, retried once.
  Running > overtime_factor x estimate -> flagged (killed only if the heartbeat is also stale).
* Exit 75 (CUDA OOM) -> retried once with half the batch size, then FAILED.
* A failure blocks only its dependents (BLOCKED).
* Guards: free disk, RAM %, GPU temperature, PAUSED file, wave pause point.
"""
import argparse
import logging
import logging.handlers
import os
import shlex
import signal
import subprocess
import sys
import time
import traceback

from dsgx import paths
from dsgx.queue import common as q
from dsgx.queue import resources
from dsgx.util import atomic_write_json, atomic_write_text, now_iso, read_json

log = logging.getLogger("dsgx.scheduler")
GPU_SLOTS = ("gpu0", "gpu1")


def setup_logging():
    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    h = logging.handlers.RotatingFileHandler(paths.logs_dir() / "scheduler.log",
                                             maxBytes=50 * 1024 * 1024, backupCount=3)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    log.handlers[:] = [h, logging.StreamHandler(sys.stdout)]
    log.setLevel(logging.INFO)


def acquire_lock() -> bool:
    lock = q.qdir() / "scheduler.lock"
    if lock.exists():
        try:
            pid = int(lock.read_text().strip())
        except ValueError:
            pid = None
        if pid and pid != os.getpid() and q.pid_alive(pid):
            return False
    atomic_write_text(lock, str(os.getpid()))
    return True


def release_lock():
    lock = q.qdir() / "scheduler.lock"
    try:
        if lock.read_text().strip() == str(os.getpid()):
            lock.unlink()
    except FileNotFoundError:
        pass


def kill_group(pgid, grace: float = 10.0):
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(int(pgid), sig)
        except (ProcessLookupError, PermissionError, TypeError):
            return
        t = time.time()
        while time.time() - t < grace:
            try:
                os.killpg(int(pgid), 0)
            except ProcessLookupError:
                return
            time.sleep(0.5)


class Scheduler:
    def __init__(self):
        q.ensure_dirs()
        self.procs = {}  # job_id -> Popen (only for jobs this process started)
        self.last_status = 0.0
        self.alerts = {}

    # ---------------- helpers ----------------
    def jobs_states(self):
        jobs = q.load_jobs()
        return jobs, {j: q.load_state(j) for j in jobs}

    def alert(self, key, msg):
        if key not in self.alerts:
            log.warning("ALERT %s: %s", key, msg)
        self.alerts[key] = {"time": now_iso(), "msg": msg}

    def clear_alert(self, key):
        self.alerts.pop(key, None)

    def _exit_code(self, job_id, job):
        p = self.procs.get(job_id)
        if p is not None:
            rc = p.poll()
            if rc is None:
                return None
            self.procs.pop(job_id, None)
        f = q.job_dir(job) / "exit_code"
        try:
            return int(f.read_text().strip())
        except (FileNotFoundError, ValueError):
            return None if p is None else 1

    # ---------------- reconcile running jobs ----------------
    def reconcile(self, jobs, states, ctl):
        for jid, st in states.items():
            if st["status"] != q.RUNNING:
                continue
            job = jobs[jid]
            alive = q.pid_alive(st.get("pid")) or (jid in self.procs and self.procs[jid].poll() is None)
            if alive:
                self.watchdog(jid, job, st, ctl)
                continue
            rc = self._exit_code(jid, job)
            st["end"] = time.time()
            st["exit_code"] = rc
            if rc == 0:
                st["status"] = q.DONE
                self.progress_line(job, st)
            elif rc == q.EXIT_OOM and st["attempts"] < ctl["max_attempts"]:
                st["status"] = q.WAITING
                st["batch_factor"] = st.get("batch_factor", 1.0) / 2
                st.setdefault("flags", []).append(f"OOM retry with batch factor {st['batch_factor']}")
            else:
                st["status"] = q.FAILED
                st["error_tail"] = q.tail(q.job_dir(job) / "job.log", 3)
                if rc == q.EXIT_DRIFT and job.get("kind") == "sanity":
                    q.set_paused(f"canary drift detected by {jid} at {now_iso()}")
                    self.alert("canary", f"sanity canary {jid} drifted; queue paused")
                self.progress_line(job, st)
            log.info("job %s finished rc=%s -> %s", jid, rc, st["status"])
            q.save_state(jid, st)

    def watchdog(self, jid, job, st, ctl):
        prog = q.progress_of(job) or {}
        last = prog.get("last_update") or st.get("start", time.time())
        age = time.time() - last
        elapsed = time.time() - st.get("start", time.time())
        flags = st.setdefault("flags", [])
        over = elapsed > ctl["overtime_factor"] * 60 * max(job.get("est_minutes", 1), 0.1)
        if over and "OVERTIME" not in flags:
            flags.append("OVERTIME")
            q.save_state(jid, st)
        if age > ctl["hung_after_seconds"]:
            log.warning("job %s heartbeat %.0fs old -> HUNG, killing pgid %s", jid, age, st.get("pgid"))
            kill_group(st.get("pgid"))
            self.procs.pop(jid, None)
            st["end"] = time.time()
            st["error_tail"] = [f"HUNG: heartbeat {age:.0f}s old"] + q.tail(q.job_dir(job) / "job.log", 2)
            flags.append(f"HUNG at {now_iso()}")
            if st["attempts"] < ctl["max_attempts"]:
                st["status"] = q.WAITING
                log.info("job %s re-queued after HUNG (attempt %d)", jid, st["attempts"])
            else:
                st["status"] = q.FAILED
                self.progress_line(job, st)
            q.save_state(jid, st)

    # ---------------- dependencies ----------------
    def mark_blocked(self, jobs, states):
        changed = True
        while changed:
            changed = False
            for jid, st in states.items():
                if st["status"] != q.WAITING:
                    continue
                bad = [d for d in jobs[jid].get("deps", []) if states.get(d, {}).get("status") in (q.FAILED, q.BLOCKED)]
                if bad:
                    st["status"] = q.BLOCKED
                    st["blocked_by"] = bad
                    q.save_state(jid, st)
                    changed = True

    # ---------------- guards ----------------
    def guards(self, ctl, res) -> str | None:
        reasons = []
        if res["disk"]["free_gb"] < ctl["min_free_disk_gb"]:
            reasons.append(f"free disk {res['disk']['free_gb']:.0f} GB < {ctl['min_free_disk_gb']}")
            self.alert("disk", reasons[-1])
        else:
            self.clear_alert("disk")
        if res["ram"]["percent"] > ctl["max_ram_percent"]:
            reasons.append(f"RAM {res['ram']['percent']:.0f}% > {ctl['max_ram_percent']}%")
            self.alert("ram", reasons[-1])
        else:
            self.clear_alert("ram")
        g = res["gpu"]
        if g.get("ok") and g["temp_c"] > ctl["max_gpu_temp_c"]:
            reasons.append(f"GPU {g['temp_c']:.0f} C > {ctl['max_gpu_temp_c']} C")
            self.alert("gpu_temp", reasons[-1])
        else:
            self.clear_alert("gpu_temp")
        p = q.paused()
        if p:
            reasons.append(f"PAUSED: {p}")
        return "; ".join(reasons) or None

    def wave_pause(self, jobs, states, ctl):
        w = ctl.get("pause_after_wave")
        if w is None or w in ctl.get("resumed_waves", []) or q.paused():
            return
        upto = [j for j in jobs if jobs[j].get("wave", 0) <= w and jobs[j].get("kind") != "sanity"]
        later = [j for j in jobs if jobs[j].get("wave", 0) > w and states[j]["status"] == q.WAITING]
        if upto and later and all(states[j]["status"] in q.TERMINAL for j in upto):
            from dsgx.queue.status import wave_report

            atomic_write_text(paths.results_dir() / f"WAVE{w}_DONE.md", wave_report(w))
            q.set_paused(f"wave {w} finished at {now_iso()}; review WAVE{w}_DONE.md, then run python -m dsgx.queue.resume")
            log.info("wave %s done -> paused", w)

    def canary(self, jobs, states, ctl):
        if not ctl.get("canary_enabled"):
            return
        last = max([jobs[j].get("created", 0) for j in jobs if jobs[j].get("kind") == "sanity"] or [0])
        if time.time() - last < ctl["canary_hours"] * 3600:
            return
        if any(states[j]["status"] in (q.WAITING, q.RUNNING) for j in jobs if jobs[j].get("kind") == "sanity"):
            return
        from dsgx.queue.enqueue import write_job

        jid = "canary-" + time.strftime("%Y%m%d-%H%M")
        write_job({"id": jid, "exp_id": "canary", "kind": "sanity", "priority": "canary", "wave": 0,
                   "branch": "v2-harness", "worktree": str(paths.REPO_ROOT), "commit": None,
                   "config_path": None, "est_vram_gb": 7.5, "est_ram_gb": 6, "est_minutes": 5,
                   "deps": [], "gpu_exclusive": False, "kind_slot": "gpu", "items_total": 843})
        log.info("enqueued canary %s", jid)

    # ---------------- launching ----------------
    def launch_ready(self, jobs, states, ctl, res, block_reason):
        if block_reason:
            return
        running = [j for j, s in states.items() if s["status"] == q.RUNNING]
        gpu_running = [j for j in running if jobs[j].get("kind_slot", "gpu") == "gpu"]
        cpu_running = [j for j in running if jobs[j].get("kind_slot", "gpu") == "cpu"]
        free_vram = res["gpu"].get("free_gb", 0) if res["gpu"].get("ok") else 0
        free_ram = res["ram"]["total_gb"] - res["ram"]["used_gb"]
        # Jobs started in the last 3 minutes may not have allocated yet: reserve their estimates.
        for j in running:
            if time.time() - states[j].get("start", 0) < 180:
                free_ram -= jobs[j].get("est_ram_gb", 0)
                if j in gpu_running:
                    free_vram -= jobs[j].get("est_vram_gb", 0)
        ready = []
        hold = ctl.get("pause_after_wave")
        hold = None if hold is None or hold in ctl.get("resumed_waves", []) else hold
        for jid, st in states.items():
            if st["status"] != q.WAITING:
                continue
            if hold is not None and jobs[jid].get("wave", 0) > hold and jobs[jid].get("kind") != "sanity":
                continue  # later waves wait for the wave pause point to be resumed
            if all(states.get(d, {}).get("status") == q.DONE for d in jobs[jid].get("deps", [])):
                ready.append(jid)
        ready.sort(key=lambda j: (q.PRIORITY_RANK.get(jobs[j].get("priority", "must"), 9), jobs[j].get("wave", 0),
                                  jobs[j].get("group_key", ""), jobs[j].get("created", 0), j))
        for jid in ready:
            job = jobs[jid]
            if free_ram < job.get("est_ram_gb", 0):
                continue
            if job.get("kind_slot", "gpu") == "cpu":
                if len(cpu_running) >= ctl["cpu_slots"]:
                    continue
                self.start(job, states[jid], "cpu")
                cpu_running.append(jid)
                free_ram -= job.get("est_ram_gb", 0)
                continue
            if any(jobs[j].get("gpu_exclusive") for j in gpu_running):
                break
            if job.get("gpu_exclusive") and gpu_running:
                continue
            if len(gpu_running) >= ctl["max_gpu_jobs"]:
                continue
            if gpu_running and (job.get("est_vram_gb", 0) > ctl["max_concurrent_vram_gb_each"]
                                or any(jobs[j].get("est_vram_gb", 0) > ctl["max_concurrent_vram_gb_each"] for j in gpu_running)):
                continue
            if free_vram < job.get("est_vram_gb", 0) + ctl["vram_margin_gb"]:
                continue
            used = {states[j].get("slot") for j in gpu_running}
            slot = next(s for s in GPU_SLOTS if s not in used)
            self.start(job, states[jid], slot)
            gpu_running.append(jid)
            free_vram -= job.get("est_vram_gb", 0)
            free_ram -= job.get("est_ram_gb", 0)

    def start(self, job, st, slot):
        jd = q.job_dir(job)
        jd.mkdir(parents=True, exist_ok=True)
        for f in ("exit_code", "DONE", "progress.json"):
            (jd / f).unlink(missing_ok=True)
        st["attempts"] = st.get("attempts", 0) + 1
        slot_log = paths.logs_dir() / f"slot_{slot}.log"
        py = job.get("python") or sys.executable
        wt = job.get("worktree") or str(paths.REPO_ROOT)
        joblog = jd / "job.log"
        inner = (f"cd {shlex.quote(wt)} && echo \"=== {now_iso()} start {job['id']} attempt {st['attempts']} ===\" >> {shlex.quote(str(joblog))}; "
                 f"{shlex.quote(py)} -u -m dsgx.queue.worker --job {shlex.quote(job['id'])} 2>&1 "
                 f"| tee -a {shlex.quote(str(slot_log))} >> {shlex.quote(str(joblog))}; "
                 f"echo ${{PIPESTATUS[0]}} > {shlex.quote(str(jd / 'exit_code'))}")
        env = dict(os.environ, DSGX_BATCH_FACTOR=str(st.get("batch_factor", 1.0)), PYTHONUNBUFFERED="1",
                   DSG_RESULTS=str(paths.results_dir()), DSG_CACHE=str(paths.cache_dir()),
                   DSG_PRIVATE=str(paths.private_dir()))
        p = subprocess.Popen(["bash", "-c", inner], start_new_session=True, env=env,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.procs[job["id"]] = p
        st.update({"status": q.RUNNING, "pid": p.pid, "pgid": p.pid, "slot": slot, "start": time.time(),
                   "end": None, "exit_code": None, "job_dir": str(jd)})
        q.save_state(job["id"], st)
        log.info("started %s on %s pid=%s attempt=%d", job["id"], slot, p.pid, st["attempts"])

    # ---------------- reporting ----------------
    def progress_line(self, job, st):
        from dsgx.queue.status import headline_for_job

        wall = (st.get("end") or time.time()) - st.get("start", time.time())
        line = (f"- {now_iso()} | {job['id']} | {st['status']} | {headline_for_job(job) or '-'} "
                f"| wall {wall / 60:.1f} min\n")
        with open(paths.results_dir() / "PROGRESS.md", "a") as f:
            f.write(line)

    def write_status(self, force=False):
        ctl = q.control()
        if not force and time.time() - self.last_status < ctl["status_every_seconds"]:
            return
        from dsgx.queue import status

        atomic_write_json(q.qdir() / "alerts.json", [{"key": k, **v} for k, v in self.alerts.items()])
        for f in paths.logs_dir().glob("slot_*.log"):  # cap tee'd slot logs at 50 MB
            if f.stat().st_size > 50 * 1024 * 1024:
                f.replace(f.with_suffix(".log.1"))
                f.touch()
        full = status.render(chat=False)
        atomic_write_text(paths.results_dir() / "STATUS.md", full)
        atomic_write_text(paths.results_dir() / "STATUS_FOR_CHAT.md", status.render(chat=True))
        snap = paths.results_dir() / "status_snapshots" / (time.strftime("%Y%m%d-%H") + ".md")
        if not snap.exists():
            atomic_write_text(snap, full)
        self.last_status = time.time()

    # ---------------- main tick ----------------
    def tick(self):
        ctl = q.control()
        jobs, states = self.jobs_states()
        self.reconcile(jobs, states, ctl)
        jobs, states = self.jobs_states()
        self.mark_blocked(jobs, states)
        self.canary(jobs, states, ctl)
        jobs, states = self.jobs_states()
        self.wave_pause(jobs, states, ctl)
        res = resources.snapshot()
        reason = self.guards(ctl, res)
        if reason:
            self.alert("paused", f"not starting new jobs: {reason}")
        else:
            self.clear_alert("paused")
        self.launch_ready(jobs, states, ctl, res, reason)
        self.write_status()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--max-ticks", type=int, default=None)
    a = ap.parse_args(argv)
    setup_logging()
    q.ensure_dirs()
    if not acquire_lock():
        log.info("another scheduler holds the lock; exiting")
        return 0
    log.info("scheduler started pid=%s code=%s", os.getpid(), paths.REPO_ROOT)
    s = Scheduler()
    stop = {"flag": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))
    ticks = 0
    try:
        while not stop["flag"]:
            s.tick()
            ticks += 1
            if a.once or (a.max_ticks and ticks >= a.max_ticks):
                break
            for _ in range(int(q.control()["loop_seconds"])):
                if stop["flag"]:
                    break
                time.sleep(1)
        s.write_status(force=True)
    except Exception:  # noqa: BLE001
        log.error("scheduler crashed:\n%s", traceback.format_exc())
        return 1
    finally:
        release_lock()
    log.info("scheduler exiting (jobs keep running; a restart re-attaches)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
