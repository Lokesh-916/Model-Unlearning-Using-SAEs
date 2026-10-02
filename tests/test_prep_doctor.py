"""Queue doctor: failure classes and re-queue with BLOCKED dependents (fake queue, no GPU)."""
import time

from dsgx.queue import common as q
from dsgx.queue import doctor
from dsgx.queue.enqueue import write_job


def _job(jid, deps=()):
    write_job({"id": jid, "exp_id": "X", "kind": "task", "deps": list(deps), "est_minutes": 1,
               "kind_slot": "cpu", "wave": 1, "priority": "must"})


def _state(jid, **kw):
    st = q.load_state(jid)
    st.update(kw)
    q.save_state(jid, st)


def _log(jid, text):
    d = q.job_dir(q.load_job(jid))
    d.mkdir(parents=True, exist_ok=True)
    (d / "job.log").write_text(text)


def test_classes_and_requeue():
    for j, deps in [("a", []), ("b", ["a"]), ("c", ["b"]), ("d", []), ("e", []), ("f", []), ("g", [])]:
        _job(j, deps)
    now = time.time()
    _state("a", status=q.FAILED, exit_code=None, start=now - 10)            # reboot
    _state("b", status=q.BLOCKED, blocked_by=["a"])
    _state("c", status=q.BLOCKED, blocked_by=["b"])
    _state("d", status=q.FAILED, exit_code=1, start=now)
    _log("d", "Traceback (most recent call last):\nKeyError: 'n_features'\n")
    _state("e", status=q.FAILED, exit_code=4, start=now)
    _state("f", status=q.FAILED, exit_code=1, start=now)
    _log("f", "x\ntorch.OutOfMemoryError: CUDA out of memory. Tried\n")
    _state("g", status=q.FAILED, exit_code=1, start=now)
    _log("g", "RuntimeError: no finished dev runs of dsg-faithful for bio in A1-dev\n")
    jobs, states, bad = doctor.diagnose()
    cls = {b["job"]: b["class"] for b in bad}
    assert cls == {"a": "interrupted", "b": "blocked", "c": "blocked", "d": "code-error", "e": "leakage",
                   "f": "oom", "g": "missing-input"}
    assert doctor.dependents(["a"], jobs, states) == ["b", "c"]
    rc = doctor.main(["--apply"])
    assert rc == 2  # leakage present -> non-zero
    st = {j: q.load_state(j)["status"] for j in "abcdefg"}
    assert st["a"] == st["b"] == st["c"] == st["f"] == q.WAITING
    assert st["d"] == q.FAILED and st["e"] == q.FAILED and st["g"] == q.FAILED


def test_hung_once_vs_twice():
    _job("h1"); _job("h2")
    _state("h1", status=q.FAILED, exit_code=None, flags=["HUNG at x"], start=time.time())
    _state("h2", status=q.FAILED, exit_code=None, flags=["HUNG at x", "HUNG at y"], start=time.time())
    _, _, bad = doctor.diagnose()
    assert {b["job"]: b["class"] for b in bad} == {"h1": "hung", "h2": "hung-repeat"}
