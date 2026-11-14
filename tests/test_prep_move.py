"""dsgx.queue.move: MOVED-TO-SERVER marking, dependency refusal, undo, doctor/status/scheduler view."""
from dsgx.queue import common as q
from dsgx.queue import doctor, move, status
from dsgx.queue.enqueue import write_job


def _job(jid, deps=(), wave=3):
    write_job({"id": jid, "exp_id": jid.split("-")[0], "kind": "task", "deps": list(deps), "est_minutes": 10,
               "kind_slot": "cpu", "wave": wave, "priority": "must"})


def test_move_refuses_stranded_dependent_then_moves_and_undoes():
    _job("C3-000"); _job("C3-auroc", ["C3-000"]); _job("N7-000")
    assert move.main(["--jobs", "C3-000", "--server-job", "c3", "--apply"]) == 1   # C3-auroc would wait forever
    assert q.load_state("C3-000")["status"] == q.WAITING
    assert move.main(["--jobs", "C3-*", "--server-job", "c3"]) == 0                # dry run writes nothing
    assert q.load_state("C3-000")["status"] == q.WAITING
    assert move.main(["--jobs", "C3-*", "--server-job", "c3", "--apply"]) == 0
    st = q.load_state("C3-auroc")
    assert st["status"] == q.MOVED and st["server_job"] == "c3" and st["moved_from_status"] == q.WAITING
    assert q.MOVED in q.TERMINAL
    jobs = q.load_jobs()
    states = {j: q.load_state(j) for j in jobs}
    assert doctor.moved_deadlocks(jobs, states) == []
    counts, pct, remaining = status.overall(status.collect()[2])
    assert counts[q.MOVED] == 2 and remaining == 10          # moved jobs leave the lab ETA
    assert "MOVED-TO-SERVER" in status.render()
    assert move.main(["--undo", "--jobs", "C3-*", "--apply"]) == 0
    assert q.load_state("C3-000")["status"] == q.WAITING
    assert move.main(["--jobs", "N7-000", "--server-job", "x", "--apply"]) == 0
    assert move.main(["--jobs", "N7-000", "--server-job", "x", "--apply"]) == 1    # already moved


def test_doctor_reports_moved_dependency(capsys):
    _job("A"); _job("B", ["A"])
    st = q.load_state("A"); st["status"] = q.MOVED; q.save_state("A", st)
    jobs = q.load_jobs()
    assert doctor.moved_deadlocks(jobs, {j: q.load_state(j) for j in jobs}) == [("B", "A")]
    doctor.main([])
    assert "moved-dependency" in capsys.readouterr().out
