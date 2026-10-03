"""dsgx.analysis.dashboard: renders a self-contained page from (possibly missing) inputs, nothing hazardous read."""
from dsgx.analysis import dashboard as db


def test_render_offline_inputs():
    lab = {"error": "queue not readable"}
    srv = {"error": "unreachable (ssh failed)"}
    page = db.render(lab, srv, [], {"labpc": None, "gpuws": None}, {}, "not refreshed", None)
    assert page.startswith("<title>DSG Project Board</title>") and "C-H7" in page and "no report" in page
    assert "http://" not in page and "<script" not in page          # self-contained, no scripts


def test_render_claims_and_jobs():
    lab = {"counts": {"DONE": 3, "MOVED-TO-SERVER": 2}, "pct": 50.0, "remaining_min": 90, "running": [],
           "failed": [{"id": "B1-000", "exp": "B1", "status": "FAILED"}], "moved": {"A6": 2}, "open_by_wave": {2: 1},
           "n_jobs": 6, "by_exp": {"A6": {"MOVED-TO-SERVER": 2}, "B1": {"FAILED": 1, "DONE": 3}}}
    srv = {"jobs": [{"id": "118", "name": "dsg-muse", "state": "PD", "time": "0:00", "limit": "3:00:00", "reason": "(Dependency)"}],
           "gpu": "1, 2, 3", "disk": {"size": "1T", "used": "1", "free": "90G", "pct": "9%"}, "ours": "59G"}
    sums = {"labpc": {"claims": [{"id": "C-H1", "verdict": "Supported", "evidence": "x"}]}, "gpuws": None}
    page = db.render(lab, srv, [], sums, {}, "ok", 15)
    assert "ON GPUWS" in page and "Supported" in page and "muse" in page and 'http-equiv="refresh" content="900"' in page
