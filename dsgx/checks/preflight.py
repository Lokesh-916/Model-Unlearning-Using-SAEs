"""Preflight (MASTER_PLAN 4.6.1): run before launching the queue.

    python -m dsgx.checks.preflight [--skip-sanity] [--skip-tests]

Checks: unit tests, sanity gate, leakage, free disk >= 60 GB, no other GPU processes, every
worktree with queued jobs clean and on the job's recorded commit.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

from dsgx import paths
from dsgx.queue import common as q
from dsgx.queue import resources
from dsgx.util import atomic_write_json, now_iso


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-sanity", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args(argv)
    results = {}

    def rec(name, ok, detail=""):
        results[name] = {"ok": bool(ok), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

    if not a.skip_tests:
        r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=paths.REPO_ROOT,
                           capture_output=True, text=True)
        rec("unit tests", r.returncode == 0, (r.stdout.strip().splitlines() or ["?"])[-1])
    if not a.skip_sanity:
        r = subprocess.run([sys.executable, "-m", "dsgx.checks.sanity", "--force"], cwd=paths.REPO_ROOT,
                           capture_output=True, text=True)
        try:
            s = json.loads((paths.results_dir() / "sanity" / "latest.json").read_text())
            det = f"WMDP {s['wmdp']['correct']}/{s['wmdp']['n']}, MMLU-u {s['mmlu_u']:.4f}, tau {s['tau']:.4f}"
        except Exception:  # noqa: BLE001
            det = r.stdout[-300:]
        rec("sanity gate", r.returncode == 0, det)
    r = subprocess.run([sys.executable, "-m", "dsgx.checks.leakage", "--corpus"], cwd=paths.REPO_ROOT,
                       capture_output=True, text=True)
    rec("leakage", r.returncode == 0, (r.stdout.strip().splitlines() or ["?"])[-1])
    d = resources.disk()
    rec("free disk >= 60 GB", d["free_gb"] >= 60, f"{d['free_gb']:.0f} GB free")
    procs = resources.gpu_processes()
    rec("no other GPU processes", not procs, ", ".join(f"{p['pid']}:{p['name']}" for p in procs) or "none")
    bad = []
    for jid, job in q.load_jobs().items():
        if q.load_state(jid)["status"] != q.WAITING or not job.get("commit"):
            continue
        wt = job["worktree"]
        head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, capture_output=True, text=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], cwd=wt,
                               capture_output=True, text=True).stdout.strip()
        if head != job["commit"] or dirty:
            bad.append(f"{jid} ({Path(wt).name}: head {head[:8]} vs {job['commit'][:8]}{', dirty' if dirty else ''})")
    rec("worktrees clean and pinned", not bad, "; ".join(bad[:10]) or "all queued jobs match")
    ok = all(v["ok"] for v in results.values())
    atomic_write_json(paths.results_dir() / "checks" / "preflight_latest.json",
                      {"time": now_iso(), "pass": ok, "results": results})
    print("PREFLIGHT", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
