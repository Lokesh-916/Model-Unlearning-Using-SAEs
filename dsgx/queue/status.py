"""Queue status (MASTER_PLAN 4.5).

    python -m dsgx.queue.status          # full table
    python -m dsgx.queue.status --chat   # <= 40 lines for the planning chat
"""
import argparse
import json
import time
from collections import defaultdict

from dsgx import paths
from dsgx.labels import reference_label
from dsgx.queue import common as q
from dsgx.queue import resources


def _fmt_dur(s):
    if s is None:
        return "-"
    s = int(max(0, s))
    if s >= 86400:
        return f"{s // 86400}d{(s % 86400) // 3600}h"
    if s >= 3600:
        return f"{s // 3600}h{(s % 3600) // 60:02d}m"
    return f"{s // 60}m{s % 60:02d}s"


def _ci(x):
    if not x or x.get("mean") is None:
        return "-"
    lo, hi = x.get("lo"), x.get("hi")
    ci = f" [{lo:.3f},{hi:.3f}]" if lo is not None else ""
    return f"{x['mean']:.4f}{ci} n={x.get('n')}"


def _run_dirs_for_job(job):
    jd = q.job_dir(job)
    out = []
    for line in q.tail(jd / "job.log", 10000):
        if line.startswith("[worker] run ") and "done: " in line:
            out.append(line.split("done: ", 1)[1].strip())
    return out


def _run_label(run_dir) -> str | None:
    try:
        cfg = json.loads(open(f"{run_dir}/config.json").read()).get("config")
    except (OSError, ValueError):
        return None
    return reference_label(cfg)


def headline_for_job(job) -> str | None:
    dirs = _run_dirs_for_job(job)
    if not dirs:
        return None
    try:
        d = json.loads(open(f"{dirs[-1]}/DONE").read())["headline"]
    except (OSError, ValueError, KeyError):
        return None
    label = _run_label(dirs[-1])
    head = f"forget {_ci(d.get('forget'))}; util {_ci(d.get('utility_pooled'))} ({d.get('view')})"
    return f"[{label}] {head}" if label else head


def collect():
    jobs = q.load_jobs()
    states = {j: q.load_state(j) for j in jobs}
    now = time.time()
    rows = []
    for jid, job in jobs.items():
        st = states[jid]
        prog = q.progress_of(job) if st["status"] == q.RUNNING else None
        rows.append({"id": jid, "job": job, "st": st, "prog": prog})
    return jobs, states, rows, now


def overall(rows):
    counts = defaultdict(int)
    tot_est = done_est = 0.0
    for r in rows:
        s = r["st"]["status"]
        counts[s] += 1
        if s == q.MOVED:
            continue  # runs on the server: not part of the lab ETA
        est = r["job"].get("est_minutes", 1)
        tot_est += est
        if s in (q.DONE, q.FAILED, q.BLOCKED):
            done_est += est
        elif s == q.RUNNING and r["prog"]:
            p = r["prog"]
            frac = (p.get("items_done") or 0) / p["items_total"] if p.get("items_total") else 0
            done_est += est * min(1.0, frac)
    remaining = tot_est - done_est
    return counts, (100 * done_est / tot_est if tot_est else 0.0), remaining


def render(chat: bool = False) -> str:
    jobs, states, rows, now = collect()
    counts, pct, remaining_min = overall(rows)
    ctl = q.control()
    gpu_par = 1  # conservative: ETA assumes serial GPU work
    L = []
    L.append(f"# DSG queue status  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    p = q.paused()
    if p:
        L.append(f"**PAUSED**: {p}")
    for a in q.alerts():
        L.append(f"**ALERT** {a.get('key')}: {a.get('msg')}")
    L.append("")
    L.append("## Overall")
    L.append(f"done {counts[q.DONE]} | running {counts[q.RUNNING]} | waiting {counts[q.WAITING]} | "
             f"failed {counts[q.FAILED]} | blocked {counts[q.BLOCKED]} | moved to server {counts[q.MOVED]} | total {len(rows)}")
    L.append(f"complete {pct:.1f}% (weighted by est. minutes) | ETA ~{_fmt_dur(remaining_min * 60 / gpu_par)}")
    L.append("")
    # per experiment
    per = defaultdict(list)
    for r in rows:
        per[r["job"]["exp_id"]].append(r)
    L.append("## Per experiment")
    L.append("| exp | done/total | % | status | latest headline |")
    L.append("|---|---|---|---|---|")
    for exp in sorted(per):
        rs = per[exp]
        d = sum(r["st"]["status"] == q.DONE for r in rs)
        sts = {r["st"]["status"] for r in rs}
        status = (q.FAILED if q.FAILED in sts else q.RUNNING if q.RUNNING in sts else
                  q.DONE if sts == {q.DONE} else q.MOVED if sts == {q.MOVED} else
                  q.BLOCKED if q.BLOCKED in sts else q.WAITING)
        if q.MOVED in sts and status != q.MOVED:
            status += f" ({sum(r['st']['status'] == q.MOVED for r in rs)} moved)"
        done_jobs = sorted([r for r in rs if r["st"]["status"] == q.DONE], key=lambda r: r["st"].get("end") or 0)
        head = headline_for_job(done_jobs[-1]["job"]) if done_jobs else None
        L.append(f"| {exp} | {d}/{len(rs)} | {100 * d / len(rs):.0f}% | {status} | {head or '-'} |")
    L.append("")
    run_rows = [r for r in rows if r["st"]["status"] == q.RUNNING]
    L.append("## Running")
    if not run_rows:
        L.append("(none)")
    else:
        L.append("| job | slot | items | % | elapsed | ETA | heartbeat age | phase | flags |")
        L.append("|---|---|---|---|---|---|---|---|---|")
        for r in run_rows:
            pr = r["prog"] or {}
            st = r["st"]
            dn, tt = pr.get("items_done") or 0, pr.get("items_total") or 0
            hb = now - pr["last_update"] if pr.get("last_update") else None
            L.append(f"| {r['id']} | {st.get('slot')} | {dn}/{tt} | {100 * dn / tt if tt else 0:.0f}% | "
                     f"{_fmt_dur(now - st.get('start', now))} | {_fmt_dur(pr.get('eta_seconds'))} | "
                     f"{_fmt_dur(hb)} | {pr.get('phase', '-')} | {','.join(f for f in st.get('flags', []) if f == 'OVERTIME') or '-'} |")
    L.append("")
    fails = [r for r in rows if r["st"]["status"] in (q.FAILED, q.BLOCKED)]
    L.append("## Failures")
    if not fails:
        L.append("(none)")
    for r in fails:
        st = r["st"]
        if st["status"] == q.BLOCKED:
            L.append(f"- {r['id']}: BLOCKED by {', '.join(st.get('blocked_by', []))}")
            continue
        L.append(f"- {r['id']}: FAILED (exit {st.get('exit_code')}, attempts {st.get('attempts')})")
        for line in (st.get("error_tail") or q.tail(q.job_dir(r['job']) / 'job.log', 3))[-3:]:
            L.append(f"    {line[:200]}")
    flagged = [r for r in rows if any(f.startswith("HUNG") or f.startswith("OOM") for f in r["st"].get("flags", []))]
    if flagged:
        L.append("")
        L.append("## Watchdog events")
        for r in flagged:
            L.append(f"- {r['id']} ({r['st']['status']}): " + "; ".join(f for f in r["st"]["flags"] if f != "OVERTIME"))
    L.append("")
    res = resources.snapshot()
    g = res["gpu"]
    L.append("## Resources")
    if g.get("ok"):
        L.append(f"GPU {g['used_gb']:.1f}/{g['total_gb']:.1f} GB, util {g['util_pct']:.0f}%, {g['temp_c']:.0f} C")
    else:
        L.append(f"GPU: unavailable ({g.get('error')})")
    L.append(f"RAM {res['ram']['used_gb']:.1f}/{res['ram']['total_gb']:.1f} GB ({res['ram']['percent']:.0f}%) | "
             f"disk free {res['disk']['free_gb']:.0f} GB")
    sl = q.qdir() / "scheduler.lock"
    pid = sl.read_text().strip() if sl.exists() else None
    L.append(f"scheduler: {'running pid ' + pid if pid and q.pid_alive(pid) else 'NOT RUNNING'}")
    if chat:
        return _compact(L)
    return "\n".join(L) + "\n"


def _compact(lines, limit=40) -> str:
    out = [l for l in lines if l.strip()]
    if len(out) > limit:
        out = out[: limit - 1] + [f"... ({len(out) - limit + 1} more lines in STATUS.md)"]
    return "\n".join(out) + "\n"


def _run_row(run_dir) -> tuple[str | None, str] | None:
    """(reference label, markdown row) for one finished run: both views with 95% CI and n."""
    try:
        m = json.loads(open(f"{run_dir}/metrics.json").read())
        c = json.loads(open(f"{run_dir}/config.json").read())
    except (OSError, ValueError):
        return None
    cfg = c.get("config") or {}
    views = m.get("views") or {}
    cells = []
    for view in ("raw", "dsg_subset"):
        v = views.get(view) or {}
        cells += [_ci(v.get("forget")), _ci((v.get("utility") or {}).get("pooled"))]
    name = str(run_dir).rstrip("/").rsplit("/", 1)[-1]
    row = (f"| {name} | {cfg.get('split', '-')} | {cfg.get('seed', '-')} | {m.get('batch_size', '-')} | "
           + " | ".join(cells) + " |")
    return reference_label(cfg), row


def wave_report(w: int) -> str:
    jobs, states, rows, _ = collect()
    L = [f"# Wave {w} finished", "", "## Jobs", "", "| job | status | headline (CI, n) |", "|---|---|---|"]
    main_rows, ref_rows = [], {}
    for r in rows:
        if r["job"].get("wave", 0) <= w:
            L.append(f"| {r['id']} | {r['st']['status']} | {headline_for_job(r['job']) or '-'} |")
            for d in _run_dirs_for_job(r["job"]):
                got = _run_row(d)
                if got:
                    (ref_rows.setdefault(got[0], []) if got[0] else main_rows).append(got[1])
    head = ["| run | split | seed | bs | raw forget | raw utility | DSG-subset forget | DSG-subset utility |",
            "|---|---|---|---|---|---|---|---|"]
    L += ["", "## Results per run (mean [95% bootstrap CI] n)", ""] + head + main_rows
    for label, rr in ref_rows.items():
        L += ["", f"## {label}: not a main comparison", ""] + head + rr
    L += ["", "Resume with: python -m dsgx.queue.resume"]
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chat", action="store_true")
    a = ap.parse_args(argv)
    q.ensure_dirs()
    print(render(chat=a.chat), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
