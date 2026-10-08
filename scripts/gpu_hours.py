#!/usr/bin/env python3
"""GPU hours per machine and experiment group, from the run logs (Responsible NLP checklist C1).

Lab PC (RTX 2000 Ada): attempts in $DSG_RESULTS/logs/scheduler.log ("started <job> on gpu0" ... "job <job> finished").
Only gpu-slot attempts count; cpu-slot jobs (tables, attack-success aggregation) use no GPU.
gpuws (RTX 6000 Ada): every Slurm log in dsg_results_cluster/logs/dsg-<job>-<id>.out; a requeued job has several
segments (one "=== dsg job" header each); a segment ends at Slurm's CANCELLED line, else at the log's mtime (rsync -a
keeps it).

Counted = the last successful attempt of every non-smoke job. Listed separately (not in the counted total): smoke
tests, failed attempts, superseded re-runs, attempts interrupted by power cuts (duration unknown, not summed).
Usage: python scripts/gpu_hours.py [--md]
"""
import collections, datetime as dt, glob, os, re, sys

P = os.path.expanduser("~/projects/mechunlearn-project")
LAB = os.environ.get("DSG_RESULTS", f"{P}/dsg_results")
SRV = f"{P}/dsg_results_cluster"
GROUPS = ["baselines", "attacks", "gates + StreamGuard", "baked erasure + relearning", "benchmarks"]


def group_lab(job):
    e = job.split("-")[0]
    if e in ("A1", "A3", "A4", "A5", "A8", "N9", "N10", "canary"):
        return "baselines"
    if e[0] == "B" or e in ("N1", "N4", "T"):
        return "attacks"
    if e[0] == "C" or e in ("X1", "N5", "N6", "N7"):
        return "gates + StreamGuard"
    if e[0] == "D" or e == "A6":
        return "baked erasure + relearning"
    if e in ("A2", "A7", "N2", "N3", "N8"):
        return "benchmarks"
    return "other"


def group_srv(job):
    if job.startswith(("validate", "rmu", "figs", "q2")):
        return "baselines"
    if job.startswith(("c3", "x1", "ph-union")):
        return "gates + StreamGuard"
    if job.startswith(("d1", "a6")):
        return "baked erasure + relearning"
    if job.startswith(("tofu", "muse", "mtbench", "a7")):
        return "benchmarks"
    return "other"


# gpuws jobs whose results were discarded (CLAUDE.md sessions 6, 13): Trainer accumulation bug (tofu-full v1/v2,
# MUSE v1, Q2 on that TOFU model) and the failed tofu-full eval. Failed/refused jobs are detected from the log tail.
SRV_SUPERSEDED = {84, 96, 113, 125, 128, 129, 130}
# 108/109: re-tries of 107's two failed jobs (107 itself finished 21 of 23 and is counted); 188: bad spec path.
SRV_FAILED = {108, 109, 188}
SRV_FAIL = re.compile(r"Traceback|Error|SANITY FAIL|partial: done \[\]|fine-tuning not finished")
# lab jobs whose (only) successful attempt was replaced by a differently named re-run (session 14)
LAB_SUPERSEDED = {"B2-attack-success", "B3-attack-success"}


def lab():
    rows, interrupted = [], []
    open_ = {}
    for line in open(f"{LAB}/logs/scheduler.log"):
        t = dt.datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
        if m := re.search(r"started (\S+) on (\S+) pid=\d+ attempt=(\d+)", line):
            if m[1] in open_:
                interrupted.append(m[1])
            open_[m[1]] = (t, m[2])
        elif m := re.search(r"job (\S+) finished rc=(-?\d+) -> (\S+)", line):
            if m[1] in open_:
                t0, slot = open_.pop(m[1])
                rows.append(dict(job=m[1], slot=slot, h=(t - t0).total_seconds() / 3600, rc=int(m[2]), end=t))
        elif "scheduler started" in line:
            interrupted += [j for j, (_, s) in open_.items() if s.startswith("gpu")]
            open_ = {}
    rows = [r for r in rows if r["slot"].startswith("gpu")]
    last_ok = {}
    for i, r in enumerate(rows):
        if r["rc"] == 0:
            last_ok[r["job"]] = i
    for i, r in enumerate(rows):
        if "smoke" in r["job"] or r["job"].startswith("watchdog"):
            r["cls"] = "smoke tests"
        elif r["rc"] != 0:
            r["cls"] = "failed / retried attempts"
        elif last_ok[r["job"]] != i or r["job"] in LAB_SUPERSEDED:
            r["cls"] = "superseded re-runs"
        else:
            r["cls"] = "counted"
        r["group"] = group_lab(r["job"])
    return rows, sorted(set(j for j in interrupted if "smoke" not in j))


def srv():
    rows = []
    for f in sorted(glob.glob(f"{SRV}/logs/dsg-*.out")):
        m = re.match(r"dsg-(.+)-(\d+)\.out$", os.path.basename(f))
        job, jid = m[1], int(m[2])
        text = open(f, errors="replace").read()
        segs = re.split(r"(?m)^(?==== dsg job)", text)
        segs = [s for s in segs if s.startswith("=== dsg job")]
        h = 0.0
        for k, s in enumerate(segs):
            t0 = dt.datetime.fromisoformat(re.search(r" at (\S+)$", s.splitlines()[0])[1]).replace(tzinfo=None)
            if c := re.search(r"CANCELLED AT (\S+) ", s):
                t1 = dt.datetime.fromisoformat(c[1])
            elif k == len(segs) - 1:
                t1 = dt.datetime.fromtimestamp(os.path.getmtime(f))
            else:
                t1 = t0
            h += max(0.0, (t1 - t0).total_seconds() / 3600)
        tail = "\n".join(text.strip().splitlines()[-3:])
        if jid in SRV_SUPERSEDED:
            cls = "superseded re-runs"
        elif jid in SRV_FAILED or SRV_FAIL.search(tail):
            cls = "failed / retried attempts"
        else:
            cls = "counted"
        rows.append(dict(job=f"{job}-{jid}", h=h, cls=cls, group=group_srv(job)))
    return rows


def table(rows_lab, rows_srv):
    def tot(rows, cls, g=None):
        return sum(r["h"] for r in rows if r["cls"] == cls and (g is None or r["group"] == g))
    out = ["| group | lab PC (RTX 2000 Ada) | gpuws (RTX 6000 Ada) | both |", "|---|---:|---:|---:|"]
    for g in GROUPS + ["other"]:
        a, b = tot(rows_lab, "counted", g), tot(rows_srv, "counted", g)
        if a or b or g != "other":
            out.append(f"| {g} | {a:.1f} | {b:.1f} | {a + b:.1f} |")
    a, b = tot(rows_lab, "counted"), tot(rows_srv, "counted")
    out.append(f"| **total (counted)** | **{a:.1f}** | **{b:.1f}** | **{a + b:.1f}** |")
    for cls in ("smoke tests", "failed / retried attempts", "superseded re-runs"):
        a, b = tot(rows_lab, cls), tot(rows_srv, cls)
        out.append(f"| excluded: {cls} | {a:.1f} | {b:.1f} | {a + b:.1f} |")
    al = sum(r["h"] for r in rows_lab); bl = sum(r["h"] for r in rows_srv)
    out.append(f"| all GPU time logged | {al:.1f} | {bl:.1f} | {al + bl:.1f} |")
    return "\n".join(out)


if __name__ == "__main__":
    rl, interrupted = lab()
    rs = srv()
    print(table(rl, rs))
    n = collections.Counter(r["cls"] for r in rl), collections.Counter(r["cls"] for r in rs)
    print(f"\nattempts: lab {dict(n[0])}; gpuws {dict(n[1])}")
    print(f"lab attempts interrupted by power cuts / restarts (no end time, not summed): {', '.join(interrupted)}")
    if "--md" in sys.argv:
        for r in sorted(rs, key=lambda r: r["job"]):
            print(f"  {r['job']:28s} {r['h']:6.2f} h  {r['cls']}  [{r['group']}]")
