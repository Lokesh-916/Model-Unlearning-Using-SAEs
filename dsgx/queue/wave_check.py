"""Scripted review of a finished wave before `python -m dsgx.queue.resume` (replaces the human check).

    python -m dsgx.queue.wave_check 1          # after WAVE1_DONE.md appears
    python -m dsgx.queue.wave_check 1 --smoke  # same checks on the smoke runs (for testing)

Checks (each PASS / WARN / FAIL):
  jobs        every job of waves <= w is DONE (FAIL if any FAILED/BLOCKED; run the doctor first)
  ci          every finished MCQ run has n > 0 and a 95% CI on forget and utility
  selection   every test run with a dev selection met its utility bound (WARN otherwise: the plan's
              fallback picked the closest config)
  utility     on TEST, each gated method's raw full-MMLU pooled utility is within 1 point of base
              of the same case (WARN otherwise; decision 4 tolerance)
  forget      each tuned DSG forget accuracy (raw, test) is below its base (FAIL if not: the guard
              does nothing; something is wrong)
  seeds       5 seeds per case for the headline DSG runs (WARN if fewer)
  canary      the last canary/sanity job is DONE (WARN if none, FAIL if failed)
  hardware    all runs carry one hardware label (FAIL if lab PC and gpuws runs are mixed)
Exit code 0 = no FAIL and finished (resume is fine), 1 = at least one FAIL (do not resume), 3 = not finished.
"""
import argparse
import sys
from collections import defaultdict

from dsgx.analysis.collect import fmt_ci, load_all
from dsgx.queue import common as q

WAVE_EXPS = {1: ["A1-dev", "A1-test"]}


def check(w: int, smoke: bool = False) -> list[tuple[str, str, str]]:
    out = []
    jobs = q.load_jobs()
    states = {j: q.load_state(j) for j in jobs}
    upto = [j for j in jobs if jobs[j].get("wave", 0) <= w and jobs[j].get("kind") != "sanity"
            and bool(jobs[j].get("smoke")) == smoke]
    bad = [j for j in upto if states[j]["status"] in (q.FAILED, q.BLOCKED)]
    notyet = [j for j in upto if states[j]["status"] not in q.TERMINAL]
    if bad:
        out.append(("FAIL", "jobs", f"{len(bad)} failed/blocked: {' '.join(bad[:8])} (python -m dsgx.queue.doctor)"))
    elif notyet:
        out.append(("WAIT", "jobs", f"{len(notyet)} job(s) of wave <= {w} not finished yet"))
    else:
        out.append(("PASS", "jobs", f"{len(upto)} job(s) of wave <= {w} DONE"))
    exps = sorted({jobs[j]["exp_id"].removesuffix("-smoke") for j in upto}) or WAVE_EXPS.get(w, [])
    runs = [r for r in load_all(smoke=smoke, exps=exps) if r.is_mcq]
    if not runs:
        out.append(("WARN", "ci", "no finished MCQ runs found for " + ", ".join(exps)))
        return out
    def _no_ci(ci):
        return bool((ci or {}).get("n")) and (ci or {}).get("lo") is None

    noci = [r.name for r in runs if _no_ci(r.forget("raw")) or _no_ci(r.utility("raw"))]
    out.append(("FAIL" if noci else "PASS", "ci", f"{len(noci)} run(s) without CI" if noci else f"{len(runs)} runs have n and CI"))
    sel = [r for r in runs if r.config.get("selection")]
    miss = [r.name for r in sel if not r.config["selection"]["info"].get("bound_met", True)]
    out.append(("WARN" if miss else "PASS", "selection",
                f"bound not met for {len(miss)}: {' '.join(miss[:4])}" if miss else f"{len(sel)} selected run(s) met the dev utility bound"))
    test = [r for r in runs if r.split == "test" and r.is_clean]
    bases = {r.case: r for r in test if r.is_base}
    groups = defaultdict(list)
    for r in test:
        if r.method not in ("base", None):
            groups[(r.case, r.label().split("/")[0], r.cfg["method"].get("retain_corpus"))].append(r)
    for (case, m, rc), rs in sorted(groups.items(), key=str):
        b = bases.get(case)
        tag = f"{case}/{m}" + (f"/{rc}" if rc else "")
        if not b:
            out.append(("WARN", "utility", f"{tag}: no base test run"))
            continue
        bu, bf = (b.utility("raw") or {}).get("mean"), (b.forget("raw") or {}).get("mean")
        us = [(r.utility("raw") or {}).get("mean") for r in rs]
        fs = [(r.forget("raw") or {}).get("mean") for r in rs]
        us, fs = [u for u in us if u is not None], [f for f in fs if f is not None]
        if bu is not None and us:
            drop = bu - sum(us) / len(us)
            out.append(("PASS" if drop <= 0.01 else "WARN", "utility", f"{tag}: mean test utility drop vs base {drop:+.4f} (limit 0.01)"))
        if bf is not None and fs:
            mf = sum(fs) / len(fs)
            out.append(("PASS" if mf < bf else "FAIL", "forget", f"{tag}: forget {mf:.4f} vs base {bf:.4f}"))
        seeds = sorted({r.seed for r in rs})
        want = 2 if smoke else 5
        out.append(("PASS" if len(seeds) >= want else "WARN", "seeds", f"{tag}: seeds {seeds}"))
    can = sorted([j for j in jobs if jobs[j].get("kind") == "sanity"], key=lambda j: jobs[j].get("created", 0))
    if not can:
        out.append(("WARN", "canary", "no canary job found"))
    else:
        s = states[can[-1]]["status"]
        out.append(({"DONE": "PASS", "FAILED": "FAIL"}.get(s, "WARN"), "canary", f"last canary {can[-1]}: {s}"))
    hw = {r.hardware for r in runs}
    out.append(("FAIL" if len(hw) > 1 else "PASS", "hardware", "labels: " + ", ".join(sorted(hw))))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("wave", type=int)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args(argv)
    res = check(a.wave, a.smoke)
    for lvl, k, msg in res:
        print(f"{lvl:<4}  {k:<9}  {msg}")
    fails = [r for r in res if r[0] == "FAIL"]
    wait = [r for r in res if r[0] == "WAIT"]
    print("\nVERDICT:", "do NOT resume (see NO_CLAUDE_RUNBOOK.md section T)" if fails
          else "wave not finished yet; check again later" if wait
          else "OK to resume: python -m dsgx.queue.resume")
    return 1 if fails else 3 if wait else 0


if __name__ == "__main__":
    sys.exit(main())
