"""The experiment catalogue (MASTER_PLAN section 5) and completeness status per experiment ID.

status: done | partial | failed | skipped | deferred | not-run
  done      every queued job DONE (or, without queue info, at least one finished run)
  partial   some jobs DONE, others waiting/running
  failed    at least one job FAILED/BLOCKED and the rest terminal
  skipped   no jobs queued (e.g. backlog items) and no runs
  deferred  MASTER_PLAN 5.8 items not run on the server yet
  moved     every queued job MOVED-TO-SERVER (results in the gpuws report: final_report --hardware gpuws)
"""
from collections import defaultdict

EXPERIMENTS = {
    # id: (name, priority, wave, part, run-dir exp ids)
    "A1": ("Unified baselines", "must", 1, "A", ["A1-dev", "A1-test"]),
    "A2": ("Open-ended QA / TOFU", "must", 5, "A", ["A2"]),
    "A3": ("Hard-negative bio utility", "must", 2, "A", ["A3"]),
    "A4": ("Knowledge depth (probes, logit lens)", "must", 2, "A", ["A4"]),
    "A5": ("Privacy / membership inference", "should", 5, "A", ["A5"]),
    "A6": ("Tampering (LoRA relearn, quant, steer, benign FT)", "must", 4, "A", ["A6"]),
    "A7": ("Generality on Gemma 3", "should", 5, "A", ["A7"]),
    "A8": ("Reporting standards / latency", "must", 6, "A", ["A8"]),
    "B1": ("Dilution", "must", 2, "B", ["B1"]),
    "B2": ("Decomposition", "must", 2, "B", ["B2"]),
    "B3": ("Cross-lingual and encoded", "must", 2, "B", ["B3"]),
    "B4": ("Black-box rewrite", "must", 2, "B", ["B4"]),
    "B5": ("White-box obfuscation", "should", 3, "B", ["B5"]),
    "B6": ("Generation-time leakage", "must", 2, "B", ["B6"]),
    "C1": ("Causal feature selection", "should", 3, "C", ["C1"]),
    "C2": ("Streaming and probe gates", "must", 3, "C", ["C2"]),
    "C3": ("Layer sweep", "should", 3, "C", ["C3"]),
    "C4": ("Transcoder gates", "stretch", 3, "C", ["C4"]),
    "C5": ("Intervention type", "should", 3, "C", ["C5"]),
    "C6": ("Two-level gate", "should", 3, "C", ["C6"]),
    "D1": ("UNDO distillation (local, LoRA)", "must", 5, "D", ["D1"]),
    "D2": ("Null-space edit", "must", 3, "D", ["D2"]),
    "D3": ("Baked-model audit", "should", 5, "D", ["D3"]),
    "N1": ("GuardBreak toolkit", "must", 6, "N", ["N1"]),
    "N2": ("OpenUnlearning adapter", "should", 6, "N", ["N2"]),
    "N3": ("Interactive demo", "should", 6, "N", ["N3"]),
    "N4": ("Red-team challenge app", "should", 6, "N", ["N4"]),
    "N5": ("Adaptive hardening loop", "should", 5, "N", ["N5"]),
    "N6": ("Conformal threshold", "must", 3, "N", ["N6"]),
    "N7": ("Multi-layer voting", "should", 3, "N", ["N7"]),
    "N8": ("Reasoning-trace gating", "stretch", 5, "N", ["N8"]),
    "N9": ("SAE-quality explanation", "must", 4, "N", ["N9"]),
    "N10": ("Audit cards", "should", 6, "N", ["N10"]),
    "T": ("Theory checks T1-T5", "must", 4, "T", ["T"]),
    "X1": ("Combination wave (pre-registered selection)", "must", 7, "X", ["X1"]),
}

DEFERRED = {
    "D1-full": ("Full 2B UNDO distillation", "server", ["D1-full"]),
    "A6-full": ("Full fine-tune relearning", "server", ["A6-full"]),
    "A2-tofu-full": ("Full TOFU fine-tune + DSG / best-gate eval", "server", ["A2-tofu-full"]),
    "A7-12b": ("Gemma 3 12B inference", "server", ["A7-12b"]),
    "A7-server": ("Gemma 3 1B / 4B on gpuws (A7 moved, + attacks)", "server", ["A7-1b", "A7-4b"]),
    "D1-v2": ("UNDO distillation v2 (alpha 0.05-0.2, 4000 steps) + relearning", "server", ["D1-v2", "A6-full-d1v2"]),
    "FP": ("DSG figure parity (clamp grid, data efficiency, static/dynamic, multi-topic, latency, TOFU highlights)",
           "server", ["FP-clamp", "FP-dataeff", "FP-static", "FP-multitopic", "FP-latency", "FP-highlight"]),
    "A5-muse": ("MUSE News/Books targets: VerbMem, KnowMem, PrivLeak", "server", ["A5-muse"]),
    "MT-Bench": ("MT-Bench with an open judge", "server", ["MTBench"]),
    "RMU-cluster": ("RMU trained on gpuws", "server", ["RMU-cluster-test"]),
    "Q2": ("Attribution graphs", "server", ["Q2-graphs"]),
}


def exp_of_job(job_exp: str) -> str | None:
    e = job_exp.removesuffix("-smoke")
    for k, v in EXPERIMENTS.items():
        if e in v[4]:
            return k
    return None


def completeness(runs, jobs: dict | None = None, states: dict | None = None) -> dict:
    """{exp_id: {status, name, priority, n_runs, jobs: {status: n}, failed_jobs}}."""
    from dsgx.queue import common as q

    n_runs = defaultdict(int)
    for r in runs:
        k = exp_of_job(r.exp)
        if k:
            n_runs[k] += 1
    jcount = defaultdict(lambda: defaultdict(int))
    failed = defaultdict(list)
    for jid, job in (jobs or {}).items():
        k = exp_of_job(job.get("exp_id", ""))
        if not k:
            continue
        s = (states or {}).get(jid, {}).get("status", "WAITING")
        jcount[k][s] += 1
        if s in (q.FAILED, q.BLOCKED):
            failed[k].append(jid)
    out = {}
    for k, (name, pri, wave, part, _) in EXPERIMENTS.items():
        jc = dict(jcount.get(k, {}))
        tot = sum(jc.values())
        if tot:
            if jc.get(q.MOVED, 0) == tot:
                st = "moved to gpuws"
            elif jc.get(q.DONE, 0) + jc.get(q.MOVED, 0) == tot:
                st = "done"
            elif failed.get(k) and all(s in q.TERMINAL for s in jc):
                st = "failed" if not jc.get(q.DONE) else "partial (failures)"
            elif jc.get(q.DONE, 0) or failed.get(k):
                st = "partial"
            else:
                st = "not-run"
            if jc.get(q.MOVED) and st != "moved to gpuws":
                st += f" ({jc[q.MOVED]} moved to gpuws)"
        else:
            st = "done" if n_runs.get(k) else "skipped"
        out[k] = {"name": name, "priority": pri, "wave": wave, "part": part, "status": st,
                  "n_runs": n_runs.get(k, 0), "jobs": jc, "failed_jobs": failed.get(k, [])}
    for k, (name, where, exps) in DEFERRED.items():
        have = sum(1 for r in runs if r.exp in exps)
        out[k] = {"name": name, "priority": "deferred", "wave": None, "part": where,
                  "status": "done (server)" if have else "deferred", "n_runs": have, "jobs": {}, "failed_jobs": []}
    return out
