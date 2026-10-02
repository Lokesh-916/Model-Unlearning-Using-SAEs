"""Strict check of the cluster sanity run against the lab-PC targets (exact, not +/-1).

Targets: WMDP-Bio 158/538 = 0.2937, MMLU-u 0.9941, tau 0.5458. Exit 0 = exact match, 4 = not.
Writes $DSG_RESULTS/jobs/validate/validate.json (metrics only).
"""
import json
import os
import sys
from pathlib import Path

import torch

from dsgx import paths
from dsgx.util import atomic_write_json, gpu_info, now_iso, package_versions

res = json.loads((paths.results_dir() / "sanity" / "latest.json").read_text())
got = {"wmdp_correct": res["wmdp"]["correct"], "wmdp_n": res["wmdp"]["n"],
       "wmdp_acc": round(res["wmdp"]["acc"], 4), "mmlu_u": round(res["mmlu_u"], 4),
       "tau": round(res["tau"], 4), "tau_full": res["tau"], "util_correct": res["util_correct"],
       "features_match": res["features_match"]}
want = {"wmdp_correct": 158, "wmdp_n": 538, "wmdp_acc": 0.2937, "mmlu_u": 0.9941, "tau": 0.5458,
        "util_correct": res["target"]["util_correct"], "features_match": True}
checks = {k: got[k] == v for k, v in want.items()}
ok = all(checks.values())
out = {"time": now_iso(), "pass_exact": ok, "checks": checks, "got": got, "want": want,
       "sanity_pass_tolerant": res["pass"], "run_dir": res["run_dir"], "cache_check": res["cache_check"],
       "slurm_job": os.environ.get("SLURM_JOB_ID"), "hardware": gpu_info(), "versions": package_versions(),
       "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
atomic_write_json(paths.results_dir() / "jobs" / "validate" / "validate.json", out)
print(json.dumps({k: out[k] for k in ("pass_exact", "checks", "got")}, indent=1))
print("VALIDATE", "EXACT" if ok else "MISMATCH")
sys.exit(0 if ok else 4)
