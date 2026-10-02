"""Strict check of the cluster sanity run against the sanity-gpuws targets (exact, not +/-1).

Targets ($DSG_SANITY_TARGET=gpuws, dsgx.checks.sanity.TARGETS): WMDP-Bio 161/538 = 0.2993,
MMLU-u 0.9971 (108/9/103/84), tau 0.5458, the 20 legacy features. An exact match also confirms
same-GPU determinism. Exit 0 = exact match, 4 = not.
Writes $DSG_RESULTS/jobs/validate/validate.json (metrics only).
"""
import json
import os
import sys
from pathlib import Path

import torch

from dsgx import paths
from dsgx.checks.sanity import TARGET, TARGET_NAME
from dsgx.util import atomic_write_json, gpu_info, now_iso, package_versions

res = json.loads((paths.results_dir() / "sanity" / "latest.json").read_text())
got = {"wmdp_correct": res["wmdp"]["correct"], "wmdp_n": res["wmdp"]["n"],
       "wmdp_acc": round(res["wmdp"]["acc"], 4), "mmlu_u": round(res["mmlu_u"], 4),
       "tau": round(res["tau"], 4), "tau_full": res["tau"], "util_correct": res["util_correct"],
       "features_match": res["features_match"]}
want = {k: TARGET[k] for k in ("wmdp_correct", "wmdp_n", "wmdp_acc", "mmlu_u", "tau", "util_correct")}
want["features_match"] = True
checks = {k: got[k] == v for k, v in want.items()}
ok = all(checks.values())
out = {"time": now_iso(), "target_name": f"sanity-{TARGET_NAME}", "pass_exact": ok, "checks": checks, "got": got, "want": want,
       "sanity_pass_tolerant": res["pass"], "run_dir": res["run_dir"], "cache_check": res["cache_check"],
       "slurm_job": os.environ.get("SLURM_JOB_ID"), "hardware": gpu_info(), "versions": package_versions(),
       "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
atomic_write_json(paths.results_dir() / "jobs" / "validate" / "validate.json", out)
print(json.dumps({k: out[k] for k in ("pass_exact", "checks", "got")}, indent=1))
print(f"VALIDATE sanity-{TARGET_NAME}", "EXACT" if ok else "MISMATCH")
sys.exit(0 if ok else 4)
