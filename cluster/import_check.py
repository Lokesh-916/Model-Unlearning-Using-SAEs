"""CPU-only import check for the server scripts (no GPU, no data, no model loading).

    CUDA_VISIBLE_DEVICES= python cluster/import_check.py     # run from ~/dsg_cluster/code after env.sh

Loads each script as a module (its main() is not run) plus the dsgx modules they import lazily.
"""
import importlib
import importlib.util
import sys
from pathlib import Path

SCRIPTS = ["cluster/validate_check.py", "cluster/rmu_train.py", "cluster/rmu_eval.py", "cluster/rmu_v2_train.py", "cluster/rmu_v2_eval.py", "cluster/jobcommon.py",
           "cluster/d1_full.py", "cluster/a6_full.py", "cluster/tofu_full.py", "cluster/a7_server.py", "cluster/muse.py",
           "cluster/mtbench_open.py", "cluster/lab_jobs.py"]  # q2_graphs.py needs the overlay venv env/q2 (checked by its --plan)
LAZY = ["dsgx.gen.stream", "dsgx.checks.leakage", "dsgx.checks.sanity", "dsgx.data.corpora", "dsgx.data.mcq", "dsgx.data.splits", "dsgx.models.loader",
        "dsgx.run", "dsgx.train.core", "dsgx.eval.stats", "dsgx.labels", "dsgx.logging.run_logger", "dsgx.queue.worker",
        "dsgx.methods.registry", "dsgx.methods.gates", "dsgx.data.activation_cache"]
ok = True
for m in LAZY:
    try:
        importlib.import_module(m)
        print("ok  ", m)
    except Exception as e:
        ok = False
        print("FAIL", m, type(e).__name__, e)
for s in SCRIPTS:
    if s.endswith("validate_check.py"):
        # top level reads results/sanity/latest.json; check its imports only
        src = [ln for ln in Path(s).read_text().splitlines() if ln.startswith(("import ", "from "))]
        try:
            exec("\n".join(src), {})
            print("ok  ", s, "(imports)")
        except Exception as e:
            ok = False
            print("FAIL", s, type(e).__name__, e)
        continue
    try:
        spec = importlib.util.spec_from_file_location(Path(s).stem, s)
        spec.loader.exec_module(importlib.util.module_from_spec(spec))
        print("ok  ", s)
    except Exception as e:
        ok = False
        print("FAIL", s, type(e).__name__, e)
print("IMPORTS", "OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
