"""X1-suite on gpuws (session 27): StreamGuard (X1 combined gate, CUSUM detector) vs DSG on the gpuws models.

The lab's six X1-suite jobs (exp/X1-suite b933e20: benign-open = A3 open-ended benign biology, leak = B6
leakage/gibberish, tofu-metrics = forget quality / model utility / truth ratio, tofu-qa-forget / -retain = streaming
TOFU QA, paired) run unchanged on gpuws through cluster/lab_jobs.py (group x1-suite, a replica: the lab keeps its
runs, labelled labpc). Their TOFU jobs read `ckpt:A2/tofu_full` and `ckpt:A2/tofu_retain`; on gpuws these are
the gpuws TOFU-full v3 models (train_version 2, same fine-tune code as A2-tofu-full):

  python cluster/x1suite_server.py models [--budget-min M]
    1. the retain-only reference model: cluster/tofu_full.finetune("retain") (deleted after tofu-full-v3 wrote its
       metrics; ~25 min). Resumable from its trainer checkpoint like tofu-full;
    2. links $DSG_CACHE/models/A2/{tofu_full,tofu_retain} -> ../A2-tofu-full/{full,retain};
    3. writes results/runs/X1-suite/MODELS.json (the X1-suite metrics.json text "lab A2 TOFU fine-tunes" is the
       job's fixed wording; on gpuws the models are these).
Exit 3 when the retain model is not finished (budget/disk): afterok then holds the X1-suite chain.
"""
import argparse
import os
from pathlib import Path

from cluster import jobcommon as jc
from cluster import tofu_full as tf
from dsgx import paths
from dsgx.util import atomic_write_json, now_iso

NAME = "x1-suite"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["models"])
    ap.add_argument("--budget-min", type=float, default=None)
    a = ap.parse_args(argv)
    full = paths.cache_dir() / "models" / "A2-tofu-full" / "full"
    if not tf.model_ok(full):
        raise SystemExit(f"{full}: no train_version {tf.TRAIN_VERSION} TOFU-full model on this machine")
    jc.require_gpu(40)
    tok = jc.load_tok()
    tok.padding_side = "right"
    targs = argparse.Namespace(epochs=tf.EPOCHS, lr=tf.LR, bs=tf.BS, accum=tf.ACCUM, maxlen=tf.MAXLEN)
    tf.NAME = NAME  # trainer state under results/jobs/x1-suite/ft-retain, log lines tagged x1-suite
    r90 = tf.tofu("retain90")
    if jc.TINY:
        r90 = r90[:8]
    retain = tf.finetune("retain", r90, targs, tok, jc.Budget(a.budget_min))
    if retain is None:
        jc.log(NAME, "retain model not finished (budget/disk); resubmit x1suite-models.sbatch")
        return 3
    link = paths.cache_dir() / "models" / "A2"
    link.mkdir(parents=True, exist_ok=True)
    for name, target in (("tofu_full", "full"), ("tofu_retain", "retain")):
        p = link / name
        if p.is_symlink() or p.exists():
            if p.resolve() != (link.parent / "A2-tofu-full" / target).resolve():
                raise SystemExit(f"{p} exists and is not the gpuws TOFU-full {target} model")
            continue
        os.symlink(Path("..") / "A2-tofu-full" / target, p)
    rec = {"time": now_iso(), "hardware_label": jc.hardware_label(),
           "note": "X1-suite TOFU jobs on gpuws read ckpt:A2/tofu_full and ckpt:A2/tofu_retain = the gpuws TOFU-full v3 "
                   "models (cluster/tofu_full.py fine-tune, train_version 2). The metrics.json 'models' text of "
                   "tofu_fix ('lab A2 TOFU fine-tunes') is the job's fixed wording and does not apply here.",
           "models": {n: {"path": str(paths.cache_dir() / "models" / "A2-tofu-full" / t),
                          "train_version": jc.read_json(paths.cache_dir() / "models" / "A2-tofu-full" / t / "train_version.json", {})}
                      for n, t in (("tofu_full", "full"), ("tofu_retain", "retain"))}}
    out = paths.runs_dir() / "X1-suite"
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out / "MODELS.json", rec)
    jc.log(NAME, "models ready: A2/tofu_full, A2/tofu_retain (gpuws TOFU-full v3)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
