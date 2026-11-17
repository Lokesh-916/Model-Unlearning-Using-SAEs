"""P1 fit task (CPU): E4a detector (a) only (SAE features frequent on the Cyber forget corpus, rare on benign computing
AND WikiText; top 20; Bernoulli LLR), per SAE, from the docs task's fire counts. -> $DSG_CACHE/E4a/<exp>/fit/<tag>.json"""
import numpy as np

from dsgx.util import atomic_write_json

from experiments.E4a.common import SAES, sha, work_dir
from experiments.E4a.docs import load_all
from experiments.E4a.fit import llr, select


def task(ctx):
    exp = ctx.args.get("docs_exp", ctx.exp_id)
    fdir = work_dir(ctx.exp_id, "fit")
    D = {c: load_all(exp, c) for c in ("forget", "benign", "wiki")}
    ntok = {c: float(D[c]["ntok"].sum()) for c in D}
    ctx.write_config(docs_exp=exp, n_docs={c: int(len(D[c]["ntok"])) for c in D}, ntok=ntok)
    out = {}
    for k, (tag, layer, sid) in enumerate(SAES):
        p, qb, qw = (D[c]["fire"][k] / ntok[c] for c in ("forget", "benign", "wiki"))
        feats, score = select(p, qb, qw)
        q = np.maximum(qb, qw)
        w1, w0 = llr(p[feats], q[feats])
        rec = {"tag": tag, "layer": layer, "sae_id": sid, "features": feats,
               "p_forget": [float(x) for x in p[feats]], "q_benign": [float(x) for x in qb[feats]],
               "q_wiki": [float(x) for x in qw[feats]], "w1": [float(x) for x in w1], "w0": [float(x) for x in w0],
               "n_eligible": int(np.isfinite(score).sum()), "fp": sha([exp, ntok, tag])}
        atomic_write_json(fdir / f"{tag}.json", rec)
        out[tag] = {"n_eligible": rec["n_eligible"], "n_features": len(feats),
                    "min_log_ratio": float(score[feats[-1]]) if feats else None}
    ctx.write_metrics(out)
    ctx.finish({"view": "p1-fit", "forget": None})
    return out
