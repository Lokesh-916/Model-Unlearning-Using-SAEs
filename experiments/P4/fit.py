"""P4 fit task (CPU): top-40 features + LLR weights per scored SAE, by E4a's fixed rule (experiments.E4a.fit.select /
llr, unchanged), and the drift unit delta.

L07 / L08 / L09 16k: from E4a's cached doc fire counts (the first 20 must equal E4a's fitted features and weights,
else STOP). L08 65k: from the docs65 counts. Writes $DSG_CACHE/P4/<exp>/fit/<tag>.json and fit/meta.json.
"""
import hashlib
import json

import numpy as np

from dsgx import paths
from dsgx.util import atomic_write_json

from experiments.E4a.common import CORPORA, SAES
from experiments.E4a.common import work_dir as e4a_dir
from experiments.E4a.fit import llr, select
from experiments.P4.common import E4A_EXP, FROZEN_FIT_SHA256, N_TOP, SAE16, SAE65, null_drift_unit, work_dir
from experiments.P4.docs65 import load_counts


def e4a_counts(tag):
    idx = [i for i, (t, _, _) in enumerate(SAES) if t == tag][0]
    out = {}
    for c in CORPORA:
        tot, nt = None, 0
        for f in sorted(e4a_dir(E4A_EXP, "docs", c).glob("chunk_*.npz")):
            with np.load(f) as p:
                fr = p["fire"][idx]
                tot = fr.copy() if tot is None else tot + fr
                nt += int(p["ntok"].sum())
        out[c] = (tot, nt)
    return out


def fit_one(tag, layer, sid, counts):
    p, qb, qw = (counts[c][0] / float(counts[c][1]) for c in ("forget", "benign", "wiki"))
    feats, score = select(p, qb, qw, k=N_TOP)
    w1, w0 = llr(p[feats], np.maximum(qb, qw)[feats])
    return {"tag": tag, "layer": layer, "sae_id": sid, "features": feats,
            "p_forget": [float(x) for x in p[feats]], "q_benign": [float(x) for x in qb[feats]],
            "q_wiki": [float(x) for x in qw[feats]], "w1": [float(x) for x in w1], "w0": [float(x) for x in w0],
            "n_eligible": int(np.isfinite(score).sum())}


def task(ctx):
    smoke = bool(ctx.args.get("smoke"))
    docs_exp = ctx.args.get("docs_exp", ctx.exp_id)
    fdir = work_dir(ctx.exp_id, "fit")
    frozen_path = paths.cache_dir() / "E4a" / E4A_EXP / "fit" / "L08.json"
    frozen_sha = hashlib.sha256(frozen_path.read_bytes()).hexdigest()
    if frozen_sha != FROZEN_FIT_SHA256:
        raise RuntimeError(f"STOP: frozen L08 fit sha256 {frozen_sha} != {FROZEN_FIT_SHA256}")
    checks = {}
    for tag, (layer, sid) in SAE16.items():
        rec = fit_one(tag, layer, sid, e4a_counts(tag))
        e4a = json.loads((paths.cache_dir() / "E4a" / E4A_EXP / "fit" / f"{tag}.json").read_text())
        same = rec["features"][:20] == e4a["features"] and rec["w1"][:20] == e4a["w1"] and rec["w0"][:20] == e4a["w0"]
        checks[tag] = {"first20_equal_e4a": bool(same), "n_eligible": rec["n_eligible"]}
        if not same:
            raise RuntimeError(f"STOP: {tag}: re-fitted first 20 features / weights differ from E4a's fit")
        atomic_write_json(fdir / f"{tag}.json", rec)
    counts = {}
    for c in CORPORA:
        lc = load_counts(docs_exp, c)
        counts[c] = (lc["fire65"], lc["ntok"])
    rec65 = fit_one(SAE65[0], SAE65[1], SAE65[2], counts)
    atomic_write_json(fdir / f"{SAE65[0]}.json", rec65)
    checks[SAE65[0]] = {"n_eligible": rec65["n_eligible"], "n_features": len(rec65["features"])}
    if len(rec65["features"]) < 20 and not smoke:
        raise RuntimeError("STOP: fewer than 20 eligible 65k features")
    l08 = json.loads((fdir / "L08.json").read_text())
    delta = null_drift_unit(l08, 20)
    meta = {"delta": delta, "frozen_fit_sha256": frozen_sha, "checks": checks, "n_top": N_TOP}
    atomic_write_json(fdir / "meta.json", meta)
    ctx.write_config(docs_exp=docs_exp)
    ctx.write_metrics(meta)
    ctx.finish({"view": "p4-fit", "forget": None})
    return meta
