"""P3 fit task (CPU): E4a's fixed selection rule on document subsets.

For every (percent, seed): forget and benign-biology document subsets (common.subset), token-fire rates from the
per-document L8 counts (docs task), WikiText rates from E4a's cached L08 chunk counts (whole corpus), then
experiments.E4a.fit.select / llr unchanged (top 20). Writes $DSG_CACHE/P3/<exp>/fit/<name>.json and fit/index.json.
STOP: the 100 % fits (every seed) must equal E4a/E4a/fit/L08.json (features, w1, w0) exactly; the frozen file's sha256
must be unchanged.
"""
import hashlib
import json

import numpy as np

from dsgx import paths
from dsgx.util import atomic_write_json

from experiments.E4a.common import N_FEATURES, SAES
from experiments.E4a.common import work_dir as e4a_dir
from experiments.E4a.fit import llr, select
from experiments.P3.common import E4A_EXP, FROZEN_FIT_SHA256, LAYER, PCTS, SAE_ID, SEEDS, fit_name, subset, work_dir
from experiments.P3.docs import load_docs

L08_IDX = [i for i, (t, _, _) in enumerate(SAES) if t == "L08"][0]


def wiki_counts():
    tot, nt = None, 0
    for f in sorted(e4a_dir(E4A_EXP, "docs", "wiki").glob("chunk_*.npz")):
        with np.load(f) as p:
            fr = p["fire"][L08_IDX]
            tot = fr.copy() if tot is None else tot + fr
            nt += int(p["ntok"].sum())
    return tot, nt


def fit_subset(docs, wiki, pct, seed, k=N_FEATURES):
    rates, sizes = {}, {}
    for c, (cnt, nt) in docs.items():
        idx = subset(len(nt), pct, seed, c)
        rates[c] = cnt[idx].sum(0, dtype=np.int64) / float(nt[idx].sum())
        sizes[c] = int(len(idx))
    qw = wiki[0] / float(wiki[1])
    p, qb = rates["forget"], rates["benign"]
    feats, score = select(p, qb, qw, k=k)
    w1, w0 = llr(p[feats], np.maximum(qb, qw)[feats])
    return {"name": fit_name(pct, seed), "pct": pct, "seed": seed, "layer": LAYER, "sae_id": SAE_ID,
            "n_docs": sizes, "features": feats, "p_forget": [float(x) for x in p[feats]],
            "q_benign": [float(x) for x in qb[feats]], "q_wiki": [float(x) for x in qw[feats]],
            "w1": [float(x) for x in w1], "w0": [float(x) for x in w0], "n_eligible": int(np.isfinite(score).sum())}


def task(ctx):
    smoke = bool(ctx.args.get("smoke"))
    docs_exp = ctx.args.get("docs_exp", ctx.exp_id)
    frozen_path = paths.cache_dir() / "E4a" / E4A_EXP / "fit" / "L08.json"
    raw = frozen_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != FROZEN_FIT_SHA256:
        raise RuntimeError("STOP: frozen L08 fit sha256 changed")
    frozen = json.loads(raw)
    docs = {c: load_docs(docs_exp, c) for c in ("forget", "benign")}
    wiki = wiki_counts()
    fdir = work_dir(ctx.exp_id, "fit")
    index, checks = {}, {}
    for pct in PCTS:
        for seed in SEEDS:
            rec = fit_subset(docs, wiki, pct, seed)
            if len(rec["features"]) < N_FEATURES and not smoke:
                raise RuntimeError(f"STOP: {rec['name']}: fewer than {N_FEATURES} eligible features")
            if pct == 100:
                same = (rec["features"] == frozen["features"] and rec["w1"] == frozen["w1"] and rec["w0"] == frozen["w0"])
                checks[rec["name"]] = bool(same)
                if not same and not smoke:
                    raise RuntimeError(f"STOP: {rec['name']} does not reproduce frozen v2's fit (features / weights)")
            atomic_write_json(fdir / f"{rec['name']}.json", rec)
            index[rec["name"]] = {"pct": pct, "seed": seed, "n_docs": rec["n_docs"], "features": rec["features"],
                                  "overlap_frozen": len(set(rec["features"]) & set(frozen["features"]))}
    union = sorted({f for v in index.values() for f in v["features"]})
    meta = {"fits": index, "frozen_reproduced": checks, "n_union_features": len(union),
            "n_docs_total": {c: int(len(d[1])) for c, d in docs.items()}, "frozen_fit_sha256": FROZEN_FIT_SHA256}
    atomic_write_json(fdir / "index.json", meta)
    ctx.write_config(docs_exp=docs_exp)
    ctx.write_metrics({k: v for k, v in meta.items() if k != "fits"} | {"overlap": {k: v["overlap_frozen"] for k, v in index.items()}})
    ctx.finish({"view": "p3-fit", "forget": None})
    return meta
