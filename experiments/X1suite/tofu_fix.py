"""X1-suite TOFU: the lab A2 TOFU models (exp/A2 ef5eb17 fine-tunes, read via ckpt:A2/...) under DSG and under the X1
combined gate, on one machine in one run, with paired tests.

Metrics as experiments.A2.tofu.metrics (unchanged definitions, same helper): truth ratio on forget10 / retain,
retain answer probability, model utility, forget quality (KS p vs the retain-only model); plus the forget answer
probability. A condition may carry `gate: {...}` (+ calib, intervention): the MCQ `gated` method's hook (dsgx.methods.gates)
with features and LLR weights from the fine-tuned model's own TOFU cache (forget tofu-forget10, retain tofu-retain90),
threshold calibrated as in X1 (5% FPR on benign MMLU DEV prompts), evaluated over the full question + answer sequence like
the DSG hook in A2. Per-item values go to items.parquet (scores only); paired bootstrap differences per condition vs
`full+dsg` and vs `full`. TOFU authors are fictitious (no hazardous text)."""
import numpy as np

from experiments.A2.tofu import CHAT, _answer_logprob  # noqa: F401  (same scoring as the A2 run)

DEFAULT_CONDS = [
    {"tag": "retain-model", "weights": "ckpt:A2/tofu_retain"},
    {"tag": "full", "weights": "ckpt:A2/tofu_full"},
    {"tag": "full+dsg", "weights": "ckpt:A2/tofu_full", "dsg": True},
    {"tag": "full+gate-cusum", "weights": "ckpt:A2/tofu_full",
     "gate": {"type": "cusum", "n_features": 20, "retain_pct": 95},
     "calib": {"fpr": 0.05, "n_max": 1000, "source": "mmlu-dev"},
     "intervention": {"type": "clamp_all", "multiplier": 500}},
]


def metrics(ctx):
    import pandas as pd
    from datasets import load_dataset
    from scipy.stats import ks_2samp

    from dsgx.data import activation_cache as ac
    from dsgx.eval.stats import bootstrap_ci, paired_bootstrap
    from dsgx.methods import dsg, gates
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    a = ctx.args
    n = a.get("limit")
    fset = load_dataset("locuslab/TOFU", "forget10_perturbed", split="train")
    rset = load_dataset("locuslab/TOFU", "retain_perturbed", split="train")
    if n:
        fset, rset = fset.select(range(int(n))), rset.select(range(int(n)))
    conds = a.get("conditions", DEFAULT_CONDS)
    ctx.write_config(None, conditions=conds)
    ctx.progress.update(items_total=len(conds) * (2 * len(fset) + 2 * len(rset)), items_done=0, force=True)
    res, per = {}, {}
    for c in conds:
        b = get_bundle(weights=resolve_weights(c["weights"], ctx.exp_id))
        m = b.model
        m.reset_hooks()
        info = {}
        if c.get("dsg") or c.get("gate"):
            ac.build_cache(b, "tofu-forget10", "tofu-retain90", 0)
        if c.get("dsg"):
            cache = ac.ActivationCache(ac.build_cache(b, "tofu-forget10", "tofu-retain90", 0))
            feats = dsg.select_features(cache, int(a.get("n_features", 20)), float(a.get("retain_pct", 95)))
            tau = dsg.calibrate_tau(cache, feats, 95)
            m.add_hook(b.hook_name, dsg.DSGHook(b.sae, feats, 500, tau, faithful=True, record=False))
            info = {"features": feats, "tau": tau}
        elif c.get("gate"):
            gm = gates.Gated({"name": "gated", "case": "bio",
                              "gate": {"forget_corpus": "tofu-forget10", "retain_corpus": "tofu-retain90", **c["gate"]},
                              "calib": c.get("calib", {}), "intervention": c.get("intervention", {"type": "clamp_all"})},
                             b, 0)
            gm.hook.record = False
            gm.install()
            info = {"features": gm.gate.features, "threshold": gm.tau, "calib": gm.calib, "gate": c["gate"]}

        def logp(q, ans):
            v = _answer_logprob(m, q, ans)
            return v

        def truth_ratios(ds):
            out = []
            for x in ds:
                para = np.exp(logp(x["question"], x["paraphrased_answer"]))
                pert = np.mean([np.exp(logp(x["question"], p)) for p in x["perturbed_answer"]])
                out.append(pert / max(para, 1e-12))
                ctx.progress.advance(1)
            return np.array(out)

        def answer_probs(ds):
            out = []
            for x in ds:
                out.append(np.exp(logp(x["question"], x["answer"])))
                ctx.progress.advance(1)
            return np.array(out)

        tr_f, tr_r = truth_ratios(fset), truth_ratios(rset)
        ans_f, ans_r = answer_probs(fset), answer_probs(rset)
        m.reset_hooks()
        per[c["tag"]] = {"tr_forget": tr_f, "tr_retain": tr_r, "ans_forget": ans_f, "ans_retain": ans_r}
        util_parts = [float(ans_r.mean()), float(np.clip(1 - tr_r, 0, 1).mean())]
        res[c["tag"]] = {"truth_ratio_forget": bootstrap_ci(tr_f), "truth_ratio_retain": bootstrap_ci(tr_r),
                         "retain_answer_prob": bootstrap_ci(ans_r), "forget": {"answer_prob": bootstrap_ci(ans_f)},
                         "model_utility": float(len(util_parts) / sum(1 / max(u, 1e-9) for u in util_parts)),
                         **info}
    ref = per.get("retain-model", {}).get("tr_forget")
    for t, v in res.items():
        v["forget_quality_ks_p"] = float(ks_2samp(per[t]["tr_forget"], ref).pvalue) if ref is not None and t != "retain-model" else None
    paired = {}
    for t in res:
        for vs in ("full+dsg", "full"):
            if t in (vs, "retain-model") or vs not in per or t == "full" and vs == "full+dsg":
                continue
            paired[f"{t} vs {vs}"] = {k: paired_bootstrap(per[t][k], per[vs][k]) for k in per[t]}
    rows = [{"condition": t, "part": part, "item": i, "value": float(x)}
            for t, d in per.items() for part, arr in d.items() for i, x in enumerate(arr)]
    pd.DataFrame(rows).to_parquet(ctx.run_dir() / "items.parquet", index=False)
    ctx.write_metrics({"conditions": res, "paired": paired, "n_forget": len(fset), "n_retain": len(rset),
                       "models": "lab A2 TOFU fine-tunes (exp/A2 ef5eb17), read via ckpt:A2/"})
    ctx.finish({"view": "tofu-metrics", "forget": None})
    return {k: v.get("forget_quality_ks_p") for k, v in res.items()}
