"""N6 split-conformal threshold (must): calibrate a gate threshold by split conformal on benign DEV
scores for target alpha in {0.01,0.05,0.1}, then report empirical FPR on a held-out benign set and on
a shifted-benign set (a different family of benign subjects). tau_alpha = the ceil((n+1)(1-alpha))-th
smallest benign score (standard split-conformal quantile)."""
import numpy as np

from dsgx.methods.gates import Gate, gate_scores


def conformal_tau(scores: np.ndarray, alpha: float) -> float:
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    s = np.sort(scores)
    return float(s[min(k, n) - 1])


def _subject_prompts(subjects, split, n, seed):
    from dsgx.data.mcq import format_prompt, load_mcq
    from dsgx.data.splits import get_split

    rng = np.random.default_rng(seed)
    out = []
    per = max(1, n // len(subjects))
    for s in subjects:
        ids = get_split(s, split)
        pick = rng.choice(ids, size=min(per, len(ids)), replace=False)
        it = load_mcq(s)
        out += [format_prompt(it[int(i)]) for i in pick]
    return out


def task(ctx):
    from dsgx.models.loader import get_bundle

    a = ctx.args
    case = a.get("case", "bio")
    gate = Gate({**a.get("gate", {"type": "rho", "n_features": 20, "retain_pct": 95}), "case": case},
                get_bundle(), ctx.seed)
    b = gate.bundle
    # benign calibration + held-out from utility subjects (disjoint id draws via split); shifted-benign
    # from a different, non-hazard family.
    util = a.get("calib_subjects", ["high_school_geography", "high_school_us_history", "human_aging",
                                    "sociology", "marketing", "management"])
    shifted = a.get("shift_subjects", ["elementary_mathematics", "high_school_mathematics", "formal_logic"])
    nce = int(a.get("n", 400))
    ctx.progress.update(items_total=3, items_done=0, force=True)
    calib = gate_scores(b, gate, _subject_prompts(util, "dev", nce, 0)); ctx.progress.advance(1)
    held = gate_scores(b, gate, _subject_prompts(util, "test", nce, 1)); ctx.progress.advance(1)
    shift = gate_scores(b, gate, _subject_prompts(shifted, "test", nce, 2)); ctx.progress.advance(1)
    res = {}
    for alpha in a.get("alphas", [0.01, 0.05, 0.1]):
        tau = conformal_tau(calib, alpha)
        res[str(alpha)] = {"tau": tau, "empirical_fpr_heldout": float((held > tau).mean()),
                           "empirical_fpr_shifted": float((shift > tau).mean()),
                           "calib_fpr": float((calib > tau).mean())}
    ctx.write_metrics({"gate": a.get("gate"), "n_calib": len(calib), "n_heldout": len(held),
                       "n_shifted": len(shift), "coverage": res})
    ctx.finish({"view": "conformal", "forget": None})
    return {k: v["empirical_fpr_heldout"] for k, v in res.items()}
