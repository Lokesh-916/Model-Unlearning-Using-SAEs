"""Statistics (decision 6): bootstrap CIs over questions, paired bootstrap and McNemar."""
import math

import numpy as np

N_BOOT = 10_000


def _boot_means(x: np.ndarray, n_boot: int, rng, chunk: int = 500) -> np.ndarray:
    """Bootstrap means of x, resampled in chunks to bound memory."""
    n = len(x)
    out = np.empty(n_boot)
    for s in range(0, n_boot, chunk):
        m = min(chunk, n_boot - s)
        out[s:s + m] = x[rng.integers(0, n, size=(m, n))].mean(axis=1)
    return out


def bootstrap_ci(x, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0) -> dict:
    """Mean and percentile bootstrap CI over items."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    if n == 0:
        return {"mean": None, "lo": None, "hi": None, "n": 0}
    rng = np.random.default_rng(seed)
    means = _boot_means(x, n_boot, rng)
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"mean": float(x.mean()), "lo": float(lo), "hi": float(hi), "n": int(n)}


def stratified_bootstrap(groups: dict, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0) -> dict:
    """Pooled (by question count) and unweighted (mean of group means) accuracy with CIs.

    Resamples items within each group, so both estimators keep group sizes fixed.
    """
    groups = {k: np.asarray(v, dtype=np.float64) for k, v in groups.items() if len(v)}
    if not groups:
        return {"pooled": bootstrap_ci([]), "unweighted": bootstrap_ci([])}
    rng = np.random.default_rng(seed)
    sums = np.zeros(n_boot)
    gmeans = []
    for v in groups.values():
        bm = _boot_means(v, n_boot, rng)
        gmeans.append(bm)
        sums += bm * len(v)
    ntot = sum(len(v) for v in groups.values())
    pooled_b = sums / ntot
    unw_b = np.mean(gmeans, axis=0)
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    pooled = float(sum(v.sum() for v in groups.values()) / ntot)
    unw = float(np.mean([v.mean() for v in groups.values()]))
    pl, ph = np.percentile(pooled_b, q)
    ul, uh = np.percentile(unw_b, q)
    return {"pooled": {"mean": pooled, "lo": float(pl), "hi": float(ph), "n": int(ntot)},
            "unweighted": {"mean": unw, "lo": float(ul), "hi": float(uh), "n": int(ntot),
                           "n_groups": len(groups)}}


def paired_bootstrap(a, b, n_boot: int = N_BOOT, alpha: float = 0.05, seed: int = 0) -> dict:
    """Difference mean(a) - mean(b) on the same items: CI and two-sided bootstrap p-value."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    assert a.shape == b.shape
    d = a - b
    n = len(d)
    rng = np.random.default_rng(seed)
    bd = _boot_means(d, n_boot, rng) if n else np.zeros(1)
    lo, hi = np.percentile(bd, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    # p-value: how often the centred bootstrap distribution is at least as extreme as observed
    obs = d.mean()
    p = float((np.abs(bd - obs) >= abs(obs)).mean()) if n else 1.0
    return {"diff": float(obs), "lo": float(lo), "hi": float(hi), "p": p, "n": int(n)}


def mcnemar(a, b) -> dict:
    """Exact (binomial) McNemar test on paired binary outcomes."""
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)
    n01 = int((~a & b).sum())
    n10 = int((a & ~b).sum())
    n = n01 + n10
    if n == 0:
        return {"n01": 0, "n10": 0, "p": 1.0}
    k = min(n01, n10)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return {"n01": n01, "n10": n10, "p": float(min(1.0, 2 * p))}


def seed_summary(values) -> dict:
    """Mean and t-based 95% CI across seeds."""
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    if len(v) == 0:
        return {"mean": None, "n_seeds": 0}
    if len(v) == 1:
        return {"mean": float(v[0]), "lo": None, "hi": None, "n_seeds": 1}
    from scipy import stats

    m, se = v.mean(), v.std(ddof=1) / math.sqrt(len(v))
    h = se * stats.t.ppf(0.975, len(v) - 1)
    return {"mean": float(m), "lo": float(m - h), "hi": float(m + h), "sd": float(v.std(ddof=1)),
            "n_seeds": int(len(v))}
