"""PH-union (POST-HOC, EXPLORATORY): the union gate = DSG's rho gate OR the CUSUM gate, as one score.
Streaming score = full-sequence Gate.score at every step; the union fires exactly when rho > tau_DSG or
cusum > threshold; one conformal threshold on the union score bounds the union's benign FPR."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dsgx.gen.stream import _GateHook
from dsgx.methods import gates
from tests.test_prep_stream_gate import FakeSAE


def _union(feats, thr, tau):
    g = SimpleNamespace(type="union", spec={"drift": 0.1}, threshold=thr, features=feats, dsg_tau=tau,
                        feat_t=torch.tensor(feats))
    g.w1 = torch.tensor([0.9, 1.4, 0.5])
    g.w0 = torch.tensor([-0.2, -0.1, -0.05])
    return g


def _full(g, sae, resid):
    a = sae.encode(resid[0])
    a[0] = 0
    return gates.Gate.score(g, resid[0], a, resid.shape[1])


@pytest.mark.parametrize("tau", [0.0, 0.3, 2.0])
def test_union_stream_matches_full_and_or_rule(tau):
    sae = FakeSAE()
    resid = torch.randn(1, 14, 8, generator=torch.Generator().manual_seed(3))
    g = _union([1, 4, 7], thr=0.5, tau=tau)
    f, thr, score_fn, token_fn = gates.stream_gate(g)
    hook = _GateHook(sae, f, 500.0, thr, score_fn, token_fn)
    hook(resid[:, :5])
    hook.commit()
    for t in range(5, 14):
        hook(resid[:, t:t + 1])
        sc, on = hook.commit()
        full = _full(g, sae, resid[:, :t + 1])
        assert sc == pytest.approx(full["score"], abs=1e-5)
        assert on == ((full["rho"] > tau) or (full["cusum_max"] > thr))
        assert full["dsg_fired"] == (full["rho"] > tau)


def test_conformal_threshold_bounds_union_fpr():
    rng = np.random.default_rng(0)
    cus = rng.exponential(1.0, 2000)
    dsg_on = rng.random(2000) < 0.02                       # DSG's own benign FPR ~2 %
    s = np.where(dsg_on, gates.UNION_BIG, cus)
    thr = gates.conformal_threshold(s[:1000], 0.05)
    assert thr < gates.UNION_BIG
    assert (s[1000:] > thr).mean() <= 0.05 + 0.02          # held-out union FPR near alpha
    assert ((s[:1000] > thr) == (dsg_on[:1000] | (cus[:1000] > thr))).all()
