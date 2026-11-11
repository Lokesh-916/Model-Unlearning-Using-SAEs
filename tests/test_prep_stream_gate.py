"""Streaming generation with a calibrated rho / window / cusum gate (gates.stream_gate + stream._GateHook token_fn):
the incremental score must equal the gate's full-sequence score (Gate.score convention) at every step."""
from types import SimpleNamespace

import pytest
import torch

from dsgx.gen.stream import _GateHook
from dsgx.methods import gates


class FakeSAE:
    def __init__(self, d=8, m=12, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.W = torch.randn(d, m, generator=g)

    def encode(self, x):
        return torch.relu(x @ self.W - 0.3)

    def decode(self, a):
        return a @ self.W.T


def _gate(kind, feats, thr):
    g = SimpleNamespace(type=kind, spec={"w": 3, "drift": 0.1}, threshold=thr, features=feats)
    g.w1 = torch.tensor([0.9, 1.4, 0.5])
    g.w0 = torch.tensor([-0.2, -0.1, -0.05])
    return g


def _full_score(kind, g, sae, resid):
    a = sae.encode(resid[0])
    a[0] = 0
    tgt = a[:, g.features] > 0
    n = resid.shape[1]
    if kind == "rho":
        return gates.score_rho(tgt.any(1), n)
    if kind == "window":
        return gates.score_window(tgt.any(1), n, 3)
    return gates.score_cusum(gates.token_llr(tgt, g.w1, g.w0), n, 0.1)[0]


@pytest.mark.parametrize("kind", ["rho", "window", "cusum"])
def test_incremental_score_matches_full(kind):
    sae = FakeSAE()
    feats = [1, 4, 7]
    resid = torch.randn(1, 14, 8, generator=torch.Generator().manual_seed(1))
    g = _gate(kind, feats, thr=0.2)
    f, thr, score_fn, token_fn = gates.stream_gate(g)
    hook = _GateHook(sae, f, 500.0, thr, score_fn, token_fn)
    hook(resid[:, :6])                    # full pass on the "prompt"
    rho, gate = hook.commit()
    assert rho == pytest.approx(_full_score(kind, g, sae, resid[:, :6]), abs=1e-5)
    for t in range(6, 14):                # one new position at a time (KV-cache path)
        hook(resid[:, t:t + 1])
        rho, gate = hook.commit()
        full = _full_score(kind, g, sae, resid[:, :t + 1])
        assert rho == pytest.approx(full, abs=1e-5)
        assert gate == (full > thr)


def test_clamp_positions_are_any_fire_for_cusum():
    sae = FakeSAE()
    g = _gate("cusum", [1, 4, 7], thr=-1.0)          # always on
    f, thr, score_fn, token_fn = gates.stream_gate(g)
    resid = torch.randn(1, 10, 8, generator=torch.Generator().manual_seed(2))
    out = _GateHook(sae, f, 500.0, thr, score_fn, token_fn)(resid.clone())
    a = sae.encode(resid)
    a[:, 0] = 0
    fire = (a[0][:, f] > 0).any(1)
    exp = a.clone()
    exp[0][:, f] = torch.where(fire[:, None], torch.full_like(a[0][:, f], -500.0), a[0][:, f])
    ref = sae.decode(exp) + (resid - sae.decode(a))
    assert torch.allclose(out, ref, atol=1e-4)


def test_stream_gate_requires_threshold():
    with pytest.raises(AssertionError):
        gates.stream_gate(_gate("cusum", [1], thr=None))


def test_model_tag_separates_smoke_checkpoints():
    from dsgx.data.activation_cache import model_tag

    def b(w):
        return SimpleNamespace(model_name="gemma-2-2b-it", meta={"weights": w})

    assert model_tag(b("/c/models/A2/tofu_full")) == "gemma-2-2b-it@tofu_full"            # unchanged key
    assert model_tag(b("/c/models/A2-smoke/tofu_full/")) == "gemma-2-2b-it@A2-smoke.tofu_full"
    assert model_tag(SimpleNamespace(model_name="gemma-2-2b-it", meta={})) == "gemma-2-2b-it"
