"""DSG-faithful must match main's code exactly; dsg-fixed must differ only where intended."""
import pickle
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dsgx.methods import dsg


class ToySAE(torch.nn.Module):
    def __init__(self, d=8, f=32, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W_enc = torch.nn.Parameter(torch.randn(d, f, generator=g))
        self.b_enc = torch.nn.Parameter(torch.randn(f, generator=g) * 0.5 - 0.3)
        self.W_dec = torch.nn.Parameter(torch.randn(f, d, generator=g) * 0.2)
        self.b_dec = torch.nn.Parameter(torch.randn(d, generator=g) * 0.1)

    def encode(self, x):
        return torch.relu(x @ self.W_enc + self.b_enc)

    def decode(self, a):
        return a @ self.W_dec + self.b_dec


@pytest.fixture
def toy():
    torch.manual_seed(0)
    return ToySAE(), torch.randn(3, 12, 8)


def test_faithful_hook_matches_legacy(toy):
    from evals.unlearning.utils.intervention import anthropic_clamp_resid_SAE_features as legacy

    sae, resid = toy
    feats = [3, 7, 11, 19, 25]
    for tau in (0.0, 0.3, 0.6, 1.1):
        ref = legacy(resid.clone(), None, sae, np.array(feats), 500, tau)
        new = dsg.DSGHook(sae, feats, 500, tau, faithful=True)(resid.clone())
        assert torch.equal(ref, new), tau


def test_hook_records_rho_and_gate(toy):
    sae, resid = toy
    feats = [3, 7, 11]
    h = dsg.DSGHook(sae, feats, 500, 0.4, faithful=True)
    h(resid.clone())
    recs = h.pop_records()
    acts = sae.encode(resid)
    acts[:, 0] = 0
    rho = ((acts[:, :, feats] > 0).any(2).sum(1) / 12).numpy()
    assert np.allclose([r["rho"] for r in recs], rho)
    assert [r["gate_fired"] for r in recs] == list(rho > 0.4)
    assert len(recs[0]["fire_trace"]) == 12 and len(recs[0]["top_other_ids"]) == 20


def test_gate_off_is_identity_up_to_sae_roundtrip(toy):
    sae, resid = toy
    out = dsg.DSGHook(sae, [3, 7], 500, 2.0, faithful=True)(resid.clone())
    assert torch.allclose(out, resid, atol=1e-5)
    out = dsg.DSGHook(sae, [3, 7], 500, 2.0, faithful=False)(resid.clone())
    assert torch.allclose(out, resid, atol=1e-5)


def test_fixed_variant_per_feature_mask_and_padding(toy):
    sae, resid = toy
    feats = [3, 7, 11, 19, 25]
    h = dsg.DSGHook(sae, feats, 500, -1.0, faithful=False, record=False)  # always active
    h.lengths = [12, 8, 5]
    out = h(resid.clone())
    acts = sae.encode(resid)
    acts[:, 0] = 0
    err = resid - sae.decode(acts)
    exp = acts.clone()
    valid = torch.zeros(3, 12, dtype=torch.bool)
    for b, n in enumerate([12, 8, 5]):
        valid[b, 1:n] = True
    t = acts[:, :, feats]
    m = (t > 0) & valid[:, :, None]
    exp[:, :, feats] = torch.where(m, torch.full_like(t, -500.0), t)
    assert torch.allclose(out, sae.decode(exp) + err, atol=1e-4)
    # rate excludes BOS and padding
    h2 = dsg.DSGHook(sae, feats, 500, 0.0, faithful=False)
    h2.lengths = [12, 8, 5]
    h2(resid.clone())
    r = h2.pop_records()
    fa = (t > 0).any(2) & valid
    assert np.allclose([x["rho"] for x in r], (fa.sum(1) / valid.sum(1)).numpy())


def test_txt_round_matches_savetxt(tmp_path):
    x = np.random.default_rng(0).random(500).astype(np.float32) * 1e-3
    np.savetxt(tmp_path / "s.txt", x, fmt="%f")
    assert np.array_equal(dsg.txt_round(x), np.loadtxt(tmp_path / "s.txt"))


class FakeCache:
    """Cache-like object over dense arrays (only what select_features / calibrate_tau need)."""

    def __init__(self, fm, rm, ret_acts):
        self._s = {"forget": {"legacy_mean": fm}, "retain": {"legacy_mean": rm}}
        self.ret = ret_acts

    def stats(self, part):
        return self._s[part]

    def seq_fire_rate(self, part, features, exclude_special=False):
        return (self.ret[:, :, features] > 0).any(2).sum(1) / self.ret.shape[1]


def test_selection_and_tau_match_legacy(tmp_path):
    from evals.unlearning.utils.feature_activation import get_top_features_percentile

    rng = np.random.default_rng(1)
    F, N, L = 200, 30, 40
    act_f = (rng.random((N, L, F)) < 0.05) * rng.random((N, L, F))
    act_r = (rng.random((N, L, F)) < rng.random(F) * 0.1) * rng.random((N, L, F))
    act_f[:, 0], act_r[:, 0] = 0, 0
    fm, rm = act_f.mean((0, 1)), act_r.mean((0, 1))
    # legacy reads the means back from '%f' text files and the dense activations from pickles
    np.savetxt(tmp_path / "f.txt", fm, fmt="%f")
    np.savetxt(tmp_path / "r.txt", rm, fmt="%f")
    with open(tmp_path / "act_fgt.pkl", "wb") as f:
        pickle.dump([a[None] for a in act_f.astype(np.float32)], f)
    with open(tmp_path / "act_ret.pkl", "wb") as f:
        pickle.dump([a[None] for a in act_r.astype(np.float32)], f)
    ns = [5, 10]
    sel, taus = get_top_features_percentile(np.loadtxt(tmp_path / "f.txt"), np.loadtxt(tmp_path / "r.txt"),
                                            ratio_percentile=90, folder_name=str(tmp_path), n_features_lst=ns)
    cache = FakeCache(fm, rm, act_r)
    for n in ns:
        mine = dsg.select_features(cache, n, 90)
        assert mine == [int(x) for x in sel[:n]]
        assert abs(dsg.calibrate_tau(cache, mine, 95) - taus[str(n)]) < 1e-12


def test_faithful_batched_equals_rowwise(toy):
    """Batched faithful hook == batch-size-1 faithful hook on every real (unpadded) position."""
    sae, resid = toy
    feats = [3, 7, 11, 19, 25]
    lengths = [12, 9, 6]
    for tau in (0.2, 0.5, 0.8):
        h = dsg.DSGHook(sae, feats, 500, tau, faithful=True)
        h.lengths = lengths
        out = h(resid.clone())
        recs = h.pop_records()
        for b, n in enumerate(lengths):
            h1 = dsg.DSGHook(sae, feats, 500, tau, faithful=True)
            ref = h1(resid[b:b + 1, :n].clone())
            r1 = h1.pop_records()[0]
            assert torch.allclose(out[b, :n], ref[0], atol=1e-5)
            assert abs(recs[b]["rho"] - r1["rho"]) < 1e-9 and recs[b]["gate_fired"] == r1["gate_fired"]
