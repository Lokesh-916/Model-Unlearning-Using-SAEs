"""The compact cache must reproduce DSG's sparsity accumulation and tau convention exactly."""
from types import SimpleNamespace

import numpy as np
import torch

from dsgx.data import activation_cache as ac
from dsgx.models.loader import Bundle
from test_dsg import ToySAE


class FakeModel:
    def __init__(self, V=50, D=8):
        g = torch.Generator().manual_seed(3)
        self.emb = torch.randn(V, D, generator=g)
        self.tokenizer = SimpleNamespace(pad_token_id=0, eos_token_id=1, bos_token_id=2)

    def run_with_cache(self, tokens, stop_at_layer=None, names_filter=None):
        return None, {names_filter: self.emb[tokens]}


def _bundle():
    return Bundle(FakeModel(), ToySAE(f=32), "toy", "toyrel", "l0", "hook", 0, "cpu", "float32")


def _tokens(n, L, seed):
    t = torch.randint(0, 50, (n, L), generator=torch.Generator().manual_seed(seed))
    t[:, 0] = 2
    return t


def test_cache_matches_dense_reference(tmp_path):
    b = _bundle()
    f, r = _tokens(6, 16, 0), _tokens(6, 16, 1)
    out = ac.build_from_tokens(b, f, r, {"toy": True}, tmp_path / "c", "toy", k=12, n_tok_sub=2)
    cache = ac.ActivationCache(out)
    for part, toks in (("forget", f), ("retain", r)):
        acts = b.sae.encode(b.model.emb[toks]).detach()  # [N, L, F]
        keep = ~((toks == 0) | (toks == 1) | (toks == 2))
        ref_sum = torch.zeros(32)
        for i in range(len(toks)):  # legacy: per-row sum in act dtype, then accumulate
            ref_sum += (acts[i:i + 1] * keep[i:i + 1, :, None]).sum(dim=(0, 1)).float()
        st = cache.stats(part)
        assert np.array_equal(st["legacy_mean"], (ref_sum / keep.sum()).numpy())
        assert np.allclose(st["fire_rate"], ((acts > 0) & keep[:, :, None]).sum((0, 1)).numpy() / keep.sum().item())
        assert np.allclose(st["max"], (acts * keep[:, :, None]).amax((0, 1)).numpy())
        assert st["hist"].sum() == ((acts > 0) & keep[:, :, None]).sum().item()
        # tau convention: position 0 zeroed, denominator = full row length
        feats = [int(x) for x in cache.candidates[:5]]
        a0 = acts.clone()
        a0[:, 0] = 0
        rho = ((a0[:, :, feats] > 0).any(2).sum(1) / toks.shape[1]).numpy()
        assert np.allclose(cache.seq_fire_rate(part, feats), rho)
        assert cache.tokacts(part).shape == (2, 16, 12)
    meta = cache.meta
    assert meta["status"] == "COMPLETE" and meta["k"] == 12


def test_candidates_cover_dsg_ranking():
    rng = np.random.default_rng(0)
    fm, rm = rng.random(500) * 1e-2, rng.random(500) * 1e-2
    from dsgx.methods.dsg import dsg_rank_features

    cand = ac.choose_candidates(fm, rm, k=100)
    ranked = dsg_rank_features(fm, rm, 90)
    assert len(cand) == 100 and len(set(cand)) == 100
    assert list(cand[: len(ranked)]) == list(ranked[:100])
    assert set(dsg_rank_features(fm, rm, 95)[:20]) <= set(cand)
