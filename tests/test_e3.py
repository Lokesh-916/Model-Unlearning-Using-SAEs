"""E3 zero-ablation hook (CPU, fake SAE): unchanged residual without features, exact zeroing with error added back."""
import torch

from experiments.E3.ablate import ZeroAblateHook


class FakeSAE(torch.nn.Module):
    def __init__(self, d=6, f=10):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.W_enc = torch.randn(d, f, generator=g)
        self.W_dec = torch.randn(f, d, generator=g)

    def encode(self, x):
        return torch.relu(x @ self.W_enc)

    def decode(self, a):
        return a @ self.W_dec


def test_zero_ablation_changes_only_the_ablated_features():
    sae = FakeSAE()
    x = torch.randn(2, 5, 6, generator=torch.Generator().manual_seed(1))
    assert torch.allclose(ZeroAblateHook(sae, [])(x), x)
    feats = [1, 4, 7]
    y = ZeroAblateHook(sae, feats)(x)
    a = sae.encode(x)
    expect = x - a[:, :, feats] @ sae.W_dec[feats]  # removing exactly those features' decoder contributions
    assert torch.allclose(y[:, 1:], expect[:, 1:], atol=1e-5)
    assert torch.allclose(y[:, 0], x[:, 0], atol=1e-5)  # BOS untouched
