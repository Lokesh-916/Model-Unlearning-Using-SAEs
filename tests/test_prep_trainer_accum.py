"""Trainer gradient accumulation: micro-batches backpropagated inside step_fn must reach the optimizer
(zero_grad runs before step_fn). Before the fix only the last micro-batch's gradient survived."""
import torch

from dsgx.train.core import Trainer


def _grad_after_one_step(tmp_path, accum):
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 1)
    x, y = torch.randn(4, 4), torch.randn(4, 1)
    tr = Trainer(model, model.parameters(), tmp_path / f"a{accum}", lr=1e-3, steps=1, ckpt_every=10,
                 grad_clip=1e9)
    seen = {}
    step = tr.opt.step

    def record(*a, **k):
        seen["g"] = model.weight.grad.detach().clone()
        return step(*a, **k)

    tr.opt.step = record

    def step_fn(_):
        n = len(x) // accum
        tot = 0.0
        for j in range(accum):
            loss = torch.nn.functional.mse_loss(model(x[j * n:(j + 1) * n]), y[j * n:(j + 1) * n]) / accum
            if j < accum - 1:
                loss.backward()
                tot = tot + loss.detach()
            else:
                tot = loss + tot
        return {"loss": tot}

    tr.run(step_fn)
    return seen["g"], tr.log[-1]["loss"]


def test_accumulated_grad_equals_full_batch(tmp_path):
    g1, l1 = _grad_after_one_step(tmp_path, 1)
    g4, l4 = _grad_after_one_step(tmp_path, 4)
    assert torch.allclose(g1, g4, atol=1e-6)
    assert abs(l1 - l4) < 1e-6
