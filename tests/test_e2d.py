"""E2d: selection rule, composite definition, delta patch/restore (CPU, no model downloads)."""
import types

import numpy as np
import torch

from experiments.E2d import common


def test_ladder():
    L = common.LADDER
    assert L["R2"]["steering_coeff"] == round(2 * L["R1"]["steering_coeff"], 2)
    assert L["R3"]["steering_coeff"] == round(4 * L["R1"]["steering_coeff"], 2)
    assert L["R4"]["max_num_batches"] == 300 and L["R4"]["steering_coeff"] == L["R2"]["steering_coeff"]
    assert L["R5"]["retain"] == "wikitext+benign_bio" and L["R5"]["max_num_batches"] == 150
    assert all(v["layer_ids"] == [1, 2, 3] and v["alpha"] == 300.0 for v in L.values())


def test_select_rule():
    c = [{"rid": "R1", "forget": 0.30, "utility": 0.56, "benign_bio": 0.60},
         {"rid": "R2", "forget": 0.25, "utility": 0.555, "benign_bio": 0.50},
         {"rid": "R3", "forget": 0.20, "utility": 0.50, "benign_bio": 0.70},   # below the floor
         {"rid": "R4", "forget": 0.25, "utility": 0.556, "benign_bio": 0.55}]  # ties R2 on forget, more benign bio
    assert common.select(c, 0.56)["rid"] == "R4"
    assert common.select(c, 0.60) is None
    assert common.select([dict(c[1]), {**c[1], "rid": "R5"}], 0.56)["rid"] == "R2"  # full tie -> lower index
    # floor is inclusive: utility exactly DSG - 0.005
    assert common.select([{"rid": "R1", "forget": 0.3, "utility": 0.555, "benign_bio": 0.5}], 0.56)["rid"] == "R1"


def test_composite_definition():
    base = np.array([0, 1, 2, 3, 0])
    resp = np.array([1, 1, 0, 0, 2])
    fired = np.array([True, False, True, False, True])
    gold = np.array([0, 1, 2, 0, 2])
    pred = np.where(fired, resp, base)
    assert pred.tolist() == [1, 1, 0, 3, 2]
    assert (pred == gold).tolist() == [False, True, False, False, True]


def test_delta_patch_restore(monkeypatch):
    from experiments.E2d import response

    d_mlp, d_model = 6, 4
    g = torch.Generator().manual_seed(0)
    base = {f"model.layers.{i}.mlp.down_proj.weight": torch.randn(d_model, d_mlp, generator=g).to(torch.bfloat16)
            for i in (1, 2)}
    new = {k: (v.float() + 0.01 * torch.randn(v.shape, generator=g)).to(torch.bfloat16) for k, v in base.items()}
    delta = {k: new[k].float() - base[k].float() for k in base}
    blocks = [types.SimpleNamespace(mlp=types.SimpleNamespace(W_out=torch.nn.Parameter(
        (base.get(f"model.layers.{i}.mlp.down_proj.weight", torch.zeros(d_model, d_mlp))).T.clone(),
        requires_grad=False))) for i in range(3)]
    model = types.SimpleNamespace(blocks=blocks, reset_hooks=lambda: None)
    monkeypatch.setattr(response, "load_meta", lambda e, r: {"delta_file_sha256": "x", "new_sha256": "y"})
    monkeypatch.setattr(response, "load_delta", lambda e, r: delta)
    import experiments.E2d.train as tr
    monkeypatch.setattr(tr, "base_tensors", lambda keys: {k: base[k] for k in keys})
    m = response.DeltaResponse({"response": "R1"}, types.SimpleNamespace(model=model))
    orig = [b.mlp.W_out.detach().clone() for b in blocks]
    m.install()
    for i in (1, 2):
        assert torch.equal(blocks[i].mlp.W_out, new[f"model.layers.{i}.mlp.down_proj.weight"].T)
    assert torch.equal(blocks[0].mlp.W_out, orig[0])
    m2 = response.DeltaResponse({"response": "R1"}, types.SimpleNamespace(model=model))
    try:
        m2.install()
        raise AssertionError("double patch must raise")
    except RuntimeError:
        pass
    m.remove()
    for b, o in zip(blocks, orig):
        assert torch.equal(b.mlp.W_out, o)
