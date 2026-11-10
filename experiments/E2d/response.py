"""Method `delta-response`: the base model with an E2d response delta applied to mlp.down_proj of layers 1-3, on
every input (forced-on, no gate, no SAE hook).

TransformerLens (from_pretrained_no_processing) stores blocks[i].mlp.W_out = HF down_proj.weight.T, in bf16, so the
patched weight bf16(W_out + delta.T) (fp64 add) equals the trained bf16 weights exactly (checked when the delta was saved).
install() patches in place and keeps the original tensors; remove() (run()'s finally) copies them back, so the shared
cached base model is unchanged for the next run of the worker. A second patch on an already patched model raises.

method config: {name: delta-response, module: experiments.E2d.response, response: R1..R5, forced_on: true}
"""
import torch

from dsgx.methods.registry import Method, register

from experiments.E2d.common import load_meta, response_dir

_PATCHED: dict = {}  # id(model) -> response id currently applied


def load_delta(exp_id: str, rid: str) -> dict:
    from safetensors.torch import load_file

    return load_file(str(response_dir(exp_id, rid) / "delta.safetensors"))


@register("delta-response")
class DeltaResponse(Method):
    def __init__(self, cfg, bundle, seed=0):
        super().__init__(cfg, bundle, seed)
        c = dict(cfg)
        self.rid = c["response"]
        self.exp = c.get("response_exp", "E2d")
        meta = load_meta(self.exp, self.rid)
        self.delta = load_delta(self.exp, self.rid)
        self.layers = sorted(int(k.split(".")[2]) for k in self.delta)
        from experiments.E2d.train import base_tensors

        self.base = base_tensors(list(self.delta))  # bf16 HF down_proj of the base checkpoint (CPU)
        self._orig = {}
        self.info = {"response": self.rid, "response_exp": self.exp, "layers": self.layers,
                     "delta_file_sha256": meta["delta_file_sha256"], "new_sha256": meta["new_sha256"],
                     "hp": meta.get("hp"), "intervention": "RMU response weights, forced-on"}

    def install(self):
        m = self.bundle.model
        m.reset_hooks()
        if id(m) in _PATCHED:
            raise RuntimeError(f"model already patched with {_PATCHED[id(m)]}")
        _PATCHED[id(m)] = self.rid
        try:  # run() calls remove() only after install() returned: restore here if patching fails midway
            with torch.no_grad():
                for i in self.layers:
                    W = m.blocks[i].mlp.W_out
                    d = self.delta[f"model.layers.{i}.mlp.down_proj.weight"].to(W.device).T
                    if tuple(d.shape) != tuple(W.shape):
                        raise ValueError(f"layer {i}: delta {tuple(d.shape)} vs W_out {tuple(W.shape)}")
                    b = self.base[f"model.layers.{i}.mlp.down_proj.weight"].to(W.device, W.dtype).T
                    if not torch.equal(W, b):
                        raise RuntimeError(f"layer {i}: W_out is not the base down_proj.T (mapping or stale patch)")
                    self._orig[i] = W.detach().clone()
                    W.copy_((W.double() + d.double()).to(W.dtype))
        except Exception:
            self.remove()
            raise

    def remove(self):
        m = self.bundle.model
        with torch.no_grad():
            for i, w in self._orig.items():
                m.blocks[i].mlp.W_out.copy_(w)
        self._orig = {}
        _PATCHED.pop(id(m), None)
        m.reset_hooks()
