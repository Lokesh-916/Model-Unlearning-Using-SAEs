"""Model + SAE loading with in-process sharing.

An eval worker calls `get_bundle(...)` for every config it runs; the model and SAE are loaded
once per (model, dtype, device) and once per (release, sae_id), then reused. Hooks are reset
between configs by the methods themselves.
"""
from dataclasses import dataclass, field

import torch

_MODELS: dict = {}
_SAES: dict = {}
LOAD_COUNTS = {"model": 0, "sae": 0}  # inspected by tests and logged per job


@dataclass
class Bundle:
    model: object
    sae: object
    model_name: str
    sae_release: str
    sae_id: str
    hook_name: str
    layer: int
    device: str
    dtype: str
    meta: dict = field(default_factory=dict)

    @property
    def key(self):
        return (self.model_name, self.sae_release, self.sae_id, self.layer)


def _dtype(s: str):
    return {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[s]


def get_model(model_name: str = "gemma-2-2b-it", dtype: str = "bfloat16", device: str = "cuda",
              weights: str | None = None):
    """weights: optional HF checkpoint dir or hub id with the same architecture (trained / edited
    models: D1, D2, A6 relearned, RMU). Only one alternate-weights model is kept in memory."""
    key = (model_name, dtype, device, weights)
    if key not in _MODELS:
        from transformer_lens import HookedTransformer

        if weights:
            for k in [k for k in _MODELS if k[3]]:
                del _MODELS[k]
            import torch as _t
            from transformers import AutoModelForCausalLM

            hf = AutoModelForCausalLM.from_pretrained(weights, torch_dtype=_t.float32)
            _MODELS[key] = HookedTransformer.from_pretrained_no_processing(
                model_name, hf_model=hf, device=device, dtype=dtype)
            del hf
        else:
            # Same call as DSG's main.py (no weight processing).
            _MODELS[key] = HookedTransformer.from_pretrained_no_processing(
                model_name, device=device, dtype=dtype)
        LOAD_COUNTS["model"] += 1
    m = _MODELS[key]
    m.reset_hooks()
    return m


def get_sae(release: str = "gemma-scope-2b-pt-res", sae_id: str = "layer_3/width_16k/average_l0_142",
            dtype: str = "bfloat16", device: str = "cuda"):
    key = (release, sae_id, dtype, device)
    if key not in _SAES:
        import warnings

        from sae_lens import SAE

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sae, _, _ = SAE.from_pretrained(release=release, sae_id=sae_id, device=device)
        # DSG casts the SAE to the LLM dtype (main.py: sae.to(device, dtype=llm_dtype)).
        _SAES[key] = sae.to(device=device, dtype=_dtype(dtype))
        LOAD_COUNTS["sae"] += 1
    return _SAES[key]


def get_bundle(model_name: str = "gemma-2-2b-it", sae_release: str = "gemma-scope-2b-pt-res",
               sae_id: str = "layer_3/width_16k/average_l0_142", dtype: str = "bfloat16",
               device: str = "cuda", weights: str | None = None) -> Bundle:
    torch.set_grad_enabled(False)
    model = get_model(model_name, dtype, device, weights)
    sae = get_sae(sae_release, sae_id, dtype, device)
    hook_name = sae.cfg.metadata.hook_name
    layer = int(hook_name.split(".")[1])
    return Bundle(model, sae, model_name, sae_release, sae_id, hook_name, layer, device, dtype,
                  meta={"weights": weights})


def clear():
    _MODELS.clear()
    _SAES.clear()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
