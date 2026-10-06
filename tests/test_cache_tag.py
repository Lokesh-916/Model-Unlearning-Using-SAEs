"""Activation-cache model tag: a smoke checkpoint must not share the cache key of the real checkpoint."""
from types import SimpleNamespace

from dsgx.data.activation_cache import model_tag


def test_model_tag_separates_smoke_checkpoints():
    def b(w):
        return SimpleNamespace(model_name="gemma-2-2b-it", meta={"weights": w})

    assert model_tag(b("/c/models/A2/tofu_full")) == "gemma-2-2b-it@tofu_full"            # unchanged key
    assert model_tag(b("/c/models/A2-smoke/tofu_full/")) == "gemma-2-2b-it@A2-smoke.tofu_full"
    assert model_tag(SimpleNamespace(model_name="gemma-2-2b-it", meta={})) == "gemma-2-2b-it"
