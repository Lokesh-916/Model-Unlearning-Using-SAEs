"""E2f tofu-model task (GPU): rebuild the lab A2 TOFU 'full' model (LoRA r 32 / alpha 64 on all projections, 1000
steps, exp/A2-openqa ef5eb17) from its step-1000 LoRA tensors in $DSG_RESULTS/runs/A2/tofu-finetune-full/last/
trainer.pt. The merged copy ($DSG_CACHE/models/A2/tofu_full) was deleted in the session-33 disk cleanup; merging the
same LoRA state into the same base with peft's merge_and_unload (bf16, GPU) reproduces it. Output:
$DSG_CACHE/models/<exp>/tofu_full (loaded with weights 'ckpt:E2f/tofu_full'). TOFU authors are fictitious (safe)."""
import hashlib

import torch

from dsgx import paths


def task(ctx):
    from peft import LoraConfig, get_peft_model

    from dsgx.train.core import load_hf

    out = ctx.cache_dir("models", ctx.exp_id, "tofu_full")
    if (out / "E2F_REBUILT").exists():
        ctx.finish({"view": "tofu-rebuild", "forget": None, "reused": True})
        return {"reused": True}
    src = paths.runs_dir() / "A2" / "tofu-finetune-full" / "last" / "trainer.pt"
    st = torch.load(src, map_location="cpu", weights_only=False)
    assert st["step"] == 1000, st["step"]
    model, tok = load_hf(dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=32, lora_alpha=64, lora_dropout=0.0,
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                           task_type="CAUSAL_LM"))
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert names == set(st["params"]), (len(names), len(st["params"]))
    res = model.load_state_dict({k: v.to(torch.bfloat16) for k, v in st["params"].items()}, strict=False)
    assert not res.unexpected_keys, res.unexpected_keys[:3]
    h = hashlib.sha256()
    for k in sorted(st["params"]):
        h.update(st["params"][k].float().numpy().tobytes())
    merged = model.merge_and_unload()
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    (out / "E2F_REBUILT").write_text(src.as_posix())
    m = {"source": str(src), "step": st["step"], "n_lora_tensors": len(st["params"]), "lora_sha256": h.hexdigest(),
         "checkpoint": str(out)}
    ctx.write_metrics(m)
    ctx.finish({"view": "tofu-rebuild", "forget": None})
    return m
