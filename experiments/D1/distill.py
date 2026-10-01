"""D1-local - UNDO-style distillation into a NOISED LoRA student (16 GB fallback of the plan's
full-parameter UNDO; the full 2B version is DEFERRED). Student = gemma-2-2b-it + LoRA with a noised
initialisation (noise alpha). Loss = KL(student || guarded teacher) on forget prompts + KL(student ||
unguarded teacher) on retain prompts. Teachers are the same base weights with the DSG hook on/off
(LoRA adapters disabled). Saves a merged checkpoint for eval + A6. Compared with lora-student-same-ref."""
import json

import torch
import torch.nn.functional as F

from dsgx.data import activation_cache as ac
from dsgx.data.mcq import FORGET_DATASET, format_prompt, load_mcq
from dsgx.data.splits import get_split
from dsgx.methods import dsg
from dsgx.models.loader import get_bundle
from dsgx.train.core import Trainer, add_dsg_hook_hf, load_hf


def _kl(student_logits, teacher_logits, last_only=True):
    if last_only:
        s, t = student_logits[:, -1], teacher_logits[:, -1]
    else:
        s, t = student_logits, teacher_logits
    return F.kl_div(F.log_softmax(s.float(), -1), F.softmax(t.float(), -1), reduction="batchmean")


def task(ctx):
    from peft import LoraConfig, get_peft_model

    a = ctx.args
    case = a.get("case", "bio")
    alpha = float(a.get("noise_alpha", 0.3))
    rank = int(a.get("rank", 16))
    steps = int(a.get("steps", 200))
    bs = int(a.get("bs", 2))
    same_ref = bool(a.get("same_ref", False))  # lora-student-same-ref baseline: no DSG teacher
    b = get_bundle()
    cache = ac.ActivationCache(ac.build_cache(b, f"{case}-forget-corpus", "wikitext", 0))
    feats = dsg.select_features(cache, 20, 95)
    tau = dsg.calibrate_tau(cache, feats, 95)
    model, tok = load_hf(dtype=torch.bfloat16)
    teach_hook, _ = add_dsg_hook_hf(model, b.sae, feats, 500, tau, b.layer)
    teach_hook.enabled = False
    lcfg = LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0, task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    with torch.no_grad():  # noised LoRA init
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.add_(torch.randn_like(p) * alpha * 0.02)
    fd = FORGET_DATASET[case]
    forget = [format_prompt(load_mcq(fd)[i]) for i in get_split(fd, "dev")[:400]]
    retain = [format_prompt(load_mcq("high_school_geography")[i]) for i in get_split("high_school_geography", "dev")]
    tok.padding_side = "right"
    g = torch.Generator().manual_seed(ctx.seed)

    def batch(src, n):
        idx = torch.randint(0, len(src), (n,), generator=g).tolist()
        return tok([src[i] for i in idx], return_tensors="pt", padding=True, truncation=True,
                   max_length=512, add_special_tokens=False).to(model.device)

    def step_fn(step):
        fb, rb = batch(forget, bs), batch(retain, bs)
        with torch.no_grad():
            model.disable_adapter_layers()
            teach_hook.enabled = not same_ref
            tf = model(**fb).logits
            teach_hook.enabled = False
            tr = model(**rb).logits
            model.enable_adapter_layers()
        sf = model(**fb).logits
        sr = model(**rb).logits
        loss = _kl(sf, tf) + _kl(sr, tr)
        return {"loss": loss, "kl_forget": _kl(sf, tf).detach(), "kl_retain": _kl(sr, tr).detach()}

    out = ctx.run_dir()
    tr = Trainer(model, model.parameters(), out, lr=float(a.get("lr", 2e-4)), steps=steps,
                 ckpt_every=max(10, steps // 4), progress=ctx.progress, optimizer="adamw")
    ctx.progress.update(steps_total=steps, force=True)
    tr.run(step_fn)
    merged = model.merge_and_unload()
    ck = ctx.cache_dir("models", ctx.exp_id, f"{'sameref' if same_ref else 'undo'}_a{alpha}")
    merged.save_pretrained(ck); tok.save_pretrained(ck)
    ctx.write_metrics({"noise_alpha": alpha, "rank": rank, "steps": steps, "same_ref": same_ref,
                       "checkpoint": str(ck), "features": [int(f) for f in feats], "tau": tau,
                       "last": tr.log[-1] if tr.log else None})
    ctx.finish({"view": "d1-distill", "forget": None})
    return {"checkpoint": str(ck)}
