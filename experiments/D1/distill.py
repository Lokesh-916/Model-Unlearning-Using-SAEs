"""D1-local - UNDO-style distillation into a NOISED LoRA student (16 GB fallback of the plan's
full-parameter UNDO; the full 2B version is DEFERRED). Student = gemma-2-2b-it + LoRA with a noised
initialisation (noise alpha). Loss = KL(student || guarded teacher) on forget prompts + KL(student ||
unguarded teacher) on retain prompts. Teachers are the same base weights with the DSG hook on/off
(LoRA adapters disabled). Saves a merged checkpoint for eval + A6. Compared with lora-student-same-ref.

Memory (16 GB card; the first version OOMed at 14.8 GB): the TransformerLens bundle is released before the HF
model loads (only the SAE is kept), so one 2B model is resident; the teacher is that same frozen bf16 model with
the adapters off. Micro-batch 1 with gradient accumulation over `bs` (same effective batch; no padding, so the KL
is taken at each prompt's real last token), gradient checkpointing, logits for the last position only, AdamW 8-bit."""
import gc
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
    sae, layer = b.sae, b.layer  # keep the SAE; release the TransformerLens model (the bundle held it)
    from dsgx.models import loader

    del b, cache
    loader._MODELS.clear()
    gc.collect()
    torch.cuda.empty_cache()
    model, tok = load_hf(dtype=torch.bfloat16)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False
    teach_hook, _ = add_dsg_hook_hf(model, sae, feats, 500, tau, layer)
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
        return [tok(src[i], return_tensors="pt", truncation=True, max_length=512,
                    add_special_tokens=False).to(model.device) for i in idx]

    def logits(enc):  # the KL uses the last position only
        return model(**enc, logits_to_keep=1).logits

    def step_fn(step):
        fbs, rbs = batch(forget, bs), batch(retain, bs)  # same sampling order as the batched version
        tot, klf, klr = 0.0, 0.0, 0.0
        for j, (fb, rb) in enumerate(zip(fbs, rbs)):     # micro-batch 1, accumulate over bs
            with torch.no_grad():
                model.disable_adapter_layers()
                teach_hook.enabled = not same_ref
                tf = logits(fb)
                teach_hook.enabled = False
                tr_ = logits(rb)
                model.enable_adapter_layers()
            kf, kr = _kl(logits(fb), tf), _kl(logits(rb), tr_)
            loss = (kf + kr) / bs
            klf, klr = klf + float(kf) / bs, klr + float(kr) / bs
            if j < bs - 1:
                loss.backward()
                tot = tot + loss.detach()
            else:
                tot = loss + tot
        return {"loss": tot, "kl_forget": klf, "kl_retain": klr}

    out = ctx.run_dir()
    tr = Trainer(model, model.parameters(), out, lr=float(a.get("lr", 2e-4)), steps=steps,
                 ckpt_every=max(10, steps // 4), progress=ctx.progress, optimizer=a.get("optimizer", "adamw8bit"))
    ctx.progress.update(steps_total=steps, force=True)
    tr.run(step_fn)
    merged = model.merge_and_unload()
    ck = ctx.cache_dir("models", ctx.exp_id, f"{'sameref' if same_ref else 'undo'}_a{alpha}")
    merged.save_pretrained(ck); tok.save_pretrained(ck)
    ctx.write_metrics({"noise_alpha": alpha, "rank": rank, "steps": steps, "same_ref": same_ref,
                       "effective_batch": bs, "micro_batch": 1, "optimizer": tr.opt.__class__.__name__,
                       "gradient_checkpointing": True, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1e9,
                       "checkpoint": str(ck), "features": [int(f) for f in feats], "tau": tau,
                       "last": tr.log[-1] if tr.log else None})
    ctx.finish({"view": "d1-distill", "forget": None})
    return {"checkpoint": str(ck)}
