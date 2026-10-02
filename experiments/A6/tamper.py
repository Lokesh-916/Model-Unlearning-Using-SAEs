"""A6 tampering (LoRA only on this PC; full fine-tuning relearning is DEFERRED).

Conditions (weights, runtime hook):
  dsg-hook    base weights, DSG hook ON in training and evaluation (guarded deployment + FT API)
  dsg-nohook  base weights, no hook (the open weights DSG leaves behind)
  student     legacy LoRA student (exp D1 'sameref'), no hook
  d1          D1-local noised student (alpha 0.3), no hook
  d2          D2 null-space edited model, no hook
Tasks: relearn (k forget passages, LoRA rank, forget accuracy logged at eval_at steps), quantize
(4/8-bit), steer (recovery vector added after the hook), benign (LoRA on Alpaca, side-effect recovery).
Evaluation: WMDP-Bio TEST (first n_eval items) and an MMLU utility mini-set, batch size 1 (faithful).
"""
import json

import numpy as np
import torch

from dsgx.data import activation_cache as ac
from dsgx.data.corpora import load_forget_docs
from dsgx.data.mcq import format_prompt, load_mcq
from dsgx.data.splits import get_split
from dsgx.methods import dsg
from dsgx.run import resolve_weights
from dsgx.train.core import Trainer, add_dsg_hook_hf, decoder_layers, lm_loss, load_hf, mcq_probs_hf

CONDS = {"dsg-hook": (None, True), "dsg-nohook": (None, False), "student": ("ckpt:D1/sameref_a0.0", False),
         "d1": ("ckpt:D1/undo_a0.3", False), "d2": ("ckpt:D2/nullspace", False)}
UTIL = ["high_school_geography", "high_school_us_history"]


def _eval_sets(n_eval, n_util):
    fd = load_mcq("wmdp-bio")
    forget = [fd[i] for i in get_split("wmdp-bio", "test")[:n_eval]]
    util = []
    for s in UTIL:
        it = load_mcq(s)
        util += [it[i] for i in get_split(s, "test")[: n_util // len(UTIL)]]
    return forget, util


def _acc(model, tok, items):
    p = mcq_probs_hf(model, tok, [format_prompt(it) for it in items], bs=1)
    return float((p.argmax(1) == np.array([it.answer for it in items])).mean())


def _setup(ctx, quant=None):
    a = ctx.args
    w, hook_on = CONDS[a.get("condition", "dsg-hook")]
    w = resolve_weights(w, ctx.exp_id)
    from dsgx.models.loader import get_bundle

    b = get_bundle()  # SAE + cache come from the base model (DSG features are defined there)
    cache = ac.ActivationCache(ac.build_cache(b, "bio-forget-corpus", "wikitext", 0))
    feats = dsg.select_features(cache, int(a.get("n_features", 20)), 95)
    tau = dsg.calibrate_tau(cache, feats, 95)
    sae = b.sae
    from dsgx.models import loader

    loader._MODELS.clear()  # free the TransformerLens copy; HF model is used from here on
    torch.cuda.empty_cache()
    if quant:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16) if quant == 4 \
            else BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(w or "google/gemma-2-2b-it", quantization_config=q,
                                                     attn_implementation="eager", device_map={"": 0})
        tok = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
    else:
        model, tok = load_hf(weights=w, dtype=torch.bfloat16)
    hook, _ = add_dsg_hook_hf(model, sae, feats, 500, tau, b.layer)
    hook.enabled = hook_on
    return model, tok, hook, {"weights": w, "hook": hook_on, "tau": tau, "features": feats}


def relearn(ctx):
    from peft import LoraConfig, get_peft_model

    a = ctx.args
    model, tok, hook, info = _setup(ctx)
    forget, util = _eval_sets(int(a.get("n_eval", 300)), int(a.get("n_util", 100)))
    docs = load_forget_docs("bio-forget-corpus")
    rng = np.random.default_rng(ctx.seed)
    k = int(a.get("k", 50))
    passages = [docs[i][:2000] for i in rng.choice(len(docs), size=k, replace=False)]
    model = get_peft_model(model, LoraConfig(r=int(a.get("rank", 8)), lora_alpha=2 * int(a.get("rank", 8)),
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                           lora_dropout=0.0, task_type="CAUSAL_LM"))
    tok.padding_side = "right"
    steps = int(a.get("steps", 1000))
    eval_at = set(int(x) for x in a.get("eval_at", [50, 200, 1000]))
    pre = {"forget_acc": _acc(model, tok, forget), "util_acc": _acc(model, tok, util)}

    def eval_fn(step):
        if step not in eval_at and step != steps:
            return {}
        return {"forget_acc": _acc(model, tok, forget), "util_acc": _acc(model, tok, util)}

    def step_fn(step):
        idx = rng.integers(0, len(passages), size=int(a.get("bs", 4)))
        return {"loss": lm_loss(model, tok, [passages[i] for i in idx], max_len=256)}

    tr = Trainer(model, model.parameters(), ctx.run_dir(), lr=float(a.get("lr", 1e-4)), steps=steps,
                 ckpt_every=max(10, steps // 4), progress=ctx.progress, eval_fn=eval_fn,
                 eval_every=min(eval_at | {steps}))
    log = tr.run(step_fn)
    curve = [{"step": r["step"], "forget_acc": r["forget_acc"], "util_acc": r["util_acc"]}
             for r in log if "forget_acc" in r]
    info.pop("features")
    ctx.write_metrics({**{k_: a.get(k_) for k_ in ("condition", "k", "rank", "steps")}, **info,
                       "before": pre, "curve": curve, "n_eval": len(forget), "n_util": len(util)})
    ctx.finish({"forget": {"mean": curve[-1]["forget_acc"], "lo": None, "hi": None, "n": len(forget)} if curve else None,
                "view": f"relearn:{a.get('condition')}"})


def quantize(ctx):
    a = ctx.args
    model, tok, hook, info = _setup(ctx, quant=int(a.get("bits", 8)))
    forget, util = _eval_sets(int(a.get("n_eval", 300)), int(a.get("n_util", 100)))
    m = {"bits": int(a.get("bits", 8)), "condition": a.get("condition"), "forget_acc": _acc(model, tok, forget),
         "util_acc": _acc(model, tok, util), "n_eval": len(forget), "weights": info["weights"], "hook": info["hook"]}
    ctx.write_metrics(m)
    ctx.finish({"forget": {"mean": m["forget_acc"], "lo": None, "hi": None, "n": len(forget)}, "view": "quantize"})


@torch.no_grad()
def steer(ctx):
    """v = mean over DEV forget prompts of (hook-off - hook-on) layer output at the last token; added
    after the hook on TEST prompts with scale alpha. Measures recovery of forget accuracy."""
    a = ctx.args
    model, tok, hook, info = _setup(ctx)
    forget, util = _eval_sets(int(a.get("n_eval", 300)), int(a.get("n_util", 100)))
    dev = [load_mcq("wmdp-bio")[i] for i in get_split("wmdp-bio", "dev")[: int(a.get("n_fit", 64))]]
    layer = decoder_layers(model)[int(a.get("layer", 3))]
    grab = {}
    h = layer.register_forward_hook(lambda m_, i_, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0, -1].float()))
    diffs = []
    for it in dev:
        ids = tok(format_prompt(it), return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
        hook.enabled = False; model(ids); off = grab["h"].clone()
        hook.enabled = True; model(ids); on = grab["h"].clone()
        diffs.append(off - on)
    h.remove()
    v = torch.stack(diffs).mean(0)
    res = {"before": {"forget_acc": _acc(model, tok, forget), "util_acc": _acc(model, tok, util)}}
    for alpha in a.get("alphas", [0.5, 1.0, 2.0]):
        def add(m_, i_, o, alpha=alpha):
            hs = o[0] if isinstance(o, tuple) else o
            hs = hs + alpha * v.to(hs.dtype)
            return (hs, *o[1:]) if isinstance(o, tuple) else hs
        hh = layer.register_forward_hook(add)
        res[f"alpha_{alpha}"] = {"forget_acc": _acc(model, tok, forget), "util_acc": _acc(model, tok, util)}
        hh.remove()
    ctx.write_metrics({"condition": a.get("condition"), "n_fit_dev": len(dev), "n_eval": len(forget), **res})
    best = max(v_["forget_acc"] for k_, v_ in res.items() if k_.startswith("alpha"))
    ctx.finish({"forget": {"mean": best, "lo": None, "hi": None, "n": len(forget)}, "view": "steer(max)"})


def benign(ctx):
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model

    a = ctx.args
    model, tok, hook, info = _setup(ctx)
    forget, util = _eval_sets(int(a.get("n_eval", 300)), int(a.get("n_util", 100)))
    alp = load_dataset("tatsu-lab/alpaca", split="train")
    texts = [f"<bos><start_of_turn>user\n{x['instruction']} {x['input']}<end_of_turn>\n<start_of_turn>model\n{x['output']}<end_of_turn>"
             for x in alp.select(range(2000))]
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"],
                                             lora_dropout=0.0, task_type="CAUSAL_LM"))
    tok.padding_side = "right"
    rng = np.random.default_rng(ctx.seed)
    pre = {"forget_acc": _acc(model, tok, forget), "util_acc": _acc(model, tok, util)}
    steps = int(a.get("steps", 500))
    tr = Trainer(model, model.parameters(), ctx.run_dir(), lr=1e-4, steps=steps, ckpt_every=max(10, steps // 4),
                 progress=ctx.progress)
    tr.run(lambda s: {"loss": lm_loss(model, tok, [texts[i] for i in rng.integers(0, len(texts), 4)], max_len=256)})
    post = {"forget_acc": _acc(model, tok, forget), "util_acc": _acc(model, tok, util)}
    ctx.write_metrics({"condition": a.get("condition"), "steps": steps, "before": pre, "after": post})
    ctx.finish({"forget": {"mean": post["forget_acc"], "lo": None, "hi": None, "n": len(forget)}, "view": "benign-ft"})
