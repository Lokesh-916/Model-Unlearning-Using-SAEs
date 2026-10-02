"""A2 TOFU: LoRA fine-tune gemma-2-2b-it on TOFU (full = forget+retain; retain90 = reference model for
forget quality), then TOFU metrics for {full, full+DSG, retain}:
  * truth ratio R per forget item = mean_perturbed P(a_pert|q)^(1/|a|) / P(a_para|q)^(1/|a|)
  * forget quality = KS-test p-value between R on forget10 for the evaluated model and the retain model
  * model utility = harmonic mean of retain answer probability and retain (1 - truth ratio, clipped)
DSG on TOFU: features from the fine-tuned model's own cache (forget = tofu-forget10, retain =
tofu-retain90), applied with the faithful hook. Data are fictitious authors (safe)."""
import numpy as np
import torch

CHAT = "<bos><start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"


def finetune(ctx):
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model

    from dsgx.train.core import Trainer, load_hf, lm_loss

    a = ctx.args
    cfg = a.get("config", "full")
    # TOFU 'full' == forget10 + retain90; build it from those (the 'full' config is not cached offline).
    cfgs = ["forget10", "retain90"] if cfg == "full" else [cfg]
    qa = [(x["question"], x["answer"]) for c in cfgs
          for x in load_dataset("locuslab/TOFU", c, split="train")]
    model, tok = load_hf(dtype=torch.bfloat16)
    model = get_peft_model(model, LoraConfig(r=int(a.get("rank", 32)), lora_alpha=64, lora_dropout=0.0,
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                           task_type="CAUSAL_LM"))
    tok.padding_side = "right"
    g = torch.Generator().manual_seed(ctx.seed)
    bs, steps = int(a.get("bs", 8)), int(a.get("steps", 1000))

    def step_fn(step):
        idx = torch.randint(0, len(qa), (bs,), generator=g).tolist()
        prefixes = [CHAT.format(q=qa[i][0]) for i in idx]
        texts = [p + qa[i][1] + "<end_of_turn>" for p, i in zip(prefixes, idx)]
        return {"loss": lm_loss(model, tok, texts, max_len=384, mask_prefix=prefixes)}

    tr = Trainer(model, model.parameters(), ctx.run_dir(), lr=float(a.get("lr", 1e-4)), steps=steps,
                 ckpt_every=max(10, steps // 5), progress=ctx.progress)
    tr.run(step_fn)
    merged = model.merge_and_unload()
    out = ctx.cache_dir("models", ctx.exp_id, a["tag"])
    merged.save_pretrained(out)
    tok.save_pretrained(out)
    ctx.write_metrics({"config": a.get("config", "full"), "steps": steps, "checkpoint": str(out),
                       "final_loss": tr.log[-1]["loss"] if tr.log else None})
    ctx.finish({"view": f"tofu-finetune:{a['tag']}", "forget": None})


@torch.no_grad()
def _answer_logprob(model, q, a):
    """Mean per-token log-prob of answer a given question q (chat format)."""
    p = model.to_tokens(CHAT.format(q=q), prepend_bos=False)
    full = model.to_tokens(CHAT.format(q=q) + a, prepend_bos=False)
    n = full.shape[1] - p.shape[1]
    lp = torch.log_softmax(model(full)[0, -n - 1:-1].float(), -1)
    return float(lp.gather(1, full[0, -n:, None]).mean())


def metrics(ctx):
    from datasets import load_dataset
    from scipy.stats import ks_2samp

    from dsgx.data import activation_cache as ac
    from dsgx.eval.stats import bootstrap_ci
    from dsgx.methods import dsg
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    a = ctx.args
    n = a.get("limit")
    fset = load_dataset("locuslab/TOFU", "forget10_perturbed", split="train")
    rset = load_dataset("locuslab/TOFU", "retain_perturbed", split="train")
    if n:
        fset, rset = fset.select(range(int(n))), rset.select(range(int(n)))
    conds = a.get("conditions", [{"tag": "retain-model", "weights": "ckpt:tofu_retain"},
                                 {"tag": "full", "weights": "ckpt:tofu_full"},
                                 {"tag": "full+dsg", "weights": "ckpt:tofu_full", "dsg": True}])
    ctx.progress.update(items_total=len(conds) * (len(fset) + len(rset)), items_done=0, force=True)
    res, tr_by = {}, {}
    for c in conds:
        b = get_bundle(weights=resolve_weights(c["weights"], ctx.exp_id))
        m = b.model
        m.reset_hooks()
        info = {}
        if c.get("dsg"):
            cache = ac.ActivationCache(ac.build_cache(b, "tofu-forget10", "tofu-retain90", 0))
            feats = dsg.select_features(cache, int(a.get("n_features", 20)), float(a.get("retain_pct", 95)))
            tau = dsg.calibrate_tau(cache, feats, 95)
            m.add_hook(b.hook_name, dsg.DSGHook(b.sae, feats, 500, tau, faithful=True, record=False))
            info = {"features": feats, "tau": tau}

        def truth_ratios(ds):
            out = []
            for x in ds:
                para = np.exp(_answer_logprob(m, x["question"], x["paraphrased_answer"]))
                pert = np.mean([np.exp(_answer_logprob(m, x["question"], p)) for p in x["perturbed_answer"]])
                out.append(pert / max(para, 1e-12))
                ctx.progress.advance(1)
            return np.array(out)

        tr_f = truth_ratios(fset)
        tr_r = truth_ratios(rset)
        ans_r = np.array([np.exp(_answer_logprob(m, x["question"], x["answer"])) for x in rset])
        m.reset_hooks()
        tr_by[c["tag"]] = tr_f
        util_parts = [float(ans_r.mean()), float(np.clip(1 - tr_r, 0, 1).mean())]
        res[c["tag"]] = {"truth_ratio_forget": bootstrap_ci(tr_f), "truth_ratio_retain": bootstrap_ci(tr_r),
                         "retain_answer_prob": bootstrap_ci(ans_r),
                         "model_utility": float(len(util_parts) / sum(1 / max(u, 1e-9) for u in util_parts)),
                         **info}
    ref = tr_by.get("retain-model")
    for t, v in res.items():
        v["forget_quality_ks_p"] = float(ks_2samp(tr_by[t], ref).pvalue) if ref is not None and t != "retain-model" else None
    ctx.write_metrics({"conditions": res, "n_forget": len(fset), "n_retain": len(rset)})
    ctx.finish({"view": "tofu-metrics", "forget": None})
    return {k: v.get("forget_quality_ks_p") for k, v in res.items()}
