"""A5 privacy and indistinguishability: membership-inference AUC on the TOFU-fine-tuned model
(exp A2 checkpoint), members = TOFU forget10, non-members = TOFU holdout10 (never trained on).
Attacks: loss, zlib (loss / zlib-entropy), min-k% prob (k=20). Computed ungated and gated (DSG hook
with TOFU features). An effective unlearner should push AUC toward 0.5 on gated outputs.
TOFU forget quality itself is reported by A2 (tofu-metrics). MUSE target models are DEFERRED."""
import zlib

import numpy as np
import torch

CHAT = "<bos><start_of_turn>user\n{q}<end_of_turn>\n<start_of_turn>model\n"


@torch.no_grad()
def token_logprobs(model, q, a):
    p = model.to_tokens(CHAT.format(q=q), prepend_bos=False)
    full = model.to_tokens(CHAT.format(q=q) + a, prepend_bos=False)
    n = full.shape[1] - p.shape[1]
    lp = torch.log_softmax(model(full)[0, -n - 1:-1].float(), -1)
    return lp.gather(1, full[0, -n:, None])[:, 0].cpu().numpy()


def scores(lp, text, k=0.2):
    loss = -lp.mean()
    zl = len(zlib.compress(text.encode()))
    mink = -np.sort(lp)[: max(1, int(len(lp) * k))].mean()
    return {"loss": -loss, "zlib": -loss / zl, "mink": -mink}  # higher = more "member"


def task(ctx):
    from datasets import load_dataset

    from dsgx.analysis.gate_quality import auroc
    from dsgx.data import activation_cache as ac
    from dsgx.methods import dsg
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    a = ctx.args
    n = a.get("limit")
    mem = load_dataset("locuslab/TOFU", "forget10", split="train")
    non = load_dataset("locuslab/TOFU", "holdout10", split="train")
    if n:
        mem, non = mem.select(range(int(n))), non.select(range(int(n)))
    b = get_bundle(weights=resolve_weights(a.get("weights", "ckpt:A2/tofu_full"), ctx.exp_id))
    cache = ac.ActivationCache(ac.build_cache(b, "tofu-forget10", "tofu-retain90", 0))
    feats = dsg.select_features(cache, 20, 95)
    tau = dsg.calibrate_tau(cache, feats, 95)
    ctx.progress.update(items_total=2 * (len(mem) + len(non)), items_done=0, force=True)
    out = {}
    for cond in ("ungated", "gated"):
        b.model.reset_hooks()
        if cond == "gated":
            b.model.add_hook(b.hook_name, dsg.DSGHook(b.sae, feats, 500, tau, faithful=True, record=False))
        S = {"member": [], "non": []}
        for lab, ds in (("member", mem), ("non", non)):
            for x in ds:
                S[lab].append(scores(token_logprobs(b.model, x["question"], x["answer"]), x["answer"]))
                ctx.progress.advance(1)
        b.model.reset_hooks()
        out[cond] = {k: auroc(np.array([s[k] for s in S["member"]]), np.array([s[k] for s in S["non"]]))
                     for k in ("loss", "zlib", "mink")}
    ctx.write_metrics({"auroc": out, "n_member": len(mem), "n_nonmember": len(non), "tau": tau,
                       "features": feats, "note": "AUROC 0.5 = indistinguishable"})
    ctx.finish({"view": "mia-auroc", "forget": None})
    return out
