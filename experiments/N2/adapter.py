"""N2 OpenUnlearning adapter: expose the DSG hook as a drop-in HF causal-LM wrapper so OpenUnlearning's
evaluators (which take an HF model + tokenizer) can score a DSG-guarded model. The wrapper registers the
faithful DSG hook on decoder layer L and forwards everything else to the base model.

OpenUnlearning itself is not installed into the shared env (its pinned stack - hydra, deepspeed,
specific transformers - would risk the harness env). The task verifies the wrapper on TOFU with our
TOFU metrics and writes $DSG_RESULTS/N2_PR.md (draft; no PR is opened)."""
import torch


class DSGGuardedModel(torch.nn.Module):
    """model(**inputs) and model.generate(...) behave like the wrapped HF model, with DSG active."""

    def __init__(self, hf_model, sae, features, tau, multiplier=500.0, layer=3):
        super().__init__()
        from dsgx.train.core import add_dsg_hook_hf

        self.model = hf_model
        self.config = hf_model.config
        self.hook, self.handle = add_dsg_hook_hf(hf_model, sae, features, multiplier, tau, layer)

    def forward(self, *a, **k):
        return self.model(*a, **k)

    def generate(self, *a, **k):
        # No KV cache: each step re-runs the full sequence so the gate sees prompt + generated text
        # (exact DSG semantics). With HF's cache the hook would see only one token per step.
        k.setdefault("use_cache", False)
        return self.model.generate(*a, **k)

    @property
    def device(self):
        return self.model.device

    def remove(self):
        self.handle.remove()


PR = """# [Draft] Add a DSG (Dynamic SAE Guardrails) method adapter

**Not opened** (MASTER_PLAN section 9: no PRs). Draft text for the OpenUnlearning maintainers.

## What
`DSGGuardedModel`: wraps any HF causal LM with an inference-time SAE gate (Dynamic SAE Guardrails):
if the fraction of tokens on which any selected SAE feature fires exceeds a calibrated threshold, the
selected features are clamped to -M at firing positions. Training-free; the base weights are unchanged.

## Why
DSG is a strong "unlearning as access control" baseline. Including it lets OpenUnlearning users compare
weight-based unlearning with activation gating on the same TOFU / MUSE / WMDP metrics, including
robustness checks (relearning, quantisation) where gating and weight methods differ sharply.

## Interface
`DSGGuardedModel(hf_model, sae, features, tau, multiplier=500, layer=3)`; features/tau come from a
forget-vs-retain calibration pass (forget split vs retain split of the benchmark).

## Verified locally
{verified}
"""


def task(ctx):
    from dsgx import paths
    from dsgx.data import activation_cache as ac
    from dsgx.methods import dsg
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights
    from dsgx.train.core import load_hf

    a = ctx.args
    w = resolve_weights(a.get("weights", "ckpt:A2/tofu_full"), ctx.exp_id)
    b = get_bundle(weights=w)
    cache = ac.ActivationCache(ac.build_cache(b, "tofu-forget10", "tofu-retain90", 0))
    feats = dsg.select_features(cache, 20, 95)
    tau = dsg.calibrate_tau(cache, feats, 95)
    sae = b.sae
    from dsgx.models import loader

    loader._MODELS.clear(); torch.cuda.empty_cache()
    hf, tok = load_hf(weights=w, dtype=torch.bfloat16)
    from datasets import load_dataset

    
    fq = load_dataset("locuslab/TOFU", "forget10", split="train").select(range(int(a.get("n", 20))))
    rows = {}
    for guarded in (False, True):
        m = DSGGuardedModel(hf, sae, feats, tau) if guarded else hf
        lps = []
        for x in fq:
            p = f"<bos><start_of_turn>user\n{x['question']}<end_of_turn>\n<start_of_turn>model\n"
            pi = tok(p, return_tensors="pt", add_special_tokens=False).input_ids
            fi = tok(p + x["answer"], return_tensors="pt", add_special_tokens=False).input_ids.to(hf.device)
            n_ans = fi.shape[1] - pi.shape[1]
            with torch.no_grad():  # full-sequence forward: the gate sees the whole prompt + answer
                lp = torch.log_softmax(m(fi).logits[0, -n_ans - 1:-1].float(), -1)
            lps.append(float(lp.gather(1, fi[0, -n_ans:, None]).mean()))
            ctx.progress.advance(1)
        if guarded:
            m.remove()
        rows["guarded" if guarded else "unguarded"] = float(sum(lps) / len(lps))
    verified = (f"- TOFU forget10 (n={len(fq)}), mean answer-token log-prob: unguarded {rows['unguarded']:.3f}, "
                f"DSG-guarded {rows['guarded']:.3f} (tau={tau:.3f}, {len(feats)} features). Generation through the "
                f"wrapper must not use HF's KV cache (the gate would see one token at a time).")
    (paths.results_dir() / ("N2_PR_smoke.md" if ctx.smoke else "N2_PR.md")).write_text(PR.format(verified=verified))
    ctx.write_metrics({"answer_logprob": rows, "n": len(fq), "tau": tau, "features": feats,
                       "openunlearning_installed": False})
    ctx.finish({"view": "n2-adapter", "forget": None})
