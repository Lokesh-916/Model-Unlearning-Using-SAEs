"""Greedy generation (batch 1) with a detect-only streaming gate and a latched switch to a response on fire.

Phase 1: the unmodified model generates with a detect-only hook at the gate's hook point (same per-token values and
score as dsgx.gen.stream with gates.stream_gate; BOS never fires). The first step whose score exceeds tau is the fire
step s (0 = the prompt already fires). Phase 1 always runs to the end, so it is also the ungated base output.
Phase 2 (per response, only if the gate fired): seq = prompt + phase-1 tokens[:s], one full pass under the response,
then greedy continuation up to the same total max_new. This equals dsgx.gen.stream.generate for the -500 clamp
(the clamp acts at the gate's own hook output, so the gate's input, and hence its running-max CUSUM decision, are
unchanged after the rebuild); for weight / steering responses the decision is latched (CUSUM is a running max).
Responses (any combination): clamp at the gate hook (DSG-faithful clamp_all at firing positions, SAE error added
back), weight patch (TransformerLens W_out swap, layers 1-3, exact bf16(W + delta)), steering (vector added at
blocks.<L>.hook_resid_post at every position).
"""
from contextlib import contextmanager

import torch


def _end_ids(model):
    tok = model.tokenizer
    ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0:
        ids.add(eot)
    return ids


class DetectHook:
    def __init__(self, sae, features, tau, score_fn=None, token_fn=None):
        self.sae, self.tau, self.score_fn, self.token_fn = sae, float(tau), score_fn, token_fn
        self.feats = [int(f) for f in features]
        self.vals = []

    @torch.no_grad()
    def __call__(self, resid, hook=None):
        acts = self.sae.encode(resid)
        if not self.vals:
            acts[:, 0, :] = 0.0
        tgt = acts[0, :, self.feats] > 0
        new = tgt.any(dim=1).tolist() if self.token_fn is None else [float(v) for v in self.token_fn(tgt)]
        self.vals += new
        return resid

    def score(self):
        v = self.vals
        return sum(v) / len(v) if self.score_fn is None else float(self.score_fn(v))

    def fired(self):
        return self.score() > self.tau


class ClampHook:
    def __init__(self, sae, features, multiplier=500.0):
        self.sae, self.mult = sae, float(multiplier)
        self.feats = [int(f) for f in features]
        self.full = True

    @torch.no_grad()
    def __call__(self, resid, hook=None):
        sae = self.sae
        acts = sae.encode(resid)
        if self.full:
            acts[:, 0, :] = 0.0
            self.full = False
        err = resid - sae.decode(acts)
        tgt = acts[:, :, self.feats]
        mask = (tgt > 0).any(dim=2, keepdim=True)
        acts[:, :, self.feats] = torch.where(mask, torch.full_like(tgt, -self.mult), tgt)
        return sae.decode(acts) + err


class Response:
    """name; clamp (sae, features, mult) | None; weights {layer: patched W_out tensor} | None; steer (layer, vec) | None."""

    def __init__(self, name, hook_name=None, clamp=None, weights=None, steer=None):
        self.name, self.hook_name, self.clamp, self.weights, self.steer = name, hook_name, clamp, weights, steer

    @contextmanager
    def active(self, model):
        saved = {}
        try:
            for i, W in (self.weights or {}).items():
                saved[i] = model.blocks[i].mlp.W_out.data
                model.blocks[i].mlp.W_out.data = W
            if self.clamp:
                model.add_hook(self.hook_name, ClampHook(*self.clamp))
            if self.steer:
                layer, vec = self.steer

                def add(resid, hook=None):
                    return resid + vec.to(resid.dtype)

                model.add_hook(f"blocks.{layer}.hook_resid_post", add)
            yield
        finally:
            model.reset_hooks()
            for i, W in saved.items():
                model.blocks[i].mlp.W_out.data = W


def patched_weights(model, delta: dict) -> dict:
    """{layer: bf16(W_out + delta.T)} (fp64 add) for HF-named down_proj deltas."""
    out = {}
    for k, d in delta.items():
        i = int(k.split(".")[2])
        W = model.blocks[i].mlp.W_out.data
        dt = d.to(W.device).T
        assert tuple(dt.shape) == tuple(W.shape), (k, tuple(dt.shape), tuple(W.shape))
        out[i] = (W.double() + dt.double()).to(W.dtype).contiguous()
    return out


@torch.no_grad()
def _continue(model, seq, n_left, end, prefix_hooks=None):
    """Full pass on seq, then greedy up to n_left tokens. -> (new tokens, stop reason)."""
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

    dev = model.cfg.device
    cache = TransformerLensKeyValueCache.init_cache(model.cfg, dev, 1)
    logits = model(seq, past_kv_cache=cache)
    last = logits[0, -1]
    out = []
    for _ in range(n_left):
        nxt = int(last.argmax())
        out.append(nxt)
        if nxt in end:
            return out, "eos"
        last = model(torch.tensor([[nxt]], device=dev), past_kv_cache=cache)[0, -1]
        if prefix_hooks:
            prefix_hooks(len(out))
    return out, "max_new"


@torch.no_grad()
def generate(model, prompt, bundle=None, gate=None, responses=(), max_new=64, forced=False):
    """gate = (features, tau, score_fn, token_fn) or None. forced: every response is on from the prompt (no gate).
    -> {"base": {...}, "<response>": {...}, "fire_step": s | None, "prompt_len": n}; each entry has tokens, stop."""
    dev = model.cfg.device
    toks = model.to_tokens(prompt, prepend_bos=False).to(dev)
    end = _end_ids(model)
    model.reset_hooks()
    res = {"prompt_len": int(toks.shape[1])}
    fire = 0 if forced else None
    if not forced:
        det = None
        if gate is not None:
            feats, tau, score_fn, token_fn = gate
            det = DetectHook(bundle.sae, feats, tau, score_fn, token_fn)
            model.add_hook(bundle.hook_name, det)
        state = {"fire": None}

        def check(step):
            if det is not None and state["fire"] is None and det.fired():
                state["fire"] = step

        try:
            from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

            cache = TransformerLensKeyValueCache.init_cache(model.cfg, dev, 1)
            last = model(toks, past_kv_cache=cache)[0, -1]
            check(0)
            out, reason = [], "max_new"
            for step in range(1, max_new + 1):
                nxt = int(last.argmax())
                out.append(nxt)
                if nxt in end:
                    reason = "eos"
                    break
                last = model(torch.tensor([[nxt]], device=dev), past_kv_cache=cache)[0, -1]
                check(step)
        finally:
            model.reset_hooks()
        res["base"] = {"tokens": out, "stop": reason}
        fire = state["fire"]
        base_out = out
    res["fire_step"] = fire
    for r in responses:
        if fire is None:
            res[r.name] = {"tokens": list(res["base"]["tokens"]), "stop": res["base"]["stop"], "switched": False}
            continue
        prefix = [] if forced else base_out[:fire]
        seq = torch.cat([toks, torch.tensor([prefix], dtype=torch.long, device=dev)], dim=1) if prefix else toks
        with r.active(model):
            new, reason = _continue(model, seq, max_new - len(prefix), end)
        res[r.name] = {"tokens": prefix + new, "stop": reason, "switched": True}
    return res
