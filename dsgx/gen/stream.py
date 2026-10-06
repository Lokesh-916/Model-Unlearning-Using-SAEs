"""Generation with the DSG gate (decision 8).

mode="stream" (canonical): the gate is re-evaluated at every step on prompt + generated text.
    The KV cache is reused and fire counts are incremental. Whenever the gate state changes, the
    cache is rebuilt with one full forward pass under the new state, so the output is identical
    to re-running the full sequence through the DSG hook at every step (the no-cache semantics of
    the legacy demo) at a fraction of the cost.
mode="prompt_only" (ablation): the gate is decided once on the prompt and kept.
mode="none": the unguarded model through the same loop.

score_fn (optional): gate score from the per-position fire flags (BOS included as False), compared with `tau`;
default None = rho (fraction of firing positions), the DSG rule. E.g. the window gate (MT-Bench, session 11):
lambda fires: gates.score_window(torch.tensor(fires), len(fires), 16) with the threshold from gates.calibrate.
token_fn (optional): per-position value from the per-feature fire pattern ([L, k] bool -> [L]) stored instead of
the any-fire flag, so score_fn can use it; e.g. the CUSUM gate's per-token LLR (gates.stream_gate builds both).
The clamp positions stay "any selected feature fires".

The clamp rule is DSG-faithful: on an active sequence every selected feature is set to
-multiplier at every position where any selected feature fires; the SAE error is added back.
"""
from dataclasses import dataclass, field

import torch


@dataclass
class GenResult:
    prompt_len: int
    tokens: list            # generated token ids
    rho_trace: list         # rho after each generated token
    gate_trace: list        # gate state used for each generated token
    fire_trace: list        # any-selected-feature fires per position (prompt + generated)
    prompt_gate: bool
    first_gate_step: int | None   # first generated step with the gate on (0 = from the prompt)
    n_rebuilds: int
    stop_reason: str
    extra: dict = field(default_factory=dict)


class _GateHook:
    def __init__(self, sae, features, multiplier, tau, score_fn=None, token_fn=None):
        self.sae = sae
        self.score_fn = score_fn
        self.token_fn = token_fn
        self.feats = list(int(f) for f in features)
        self.mult = float(multiplier)
        self.tau = float(tau)
        self.fires = []          # committed per-position fire flags
        self.pending = None      # fires of the last incremental call
        self.forced = None       # gate state override (prompt_only mode)
        self.last_gate = None

    def reset(self):
        self.fires, self.pending = [], None

    @torch.no_grad()
    def __call__(self, resid, hook=None):
        sae, feats = self.sae, self.feats
        acts = sae.encode(resid)
        full = len(self.fires) == 0
        if full:
            acts[:, 0, :] = 0.0
        err = resid - sae.decode(acts)
        tgt = acts[:, :, feats]
        any_fire = (tgt > 0).any(dim=2)[0]
        new = any_fire.tolist() if self.token_fn is None else [float(v) for v in self.token_fn(tgt[0] > 0)]
        all_f = (self.fires if not full else []) + new
        rho = sum(all_f) / len(all_f) if self.score_fn is None else float(self.score_fn(all_f))
        gate = (rho > self.tau) if self.forced is None else self.forced
        self.pending = (new, rho, gate, full)
        self.last_gate = gate
        if not gate:
            return resid
        mask = any_fire[None, :, None]
        acts[:, :, feats] = torch.where(mask, torch.full_like(tgt, -self.mult), tgt)
        return sae.decode(acts) + err

    def commit(self):
        new, rho, gate, full = self.pending
        self.fires = new if full else self.fires + new
        return rho, gate


def _end_ids(model):
    tok = model.tokenizer
    ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0:
        ids.add(eot)
    return ids


@torch.no_grad()
def generate(model, prompt: str, bundle=None, features=None, multiplier=500.0, tau=None,
             mode: str = "stream", max_new: int = 64, score_fn=None, token_fn=None) -> GenResult:
    """Greedy generation (batch 1). `prompt` must already contain the chat template and <bos>."""
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

    dev = model.cfg.device
    toks = model.to_tokens(prompt, prepend_bos=False).to(dev)
    end = _end_ids(model)
    hook = None
    model.reset_hooks()
    if mode != "none":
        hook = _GateHook(bundle.sae, features, multiplier, tau, score_fn, token_fn)
        model.add_hook(bundle.hook_name, hook)

    def full_pass(seq):
        cache = TransformerLensKeyValueCache.init_cache(model.cfg, dev, 1)
        if hook:
            hook.reset()
        logits = model(seq, past_kv_cache=cache)
        st = hook.commit() if hook else (0.0, False)
        return cache, logits[0, -1], st

    try:
        cache, last, (rho, gate) = full_pass(toks)
        prompt_gate = bool(gate)
        if hook and mode == "prompt_only":
            hook.forced = prompt_gate
        seq = toks
        out, rhos, gates = [], [], []
        first = 0 if prompt_gate else None
        n_rebuilds, reason = 0, "max_new"
        for step in range(1, max_new + 1):
            nxt = int(last.argmax())
            out.append(nxt)
            seq = torch.cat([seq, torch.tensor([[nxt]], device=dev)], dim=1)
            if nxt in end:
                rhos.append(rho)
                gates.append(bool(gate))
                reason = "eos"
                break
            logits = model(seq[:, -1:], past_kv_cache=cache)
            if hook:
                _, _, g_new, _ = hook.pending
                if mode == "stream" and bool(g_new) != bool(gate):
                    # Gate flipped: recompute everything under the new state (exact semantics).
                    cache, last, (rho, gate) = full_pass(seq)
                    n_rebuilds += 1
                else:
                    rho, gate = hook.commit()
                    last = logits[0, -1]
            else:
                last = logits[0, -1]
            rhos.append(float(rho))
            gates.append(bool(gate))
            if gate and first is None:
                first = step
        return GenResult(toks.shape[1], out, rhos, gates, list(hook.fires) if hook else [],
                         prompt_gate, first, n_rebuilds, reason)
    finally:
        model.reset_hooks()


def decode(model, ids) -> str:
    return model.tokenizer.decode(ids, skip_special_tokens=True)
