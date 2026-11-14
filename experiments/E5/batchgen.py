"""E5 generation: the frozen v2 L08 gate in streaming detect mode, and a batched greedy continuation in which every
row (arm) has its own response (DSG -500 clamp at layer 3 on or off, steering vector at one layer or none).

Gate (frozen StreamGuard v2): per position, the canonical Gemma Scope L8 SAE (bundle dtype) encodes
blocks.8.hook_resid_post; x = 1[act > 0] of E4a's 20 L08 features; LLR = x @ w1 + (1 - x) @ w0; score = CUSUM running
max over all non-BOS positions (the ops of experiments.V2.method.l08_score, which the prompt score is checked
against item by item). Fire iff score > the frozen threshold, checked after the prompt pass and after every
generated token; CUSUM is a running max, so a fire is latched.

Phase 1: the unmodified model generates greedily (batch 1, KV cache) with the detector; it is also the ungated base
output. Phase 2 (only if fired at step s; s = 0: the prompt already fired): seq = prompt + phase-1 tokens[:s] is
repeated once per arm, one full pass with the arm hooks, then greedy continuation up to the same total max_new.
"""
import torch

from experiments.V2.common import THRESHOLD


def end_ids(model):
    tok = model.tokenizer
    ids = {tok.eos_token_id}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    if isinstance(eot, int) and eot >= 0:
        ids.add(eot)
    return ids


class L08Detector:
    def __init__(self, sae8, feats, w1, w0, threshold=THRESHOLD):
        self.sae8, self.feats, self.w1, self.w0, self.threshold = sae8, feats, w1, w0, float(threshold)
        self.first, self.s, self.best = True, 0.0, 0.0

    @torch.no_grad()
    def __call__(self, resid, hook=None):
        acts = self.sae8.encode(resid[0])
        if self.first:
            acts = acts[1:]
            self.first = False
        x = (acts[:, self.feats] > 0).float()
        for v in (x @ self.w1 + (1 - x) @ self.w0).tolist():
            self.s = max(0.0, self.s + float(v))
            self.best = max(self.best, self.s)
        return resid

    def fired(self):
        return self.best > self.threshold


class BatchClamp:
    """DSG-faithful clamp_all (-mult at positions where any of the features fires; SAE error added back; BOS kept)
    on the rows in `rows` only; the other rows pass through unchanged."""

    def __init__(self, sae, feats, rows, mult=500.0):
        self.sae, self.feats, self.mult = sae, [int(f) for f in feats], float(mult)
        self.rows = rows
        self.full = True

    @torch.no_grad()
    def __call__(self, resid, hook=None):
        acts = self.sae.encode(resid)
        if self.full:
            acts[:, 0, :] = 0.0
            self.full = False
        err = resid - self.sae.decode(acts)
        tgt = acts[:, :, self.feats]
        mask = (tgt > 0).any(dim=2, keepdim=True)
        acts[:, :, self.feats] = torch.where(mask, torch.full_like(tgt, -self.mult), tgt)
        new = self.sae.decode(acts) + err
        return torch.where(self.rows.view(-1, 1, 1), new, resid)


class BatchSteer:
    def __init__(self, vecs):
        self.vecs = vecs  # [R, d] float32 (zero rows = no steering)

    def __call__(self, resid, hook=None):
        return resid + self.vecs.to(resid.dtype).unsqueeze(1)


class StepSteer:
    """E5b: per-row steering with an optional limit on generated positions. t = number of generated tokens fed so
    far (0 during the pass over the prompt / re-fed sequence; k while generated token k is fed). Row i is steered
    iff limits[i] is None (mode i: every position) or t <= limits[i] (mode ii: prompt + first `limit` generated)."""

    def __init__(self, vecs, limits):
        self.vecs = vecs
        self.limits = list(limits)
        self.t = 0

    def __call__(self, resid, hook=None):
        if self.t == 0:
            v = self.vecs
        else:
            m = torch.tensor([lim is None or self.t <= lim for lim in self.limits], device=self.vecs.device)
            v = self.vecs * m.view(-1, 1).to(self.vecs.dtype)
        return resid + v.to(resid.dtype).unsqueeze(1)


def _steer_hook(vecs, limits):
    if limits is None or all(lim is None for lim in limits):
        return BatchSteer(vecs)
    return StepSteer(vecs, limits)


def _set_t(hooks, t):
    for h in hooks:
        if isinstance(h, StepSteer):
            h.t = t


@torch.no_grad()
def phase1(model, toks, det_hook_name=None, det=None, max_new=96):
    """-> (tokens, stop, fire_step | None, fire_score, prompt_score)."""
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

    dev = model.cfg.device
    end = end_ids(model)
    model.reset_hooks()
    if det is not None:
        model.add_hook(det_hook_name, det)
    fire, fire_score, prompt_score = None, None, None
    try:
        cache = TransformerLensKeyValueCache.init_cache(model.cfg, dev, 1)
        last = model(toks, past_kv_cache=cache)[0, -1]
        if det is not None:
            prompt_score = det.best
            if det.fired():
                fire, fire_score = 0, det.best
        out, reason = [], "max_new"
        for step in range(1, max_new + 1):
            nxt = int(last.argmax())
            out.append(nxt)
            if nxt in end:
                reason = "eos"
                break
            last = model(torch.tensor([[nxt]], device=dev), past_kv_cache=cache)[0, -1]
            if det is not None and fire is None and det.fired():
                fire, fire_score = step, det.best
    finally:
        model.reset_hooks()
    return out, reason, fire, fire_score, prompt_score


@torch.no_grad()
def continue_batch(model, seq, n_left, rows, clamp=None, steer_layer=None):
    """seq [1, n]; rows: list of {"clamp": bool, "vec": tensor [d] | None, "limit": int | None (E5b mode ii; optional)};
    clamp = (sae, feats, mult, hook_name). -> list of (new tokens, stop) per row."""
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

    dev = model.cfg.device
    end = end_ids(model)
    R = len(rows)
    hooks = []
    model.reset_hooks()
    try:
        if any(r["clamp"] for r in rows):
            sae, feats, mult, hook_name = clamp
            mask = torch.tensor([bool(r["clamp"]) for r in rows], device=dev)
            model.add_hook(hook_name, BatchClamp(sae, feats, mask, mult))
        if any(r["vec"] is not None for r in rows):
            d = model.cfg.d_model
            lay = lambda r: r.get("layer", steer_layer)  # noqa: E731  (E5b arm F may steer another layer)
            for Ls in sorted({lay(r) for r in rows if r["vec"] is not None}):
                V = torch.stack([r["vec"].float().to(dev) if (r["vec"] is not None and lay(r) == Ls)
                                 else torch.zeros(d, device=dev) for r in rows])
                steer = _steer_hook(V, [r.get("limit") for r in rows])
                model.add_hook(f"blocks.{Ls}.hook_resid_post", steer)
                hooks.append(steer)
        cache = TransformerLensKeyValueCache.init_cache(model.cfg, dev, R)
        last = model(seq.repeat(R, 1), past_kv_cache=cache)[:, -1]
        outs = [[] for _ in range(R)]
        stops = ["max_new"] * R
        live = [True] * R
        for k in range(1, n_left + 1):
            nxt = last.argmax(dim=-1)
            for i, t in enumerate(nxt.tolist()):
                if live[i]:
                    outs[i].append(t)
                    if t in end:
                        live[i], stops[i] = False, "eos"
            if not any(live):
                break
            _set_t(hooks, k)
            last = model(nxt.view(R, 1), past_kv_cache=cache)[:, -1]
    finally:
        model.reset_hooks()
    return list(zip(outs, stops))


@torch.no_grad()
def forced_batch(model, prompt, rows, steer_layers, max_new, limits=None):
    """Forced-on steering from the prompt (Steps 1-2, no gate): rows = list of (layer | None, vec | None); one hook
    per layer adds each row's vector (zero for rows of other layers). limits (E5b, optional): per-row None (every
    position) or n (prompt + first n generated positions). -> list of token lists."""
    from transformer_lens.cache.key_value_cache import TransformerLensKeyValueCache

    dev = model.cfg.device
    end = end_ids(model)
    R = len(rows)
    toks = model.to_tokens(prompt, prepend_bos=False).to(dev)
    hooks = []
    model.reset_hooks()
    try:
        d = model.cfg.d_model
        for L in steer_layers:
            V = torch.stack([v.float().to(dev) if (l == L and v is not None) else torch.zeros(d, device=dev)
                             for l, v in rows])
            if V.abs().sum() > 0:
                h = _steer_hook(V, limits)
                model.add_hook(f"blocks.{L}.hook_resid_post", h)
                hooks.append(h)
        cache = TransformerLensKeyValueCache.init_cache(model.cfg, dev, R)
        last = model(toks.repeat(R, 1), past_kv_cache=cache)[:, -1]
        outs = [[] for _ in range(R)]
        live = [True] * R
        for k in range(1, max_new + 1):
            nxt = last.argmax(dim=-1)
            for i, t in enumerate(nxt.tolist()):
                if live[i]:
                    outs[i].append(t)
                    if t in end:
                        live[i] = False
            if not any(live):
                break
            _set_t(hooks, k)
            last = model(nxt.view(R, 1), past_kv_cache=cache)[:, -1]
    finally:
        model.reset_hooks()
    return outs
