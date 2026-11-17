"""P9 redaction engine (GPU): batched layer-0..8 gate scoring, blame, raw evidence, the four variants with their
re-check rounds, attention blocking, and a streaming detector that can skip blocked positions.

Gate = frozen StreamGuard v2 (experiments.V2): canonical Gemma Scope L8 SAE, E4a's 20 L08 features, Bernoulli LLR,
CUSUM max over the non-BOS positions, fire iff > 7.084529876708984. The FIRST decision on an unredacted prompt always
uses experiments.V2.method.l08_score (bit-identical to v2). Re-checks of redacted sequences use `Scorer.scores`
(batch of one for the decision; batched rows only to search for the count), CUSUM in float64.
"""
import hashlib
import time

import numpy as np
import torch

from experiments.P9.common import MAX_ROUNDS, NEUTRAL_ID, TOKEN_BUDGET
from experiments.V2.common import THRESHOLD

GATE_LAYER = 8


def cusum_max_rows(llr: np.ndarray) -> np.ndarray:
    """Row-wise CUSUM max (drift 0, start 0): max_t (C_t - min_{s<=t} C_s), C_0 = 0. float64."""
    c = np.concatenate([np.zeros((llr.shape[0], 1)), np.cumsum(llr.astype(np.float64), axis=1)], axis=1)
    return (c - np.minimum.accumulate(c, axis=1)).max(axis=1)


class AttnBlock:
    """hook_attn_scores [B, H, Q, K] (after the causal mask): -inf where key position k is blocked for this row and
    the query's absolute position is > k (later positions cannot read a blocked token; the token itself can)."""

    def __init__(self, mask: torch.Tensor):
        self.mask = mask          # [B, n] bool (absolute positions; positions beyond n are never blocked)

    def __call__(self, scores, hook=None):
        B, H, Q, K = scores.shape
        m = self.mask
        if m.shape[1] < K:
            m = torch.cat([m, torch.zeros(m.shape[0], K - m.shape[1], dtype=torch.bool, device=m.device)], 1)
        m = m[:, :K]
        if m.shape[0] != B:
            m = m.expand(B, -1)
        q = torch.arange(K - Q, K, device=scores.device)
        k = torch.arange(K, device=scores.device)
        later = q[:, None] > k[None, :]
        blk = m[:, None, None, :] & later[None, None]
        return scores.masked_fill(blk, float("-inf"))


def add_block_hooks(model, mask, layers=None):
    h = AttnBlock(mask)
    for l in (range(model.cfg.n_layers) if layers is None else layers):
        model.add_hook(f"blocks.{l}.attn.hook_attn_scores", h)
    return h


class Scorer:
    def __init__(self, model, sae8, feats, w1, w0, threshold=THRESHOLD):
        self.m, self.sae8, self.feats, self.w1, self.w0 = model, sae8, feats, w1, w0
        self.threshold = float(threshold)

    @torch.no_grad()
    def llr(self, toks: torch.Tensor, block: torch.Tensor | None = None) -> np.ndarray:
        """toks [B, n] (no padding, equal length) -> per non-BOS position LLR [B, n-1] float64, in token batches.
        block [B, n] bool: attention blocking at layers 0..8 for each row."""
        B, n = toks.shape
        step = max(1, TOKEN_BUDGET // n)
        out = []
        for s in range(0, B, step):
            t = toks[s:s + step]
            self.m.reset_hooks()
            if block is not None:
                add_block_hooks(self.m, block[s:s + step], range(GATE_LAYER + 1))
            try:
                r = self.m(t, stop_at_layer=GATE_LAYER + 1)
            finally:
                self.m.reset_hooks()
            x = (self.sae8.encode(r)[:, 1:, self.feats] > 0).float()
            out.append((x @ self.w1 + (1 - x) @ self.w0).double().cpu().numpy())
            del r, x
        return np.concatenate(out)

    def scores(self, toks, block=None, exclude=None) -> np.ndarray:
        """Gate scores [B]; exclude [B, n] bool: those positions' own increments are dropped (set to 0)."""
        l = self.llr(toks, block)
        if exclude is not None:
            l = np.where(exclude[:, 1:].cpu().numpy(), 0.0, l)
        return cusum_max_rows(l)


def replaced(toks: torch.Tensor, pos) -> torch.Tensor:
    t = toks.clone()
    if len(pos):
        t[0, torch.as_tensor(list(pos), device=t.device)] = NEUTRAL_ID
    return t


def pos_mask(n: int, pos, device) -> torch.Tensor:
    m = torch.zeros(1, n, dtype=torch.bool, device=device)
    if len(pos):
        m[0, torch.as_tensor(list(pos), device=device)] = True
    return m


def blame(sc: Scorer, toks, cand, s0: float) -> np.ndarray:
    """s0 - score(toks with candidate i alone replaced), for every candidate (batched rows)."""
    if not len(cand):
        return np.zeros(0)
    rows = toks.repeat(len(cand), 1)
    idx = torch.arange(len(cand), device=toks.device)
    rows[idx, torch.as_tensor(cand, device=toks.device)] = NEUTRAL_ID
    return s0 - sc.scores(rows)


def greedy_count(sc: Scorer, toks, order, base=(), mode="replace"):
    """Smallest k such that applying base + order[:k] gives a score <= threshold (rows k = 1..len(order), scanned in
    token batches; a batch-of-one re-check confirms the decision). mode replace | block.
    -> (k | None if exhausted, final score of the applied set)."""
    n = toks.shape[1]
    base = list(base)
    step = max(1, TOKEN_BUDGET // n)
    k0 = 1
    while k0 <= len(order):
        ks = list(range(k0, min(len(order), k0 + step - 1) + 1))
        if mode == "replace":
            rows = torch.cat([replaced(toks, base + list(order[:k])) for k in ks])
            s = sc.scores(rows)
        else:
            rows = toks.repeat(len(ks), 1)
            m = torch.cat([pos_mask(n, base + list(order[:k]), toks.device) for k in ks])
            s = sc.scores(rows, block=m, exclude=m)
        for k, v in zip(ks, s):
            if v <= sc.threshold:
                final = recheck(sc, toks, base + list(order[:k]), mode)
                if final <= sc.threshold:
                    return k, final
        k0 = ks[-1] + 1
    return None, recheck(sc, toks, base + list(order), mode)


def recheck(sc: Scorer, toks, pos, mode) -> float:
    if mode == "replace":
        return float(sc.scores(replaced(toks, pos))[0])
    m = pos_mask(toks.shape[1], pos, toks.device)
    return float(sc.scores(toks, block=m, exclude=m)[0])


def item_rng(key: str) -> np.random.Generator:
    return np.random.default_rng(int(hashlib.sha256(f"P9|{key}".encode()).hexdigest()[:12], 16))


class Redactor:
    """Per fired sequence: blame once, then each variant's state. A state = {pos, mode, rounds, fallback, score}."""

    def __init__(self, sc: Scorer, toks, cand, s0: float, key: str):
        self.sc, self.toks, self.cand, self.s0, self.key = sc, toks, list(cand), float(s0), key
        t0 = time.perf_counter()
        self.blame = blame(sc, toks, self.cand, s0)
        # raw evidence: the gate's own increment at each candidate position (LLR index = position - 1)
        self.llr = sc.llr(toks)[0]
        self.ev = np.array([self.llr[p - 1] for p in self.cand])
        torch.cuda.synchronize()
        self.t_blame = time.perf_counter() - t0
        self.order_pos = [self.cand[i] for i in np.lexsort((np.array(self.cand), -self.blame))
                          if self.blame[i] > 0] if self.cand else []
        self.order_all = [self.cand[i] for i in np.lexsort((np.array(self.cand), -self.blame))] if self.cand else []
        self.k1 = None
        self.timing = {}
        self.rng = item_rng(key)

    # ---- round 1 -------------------------------------------------------------------------------------------
    def ra1(self):
        t0 = time.perf_counter()
        k, s = greedy_count(self.sc, self.toks, self.order_pos)
        pos = self.order_pos if k is None else self.order_pos[:k]
        self.k1 = len(pos)
        st = {"pos": list(pos), "mode": "replace", "rounds": 1, "fallback": False, "score": s, "exhausted1": k is None}
        torch.cuda.synchronize()
        self.timing["R-a"] = self.t_blame + time.perf_counter() - t0
        return st

    def rb1(self, ra):
        pos = list(ra["pos"][: self.k1])
        s = recheck(self.sc, self.toks, pos, "block")
        return {"pos": pos, "mode": "block", "rounds": 1, "fallback": False, "score": s}

    def rc1(self):
        k = self.k1 or 0
        idx = np.lexsort((np.array(self.cand), -self.ev))[:k] if k else []
        pos = [self.cand[i] for i in idx]
        return {"pos": pos, "mode": "replace", "rounds": 1, "fallback": False,
                "score": recheck(self.sc, self.toks, pos, "replace")}

    def rd1(self):
        k = min(self.k1 or 0, len(self.cand))
        pos = sorted(int(x) for x in self.rng.choice(self.cand, size=k, replace=False)) if k else []
        return {"pos": pos, "mode": "replace", "rounds": 1, "fallback": False,
                "score": recheck(self.sc, self.toks, pos, "replace")}

    def first_round(self) -> dict:
        ra = self.ra1()
        return {"R-a": ra, "R-b": self.rb1(ra), "R-c": self.rc1(), "R-d": self.rd1()}

    # ---- further rounds (on the prompt or mid-generation; seq may be longer than the original) ----------------
    def next_round(self, v: str, st: dict, seq: torch.Tensor, cand: list[int]) -> dict:
        """One more round for variant v on sequence seq (prompt + any generated tokens), candidates cand (not yet
        redacted). Returns the new state; fallback=True if the round budget is used up or the round fails."""
        if st["rounds"] >= MAX_ROUNDS:
            return {**st, "fallback": True}
        sc, done = self.sc, set(st["pos"])
        rest = [p for p in cand if p not in done]
        mode = st["mode"]
        if v == "R-a":
            cur = replaced(seq, st["pos"])
            s_cur = float(sc.scores(cur)[0])
            b2 = blame(sc, cur, rest, s_cur)
            order = [rest[i] for i in np.lexsort((np.array(rest), -b2)) if b2[i] > 0] if rest else []
            k, s = greedy_count(sc, seq, order, base=st["pos"])
            add = order if k is None else order[:k]
        elif v == "R-b":
            orig = [p for p in self.order_all if p in set(rest)]
            extra = [p for p in rest if p not in set(orig)]          # generated tokens: by their own replace-blame
            if extra:
                cur = seq
                b2 = blame(sc, cur, extra, float(sc.scores(cur)[0]))
                extra = [extra[i] for i in np.lexsort((np.array(extra), -b2))]
            order = orig + extra
            k, s = greedy_count(sc, seq, order, base=st["pos"], mode="block")
            add = order if k is None else order[:k]
        elif v == "R-c":
            cur = replaced(seq, st["pos"])
            l = sc.llr(cur)[0]
            ev = np.array([l[p - 1] for p in rest]) if rest else np.zeros(0)
            k = min(self.k1 or 0, len(rest))
            add = [rest[i] for i in np.lexsort((np.array(rest), -ev))[:k]] if k else []
            s = recheck(sc, seq, st["pos"] + add, mode)
        else:
            k = min(self.k1 or 0, len(rest))
            add = sorted(int(x) for x in self.rng.choice(rest, size=k, replace=False)) if k else []
            s = recheck(sc, seq, st["pos"] + add, mode)
        new = {**st, "pos": st["pos"] + list(add), "rounds": st["rounds"] + 1, "score": s}
        new["fallback"] = bool(s > sc.threshold)
        return new


def resolve_prompt(red: Redactor, states: dict) -> dict:
    """Prompt-level re-check: a variant whose round-1 score still fires gets round 2 on the prompt; still firing ->
    fallback (v2 clamp)."""
    out = {}
    for v, st in states.items():
        if st["score"] > red.sc.threshold:
            st = red.next_round(v, st, red.toks, red.cand)
        out[v] = st
    return out
