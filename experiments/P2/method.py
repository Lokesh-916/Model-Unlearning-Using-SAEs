"""The two frozen gates side by side (P2). Nothing here changes either frozen method: it instantiates
experiments.V2.method.StreamGuardV2 (frozen v2) and experiments.P1T.method.P1TCyberL08 (frozen Cyber gate) with their
frozen configs and uses their own pass-1 score functions and clamp hooks.

  scores(tokens, n)      -> (bio L08 score, cyber L08 score), each by its own method's l08_score (exact ops of the
                            E4a / P1 questions tasks, so they reproduce the logged DEV scores bit for bit)
  install(set, lengths)  -> clamp hooks of the given set at blocks.3 (bio = v2's forced clamp_all -500 hook,
                            cyber = P1-T's forced DSG Cyber clamp hook); bio first, then cyber
  clear()                -> remove hooks, drop hook records

Combined method: set = (bio if bio score > 7.084529876708984) + (cyber if cyber score > 13.750207901000977).
Batch size 1 only.
"""
import torch

from experiments.P1T.method import P1TCyberL08
from experiments.P1T.method import l08_score as cyber_l08_score
from experiments.P2.common import BIO_METHOD, BIO_THRESHOLD, CYBER_METHOD, CYBER_THRESHOLD, clamp_set
from experiments.V2.method import StreamGuardV2
from experiments.V2.method import l08_score as bio_l08_score


class TwoGates:
    def __init__(self, bundle):
        self.bundle = bundle
        self.bio = StreamGuardV2(BIO_METHOD, bundle)
        self.cy = P1TCyberL08(CYBER_METHOD, bundle)
        if self.bio.threshold != BIO_THRESHOLD or self.cy.threshold != CYBER_THRESHOLD:
            raise RuntimeError("frozen thresholds changed")
        self.info = {"bio": self.bio.info, "cyber": self.cy.info, "order": ["bio", "cyber"]}

    @torch.no_grad()
    def scores(self, tokens, n: int):
        m = self.bundle.model
        m.reset_hooks()
        b, c = self.bio, self.cy
        sb = bio_l08_score(m, b.sae8, b.layer, b.feats, b.w1, b.w0, tokens[0:1, :n])
        sc = cyber_l08_score(m, c.sae8, c.layer, c.feats, c.w1, c.w0, tokens[0:1, :n])
        return float(sb), float(sc)

    @staticmethod
    def decide(sb: float, sc: float) -> tuple:
        return clamp_set(sb > BIO_THRESHOLD, sc > CYBER_THRESHOLD)

    def install(self, s: tuple, lengths):
        m, hook = self.bundle.model, self.bundle.hook_name
        m.reset_hooks()
        if "bio" in s:
            self.bio.hook.lengths = lengths
            self.bio.hook.forced = [True] * len(lengths)
            m.add_hook(hook, self.bio.hook)
        if "cyber" in s:
            self.cy.resp.set_lengths(lengths)
            m.add_hook(hook, self.cy.resp.hook)

    def clear(self):
        self.bundle.model.reset_hooks()
        self.bio.hook.records = []
        self.cy.resp.pop_records()
