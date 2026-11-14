"""Method `sg2-arm`: frozen StreamGuard v2 (experiments.V2.method.StreamGuardV2: same pass-1 L08 score, same frozen
threshold, same layer-3 clamp object) with an E5 arm as the response when the gate fires. MCQ knowledge probe only.

cfg: experiments.V2.common.METHOD + {name: sg2-arm, module: experiments.E5.method, arm: B | C | D | E, e5_exp: E5}.
  B  refusal direction at a* (no clamp)      C  top SAE refusal latent at its Step-2 level (no clamp)
  D  clamp + direction at a*                 E  direction at a* x (0.25 + 0.75 p), p from the prompt's L08 score
The steering vector is added at blocks.L.hook_resid_post on all positions of the rows whose gate fired. The clamp
(D) is v2's own _GateHook with forced = the decision; for B / C / E the same hook runs with forced all-False (it then
only records), and pop_records restores gate_fired = the decision. Arm A is v2 itself (V2-devcheck clean run).
"""
import json

import torch

from dsgx import paths
from dsgx.methods.registry import register

from experiments.E5.common import CANONICAL_L0, SAE_RELEASE, prop_strength
from experiments.V2.method import StreamGuardV2


class _RowSteer:
    def __init__(self):
        self.vecs = None   # [B, d] or None

    def __call__(self, resid, hook=None):
        if self.vecs is None:
            return resid
        return resid + self.vecs.to(resid.dtype).unsqueeze(1)


@register("sg2-arm")
class StreamGuardV2Arm(StreamGuardV2):
    def __init__(self, cfg, bundle, seed=0):
        from dsgx.models.loader import get_sae

        from experiments.E5.gen import ref_scores
        from experiments.E5.refusal import load_dir

        cfg = dict(cfg)
        super().__init__(cfg, bundle, seed)
        self.arm = cfg["arm"]
        if self.arm not in ("B", "C", "D", "E"):
            raise ValueError(f"arm {self.arm}")
        e5 = cfg.get("e5_exp", "E5")
        rd = paths.runs_dir() / e5
        s1 = json.loads((rd / "refusal" / "metrics.json").read_text())
        if not s1["selection"]["eligible"]:
            raise RuntimeError("E5 Step 1 selected no cell: the MCQ arms do not run")
        s2 = json.loads((rd / "sae" / "metrics.json").read_text())
        self.L, self.s_star = int(s1["selection"]["layer"]), float(s1["selection"]["s"])
        d, norm = load_dir(e5, "base", self.L)
        self.v_star = self.s_star * norm * torch.tensor(d, dtype=torch.float32, device=bundle.device)
        if self.arm == "C":
            sae = get_sae(SAE_RELEASE, f"layer_{self.L}/width_16k/average_l0_{CANONICAL_L0[self.L]}", "float32", bundle.device)
            j0 = int(s2["top"][0]["latent"])
            self.v_arm = float(s2["chosen"]["level"]) * sae.W_dec[j0].detach().float().to(bundle.device)
        else:
            self.v_arm = self.v_star
        self.ref = ref_scores() if self.arm == "E" else None
        self.steer = _RowSteer()
        self._dec = []
        self.info.update({"arm": self.arm, "e5_exp": e5, "steer_layer": self.L, "s_star": self.s_star,
                          "latent": s2["top"][0]["latent"] if self.arm == "C" else None})

    def install(self):
        m = self.bundle.model
        m.reset_hooks()
        m.add_hook(self.bundle.hook_name, self.hook)
        m.add_hook(f"blocks.{self.L}.hook_resid_post", self.steer)

    @torch.no_grad()
    def before_forward(self, tokens, lengths):
        super().before_forward(tokens, lengths)   # pass 1 scores, hook.forced = decisions, install()
        self._dec = list(self.hook.forced)
        if self.arm != "D":
            self.hook.forced = [False] * len(self._dec)
        rows = []
        self._E = []
        for s, on in zip(self._scores, self._dec):
            if not on:
                rows.append(torch.zeros_like(self.v_star))
                self._E.append(None)
            elif self.arm == "E":
                st, p = prop_strength(s, self.ref, self.s_star)
                rows.append((st / self.s_star) * self.v_star)
                self._E.append(st)
            else:
                rows.append(self.v_arm)
                self._E.append(None)
        self.steer.vecs = torch.stack(rows)

    def pop_records(self):
        dec, E = self._dec, self._E
        recs = super().pop_records()
        for r, on, st in zip(recs, dec, E):
            r["gate_fired"] = bool(on)
            r["e5_strength"] = st
        self.steer.vecs = None
        return recs
