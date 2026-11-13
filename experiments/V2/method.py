"""Method `sg2-l08` = StreamGuard v2 (frozen; see experiments.V2.common).

Two passes per batch (MCQ scoring calls before_forward before the scored forward):
  pass 1  no hooks; per sequence, the residual at blocks.8.hook_resid_post up to the true length, encoded by the
          canonical Gemma Scope L8 SAE; score = CUSUM max of the Bernoulli LLR of E4a's 20 L08 features over the
          non-BOS positions -- the exact operations of experiments.E4b.scores / E4a.questions (same SAE load, same
          dtype, same float32 weights, same cusum_max), so DEV scores reproduce the logged ones;
  pass 2  the scored forward with DSG's layer-3 clamp (clamp_all -500 on DSG's 20 features at the positions where
          any of them fires) forced on iff score > the frozen threshold.
Gate records: gate_score = cusum_max = the L08 score, gate_fired = the decision, rho / fire_trace = layer-3 DSG
feature fires (as every other gated run). The fit file is checked against its pinned sha256.

method config: experiments.V2.common.METHOD
"""
import hashlib
import json

import torch

from dsgx import paths
from dsgx.methods import gates
from dsgx.methods.registry import Method, register

from experiments.E4a.questions import cusum_max


def load_fit(g: dict) -> dict:
    p = paths.cache_dir() / g["fit"]
    raw = p.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    if h != g["fit_sha256"]:
        raise RuntimeError(f"L08 fit file {p} sha256 {h} != frozen {g['fit_sha256']}")
    return json.loads(raw)


@torch.no_grad()
def l08_score(model, sae, layer: int, feats, w1, w0, tokens_row) -> float:
    """tokens_row [1, n] (no padding). Same ops as experiments.E4b.scores.task.one."""
    name = f"blocks.{layer}.hook_resid_post"
    n = tokens_row.shape[1]
    _, cache = model.run_with_cache(tokens_row, stop_at_layer=layer + 1, names_filter=lambda nm: nm == name)
    r = cache[name][0, :n]
    x = (sae.encode(r)[1:, feats] > 0).float()
    return cusum_max((x @ w1 + (1 - x) @ w0).tolist())


@register("sg2-l08")
class StreamGuardV2(Method):
    records_gate = True

    def __init__(self, cfg, bundle, seed=0):
        from dsgx.models.loader import get_sae

        super().__init__(cfg, bundle, seed)
        g, r = cfg["gate"], cfg["response"]
        if g.get("type") != "l08-sae-cusum" or r.get("type") != "clamp_all":
            raise ValueError("sg2-l08 supports only the frozen gate / response")
        rec = load_fit(g)
        dev = bundle.device
        self.layer = int(g["layer"])
        self.sae8 = get_sae(g["release"], g["sae_id"], bundle.dtype, dev)
        self.feats = torch.tensor(rec["features"], device=dev)
        self.w1 = torch.tensor(rec["w1"], dtype=torch.float32, device=dev)
        self.w0 = torch.tensor(rec["w0"], dtype=torch.float32, device=dev)
        self.threshold = float(g["threshold"])
        if int(r.get("layer", 3)) != bundle.layer:
            raise ValueError(f"response layer {r.get('layer')} != bundle layer {bundle.layer}")
        # layer-3 response: DSG's 20 features (frozen list), clamp_all -500; the gate object is used only for the
        # fire positions (type rho needs no LLR weights); the decision always comes from pass 1
        self.rgate = gates.Gate({"type": "rho", "features": list(r["features"]), "calib_seed": int(r.get("calib_seed", 0))},
                                bundle, 0)
        self.hook = gates._GateHook(self.rgate, gates.Intervention({"type": "clamp_all",
                                                                    "multiplier": float(r["multiplier"])}, self.rgate))
        self.tau = self.threshold
        self._scores = []
        self.info = {"variant": cfg.get("variant"), "gate": g, "response": r, "l08_features": rec["features"],
                     "fit_sha256": g["fit_sha256"], "threshold": self.threshold, "hook": bundle.hook_name,
                     "features": list(r["features"]), "two_pass": True}

    def install(self):
        self.bundle.model.reset_hooks()
        self.bundle.model.add_hook(self.bundle.hook_name, self.hook)

    def set_lengths(self, lengths):
        self.hook.lengths = lengths

    @torch.no_grad()
    def before_forward(self, tokens, lengths):
        m = self.bundle.model
        m.reset_hooks()
        self._scores = [l08_score(m, self.sae8, self.layer, self.feats, self.w1, self.w0, tokens[b:b + 1, :n])
                        for b, n in enumerate(lengths)]
        self.hook.forced = [s > self.threshold for s in self._scores]
        self.install()

    def pop_records(self):
        recs, self.hook.records = self.hook.records, []
        if len(recs) != len(self._scores):
            raise RuntimeError(f"{len(recs)} hook records vs {len(self._scores)} pass-1 scores")
        for rec, s in zip(recs, self._scores):
            rec["gate_score"] = s
            rec["cusum_max"] = s
        self._scores = []
        return recs
