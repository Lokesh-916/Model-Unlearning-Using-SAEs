"""Method `p1t-l08` (frozen; see experiments.P1T.common).

Pass 1: no hooks; the prompt's residual at blocks.8.hook_resid_post (true length) is encoded by the canonical Gemma
Scope L8 SAE; score = CUSUM max of the Bernoulli LLR of P1's 20 L08 features over the non-BOS positions (the exact
operations of experiments.P1.questions, so DEV scores reproduce P1's logged ones). Pass 2: the scored forward with
DSG's Cyber clamp hook (dsg-faithful paper recipe, tau -1 = forced) installed iff score > THRESHOLD; otherwise the
plain model. Batch size 1 only (a batch mixing fired and unfired prompts cannot switch the hook per row).

Records: gate_score = cusum_max = the L08 score; gate_fired = the L08 decision; for fired prompts the layer-3 DSG
records (rho, fire_trace, ...) of the clamp hook; for unfired prompts rho / recon_mse / l0 are NaN (no layer-3 hook ran).
"""
import hashlib
import json

import numpy as np
import torch

from dsgx import paths
from dsgx.methods.registry import Method, DSGFaithful, register

from experiments.E4a.questions import cusum_max
from experiments.P1T.common import CLAMP_FEATURES


def load_fit(g: dict) -> dict:
    p = paths.cache_dir() / g["fit"]
    raw = p.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    if h != g["fit_sha256"]:
        raise RuntimeError(f"P1 L08 fit file {p} sha256 {h} != frozen {g['fit_sha256']}")
    return json.loads(raw)


@torch.no_grad()
def l08_score(model, sae, layer: int, feats, w1, w0, tokens_row) -> float:
    name = f"blocks.{layer}.hook_resid_post"
    n = tokens_row.shape[1]
    _, cache = model.run_with_cache(tokens_row, stop_at_layer=layer + 1, names_filter=lambda nm: nm == name)
    x = (sae.encode(cache[name][0, :n])[1:, feats] > 0).float()
    return cusum_max((x @ w1 + (1 - x) @ w0).tolist())


@register("p1t-l08")
class P1TCyberL08(Method):
    records_gate = True

    def __init__(self, cfg, bundle, seed=0):
        from dsgx.models.loader import get_sae

        super().__init__(cfg, bundle, seed)
        g, r = cfg["gate"], dict(cfg["response"])
        if g.get("type") != "sae-cusum" or r.get("name") != "dsg-faithful" or float(r.get("tau", 0)) >= 0:
            raise ValueError("p1t-l08 supports only the frozen gate / forced DSG clamp")
        rec = load_fit(g)
        dev = bundle.device
        self.layer = int(g["layer"])
        if int(rec["layer"]) != self.layer or rec["sae_id"] != g["sae_id"]:
            raise RuntimeError("fit file layer / SAE does not match the frozen gate")
        self.sae8 = get_sae(g["release"], g["sae_id"], bundle.dtype, dev)
        self.feats = torch.tensor(rec["features"], device=dev)
        self.w1 = torch.tensor(rec["w1"], dtype=torch.float32, device=dev)
        self.w0 = torch.tensor(rec["w0"], dtype=torch.float32, device=dev)
        self.threshold = float(g["threshold"])
        self.resp = DSGFaithful(r, bundle, seed=int(r["calib_seed"]))      # calib_seed 0, independent of run seed
        if list(map(int, self.resp.features)) != CLAMP_FEATURES:
            raise RuntimeError("DSG Cyber clamp features differ from the frozen list")
        self.tau = self.threshold
        self._scores, self._fired, self._lens = [], [], []
        self.info = {"variant": cfg.get("variant"), "gate": g, "response": r, "l08_features": rec["features"],
                     "fit_sha256": g["fit_sha256"], "threshold": self.threshold, "hook": bundle.hook_name,
                     "features": list(self.resp.features), "clamp_tau": self.resp.tau,
                     "multiplier": self.resp.multiplier, "cache": self.resp.info.get("cache"), "two_pass": True}

    def install(self):
        self.bundle.model.reset_hooks()

    def remove(self):
        self.bundle.model.reset_hooks()

    def set_lengths(self, lengths):
        self.resp.set_lengths(lengths)

    @torch.no_grad()
    def before_forward(self, tokens, lengths):
        if len(lengths) != 1:
            raise ValueError("p1t-l08 requires batch size 1")
        m = self.bundle.model
        m.reset_hooks()
        s = l08_score(m, self.sae8, self.layer, self.feats, self.w1, self.w0, tokens[0:1, :lengths[0]])
        fired = s > self.threshold
        self._scores.append(s)
        self._fired.append(fired)
        self._lens.append(int(lengths[0]))
        if fired:
            m.add_hook(self.bundle.hook_name, self.resp.hook)

    def pop_records(self):
        hook_recs = self.resp.pop_records()
        if len(hook_recs) != sum(self._fired):
            raise RuntimeError(f"{len(hook_recs)} clamp records vs {sum(self._fired)} fired prompts")
        it = iter(hook_recs)
        out = []
        for s, f, n in zip(self._scores, self._fired, self._lens):
            if f:
                r = next(it)
            else:
                r = {"rho": float("nan"), "n_sel_fired": 0, "sel_max": [], "top_other_ids": [], "top_other_vals": [],
                     "recon_mse": float("nan"), "l0": float("nan"), "fire_trace": np.zeros(n, dtype=np.uint8)}
            r["gate_fired"] = bool(f)
            r["gate_score"] = float(s)
            r["cusum_max"] = float(s)
            out.append(r)
        self._scores, self._fired, self._lens = [], [], []
        self.bundle.model.reset_hooks()
        return out
