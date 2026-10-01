"""DSG: feature selection, threshold calibration and the clamp hook (one implementation, flags).

dsg-faithful reproduces main's logic exactly:
  * features: get_top_features_percentile (score = mean activation squared, forget >= 5th pct,
    forget/retain ratio >= retain_pct-th pct, sorted by forget score), on scores read back from
    the '%f'-formatted sparsity text files, which we emulate by rounding to 6 decimals;
  * tau: 95th percentile over retain rows of (# positions where any selected feature > 0) / L,
    position 0 zeroed, denominator = full row length;
  * hook: anthropic_clamp_resid_SAE_features (BOS zeroed, rate over the whole padded length,
    on active sequences every selected feature is set to -multiplier at every position where
    any selected feature fires, SAE error added back).

dsg-fixed changes two things (ablation): BOS and right-padding are excluded from the rate
denominator, and only the features that actually fire at a position are clamped.
"""
import numpy as np
import torch

from dsgx.util import debug_enabled


def txt_round(x: np.ndarray) -> np.ndarray:
    """Values as written by np.savetxt(fmt='%f') and read back by np.loadtxt."""
    return np.array([float(f"{float(v):f}") for v in np.asarray(x)], dtype=np.float64)


def dsg_rank_features(forget_mean, retain_mean, ratio_pct: float, forget_pct: float = 5,
                      retain_pct: float = 100, round_like_txt: bool = True) -> np.ndarray:
    """Exact port of get_top_features_percentile's feature ordering (sel_ind_out)."""
    fm = txt_round(forget_mean) if round_like_txt else np.asarray(forget_mean, dtype=np.float64)
    rm = txt_round(retain_mean) if round_like_txt else np.asarray(retain_mean, dtype=np.float64)
    forget_score = fm ** 2
    retain_score = rm ** 2
    ratio = forget_score / (retain_score + 1e-21)
    sel = np.where((forget_score >= np.percentile(forget_score, forget_pct))
                   & (retain_score <= np.percentile(retain_score, retain_pct))
                   & (ratio >= np.percentile(ratio, ratio_pct)))[0]
    return sel[np.argsort(-forget_score[sel])]


def select_features(cache, n_features: int, ratio_pct: float, round_like_txt: bool = True):
    f = cache.stats("forget")["legacy_mean"]
    r = cache.stats("retain")["legacy_mean"]
    return [int(x) for x in dsg_rank_features(f, r, ratio_pct, round_like_txt=round_like_txt)[:n_features]]


def calibrate_tau(cache, features, pct: float = 95.0, exclude_special: bool = False) -> float:
    rho = cache.seq_fire_rate("retain", features, exclude_special=exclude_special)
    return float(np.percentile(rho, pct))


class DSGHook:
    """Forward hook on the SAE's residual hook point.

    The evaluator sets `lengths` (real token counts per row) before each forward when batching,
    and reads `records` afterwards. Recording never changes the output.
    """

    def __init__(self, sae, features, multiplier: float, tau: float, faithful: bool = True,
                 record: bool = True, n_top_other: int = 20):
        self.sae = sae
        self.features = list(int(f) for f in features)
        self.feat_t = torch.tensor(self.features, device=sae.W_dec.device)
        self.multiplier = float(multiplier)
        self.tau = float(tau)
        self.faithful = faithful
        self.record = record
        self.n_top_other = n_top_other
        self.lengths = None
        self.records = []
        self.enabled = True

    def __call__(self, resid, hook=None):
        if not self.enabled or len(self.features) == 0:
            return resid
        return self._faithful(resid) if self.faithful else self._fixed(resid)

    # ---- dsg-faithful: verbatim logic of anthropic_clamp_resid_SAE_features ----
    def _faithful(self, resid):
        sae, feats = self.sae, self.features
        feature_activations = sae.encode(resid)
        feature_activations[:, 0, :] = 0.0
        reconstruction = sae.decode(feature_activations)
        error = resid - reconstruction
        target_features = feature_activations[:, :, feats]
        activation_mask = target_features > 0
        activation_mask = activation_mask.sum(dim=2) > 0
        batch_activation_rates = activation_mask.sum(dim=1) / activation_mask.shape[1]
        active_batches = batch_activation_rates > self.tau
        if self.record:
            self._record(resid, feature_activations, reconstruction, target_features,
                         activation_mask, batch_activation_rates, active_batches)
        if debug_enabled():
            self._debug(resid, target_features, batch_activation_rates, active_batches)
        final_mask = activation_mask.unsqueeze(2) & active_batches.unsqueeze(1).unsqueeze(2)
        feature_activations[:, :, feats] = torch.where(
            final_mask, torch.full_like(target_features, -self.multiplier),
            feature_activations[:, :, feats])
        return sae.decode(feature_activations) + error

    # ---- dsg-fixed: BOS/pad excluded from the rate, per-feature mask ----
    def _fixed(self, resid):
        sae, feats = self.sae, self.features
        B, L, _ = resid.shape
        feature_activations = sae.encode(resid)
        feature_activations[:, 0, :] = 0.0
        reconstruction = sae.decode(feature_activations)
        error = resid - reconstruction
        target_features = feature_activations[:, :, feats]
        lengths = self.lengths if self.lengths is not None else [L] * B
        valid = torch.zeros(B, L, dtype=torch.bool, device=resid.device)
        for b, n in enumerate(lengths):
            valid[b, 1:n] = True  # exclude BOS and right padding
        per_feat = (target_features > 0) & valid[:, :, None]
        activation_mask = per_feat.any(dim=2)
        rates = activation_mask.sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        active = rates > self.tau
        if self.record:
            self._record(resid, feature_activations, reconstruction, target_features,
                         activation_mask, rates, active)
        if debug_enabled():
            self._debug(resid, target_features, rates, active)
        final_mask = per_feat & active[:, None, None]
        feature_activations[:, :, feats] = torch.where(
            final_mask, torch.full_like(target_features, -self.multiplier),
            feature_activations[:, :, feats])
        return sae.decode(feature_activations) + error

    @torch.no_grad()
    def _record(self, resid, acts, recon, target, any_fire, rates, active):
        # Recording is detached so the hook stays differentiable for white-box attacks.
        resid, acts, recon, target = resid.detach(), acts.detach(), recon.detach(), target.detach()
        lengths = self.lengths if self.lengths is not None else [resid.shape[1]] * resid.shape[0]
        other = acts.clone()
        other[:, :, self.feat_t] = 0
        omax = other.max(dim=1).values  # [B, F]
        top_v, top_i = omax.topk(self.n_top_other, dim=1)
        err = (resid - recon).float()
        for b in range(resid.shape[0]):
            n = lengths[b]
            self.records.append({
                "rho": rates[b],
                "gate_fired": active[b],
                "n_sel_fired": (target[b, 1:n] > 0).any(dim=0).sum(),
                "sel_max": target[b, :n].max(dim=0).values,
                "top_other_ids": top_i[b],
                "top_other_vals": top_v[b],
                "recon_mse": (err[b, 1:n] ** 2).mean(),
                "l0": (acts[b, 1:n] > 0).sum(dim=-1).float().mean(),
                "fire_trace": any_fire[b, :n],
            })

    def _debug(self, resid, target, rates, active):
        print(f"[DSG DEBUG] activation rate: min={rates.min().item():.4f}, max={rates.max().item():.4f}, "
              f"mean={rates.float().mean().item():.4f}, active={active.sum().item()}/{active.numel()}")
        pf = (target > 0).float().mean(dim=(0, 1))
        v, i = pf.topk(min(10, len(self.features)))
        print("[DSG DEBUG] top firing features (feature_id: rate): "
              + ", ".join(f"{self.features[j.item()]}:{x.item():.4f}" for x, j in zip(v, i)))
        print(f"[DSG DEBUG] resid norm before clamp: mean={resid.norm(dim=-1).mean().item():.4f}")

    def pop_records(self) -> list[dict]:
        """Move recorded tensors to CPU python values."""
        out = []
        for r in self.records:
            out.append({
                "rho": float(r["rho"]), "gate_fired": bool(r["gate_fired"]),
                "n_sel_fired": int(r["n_sel_fired"]),
                "sel_max": r["sel_max"].float().cpu().numpy().tolist(),
                "top_other_ids": r["top_other_ids"].cpu().numpy().tolist(),
                "top_other_vals": r["top_other_vals"].float().cpu().numpy().tolist(),
                "recon_mse": float(r["recon_mse"]), "l0": float(r["l0"]),
                "fire_trace": r["fire_trace"].cpu().numpy().astype(np.uint8),
            })
        self.records = []
        return out
