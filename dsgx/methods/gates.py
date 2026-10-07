"""Gates, calibration and interventions shared by every experiment (rule 3.3, rule 3.7).

A *gate* turns the SAE activations of one prompt at one hook point into a scalar score; the
sequence is active if score > threshold. Thresholds are calibrated to a target benign FPR on
DEV data only (default 5%).

Score types (per sequence, BOS excluded unless noted):
  rho         DSG rate (faithful convention: denominator = full length incl. BOS)
  window      max over sliding windows of w tokens of the firing fraction
  cusum       max_t S_t, S_t = max(0, S_{t-1} + LLR_t - drift); LLR_t is a Bernoulli
              naive-Bayes log-likelihood ratio of the token's selected-feature fire pattern
              (forget vs retain fire rates from the activation cache)
  probe_sae   logistic regression on per-sequence fire rates of K candidate features
  probe_resid logistic regression on the mean residual at the hook layer
  union       POST-HOC, EXPLORATORY (PH-union): DSG's rho gate OR the cusum gate. score = UNION_BIG if
              rho > DSG's tau (dsg.calibrate_tau, 95th pct of retain-cache rates, the dsg-faithful rule) else
              cusum_max, so one threshold on the score (conformal at alpha) bounds the union's benign FPR

Interventions (applied on active sequences, at positions where a selected feature fires):
  clamp_all          every selected feature -> -multiplier (DSG-faithful)
  clamp_per_feature  only the features that fire -> -multiplier
  clamp_scaled       per-feature value multiplier * max_forget_i / mean(max_forget)
  mean_ablate        selected features -> their retain mean activation
  steer              add alpha * v (vector from `vector_path`) to the residual
  none               detection only
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from dsgx import paths
from dsgx.data import activation_cache as ac
from dsgx.methods import dsg
from dsgx.methods.registry import Method, register
from dsgx.util import atomic_write_json

EPS = 1e-4
UNION_BIG = 1e9  # union score when DSG's rho gate fires (finite: JSON-safe)


# ----------------------------------------------------------------------------- scores
def score_rho(fire: torch.Tensor, n: int) -> float:
    return float(fire[:n].sum()) / n


def score_window(fire: torch.Tensor, n: int, w: int) -> float:
    f = fire[1:n].float()
    if len(f) == 0:
        return 0.0
    if len(f) <= w:
        return float(f.mean())
    c = torch.cat([torch.zeros(1, device=f.device), f.cumsum(0)])
    return float(((c[w:] - c[:-w]) / w).max())


def llr_weights(cache, feats):
    """Per-feature Bernoulli LLR terms from forget/retain fire rates (masked tokens)."""
    p = np.clip(cache.stats("forget")["fire_rate"][feats], EPS, 1 - EPS)
    q = np.clip(cache.stats("retain")["fire_rate"][feats], EPS, 1 - EPS)
    return np.log(p / q), np.log((1 - p) / (1 - q))


def token_llr(fire_feat: torch.Tensor, w1, w0) -> torch.Tensor:
    """fire_feat [L, k] bool -> per-token LLR [L]."""
    x = fire_feat.float()
    return x @ w1 + (1 - x) @ w0


def score_cusum(llr: torch.Tensor, n: int, drift: float = 0.0) -> tuple[float, torch.Tensor]:
    s, best, trace = 0.0, 0.0, []
    for v in llr[1:n].tolist():
        s = max(0.0, s + v - drift)
        best = max(best, s)
        trace.append(s)
    return best, torch.tensor(trace)


# ----------------------------------------------------------------------------- gate spec
class Gate:
    """One gate at one hook point. Holds features, score type, threshold and probe weights."""

    def __init__(self, spec: dict, bundle, seed: int = 0):
        self.spec = dict(spec)
        self.bundle = bundle
        self.type = spec.get("type", "rho")
        self.cache = ac.open_cache(ac.model_tag(bundle), bundle.sae_release, bundle.sae_id,
                                   spec.get("forget_corpus", "bio-forget-corpus"),
                                   spec.get("retain_corpus", "wikitext"), spec.get("calib_seed", seed))
        if "features" in spec:
            self.features = [int(f) for f in spec["features"]]
        elif "features_file" in spec:
            ff = Path(spec["features_file"])
            if not ff.is_absolute() and not ff.exists():
                ff = paths.cache_dir() / "features" / spec["features_file"]
            self.features = [int(f) for f in json.loads(ff.read_text())["features"]][
                : int(spec.get("n_features", 10 ** 9))]
        else:
            self.features = dsg.select_features(self.cache, int(spec.get("n_features", 20)),
                                                float(spec.get("retain_pct", 95)))
        dev = bundle.sae.W_dec.device
        self.feat_t = torch.tensor(self.features, device=dev)
        if self.type in ("cusum", "union"):
            w1, w0 = llr_weights(self.cache, self.features)
            self.w1 = torch.tensor(w1, dtype=torch.float32, device=dev)
            self.w0 = torch.tensor(w0, dtype=torch.float32, device=dev)
        if self.type == "union":
            self.dsg_tau = float(spec["dsg_tau"]) if "dsg_tau" in spec else \
                dsg.calibrate_tau(self.cache, self.features, float(spec.get("dsg_tau_pct", 95)))
        self.probe = None
        if self.type in ("probe_sae", "probe_resid"):
            self.probe = self._fit_probe()
        self.threshold = spec.get("threshold")

    # -- probes trained on the calibration cache (forget rows vs retain rows; no eval data) --
    def _fit_probe(self):
        from sklearn.linear_model import LogisticRegression

        src = self.spec.get("probe_train", "cache")
        key = hashlib.sha256(json.dumps([self.type, self.cache.meta["key"], self.features, src,
                                         self.spec.get("probe_k", 256), self.spec.get("case", "bio"),
                                         self.spec.get("probe_n", 600)]).encode()).hexdigest()[:16]
        f = paths.cache_dir() / "gates" / f"probe_{key}.npz"
        if f.exists():
            z = np.load(f)
            return {k: z[k] for k in z.files}
        k = int(self.spec.get("probe_k", 256))
        cand = np.asarray(self.cache.candidates[:k])
        extra = {"cand": cand} if self.type == "probe_sae" else {}
        if src == "mcq-dev":
            # DEV-split prompts: forget dataset (positive) vs utility subjects (negative).
            X, n_f = self._mcq_dev_features(cand)
        elif self.type == "probe_sae":
            pos = self.cache.candidate_positions(cand)
            X = np.concatenate([np.asarray(self.cache.seqfire("forget"))[:, pos],
                                np.asarray(self.cache.seqfire("retain"))[:, pos]]).astype(np.float32)
            n_f = len(X) // 2
        else:
            X = self._resid_chunks()
            n_f = self._n_forget_chunks
        y = np.r_[np.ones(n_f), np.zeros(len(X) - n_f)]
        mu, sd = X.mean(0), X.std(0) + 1e-6
        clf = LogisticRegression(C=float(self.spec.get("C", 1.0)), max_iter=2000).fit((X - mu) / sd, y)
        out = {"coef": clf.coef_[0].astype(np.float32), "intercept": np.float32(clf.intercept_[0]),
               "mu": mu.astype(np.float32), "sd": sd.astype(np.float32), **extra}
        f.parent.mkdir(parents=True, exist_ok=True)
        np.savez(f, **out)
        return out

    @torch.no_grad()
    def _mcq_dev_features(self, cand):
        from dsgx.data.mcq import FORGET_DATASET, format_prompt, load_mcq
        from dsgx.data.splits import get_split

        case = self.spec.get("case", "bio")
        n = int(self.spec.get("probe_n", 600))
        fd = FORGET_DATASET[case]
        items = load_mcq(fd)
        pos = [format_prompt(items[i]) for i in get_split(fd, "dev")[:n]]
        neg = benign_calibration_prompts(case, n, seed=1)
        b = self.bundle
        cand_t = torch.tensor(cand, device=b.device)
        X = []
        for p in pos + neg:
            t = b.model.to_tokens(p, prepend_bos=False).to(b.device)
            _, c = b.model.run_with_cache(t, stop_at_layer=b.layer + 1, names_filter=b.hook_name)
            r = c[b.hook_name][0]
            if self.type == "probe_sae":
                a = b.sae.encode(r)
                a[0] = 0
                X.append(((a[:, cand_t] > 0).float().sum(0) / t.shape[1]).cpu().numpy())
            else:
                X.append(r[1:].float().mean(0).cpu().numpy())
        return np.array(X, dtype=np.float32), len(pos)

    @torch.no_grad()
    def _resid_chunks(self, chunk: int = 64, max_rows: int = 128):
        b = self.bundle
        feats = []
        counts = {}
        for part in ("forget", "retain"):
            toks = np.asarray(self.cache.tokens(part))[:max_rows]
            n = 0
            for row in toks:
                t = torch.tensor(row[None].astype(np.int64), device=b.device)
                _, c = b.model.run_with_cache(t, stop_at_layer=b.layer + 1, names_filter=b.hook_name)
                r = c[b.hook_name][0, 1:].float()
                r = r[: (len(r) // chunk) * chunk].reshape(-1, chunk, r.shape[-1]).mean(1)
                feats.append(r.cpu().numpy())
                n += len(r)
            counts[part] = n
        self._n_forget_chunks = counts["forget"]
        return np.concatenate(feats).astype(np.float32)

    # -- scoring --
    def score(self, resid_row: torch.Tensor, acts_row: torch.Tensor, n: int) -> dict:
        """Scores for one sequence (resid [L, d], acts [L, F] with position 0 zeroed)."""
        tgt = acts_row[:n, self.feat_t] > 0
        fire = tgt.any(dim=1)
        out = {"rho": score_rho(fire, n)}
        if self.type == "rho":
            out["score"] = out["rho"]
        elif self.type == "window":
            out["score"] = out["window_max"] = score_window(fire, n, int(self.spec.get("w", 24)))
        elif self.type == "cusum":
            llr = token_llr(tgt, self.w1, self.w0)
            out["score"], _ = score_cusum(llr, n, float(self.spec.get("drift", 0.0)))
            out["cusum_max"] = out["score"]
        elif self.type == "union":
            llr = token_llr(tgt, self.w1, self.w0)
            out["cusum_max"], _ = score_cusum(llr, n, float(self.spec.get("drift", 0.0)))
            out["dsg_fired"] = out["rho"] > self.dsg_tau
            out["score"] = UNION_BIG if out["dsg_fired"] else out["cusum_max"]
        elif self.type == "probe_sae":
            cand = torch.tensor(self.probe["cand"], device=acts_row.device)
            x = (acts_row[:n, cand] > 0).float().sum(0).cpu().numpy() / n
            out["score"] = out["probe_score"] = self._logit(x)
        elif self.type == "probe_resid":
            x = resid_row[1:n].float().mean(0).cpu().numpy()
            out["score"] = out["probe_score"] = self._logit(x)
        else:
            raise ValueError(f"unknown gate type {self.type}")
        out["fire"] = fire
        return out

    def _logit(self, x):
        p = self.probe
        return float(((x - p["mu"]) / p["sd"]) @ p["coef"] + p["intercept"])

    def active(self, s: dict) -> bool:
        return s["score"] > self.threshold


# ----------------------------------------------------------------------------- calibration
def benign_calibration_prompts(case: str, n_max: int = 1000, seed: int = 0) -> list[str]:
    """DEV-split utility MMLU prompts (never test), stratified across subjects."""
    from dsgx.data.mcq import format_prompt, load_mcq, utility_subjects
    from dsgx.data.splits import get_split

    rng = np.random.default_rng(seed)
    subs = utility_subjects(case)
    per = max(1, n_max // len(subs))
    out = []
    for s in subs:
        ids = get_split(s, "dev")
        pick = rng.choice(ids, size=min(per, len(ids)), replace=False)
        items = load_mcq(s)
        out += [format_prompt(items[int(i)]) for i in sorted(pick)]
    return out[:n_max]


@torch.no_grad()
def tofu_retain_dev_prompts(n_max: int = 1000, seed: int = 0) -> list[str]:
    """TOFU retain DEV (PH-tofucal): retain90 QA pairs whose question is in neither TOFU test set (retain_perturbed,
    forget10_perturbed), as question + answer in the TOFU fine-tune chat format (the sequence a TOFU gate sees)."""
    from datasets import load_dataset

    from experiments.A2.tofu import CHAT

    test = {x["question"] for c in ("retain_perturbed", "forget10_perturbed")
            for x in load_dataset("locuslab/TOFU", c, split="train")}
    pool = [x for x in load_dataset("locuslab/TOFU", "retain90", split="train") if x["question"] not in test]
    pick = sorted(np.random.default_rng(seed).choice(len(pool), size=min(n_max, len(pool)), replace=False))
    return [CHAT.format(q=pool[int(i)]["question"]) + pool[int(i)]["answer"] for i in pick]


def gate_scores(bundle, gate: Gate, prompts: list[str]) -> np.ndarray:
    m = bundle.model
    out = []
    for p in prompts:
        t = m.to_tokens(p, prepend_bos=False).to(bundle.device)
        _, c = m.run_with_cache(t, stop_at_layer=bundle.layer + 1, names_filter=bundle.hook_name)
        r = c[bundle.hook_name][0]
        a = bundle.sae.encode(r)
        a[0] = 0
        out.append(gate.score(r, a, t.shape[1])["score"])
    return np.array(out)


def conformal_threshold(scores, alpha: float) -> float:
    """Split-conformal threshold (N6): the ceil((n+1)(1-alpha))-th smallest benign score, so a new
    exchangeable benign prompt exceeds it with probability <= alpha."""
    s = np.sort(np.asarray(scores, dtype=float))
    n = len(s)
    if n == 0:
        return float("inf")
    k = int(np.ceil((n + 1) * (1 - alpha)))
    return float(s[min(k, n) - 1]) if k <= n else float("inf")


def calibrate(bundle, gate: Gate, case: str, fpr: float = 0.05, n_max: int = 1000, source="mmlu-dev",
              rule: str = "quantile"):
    """Threshold on benign DEV scores, cached per gate definition.
    rule 'quantile' (default): the (1 - fpr) quantile; 'conformal': split-conformal at alpha = fpr (N6)."""
    parts = [gate.spec, gate.features, gate.cache.meta["key"], case, fpr, n_max, source]
    if rule != "quantile":  # keeps the cache key (and threshold) of every existing quantile gate unchanged
        parts.append(rule)
    key = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:16]
    f = paths.cache_dir() / "gates" / f"thr_{key}.json"
    if f.exists():
        rec = json.loads(f.read_text())
    else:
        if source == "cache-retain" and gate.type == "rho":
            s = gate.cache.seq_fire_rate("retain", gate.features)
        elif source == "tofu-retain-dev":
            s = gate_scores(bundle, gate, tofu_retain_dev_prompts(n_max))
        else:
            s = gate_scores(bundle, gate, benign_calibration_prompts(case, n_max))
        if rule == "conformal":
            thr = conformal_threshold(s, fpr)
        else:
            thr = float(np.quantile(s, 1 - fpr, method="higher")) if len(s) else float("inf")
        rec = {"threshold": thr, "fpr_target": fpr, "n": int(len(s)), "source": source, "rule": rule,
               "empirical_fpr": float((s > thr).mean()) if len(s) else None}
        atomic_write_json(f, rec)
    gate.threshold = rec["threshold"]
    return rec


def stream_gate(gate: Gate):
    """(features, threshold, score_fn, token_fn) for dsgx.gen.stream.generate, so a calibrated rho / window /
    cusum gate decides exactly as Gate.score does on the same positions (position 0 = BOS, never firing).
    The threshold must be set (calibrate first)."""
    assert gate.threshold is not None, "calibrate the gate first"
    if gate.type == "rho":
        return gate.features, float(gate.threshold), None, None
    if gate.type == "window":
        w = int(gate.spec.get("w", 24))
        return gate.features, float(gate.threshold), (lambda v: score_window(torch.tensor(v), len(v), w)), None
    if gate.type == "cusum":
        drift = float(gate.spec.get("drift", 0.0))

        def token_fn(fire_feat):
            return token_llr(fire_feat, gate.w1, gate.w0).tolist()

        return gate.features, float(gate.threshold), (lambda v: score_cusum(torch.tensor(v), len(v), drift)[0]), token_fn
    if gate.type == "union":
        drift, tau_dsg = float(gate.spec.get("drift", 0.0)), float(gate.dsg_tau)

        def token_fn(fire_feat):  # (any selected feature fires, cusum LLR) per position
            return list(zip(fire_feat.any(dim=1).tolist(), token_llr(fire_feat, gate.w1, gate.w0).tolist()))

        def score_fn(v):
            if sum(f for f, _ in v) / len(v) > tau_dsg:
                return UNION_BIG
            return score_cusum(torch.tensor([x for _, x in v]), len(v), drift)[0]

        return gate.features, float(gate.threshold), score_fn, token_fn
    raise ValueError(f"gate type {gate.type} has no streaming form")


# ----------------------------------------------------------------------------- interventions
class Intervention:
    def __init__(self, spec: dict, gate: Gate):
        self.type = spec.get("type", "clamp_all")
        self.mult = float(spec.get("multiplier", 500))
        self.gate = gate
        dev = gate.feat_t.device
        if self.type == "clamp_scaled":
            mx = gate.cache.stats("forget")["max"][gate.features]
            self.values = torch.tensor(-self.mult * mx / max(mx.mean(), 1e-9), dtype=torch.float32, device=dev)
        if self.type == "mean_ablate":
            self.values = torch.tensor(gate.cache.stats("retain")["mean"][gate.features], dtype=torch.float32, device=dev)
        if self.type == "steer":
            self.vec = torch.tensor(np.load(spec["vector_path"]), dtype=torch.float32, device=dev)
            self.alpha = float(spec.get("alpha", 1.0))

    def apply(self, resid, acts, err, b, n, fire):
        """Modify sequence b in place; returns the new residual row."""
        f = self.gate.feat_t
        if self.type == "none":
            return resid[b]
        if self.type == "steer":
            out = resid[b].clone()
            out[:n][fire] = out[:n][fire] + self.alpha * self.vec.to(out.dtype)
            return out
        a = acts[b].clone()
        tgt = a[:n, f]
        if self.type == "clamp_all":
            new = torch.where(fire[:, None], torch.full_like(tgt, -self.mult), tgt)
        elif self.type == "clamp_per_feature":
            new = torch.where(tgt > 0, torch.full_like(tgt, -self.mult), tgt)
        elif self.type in ("clamp_scaled", "mean_ablate"):
            vals = self.values.to(tgt.dtype)[None].expand_as(tgt)
            m = fire[:, None] if self.type == "mean_ablate" else (tgt > 0)
            new = torch.where(m, vals, tgt)
        else:
            raise ValueError(self.type)
        a[:n, f] = new
        return self.gate.bundle.sae.decode(a) + err[b]


# ----------------------------------------------------------------------------- method
class _GateHook:
    def __init__(self, gate: Gate, interv: Intervention, record=True):
        self.gate, self.interv, self.record = gate, interv, record
        self.lengths = None
        self.forced = None   # list of bools from a composite decision (two-pass gates)
        self.records = []

    def __call__(self, resid, hook=None):
        sae = self.gate.bundle.sae
        acts = sae.encode(resid)
        acts[:, 0, :] = 0.0
        err = resid - sae.decode(acts)
        B, L, _ = resid.shape
        lengths = self.lengths or [L] * B
        out = resid.clone()
        for b in range(B):
            n = lengths[b]
            s = self.gate.score(resid[b], acts[b], n)
            on = self.gate.active(s) if self.forced is None else bool(self.forced[b])
            if on:
                out[b] = self.interv.apply(resid, acts, err, b, n, s["fire"])
            if self.record:
                self.records.append({"rho": s["rho"], "gate_score": s["score"], "gate_fired": bool(on),
                                     "window_max": s.get("window_max"), "cusum_max": s.get("cusum_max"),
                                     "probe_score": s.get("probe_score"),
                                     "n_sel_fired": int((acts[b, 1:n][:, self.gate.feat_t] > 0).any(0).sum()),
                                     "fire_trace": s["fire"].cpu().numpy().astype(np.uint8)})
        return out


def _bundle_for(spec, base_bundle):
    from dsgx.models.loader import get_bundle

    if "sae_id" not in spec and "sae_release" not in spec:
        return base_bundle
    return get_bundle(base_bundle.model_name, spec.get("sae_release", base_bundle.sae_release),
                      spec.get("sae_id", base_bundle.sae_id), base_bundle.dtype, base_bundle.device)


@register("gated")
class Gated(Method):
    """cfg: {name: gated, gate: {...}, intervention: {...}, calib: {fpr, n_max, source}}"""
    records_gate = True

    def __init__(self, cfg, bundle, seed=0):
        super().__init__(cfg, bundle, seed)
        gspec = dict(cfg.get("gate", {"type": "rho"}))
        gspec.setdefault("case", cfg.get("case", "bio"))
        self.gbundle = _bundle_for(gspec, bundle)
        self.gate = Gate(gspec, self.gbundle, seed)
        cal = cfg.get("calib", {})
        case = cfg.get("case", "bio")
        if self.gate.threshold is None:
            self.calib = calibrate(self.gbundle, self.gate, case, float(cal.get("fpr", 0.05)),
                                   int(cal.get("n_max", 1000)), cal.get("source", "mmlu-dev"),
                                   cal.get("rule", "quantile"))
        else:
            self.calib = {"threshold": self.gate.threshold, "source": "given"}
        self.tau = self.gate.threshold
        self.hook = _GateHook(self.gate, Intervention(cfg.get("intervention", {"type": "clamp_all"}), self.gate))
        self.info = {"features": self.gate.features, "gate": gspec, "calib": self.calib,
                     "intervention": cfg.get("intervention", {"type": "clamp_all"}),
                     "hook": self.gbundle.hook_name, "cache": str(self.gate.cache.path)}

    def install(self):
        self.bundle.model.reset_hooks()
        self.bundle.model.add_hook(self.gbundle.hook_name, self.hook)

    def set_lengths(self, lengths):
        self.hook.lengths = lengths

    def pop_records(self):
        r, self.hook.records = self.hook.records, []
        return r


@register("composite")
class Composite(Method):
    """Several gates (possibly at different layers) combined by 'any' / 'all' / 'majority'.

    Two passes per batch: pass 1 (no intervention, stopped after the deepest gate layer) scores
    every gate; pass 2 runs with each gate's intervention forced to the combined decision.
    cfg: {name: composite, combine: any|all|majority, gates: [{gate: {...}, intervention: {...},
          calib: {...}, intervene: true}, ...]}
    """
    records_gate = True

    def __init__(self, cfg, bundle, seed=0):
        super().__init__(cfg, bundle, seed)
        self.combine = cfg.get("combine", "any")
        self.members = []
        for g in cfg["gates"]:
            sub = Gated({"name": "gated", "case": cfg.get("case", "bio"), **g}, bundle, seed)
            sub.intervene = g.get("intervene", True)
            self.members.append(sub)
        self.tau = None
        self.info = {"combine": self.combine, "members": [m.info for m in self.members]}
        self._pending = []

    def install(self):
        self.bundle.model.reset_hooks()
        for m in self.members:
            if m.intervene:
                self.bundle.model.add_hook(m.gbundle.hook_name, m.hook)

    def set_lengths(self, lengths):
        for m in self.members:
            m.hook.lengths = lengths

    @torch.no_grad()
    def before_forward(self, tokens, lengths):
        model = self.bundle.model
        votes = []
        for m in self.members:
            m.hook.forced = None
        names = {m.gbundle.hook_name for m in self.members}
        deepest = max(m.gbundle.layer for m in self.members)
        model.reset_hooks()
        _, c = model.run_with_cache(tokens, stop_at_layer=deepest + 1, names_filter=lambda n: n in names)
        per = []
        for m in self.members:
            r = c[m.gbundle.hook_name]
            a = m.gbundle.sae.encode(r)
            a[:, 0] = 0
            ss = [m.gate.score(r[b], a[b], lengths[b]) for b in range(len(lengths))]
            per.append(ss)
            votes.append([m.gate.active(s) for s in ss])
        v = np.array(votes)  # [G, B]
        dec = {"any": v.any(0), "all": v.all(0), "majority": v.sum(0) * 2 > len(self.members)}[self.combine]
        for m in self.members:
            m.hook.forced = list(dec)
            m.hook.record = False
        self._pending = [{"rho": per[0][b]["rho"], "gate_score": float(np.mean([p[b]["score"] for p in per])),
                          "gate_fired": bool(dec[b]), "layer_gate_scores": [p[b]["score"] for p in per],
                          "member_votes": v[:, b].tolist(), "fire_trace": per[0][b]["fire"].cpu().numpy().astype(np.uint8)}
                         for b in range(len(lengths))]
        self.install()

    def pop_records(self):
        r, self._pending = self._pending, []
        return r
