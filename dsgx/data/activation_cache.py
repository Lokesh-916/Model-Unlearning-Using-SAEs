"""Compact, memory-mapped SAE activation cache (MASTER_PLAN section 4.3).

Replaces the legacy act_fgt.pkl / act_ret.pkl (18 GB each, dense [275, 1024, 16384] float32).
The model is streamed over the calibration rows twice, one row at a time, and only these are kept:

  stats_<part>.npz     per-feature (all 16,384): legacy_mean (bit-exact replay of DSG's sparsity
                       accumulation), mean, mean_sq, fire_rate, max, n_tokens, and a log-binned
                       histogram of positive activations (quantile sketch)
  candidates.npy       top-K candidate features (default K=2048, see `choose_candidates`)
  firebits_<part>.npy  packed per-token fire bits (act > 0) for the candidates, [N, L, K/8] uint8,
                       position 0 zeroed as in the legacy pickles
  seqfire_<part>.npy   per-sequence fire-rate vectors for the candidates, [N, K] float16
  tokacts_<part>.npy   per-token activations for the candidates on the first S rows, float16
  special_<part>.npy   [N, L] bool, token is BOS / PAD / EOS
  tokens_<part>.npy    the token rows, int32

`part` is 'forget' or 'retain'. All arrays are opened with mmap_mode='r'; a typical job uses
well under 1 GB of RAM. The old artifacts are never read or deleted.
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from dsgx import paths
from dsgx.util import atomic_write_json, package_versions, stable_hash

HIST_EDGES = np.logspace(-3, 3, 65).astype(np.float32)  # 64 log bins for positive activations
PARTS = ("forget", "retain")


def cache_key(model_name, sae_release, sae_id, forget, retain, seed, dataset_size, seq_len) -> str:
    return "__".join([model_name, sae_release, sae_id.replace("/", "-"), forget, retain,
                      f"s{seed}", f"n{dataset_size}", f"L{seq_len}"])


def cache_path(key: str) -> Path:
    return paths.cache_dir() / "actcache" / key


def _special_mask(tokens: torch.Tensor, tok) -> torch.Tensor:
    return (tokens == tok.pad_token_id) | (tokens == tok.eos_token_id) | (tokens == tok.bos_token_id)


@torch.no_grad()
def _encode_row(bundle, row: torch.Tensor) -> torch.Tensor:
    """SAE activations [L, F] for one token row, exactly as DSG computes them (bs=1, bf16)."""
    _, cache = bundle.model.run_with_cache(row[None].to(bundle.device), stop_at_layer=bundle.layer + 1,
                                           names_filter=bundle.hook_name)
    return bundle.sae.encode(cache[bundle.hook_name])[0]


@torch.no_grad()
def _pass_stats(bundle, tokens: torch.Tensor, progress=None, lengths=None) -> dict:
    tok = bundle.model.tokenizer
    F = bundle.sae.W_dec.shape[0]
    dev = bundle.device
    legacy_sum = torch.zeros(F, dtype=torch.float32, device=dev)
    s1 = torch.zeros(F, dtype=torch.float64, device=dev)
    s2 = torch.zeros(F, dtype=torch.float64, device=dev)
    fires = torch.zeros(F, dtype=torch.float64, device=dev)
    mx = torch.zeros(F, dtype=torch.float32, device=dev)
    hist = torch.zeros(F * (len(HIST_EDGES) - 1), dtype=torch.int64, device=dev)
    edges = torch.tensor(HIST_EDGES, device=dev)
    n_tok = 0
    for i in range(tokens.shape[0]):
        row = tokens[i] if lengths is None else tokens[i, : int(lengths[i])]
        acts = _encode_row(bundle, row)  # [L, F] bf16
        keep = ~_special_mask(row, tok).to(dev)
        # Legacy: sum in the activation dtype per batch (bs=1), then add to a float32 buffer.
        legacy_sum += (acts[None] * keep[None, :, None]).sum(dim=(0, 1)).float()
        a = acts.float()[keep]
        n_tok += int(keep.sum())
        s1 += a.sum(0, dtype=torch.float64)
        s2 += (a.double() ** 2).sum(0)
        pos = a > 0
        fires += pos.sum(0, dtype=torch.float64)
        mx = torch.maximum(mx, a.max(0).values)
        r, c = pos.nonzero(as_tuple=True)
        if r.numel():
            b = torch.bucketize(a[r, c], edges).clamp_(1, len(HIST_EDGES) - 1) - 1
            hist.index_add_(0, c * (len(HIST_EDGES) - 1) + b, torch.ones_like(b))
        if progress:
            progress(i + 1)
    return {
        "legacy_mean": (legacy_sum / n_tok).cpu().numpy(),
        "mean": (s1 / n_tok).cpu().numpy(),
        "mean_sq": (s2 / n_tok).cpu().numpy(),
        "fire_rate": (fires / n_tok).cpu().numpy(),
        "max": mx.cpu().numpy(),
        "n_tokens": np.array(n_tok),
        "hist": hist.reshape(F, -1).cpu().numpy().astype(np.int32),
        "hist_edges": HIST_EDGES,
    }


def choose_candidates(forget_mean, retain_mean, k: int = 2048, min_ratio_pct: float = 90.0):
    """Candidate features for the per-token stores.

    First, every feature that passes DSG's own filter at the loosest ratio percentile used in
    the plan (90), ordered by forget score. That guarantees exact taus for every DSG config with
    retain percentile >= 90. Then fill to k with the highest forget score, then highest
    forget fire rate. Uses DSG's rounded scores so the faithful ordering is preserved.
    """
    from dsgx.methods.dsg import dsg_rank_features

    ranked = list(dsg_rank_features(forget_mean, retain_mean, min_ratio_pct))
    seen = set(ranked)
    for f in np.argsort(-forget_mean, kind="stable"):
        if len(ranked) >= k:
            break
        if int(f) not in seen:
            ranked.append(int(f))
            seen.add(int(f))
    return np.array(ranked[:k], dtype=np.int64)


@torch.no_grad()
def _pass_bits(bundle, tokens, cand, n_tok_sub, progress=None, lengths=None):
    N, L = tokens.shape
    K = len(cand)
    cand_t = torch.tensor(cand, device=bundle.device)
    bits = np.zeros((N, L, (K + 7) // 8), dtype=np.uint8)
    seqfire = np.zeros((N, K), dtype=np.float16)
    tokacts = np.zeros((min(n_tok_sub, N), L, K), dtype=np.float16)
    for i in range(N):
        n = L if lengths is None else int(lengths[i])
        acts = _encode_row(bundle, tokens[i, :n])[:, cand_t].float()
        acts[0] = 0.0  # legacy pickles zero position 0 only
        fire = (acts > 0).cpu().numpy()
        bits[i, :n] = np.packbits(fire, axis=1)
        seqfire[i] = fire.sum(0) / n
        if i < tokacts.shape[0]:
            tokacts[i, :n] = acts.cpu().numpy().astype(np.float16)
        if progress:
            progress(i + 1)
    return bits, seqfire, tokacts


def build_cache(bundle, forget: str = "bio-forget-corpus", retain: str = "wikitext", seed: int = 0,
                dataset_size: int = 1024, seq_len: int = 1024, k: int = 2048, n_tok_sub: int = 32,
                progress=None, force: bool = False) -> Path:
    from dsgx.data.corpora import legacy_calibration_tokens

    key = cache_key(bundle.model_name, bundle.sae_release, bundle.sae_id, forget, retain, seed,
                    dataset_size, seq_len)
    import fcntl

    out = cache_path(key)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)  # two jobs never build the same cache at once
        if (out / "meta.json").exists() and not force:
            meta = json.loads((out / "meta.json").read_text())
            if meta.get("status") == "COMPLETE":
                return out
        res = legacy_calibration_tokens(bundle.model.tokenizer, forget, retain, seed, dataset_size, seq_len)
        lengths = {"retain": res[3]} if len(res) == 4 else None
        return build_from_tokens(bundle, res[0], res[1], res[2], out, key, k, n_tok_sub, progress,
                                 lengths=lengths)


def build_from_tokens(bundle, f_tok, r_tok, info: dict, out: Path, key: str, k: int = 2048,
                      n_tok_sub: int = 32, progress=None, lengths: dict | None = None) -> Path:
    """lengths: optional {part: LongTensor[N]} for right-padded prompt rows (each row is encoded
    on its own length, i.e. batch-size-1 semantics)."""
    out = Path(out)
    lengths = {p: (lengths or {}).get(p) for p in PARTS}
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    toks = {"forget": f_tok, "retain": r_tok}
    N = max(f_tok.shape[0], r_tok.shape[0])
    total = 4 * N

    def _p(base):
        return (lambda j: progress(base + j, total)) if progress else None

    stats = {}
    for j, part in enumerate(PARTS):
        np.save(out / f"tokens_{part}.npy", toks[part].numpy().astype(np.int32))
        np.save(out / f"special_{part}.npy",
                _special_mask(toks[part], bundle.model.tokenizer).numpy())
        if lengths[part] is not None:
            np.save(out / f"lengths_{part}.npy", lengths[part].numpy().astype(np.int32))
        stats[part] = _pass_stats(bundle, toks[part], _p(j * N), lengths[part])
        np.savez(out / f"stats_{part}.npz", **stats[part])
    cand = choose_candidates(stats["forget"]["legacy_mean"], stats["retain"]["legacy_mean"], k)
    np.save(out / "candidates.npy", cand)
    for j, part in enumerate(PARTS):
        bits, seqfire, tokacts = _pass_bits(bundle, toks[part], cand, n_tok_sub, _p((2 + j) * N),
                                            lengths[part])
        np.save(out / f"firebits_{part}.npy", bits)
        np.save(out / f"seqfire_{part}.npy", seqfire)
        np.save(out / f"tokacts_{part}.npy", tokacts)
    meta = {"status": "COMPLETE", "key": key, "model": bundle.model_name,
            "sae_release": bundle.sae_release, "sae_id": bundle.sae_id, "hook": bundle.hook_name,
            "corpus": info, "k": int(len(cand)), "n_tok_sub": n_tok_sub,
            "token_hash": {p: stable_hash(toks[p].numpy().tolist(), 16) for p in PARTS},
            "build_seconds": round(time.time() - t0, 1), "versions": package_versions()}
    atomic_write_json(out / "meta.json", meta)
    return out


class ActivationCache:
    """Read-only, memory-mapped view of a built cache."""

    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        if self.meta.get("status") != "COMPLETE":
            raise RuntimeError(f"cache {self.path} is incomplete")
        self.candidates = np.load(self.path / "candidates.npy")
        self._cand_pos = {int(f): i for i, f in enumerate(self.candidates)}
        self._stats = {}

    def stats(self, part: str) -> dict:
        if part not in self._stats:
            with np.load(self.path / f"stats_{part}.npz") as z:
                self._stats[part] = {k: z[k] for k in z.files}
        return self._stats[part]

    def _load(self, name):
        return np.load(self.path / name, mmap_mode="r")

    def tokens(self, part):
        return self._load(f"tokens_{part}.npy")

    def special(self, part):
        return self._load(f"special_{part}.npy")

    def seqfire(self, part):
        return self._load(f"seqfire_{part}.npy")

    def tokacts(self, part):
        return self._load(f"tokacts_{part}.npy")

    def candidate_positions(self, features) -> np.ndarray:
        missing = [int(f) for f in features if int(f) not in self._cand_pos]
        if missing:
            raise KeyError(f"features not in cache candidates (rebuild with larger k): {missing[:10]}")
        return np.array([self._cand_pos[int(f)] for f in features])

    def fire_any(self, part: str, features, chunk: int = 32) -> np.ndarray:
        """[N, L] bool: any of `features` fires at the token (position 0 is always False)."""
        pos = self.candidate_positions(features)
        bits = self._load(f"firebits_{part}.npy")
        N, L, _ = bits.shape
        out = np.zeros((N, L), dtype=bool)
        for s in range(0, N, chunk):
            un = np.unpackbits(bits[s:s + chunk], axis=2, count=len(self.candidates))
            out[s:s + chunk] = un[:, :, pos].any(axis=2)
        return out

    def seq_fire_rate(self, part: str, features, exclude_special: bool = False) -> np.ndarray:
        """Per-sequence rho: fraction of positions where any selected feature fires.

        exclude_special=False is DSG's tau convention (denominator = full row length).
        """
        fa = self.fire_any(part, features)
        if not exclude_special:
            lp = self.path / f"lengths_{part}.npy"
            return fa.sum(1) / (np.load(lp) if lp.exists() else fa.shape[1])
        keep = ~np.asarray(self.special(part))
        return (fa & keep).sum(1) / keep.sum(1)


def open_cache(model_name="gemma-2-2b-it", sae_release="gemma-scope-2b-pt-res",
               sae_id="layer_3/width_16k/average_l0_142", forget="bio-forget-corpus",
               retain="wikitext", seed=0, dataset_size=1024, seq_len=1024) -> ActivationCache:
    key = cache_key(model_name, sae_release, sae_id, forget, retain, seed, dataset_size, seq_len)
    return ActivationCache(cache_path(key))
