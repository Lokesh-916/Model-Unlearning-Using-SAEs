"""Fixed, seeded dev/test splits (decision 5) and the DSG-subset view.

Splits live in data/splits/<dataset>.json (committed). dev is for tuning and selection,
test is for reporting only. Each file records a sha256 of its id lists so that the leakage
check and every run's config.json can prove which split was used.
"""
import hashlib
import json
import zlib
from pathlib import Path

import numpy as np

from dsgx import paths
from dsgx.data.mcq import MMLU_SUBJECTS, legacy_name

SPLIT_SEED = 0
ALL_DATASETS = ["wmdp-bio", "wmdp-cyber", *MMLU_SUBJECTS]


def _hash_ids(dev, test) -> str:
    s = json.dumps({"dev": list(map(int, dev)), "test": list(map(int, test))})
    return hashlib.sha256(s.encode()).hexdigest()


def split_ids(n: int, dataset: str, seed: int = SPLIT_SEED):
    """Deterministic 50/50 split of range(n). Independent of numpy's global RNG."""
    rng = np.random.default_rng([seed, zlib.crc32(dataset.encode())])
    perm = rng.permutation(n)
    half = n // 2
    return sorted(int(i) for i in perm[:half]), sorted(int(i) for i in perm[half:])


def make_split_file(dataset: str, n: int, seed: int = SPLIT_SEED, out_dir: Path | None = None):
    dev, test = split_ids(n, dataset, seed)
    rec = {"dataset": dataset, "n": n, "seed": seed, "method": "default_rng([seed, crc32(name)]) 50/50",
           "dev": dev, "test": test, "sha256": _hash_ids(dev, test)}
    out_dir = Path(out_dir or paths.splits_dir())
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{dataset}.json").write_text(json.dumps(rec))
    return rec


def load_split_file(dataset: str, splits_dir: Path | None = None) -> dict:
    p = Path(splits_dir or paths.splits_dir()) / f"{dataset}.json"
    rec = json.loads(p.read_text())
    if _hash_ids(rec["dev"], rec["test"]) != rec["sha256"]:
        raise ValueError(f"split file {p} hash mismatch: it was edited")
    return rec


def get_split(dataset: str, split: str, splits_dir: Path | None = None) -> list[int]:
    if split not in ("dev", "test", "all"):
        raise ValueError(f"split must be dev/test/all, got {split!r}")
    rec = load_split_file(dataset, splits_dir)
    if split == "all":
        return sorted(rec["dev"] + rec["test"])
    return list(rec[split])


def split_hash(dataset: str, splits_dir: Path | None = None) -> str:
    return load_split_file(dataset, splits_dir)["sha256"][:16]


def dsg_subset_ids(case: str, dataset: str) -> list[int] | None:
    """Question ids the base model answers correctly under all 24 permutations.

    Legacy ids come from the case's artifacts (they differ slightly between the bio and cyber
    runs for shared subjects). Ids computed by the harness are read from $DSG_CACHE/dsg_subset.
    Returns None if no ids exist for this dataset.
    """
    legacy = paths.legacy_artifacts(case) / "data" / "question_ids" / "all" / f"{legacy_name(dataset)}_correct.csv"
    if legacy.exists():
        return sorted(int(x) for x in np.atleast_1d(np.genfromtxt(legacy, dtype=int)))
    new = paths.cache_dir() / "dsg_subset" / f"{dataset}.json"
    if new.exists():
        return sorted(json.loads(new.read_text())["ids"])
    return None
