"""Shared loaders and measures for Q5 / Q6 on the A4 / D3 residual captures
($DSG_CACHE/residuals/<exp>/<tag>/{X.npy [N, layers, d] float16, logit_lens.npy [N, layers, 4], meta.json})."""
import json
from pathlib import Path

import numpy as np

from dsgx import paths


def captures(exp: str) -> dict:
    root = paths.cache_dir() / "residuals" / exp
    out = {}
    for d in sorted(root.glob("*")) if root.exists() else []:
        if (d / "X.npy").exists() and (d / "meta.json").exists():
            out[d.name] = {"X": np.load(d / "X.npy", mmap_mode="r"), "meta": json.loads((d / "meta.json").read_text()),
                           "lens": np.load(d / "logit_lens.npy", mmap_mode="r") if (d / "logit_lens.npy").exists() else None}
    return out


def linear_cka(X, Y):
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    hsic = np.linalg.norm(X.T @ Y, "fro") ** 2
    den = np.linalg.norm(X.T @ X, "fro") * np.linalg.norm(Y.T @ Y, "fro")
    return float(hsic / den) if den > 0 else float("nan")


def fisher_ratio(X, y):
    """Between-class / within-class variance (trace ratio) of labels y."""
    mu = X.mean(0)
    sb = sw = 0.0
    for c in np.unique(y):
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        sb += len(Xc) * float(((Xc.mean(0) - mu) ** 2).sum())
        sw += float(((Xc - Xc.mean(0)) ** 2).sum())
    return sb / sw if sw > 0 else float("nan")


def mean_cos(X, Y):
    a = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    b = Y / (np.linalg.norm(Y, axis=1, keepdims=True) + 1e-9)
    return float((a * b).sum(1).mean())


def is_forget(meta):
    return np.array([str(i).startswith("wmdp") for i in meta["items"]])


def pca2(X):
    Xc = X - X.mean(0)
    _, _, vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ vt[:2].T


def layer_matrix(cap, L):
    return np.asarray(cap["X"][:, L, :], dtype=np.float32)


def out_paths(out: Path, stem: str):
    return out / f"{stem}.json", out / f"{stem}.md"
