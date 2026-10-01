"""Small shared helpers: atomic writes, hashing, git and environment info."""
import hashlib
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


def _json_default(o: Any):
    try:
        import numpy as np

        if isinstance(o, np.generic):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


_UMASK = os.umask(0)
os.umask(_UMASK)


def atomic_write_text(path, text: str) -> None:
    """Write via a temp file in the same directory, then rename (atomic on POSIX)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.chmod(tmp, 0o666 & ~_UMASK)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise


def atomic_write_json(path, obj: Any, indent: int = 2) -> None:
    atomic_write_text(path, json.dumps(obj, indent=indent, default=_json_default, sort_keys=False))


def read_json(path, default=None):
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def stable_hash(obj: Any, n: int = 8) -> str:
    """Short sha256 of a JSON-canonicalised object."""
    s = json.dumps(obj, sort_keys=True, default=_json_default)
    return hashlib.sha256(s.encode()).hexdigest()[:n]


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info(cwd=None) -> dict:
    cwd = str(cwd or Path(__file__).resolve().parent.parent)

    def _git(*args):
        try:
            return subprocess.run(
                ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=20
            ).stdout.strip()
        except Exception:
            return ""

    status = _git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": _git("rev-parse", "HEAD"),
        "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status),
        "cwd": cwd,
    }


def package_versions() -> dict:
    out = {}
    import importlib.metadata as md

    for p in ["torch", "transformers", "transformer-lens", "sae-lens", "numpy", "datasets",
              "pandas", "pyarrow", "scikit-learn", "peft"]:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = None
    import platform

    out["python"] = platform.python_version()
    return out


def gpu_info() -> dict:
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        name, driver, mem = [x.strip() for x in r.stdout.strip().splitlines()[0].split(",")]
        return {"gpu": name, "driver": driver, "memory_total": mem}
    except Exception:
        return {"gpu": None}


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def debug_enabled() -> bool:
    return os.environ.get("DSG_DEBUG", "0") == "1"
