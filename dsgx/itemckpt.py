"""Item-level checkpoints for task loops (power cuts): `ItemCheckpoints.map` runs a per-item function over a
list in chunks of ~DSGX_RESUME_EVERY items (default 200, 0 = off); each finished chunk is written atomically
(fsync + rename) and reused on restart when its fingerprint (caller's run identity + chunk bounds + item keys)
matches. The function must depend only on its item (no state carried across items), so a resumed run returns the
same values as an uninterrupted one. MCQ grids use the batch-aware twin in dsgx.run (`_score_resumable`).

Chunks may hold generated text: put them under ctx.private_dir(...) when they do (rule 3.8). `clear()` at DONE."""
import hashlib
import json
import os
import pickle
import shutil
from pathlib import Path

from dsgx.util import atomic_write_bytes


def resume_every() -> int:
    return int(os.environ.get("DSGX_RESUME_EVERY", "200"))


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


class ItemCheckpoints:
    def __init__(self, directory, fingerprint, every: int | None = None):
        self.dir = Path(directory)
        self.every = resume_every() if every is None else int(every)
        self.tag = _sha(fingerprint)
        self.reused = 0

    def _load(self, f: Path, tag: str):
        try:
            part = pickle.loads(f.read_bytes())
            return part["values"] if part.get("tag") == tag else None
        except (OSError, pickle.UnpicklingError, EOFError, KeyError, AttributeError, ValueError, TypeError):
            return None

    def map(self, items, fn, key=str, on_done=None) -> list:
        """[fn(it) for it in items], checkpointed; on_done(n) after every n items finished or reused."""
        items = list(items)
        step = self.every if self.every > 0 else max(len(items), 1)
        out = []
        for s in range(0, len(items), step):
            e = min(s + step, len(items))
            f = self.dir / f"chunk_{s:06d}.pkl"
            tag = _sha([self.tag, s, e, [key(it) for it in items[s:e]]])
            vals = self._load(f, tag) if self.every > 0 else None
            if vals is None:
                vals = []
                for it in items[s:e]:
                    vals.append(fn(it))
                    if on_done:
                        on_done(1)
                if self.every > 0:
                    atomic_write_bytes(f, pickle.dumps({"tag": tag, "values": vals}, protocol=4))
            else:
                self.reused += e - s
                if on_done:
                    on_done(e - s)
            out.extend(vals)
        return out

    def clear(self):
        shutil.rmtree(self.dir, ignore_errors=True)
