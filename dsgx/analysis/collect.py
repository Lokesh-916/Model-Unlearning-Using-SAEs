"""Read-only access to finished runs under $DSG_RESULTS/runs (MASTER_PLAN section 7 layout).

Two kinds of run directories exist:
  * MCQ runs  `<exp>/<exp>__<method>__<attack>__<dataset>__<split>__s<seed>__<hash>/` with
    config.json {"config": {...}}, metrics.json {"views": ...}, items.parquet;
  * task runs `<exp>/<task_id>[__<name>]/` with config.json {"task_id", "args"} (optional) and metrics.json.
Never reads $DSG_PRIVATE; never returns item text (items.parquet has ids, predictions and scores only).
"""
import json
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

from dsgx import paths


@dataclass
class Run:
    dir: Path
    exp: str
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.dir.name

    @property
    def is_mcq(self) -> bool:
        return "views" in self.metrics

    @property
    def cfg(self) -> dict:
        """The resolved run config (MCQ) or {} (task)."""
        return self.config.get("config") or {}

    @property
    def smoke(self) -> bool:
        return self.exp.endswith("-smoke")

    @property
    def base_exp(self) -> str:
        return self.exp[:-6] if self.smoke else self.exp

    @property
    def hardware(self) -> str:
        return (self.config.get("hardware") or {}).get("label") or "labpc"

    @property
    def method(self) -> str | None:
        return (self.cfg.get("method") or {}).get("name")

    @property
    def weights(self):
        return (self.cfg.get("model") or {}).get("weights")

    @property
    def split(self):
        return self.cfg.get("split")

    @property
    def case(self):
        return self.cfg.get("case")

    @property
    def seed(self):
        return self.cfg.get("seed", 0)

    @property
    def attack(self) -> dict:
        a = dict(self.cfg.get("attack") or {"name": "none"})
        a.pop("_exp_id", None)
        return a

    @property
    def is_clean(self) -> bool:
        a = self.attack
        return a.get("name", "none") == "none" or (a.get("name") == "dilution" and int(a.get("pad", 0) or 0) == 0)

    @property
    def is_base(self) -> bool:
        return self.method == "base" and not self.weights

    def view(self, name: str | None = None) -> dict:
        v = self.metrics.get("views") or {}
        if name:
            return v.get(name) or {}
        return v.get("dsg_subset") or v.get("raw") or {}

    def forget(self, view="raw") -> dict | None:
        return self.view(view).get("forget")

    def utility(self, view="raw") -> dict | None:
        return ((self.view(view).get("utility") or {}).get("pooled"))

    @property
    def gate(self) -> dict:
        return self.metrics.get("gate") or {}

    @cached_property
    def items(self):
        import pandas as pd

        p = self.dir / "items.parquet"
        return pd.read_parquet(p) if p.exists() else None

    def label(self) -> str:
        """Short human label of the condition (method + key params + weights tag + attack)."""
        m = dict(self.cfg.get("method") or {})
        bits = [m.get("name", "?")]
        g = m.get("gate") or {}
        if g:
            bits.append(g.get("type", "rho") + (f"-w{g['w']}" if "w" in g else ""))
            if g.get("sae_id"):
                bits.append(g["sae_id"].split("/")[0])
            if g.get("features_file"):
                bits.append(Path(g["features_file"]).stem)
        if m.get("intervention"):
            bits.append(m["intervention"].get("type"))
        if m.get("combine"):
            bits.append(m["combine"])
        for k in ("n_features", "retain_pct", "multiplier"):
            if k in m:
                bits.append(f"{k[0]}{m[k]}")
        if self.cfg.get("dataset_label"):
            bits.append(self.cfg["dataset_label"])
        a = self.attack
        if a.get("name", "none") != "none":
            bits.append(a["name"] + "".join(f"-{k}{v}" for k, v in sorted(a.items()) if k != "name" and not isinstance(v, (dict, list))))
        return "/".join(str(b) for b in bits)


def _load(d: Path) -> Run | None:
    if not (d / "DONE").exists() or not (d / "metrics.json").exists():
        return None
    try:
        met = json.loads((d / "metrics.json").read_text())
    except (OSError, ValueError):
        return None
    try:
        cfg = json.loads((d / "config.json").read_text())
    except (OSError, ValueError):
        cfg = {}
    return Run(dir=d, exp=d.parent.name, config=cfg, metrics=met)


def iter_runs(root: Path | None = None, smoke: bool = False, exps=None):
    """Every finished run. smoke=True: only `*-smoke` experiments; False: only full ones."""
    root = Path(root or paths.runs_dir())
    for ed in sorted(root.iterdir()) if root.exists() else []:
        if not ed.is_dir() or ed.name.startswith("_") or ed.name in ("sanity", "sanity-batched", "canary",
                                                                     "watchdog-test"):
            continue
        if ed.name.endswith("-smoke") != smoke:
            continue
        base = ed.name[:-6] if ed.name.endswith("-smoke") else ed.name
        if exps and base not in exps and ed.name not in exps:
            continue
        for d in sorted(ed.iterdir()):
            if d.is_dir() and not d.name.startswith("_"):
                r = _load(d)
                if r:
                    yield r


def load_all(root=None, smoke=False, exps=None) -> list[Run]:
    return list(iter_runs(root, smoke, exps))


def by_exp(runs) -> dict[str, list[Run]]:
    out = {}
    for r in runs:
        out.setdefault(r.base_exp, []).append(r)
    return out


def task_run(runs, exp: str, task: str) -> Run | None:
    """The task run named `task` (exact dir name) of experiment `exp`."""
    for r in runs:
        if r.base_exp == exp and r.name == task:
            return r
    return None


def fmt_ci(ci, digits=3) -> str:
    if not isinstance(ci, dict) or ci.get("mean") is None:
        return "n/a"
    m, lo, hi, n = ci["mean"], ci.get("lo"), ci.get("hi"), ci.get("n", ci.get("n_seeds"))
    s = f"{m:.{digits}f}"
    if lo is not None and hi is not None:
        s += f" [{lo:.{digits}f}, {hi:.{digits}f}]"
    if n is not None:
        s += f" (n={n})"
    return s
