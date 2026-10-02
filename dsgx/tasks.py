"""Generic task jobs: experiments that are not MCQ grids (training, probes, generation, analysis).

An experiment config declares
    tasks:
      - id: translate
        entry: experiments.B3.translate:main     # function(ctx) -> headline dict
        args: {...}
        smoke_args: {...}                         # merged over args for the smoke config
        skip_smoke: true / smoke_only: true       # task only in the full / only in the smoke list
        kind_slot: gpu | cpu
        est_minutes / smoke_est_minutes / est_vram_gb / est_ram_gb / gpu_exclusive
        deps: [other_task_id, "exp:A1-test"]     # "exp:X" = every job of experiment X
and the worker calls the entry with a TaskContext. Entries are imported from the job's worktree.
"""
import importlib
import json
from pathlib import Path

from dsgx import paths
from dsgx.util import atomic_write_json, git_info, now_iso, package_versions, gpu_info


class TaskContext:
    def __init__(self, job: dict, args: dict, progress, smoke: bool):
        self.job, self.args, self.progress, self.smoke = job, args, progress, smoke
        self.exp_id = job["exp_id"]
        self.task_id = job["task_id"]
        self.seed = int(args.get("seed", 0))

    # ---- locations ----
    def run_dir(self, name: str | None = None) -> Path:
        d = paths.runs_dir() / self.exp_id / (self.task_id + (f"__{name}" if name else ""))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def private_dir(self, name: str | None = None) -> Path:
        """Hazardous text (generations, prompts that elicit them) goes only here (rule 3.8)."""
        d = paths.private_dir() / self.exp_id / (self.task_id + (f"__{name}" if name else ""))
        d.mkdir(parents=True, exist_ok=True)
        return d

    def cache_dir(self, *parts) -> Path:
        d = paths.cache_dir().joinpath(*parts)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def results_of(self, exp_id: str) -> Path:
        """Run directory root of another experiment (data dependency, never code)."""
        return paths.runs_dir() / (exp_id + ("-smoke" if self.smoke and not exp_id.endswith("-smoke") else ""))

    # ---- logging ----
    def write_config(self, name=None, **extra):
        atomic_write_json(self.run_dir(name) / "config.json", {
            "exp_id": self.exp_id, "task_id": self.task_id, "args": self.args, "smoke": self.smoke,
            "git": git_info(), "versions": package_versions(), "hardware": gpu_info(),
            "start_time": now_iso(), **extra})

    def write_metrics(self, metrics: dict, name=None):
        atomic_write_json(self.run_dir(name) / "metrics.json", metrics)

    def finish(self, headline: dict, name=None):
        d = self.run_dir(name)
        atomic_write_json(d / "DONE", {"time": now_iso(), "headline": headline})
        print(f"[worker] run task done: {d}", flush=True)
        return d

    def bundle(self, **kw):
        from dsgx.models.loader import get_bundle

        return get_bundle(**kw)


def resolve_entry(entry: str):
    mod, fn = entry.split(":")
    return getattr(importlib.import_module(mod), fn)


def run_task(job: dict, progress) -> dict:
    args = dict(job.get("args") or {})
    ctx = TaskContext(job, args, progress, job.get("smoke", False))
    progress.update(phase=f"task {job['task_id']}", force=True)
    out = resolve_entry(job["entry"])(ctx)
    return out or {}


def load_json(p):
    return json.loads(Path(p).read_text())
