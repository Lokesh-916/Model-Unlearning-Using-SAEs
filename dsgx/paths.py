"""Shared filesystem locations (MASTER_PLAN section 3.2).

Every worktree reads and writes the same absolute paths. Each can be overridden by an
environment variable of the same name (tests point them at a temporary directory).
"""
import os
from pathlib import Path

PROJECT = Path(os.environ.get("DSG_PROJECT", "/home/amaloch/projects/mechunlearn-project"))

# Main checkout: holds the git-ignored legacy artifacts and the forget corpus. Worktrees do not.
LEGACY_ROOT = Path(os.environ.get("DSG_LEGACY_ROOT", PROJECT / "baselines_DSG"))

# This checkout (main repo or a worktree): the code that is running.
REPO_ROOT = Path(__file__).resolve().parent.parent


def cache_dir() -> Path:
    return Path(os.environ.get("DSG_CACHE", PROJECT / "dsg_cache"))


def results_dir() -> Path:
    return Path(os.environ.get("DSG_RESULTS", PROJECT / "dsg_results"))


def private_dir() -> Path:
    return Path(os.environ.get("DSG_PRIVATE", PROJECT / "dsg_private"))


def queue_dir() -> Path:
    return results_dir() / "queue"


def runs_dir() -> Path:
    return results_dir() / "runs"


def logs_dir() -> Path:
    return results_dir() / "logs"


def splits_dir() -> Path:
    return REPO_ROOT / "data" / "splits"


def legacy_artifacts(case: str) -> Path:
    """Legacy DSG artifact folder for a case ('bio' or 'cyber'). Read-only; never delete."""
    return LEGACY_ROOT / f"artifacts_dynamic_bs1_{case}" / "unlearning" / "gemma-2-2b-it"


def forget_corpus_jsonl(name: str = "bio-forget-corpus") -> Path:
    return LEGACY_ROOT / "dynamic_sae_guardrails" / "evals" / "unlearning" / "data" / f"{name}.jsonl"


def worktrees_root() -> Path:
    return Path(os.environ.get("DSG_WORKTREES", PROJECT / "dsg_worktrees"))
