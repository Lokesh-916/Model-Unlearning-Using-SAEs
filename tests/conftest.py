"""Every test runs against throw-away DSG_RESULTS / DSG_CACHE / DSG_PRIVATE directories."""
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "dynamic_sae_guardrails"))


@pytest.fixture(autouse=True)
def tmp_dsg_dirs(tmp_path, monkeypatch):
    for k in ("DSG_RESULTS", "DSG_CACHE", "DSG_PRIVATE"):
        d = tmp_path / k.lower()
        d.mkdir()
        monkeypatch.setenv(k, str(d))
    monkeypatch.setenv("DSGX_HEARTBEAT_SECONDS", "1")
    yield tmp_path
