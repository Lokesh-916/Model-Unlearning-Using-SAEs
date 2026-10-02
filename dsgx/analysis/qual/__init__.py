"""Qualitative track (Q1-Q8). Every tool reads the queue's existing outputs (run dirs, activation caches,
residual captures) and writes to $DSG_RESULTS/qual/ (or --out). Hazardous text is never read or written:
WMDP items appear only as ids; Q4/Q8 annotation material is TOFU-only (fictitious authors).
"""
from pathlib import Path

from dsgx import paths


def out_dir(out=None) -> Path:
    d = Path(out) if out else paths.results_dir() / "qual"
    d.mkdir(parents=True, exist_ok=True)
    return d
