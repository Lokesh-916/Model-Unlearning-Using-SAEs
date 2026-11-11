"""dsgx.analysis.appendix_tables: tables come only from files on disk and compile as booktabs floats."""
from pathlib import Path

from dsgx.analysis import appendix_tables as at


def test_splits_and_hardware(tmp_path):
    s = at.splits_table(at.REPO / "data/splits")
    assert s and "wmdp-bio" in s and "cdd521a9dc598fa2" in s and r"\label{tab:app-splits}" in s
    h = at.hardware_table()
    assert h and "158/538" in h and "161/538" in h


def test_main_writes_or_skips(tmp_path):
    assert at.main(["--out", str(tmp_path), "--cluster-results", str(tmp_path / "missing")]) == 0
    written = {p.stem for p in (tmp_path / "tables").glob("*.tex")}
    assert {"app_splits", "app_hardware"} <= written
    assert "app_rmu_test" not in written  # no cluster results -> no table, never invented numbers


def test_p_format():
    assert at._p(4.3e-32) == "<10^{-31}" and at._p(0.1711) == "=0.171"
