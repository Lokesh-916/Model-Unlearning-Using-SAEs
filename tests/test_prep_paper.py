"""paper_assets: LaTeX escaping, booktabs tables and paper-size PDFs from fake runs (no GPU)."""
from dsgx.analysis import paper_assets
from tests.fakeruns import bern, mcq_run

DSG = {"name": "dsg-faithful", "n_features": 20, "retain_pct": 95, "multiplier": 500}


def test_escape():
    assert paper_assets.esc("a_b & 5% #1") == r"a\_b \& 5\% \#1"


def test_paper_assets(tmp_path):
    mcq_run("A1-test", DSG, forget=bern(0.3, 80, 1), util=bern(0.8, 80, 2))
    mcq_run("A1-test", {"name": "base"}, forget=bern(0.6, 80, 1), util=bern(0.8, 80, 2))
    mcq_run("B1", DSG, attack={"name": "dilution", "pad": 400}, forget=bern(0.5, 80, 3), util=[])
    assert paper_assets.main(["--out", str(tmp_path), "--n-boot", "100"]) == 0
    t = (tmp_path / "tables" / "baselines.tex").read_text()
    assert r"\toprule" in t and r"\bottomrule" in t and "dsg-faithful" in t
    assert (tmp_path / "tables" / "claims.tex").exists() and (tmp_path / "paper_assets.tex").exists()
    assert not list((tmp_path / "figures").glob("*.png"))  # PDFs only
