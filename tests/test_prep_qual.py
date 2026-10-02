"""Qualitative track: annotation sheet / kappa / gallery (TOFU only), Q5/Q6 on fake residual captures."""
import csv
import json

import numpy as np
import pandas as pd

from dsgx import paths
from dsgx.analysis.qual import annotate, geometry
from tests.fakeruns import task_run


def _fake_tofu(exp="A2", conds=("base-stream", "dsg-faithful-stream"), n=6, items="tofu-forget10"):
    ids = [f"tofu-forget10:{i}" for i in range(n)]
    for k, c in enumerate(conds):
        d = paths.runs_dir() / exp / f"tofu-qa__{c}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "config.json").write_text(json.dumps({"args": {"items": items}}))
        (d / "DONE").write_text("{}")
        (d / "metrics.json").write_text("{}")
        pd.DataFrame({"item_id": ids, "rougeL_recall": np.linspace(0, 1, n) * k, "gate_fired": [bool(k)] * n,
                      "token_f1": 0.0, "gibberish": False, "match": False}).to_parquet(d / "items.parquet")
        p = paths.private_dir() / exp / f"tofu-qa__{c}"
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "generations.jsonl", "w") as f:
            for i in ids:
                f.write(json.dumps({"item_id": i, "text": f"answer {c} {i}"}) + "\n")
    return ids


def test_sheet_kappa_gallery(tmp_path):
    _fake_tofu()
    out = tmp_path / "q"
    assert annotate.main(["sheet", "--n", "4", "--out", str(out)]) == 0
    rows = list(csv.DictReader(open(out / "annotation" / "SHEET.csv")))
    assert len(rows) == 8 and {r["condition_code"] for r in rows} == {"C1", "C2"}
    assert all("dsg" not in r["condition_code"] for r in rows)  # blinded
    key = json.loads((out / "annotation" / "KEY.json").read_text())
    assert set(key["codes"].values()) == {"base-stream", "dsg-faithful-stream"}
    # two annotators, agreeing on 6 of 8
    for name, flip in (("A", set()), ("B", {0, 1})):
        with open(tmp_path / f"{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                lab = "correct" if int(r["row_id"]) % 2 else "refusal_or_idk"
                if int(r["row_id"]) in flip:
                    lab = "gibberish"
                w.writerow({**r, "label": lab})
    assert annotate.main(["kappa", str(tmp_path / "A.csv"), str(tmp_path / "B.csv"), "--out", str(out)]) == 0
    k = json.loads((out / "annotation" / "kappa.json").read_text())
    assert k["n_rows"] == 8 and abs(k["agreement"] - 0.75) < 1e-9 and 0 < k["kappa"] < 1
    assert annotate.main(["gallery", "--k", "3", "--out", str(out), "--sheet", str(out / "annotation" / "SHEET.csv"),
                          "--key", str(out / "annotation" / "KEY.json"), "--labels", str(tmp_path / "A.csv")]) == 0
    assert "## Case 1" in (out / "CASE_GALLERY.md").read_text()


def test_non_tofu_runs_are_refused(tmp_path):
    _fake_tofu(items="wmdp-bio-open")
    assert annotate.tofu_conditions("A2") == {}
    assert annotate.main(["sheet", "--out", str(tmp_path)]) == 1


def test_kappa_values():
    assert annotate.cohen_kappa(list("aabb"), list("aabb")) == 1.0
    assert abs(annotate.cohen_kappa(list("abab"), list("aabb"))) < 1e-9


def _fake_caps(exp, tags=("base", "dsg"), n=12, L=4, d=8):
    rng = np.random.default_rng(0)
    base = rng.normal(size=(n, L, d)).astype(np.float16)
    items = [f"wmdp-bio:{i}" for i in range(n // 2)] + [f"high_school_geography:{i}" for i in range(n // 2)]
    for t in tags:
        dd = paths.cache_dir() / "residuals" / exp / t
        dd.mkdir(parents=True, exist_ok=True)
        X = base if t == "base" else base + rng.normal(scale=0.5, size=base.shape).astype(np.float16)
        np.save(dd / "X.npy", X)
        np.save(dd / "logit_lens.npy", np.zeros((n, L, 4), np.float32))
        (dd / "meta.json").write_text(json.dumps({"items": items, "gold": [i % 4 for i in range(n)], "split": ["test"] * n}))


def test_q5_q6_on_fake_captures(tmp_path):
    from dsgx.analysis.qual import q5_geometry, q6_trajectory

    _fake_caps("A4")
    task_run("A4", "capture__base", {"logit_lens_acc_by_layer": [0.25, 0.3, 0.4, 0.5]})
    task_run("A4", "capture__dsg", {"logit_lens_acc_by_layer": [0.25, 0.3, 0.3, 0.3]})
    assert q5_geometry.main(["--exp", "A4", "--out", str(tmp_path)]) == 0
    r = json.loads((tmp_path / "q5_geometry.json").read_text())["results"]["A4"]
    assert abs(r["base"][0]["cka"] - 1) < 1e-6 and r["dsg"][0]["cka"] < 1 and "forget_axis_cos" in r["dsg"][0]
    assert q6_trajectory.main(["--exp", "A4", "--out", str(tmp_path)]) == 0
    assert (tmp_path / "q6_trajectory_A4.png").exists()


def test_geometry_measures():
    X = np.random.default_rng(1).normal(size=(50, 6))
    assert abs(geometry.linear_cka(X, X) - 1) < 1e-9
    y = np.r_[np.zeros(25), np.ones(25)]
    Xs = X + y[:, None] * 5
    assert geometry.fisher_ratio(Xs, y) > geometry.fisher_ratio(X, y)
