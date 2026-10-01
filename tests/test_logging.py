import json
import time

import pandas as pd

from dsgx.logging.progress import Progress
from dsgx.logging.run_logger import ITEM_COLUMNS, RunLogger, make_run_id

CFG = {"exp_id": "A1", "method": {"name": "dsg-faithful", "n_features": 20}, "attack": {"name": "none"},
       "datasets": ["wmdp-bio"], "split": "dev", "seed": 3}


def test_run_id_format():
    rid = make_run_id(CFG)
    parts = rid.split("__")
    assert parts[:6] == ["A1", "dsg-faithful", "none", "wmdp-bio", "dev", "s3"] and len(parts[6]) == 8
    assert make_run_id(dict(CFG, seed=4)) != rid
    many = dict(CFG, datasets=["a", "b", "c"])
    assert make_run_id(many).split("__")[3] == "3sets"


def test_run_logger_files(tmp_path):
    log = RunLogger(CFG, root=tmp_path)
    log.write_config({"x": 1})
    log.write_items([{"item_id": "wmdp-bio:1", "correct": True, "sel_max": [1.0, 2.0]}])
    log.write_traces({"wmdp-bio:1": [0, 1, 1]})
    log.write_metrics({"a": 1})
    log.mark_done({"forget": None})
    df = pd.read_parquet(log.dir / "items.parquet")
    assert list(df.columns[: len(ITEM_COLUMNS)]) == ITEM_COLUMNS
    assert json.loads(df.loc[0, "sel_max"]) == [1.0, 2.0]
    c = json.loads((log.dir / "config.json").read_text())
    assert c["end_time"] and c["git"]["commit"] and "torch" in c["versions"]
    assert log.done


def test_progress_heartbeat(tmp_path):
    p = tmp_path / "progress.json"
    pr = Progress([str(p)], "job1", items_total=10, interval=0.2)
    pr.advance(4)
    t0 = json.loads(p.read_text())["last_update"]
    time.sleep(0.6)
    d = json.loads(p.read_text())
    assert d["last_update"] > t0 and d["items_done"] == 4 and d["items_total"] == 10
    pr.close()
    assert json.loads(p.read_text())["phase"] == "done"
    assert not list(tmp_path.glob(".*tmp"))
