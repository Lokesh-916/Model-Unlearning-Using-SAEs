"""CPU: argument parsing of scripts/refusal_demo.py (no model is loaded)."""
import importlib.util
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location("refusal_demo", Path(__file__).parents[1] / "scripts" / "refusal_demo.py")
demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(demo)


def test_parse_args(tmp_path):
    a = demo.parse_args(["--prompt", "x", "--prompt", "y", "--force-fire", "--sample", "3"])
    assert a.prompt == ["x", "y"] and a.force_fire and a.sample == 3 and a.max_new_tokens == 96
    assert a.wmdp_dev_index == [] and a.file is None
    f = tmp_path / "p.txt"
    f.write_text("# comment\nfirst\n\nsecond\n")
    a = demo.parse_args(["--file", str(f), "--wmdp-dev-index", "4", "--max-new-tokens", "32"])
    assert demo.read_prompts(a) == ["first", "second"] and a.wmdp_dev_index == [4] and a.max_new_tokens == 32
    assert not a.force_fire and a.sample == 0
    for bad in ([], ["--prompt", "x", "--sample", "-1"], ["--wmdp-dev-index", "-2"], ["--prompt", "x", "--max-new-tokens", "0"]):
        with pytest.raises(SystemExit):
            demo.parse_args(bad)
