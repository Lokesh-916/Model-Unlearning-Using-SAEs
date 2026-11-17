"""P9 CPU tests: CUSUM, attention blocking, prompt regions, clarification grader (no model)."""
import numpy as np
import torch

from experiments.P9.common import content_spans, is_clarification, label_tokens, region_spans
from experiments.P9.core import AttnBlock, cusum_max_rows
from experiments.V2.common import cusum_max


def test_cusum_rows_matches_v2():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 2, size=(5, 40))
    got = cusum_max_rows(x)
    want = [cusum_max(r.tolist()) for r in x]
    assert np.allclose(got, want, atol=1e-9)
    assert cusum_max_rows(np.zeros((1, 0)))[0] == 0.0


def test_attn_block_later_positions_only():
    s = torch.zeros(2, 1, 4, 4)
    mask = torch.tensor([[False, True, False, False], [False, False, False, False]])
    out = AttnBlock(mask)(s.clone())
    assert torch.isinf(out[0, 0, 2, 1]) and torch.isinf(out[0, 0, 3, 1])
    assert out[0, 0, 1, 1] == 0 and out[0, 0, 0, 1] == 0      # the token itself / earlier queries untouched
    assert not torch.isinf(out[1]).any()
    # KV-cache step: one query at absolute position 5, keys 0..5; mask shorter than K is padded with False
    out = AttnBlock(mask[:1])(torch.zeros(1, 1, 1, 6))
    assert torch.isinf(out[0, 0, 0, 1]) and not torch.isinf(out[0, 0, 0, 4])


def test_regions():
    pre = "The following are multiple choice questions (with answers) about biology.\n"
    p = "<bos><start_of_turn>user\nFILLER TEXT\n\n" + pre + "What is X?\nA. a\nB. b\nC. c\nD. d<end_of_turn>\n<start_of_turn>model\nAnswer: ("
    spans = content_spans(p)
    assert len(spans) == 1
    r = region_spans(p, "pad400", pre)
    i = p.index("FILLER")
    q = p.index("What")
    o = p.index("\nA. ")
    lab = label_tokens([(0, 5), (i, i + 6), (q, q + 4), (o + 1, o + 2), (p.index("model"), p.index("model") + 5)], spans, r)
    assert lab == ["template", "filler", "question", "options", "template"]


def test_clarification():
    assert is_clarification("It seems the question is incomplete. Could you please provide the missing word?")
    assert not is_clarification("Photosynthesis converts light energy into chemical energy.")
