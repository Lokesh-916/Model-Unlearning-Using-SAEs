"""E2e augmentation + masked forget loss (CPU; benign WikiText text stands in for forget docs)."""
import torch

from experiments.E2e.augment import build_stream, forms_for
from experiments.E2e.common import FORMS, MAX_LEN_AUG, PAD_TOKENS


def _tok():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained("google/gemma-2-2b-it")


def _docs(n):
    from experiments.E2e.augment import filler_texts

    t = filler_texts("test")
    return [" ".join(t[i:i + 6]) for i in range(0, 6 * n, 6)]


def test_forms_masks_and_orig_equals_r5_tokenisation():
    tok = _tok()
    doc = _docs(1)[0]
    f = forms_for(tok, doc, 0)
    assert list(f) == FORMS
    r5 = tok(doc, truncation=True, max_length=512)["input_ids"]
    assert f["orig"][0] == r5 and all(f["orig"][1])
    E = len(r5) - 1
    for k in ("pad_before", "pad_middle", "pad_after"):
        ids, m = f[k]
        assert sum(m) == E and len(ids) <= MAX_LEN_AUG
        assert len(ids) - E - 1 >= PAD_TOKENS            # filler + separators carry no loss
    ids, m = f["pad_before"]
    assert [i for i, x in zip(ids, m) if x] == r5[1:]     # forget tokens are exactly the excerpt
    ids, m = f["chat3"]
    s = tok.decode(ids)
    assert s.count("<start_of_turn>user") == 3 and s.endswith("<start_of_turn>model\n")
    assert 0.9 * E <= sum(m) <= 1.1 * E


def test_stream_deterministic_and_counts():
    tok = _tok()
    docs = _docs(3)
    a, ca = build_stream(tok, docs, 3, {0: docs[2]}, seed=42)
    b, _ = build_stream(tok, docs, 3, {0: docs[2]}, seed=42)
    assert [x[0] for x in a] == [x[0] for x in b]
    assert ca == {**{f: 3 for f in FORMS}, "hindi": 1} and len(a) == 16


def test_masked_loss_equals_mse_when_unmasked():
    torch.manual_seed(0)
    h, u = torch.randn(3, 7, 5), torch.randn(1, 1, 5)
    m = torch.ones(3, 7)
    per = (((h - u) ** 2).mean(-1) * m).sum(1) / m.sum(1)
    assert torch.allclose(per.mean(), torch.nn.functional.mse_loss(h, u.expand_as(h)))
