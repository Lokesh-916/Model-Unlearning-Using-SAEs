"""E2e forget-set augmentation (R6 / R7). Token-level, deterministic. The forget text is never printed or logged.

Each forget doc d (first N_AUG_DOCS docs of the bio forget corpus, docs > 50 chars, as R5's stream) gives an excerpt E =
its first 511 Gemma tokens (R5's truncation at 512 incl. <bos>) and five sequences:
    orig        <bos> E                                     (loss on every real token, as R5)
    pad_before  <bos> F "\\n\\n" E                            (F = 400 WikiText-2 validation tokens)
    pad_middle  <bos> E[:h] "\\n\\n" F "\\n\\n" E[h:]          (h = len(E) // 2)
    pad_after   <bos> E "\\n\\n" F
    chat3       B2's 3-turn split template over E's words, Gemma chat turns, ending with "<start_of_turn>model\\n"
For the four augmented forms the forget loss mask covers E's tokens only (filler and template tokens get no
unlearning loss). R7 adds <bos> + the first 511 tokens of a Hindi translation of E (loss on every real token).
Sequence = (ids: list[int], mask: list[int]) with len(ids) <= MAX_LEN_AUG.
"""
import hashlib
import random
from functools import lru_cache

from experiments.E2e.common import FILLER_SPLIT, FORMS, MAX_LEN_AUG, PAD_TOKENS

ACKS = ["Okay, go on.", "Understood. Please continue.", "Sure."]  # B2's acknowledgements


@lru_cache(maxsize=2)
def filler_texts(split: str = FILLER_SPLIT) -> tuple:
    from datasets import load_dataset

    d = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    return tuple(t for t in d["text"] if len(t) > 200)


def filler_ids(tok, n: int, key: str, split: str = FILLER_SPLIT) -> list[int]:
    texts = filler_texts(split)
    rng = random.Random(int(hashlib.sha256(f"e2e|{split}|{key}".encode()).hexdigest()[:12], 16))
    ids = []
    while len(ids) < n:
        ids += tok(texts[rng.randrange(len(texts))] + "\n", add_special_tokens=False)["input_ids"]
    return ids[:n]


def excerpt_ids(tok, doc: str, n: int = 511) -> list[int]:
    return tok(doc, add_special_tokens=False, truncation=True, max_length=n)["input_ids"][:n]


def _split_words(text: str, k: int) -> list[str]:
    w = text.split()
    m = max(1, len(w) // k)
    parts = [" ".join(w[i * m:(i + 1) * m]) for i in range(k - 1)] + [" ".join(w[(k - 1) * m:])]
    return [p for p in parts if p]


def _pieces(tok, pieces):
    """pieces: [(text, is_forget)] -> ids, mask (after <bos>)."""
    ids, mask = [tok.bos_token_id], [0]
    for text, f in pieces:
        t = tok(text, add_special_tokens=False)["input_ids"]
        ids += t
        mask += [int(f)] * len(t)
    return ids, mask


def forms_for(tok, doc: str, i: int, pad: int = PAD_TOKENS) -> dict:
    E = excerpt_ids(tok, doc)
    sep = tok("\n\n", add_special_tokens=False)["input_ids"]
    bos = [tok.bos_token_id]
    out = {"orig": (bos + E, [1] * (len(E) + 1))}
    for form in ("pad_before", "pad_middle", "pad_after"):
        F = filler_ids(tok, pad, f"{i}|{form}")
        if form == "pad_before":
            segs = [(F, 0), (sep, 0), (E, 1)]
        elif form == "pad_after":
            segs = [(E, 1), (sep, 0), (F, 0)]
        else:
            h = len(E) // 2
            segs = [(E[:h], 1), (sep, 0), (F, 0), (sep, 0), (E[h:], 1)]
        ids, mask = list(bos), [0]
        for s, f in segs:
            ids += s
            mask += [f] * len(s)
        out[form] = (ids, mask)
    parts = _split_words(tok.decode(E), 3)
    pieces = []
    for j, p in enumerate(parts):
        pre = f"I'll send a text in {len(parts)} parts. Part 1: " if j == 0 else f"Part {j + 1}: "
        pieces += [("<start_of_turn>user\n" + pre, 0), (p, 1), ("<end_of_turn>\n<start_of_turn>model\n", 0)]
        if j < len(parts) - 1:
            pieces += [(ACKS[j % len(ACKS)] + "<end_of_turn>\n", 0)]
    out["chat3"] = _pieces(tok, pieces)
    for k, (ids, mask) in out.items():
        if len(ids) > MAX_LEN_AUG or len(ids) != len(mask) or sum(mask) == 0:
            raise ValueError(f"doc {i} form {k}: len {len(ids)}, mask {sum(mask)}")
    assert list(out) == FORMS
    return out


def hindi_seq(tok, text: str):
    E = excerpt_ids(tok, text)
    return [tok.bos_token_id] + E, [1] * (len(E) + 1)


def build_stream(tok, docs, n_docs: int, hindi: dict | None = None, seed: int = 42):
    """docs: forget docs (R5 order). hindi: {doc_index: hindi text} (R7) or None. Returns (seqs, counts):
    seqs = shuffled list of (ids, mask, form)."""
    import numpy as np

    seqs, counts = [], {f: 0 for f in FORMS}
    for i in range(n_docs):
        for f, (ids, m) in forms_for(tok, docs[i], i).items():
            seqs.append((ids, m, f))
            counts[f] += 1
    if hindi:
        counts["hindi"] = 0
        for i in sorted(hindi):
            ids, m = hindi_seq(tok, hindi[i])
            seqs.append((ids, m, "hindi"))
            counts["hindi"] += 1
    perm = np.random.default_rng(seed).permutation(len(seqs))
    return [seqs[k] for k in perm], counts
