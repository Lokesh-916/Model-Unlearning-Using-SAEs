"""Forget / retain calibration corpora.

`legacy_calibration_tokens` replays the exact sampling of the DSG code path
(get_forget_retain_data -> get_shuffled_forget_retain_tokens -> tokenize_and_concat_dataset),
including its RNG consumption order, so the new activation cache sees the same token rows as
the legacy 18 GB pickles:

    random.seed(s); torch.manual_seed(s)       # run_eval
    random.sample(forget_docs, 1024)           # python RNG
    torch.randperm(n_forget_rows)              # torch CPU RNG
    torch.randperm(n_retain_rows)
    keep the first min(1024, n_forget_rows, n_retain_rows) rows of each

Loading the model and SAE does not consume either RNG (checked for the pinned versions).
"""
import json
import random

import torch

from dsgx import paths


def load_forget_docs(name: str = "bio-forget-corpus", min_len: int = 50) -> list[str]:
    docs = []
    if "bio-forget-corpus" in name:
        with open(paths.forget_corpus_jsonl(name)) as f:
            for line in f:
                t = json.loads(line)["text"]
                if len(t) > min_len:
                    docs.append(str(t))
    elif "cyber-forget-corpus" in name:
        from datasets import load_dataset

        ds = load_dataset("cais/wmdp-corpora", "cyber-forget-corpus", split="train")
        ds = ds.filter(lambda x: len(x["text"]) > min_len)
        docs = list(ds["text"])  # legacy .shuffle(seed=42) result was discarded: no shuffle
    elif name.startswith("tofu-"):
        docs = _tofu_docs(name[5:])
    else:
        raise ValueError(f"unknown forget corpus {name!r}")
    return docs


def load_retain_docs(name: str = "wikitext", min_len: int = 50) -> list[str]:
    from datasets import load_dataset

    if name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return [str(x["text"]) for x in ds if len(x["text"]) > min_len]
    if name.startswith("tofu-"):
        return _tofu_docs(name[5:])
    raise ValueError(f"unknown retain corpus {name!r} (chat-retain is built by exp/A1)")


def _tofu_docs(config: str) -> list[str]:
    """TOFU QA pairs as plain text documents (fictitious authors; safe to store anywhere)."""
    from datasets import load_dataset

    d = load_dataset("locuslab/TOFU", config, split="train")
    return [f"Question: {x['question']}\nAnswer: {x['answer']}" for x in d]


def tokenize_and_concat(tokenizer, docs: list[str], seq_len: int = 1024, add_bos: bool = True):
    """Exact port of dsg_utils.dataset_utils.tokenize_and_concat_dataset (max_tokens=None)."""
    full_text = tokenizer.eos_token.join(docs)
    num_chunks = 20
    chunk_length = (len(full_text) - 1) // num_chunks + 1
    chunks = [full_text[i * chunk_length:(i + 1) * chunk_length] for i in range(num_chunks)]
    tokens = tokenizer(chunks, return_tensors="pt", padding=True)["input_ids"].flatten()
    tokens = tokens[tokens != tokenizer.pad_token_id]
    num_batches = len(tokens) // seq_len
    tokens = tokens[: num_batches * seq_len].reshape(num_batches, seq_len)
    if add_bos:
        tokens[:, 0] = tokenizer.bos_token_id
    return tokens


def legacy_calibration_tokens(tokenizer, forget: str = "bio-forget-corpus", retain: str = "wikitext",
                              seed: int = 0, dataset_size: int = 1024, seq_len: int = 1024):
    """Return (forget_tokens [N, L], retain_tokens [N, L], info) exactly as the DSG code builds them.

    For the chat retain corpus, also returns per-row retain lengths (prompt rows are padded).
    """
    retain_mode = "prompts" if retain == CHAT_RETAIN else "rows"
    forget_docs = load_forget_docs(forget)
    retain_docs = load_retain_docs("wikitext" if retain_mode == "prompts" else retain)
    random.seed(seed)
    torch.manual_seed(seed)
    sampled = random.sample(forget_docs, min(dataset_size, len(forget_docs)))
    f_tok = tokenize_and_concat(tokenizer, sampled, seq_len)
    r_tok = tokenize_and_concat(tokenizer, retain_docs, seq_len)
    f_tok = f_tok[torch.randperm(f_tok.shape[0])]
    r_tok = r_tok[torch.randperm(r_tok.shape[0])]
    n = min(dataset_size, f_tok.shape[0], r_tok.shape[0])
    info = {"forget": forget, "retain": retain, "seed": seed, "n_forget_docs": len(forget_docs),
            "n_retain_docs": len(retain_docs), "forget_rows_total": int(f_tok.shape[0]),
            "retain_rows_total": int(r_tok.shape[0]), "rows_used": int(n), "seq_len": seq_len}
    if retain_mode == "prompts":
        # Forget rows are exactly those of the WikiText-retain run (same RNG replay); the retain
        # side is swapped for the chat corpus, as in the legacy Cyber swap.
        rec = build_chat_retain(seed=0, n=400)
        r_rows, lengths = prompt_rows(tokenizer, [x["prompt"] for x in rec["items"]])
        info.update({"retain": retain, "retain_rows_total": int(r_rows.shape[0]),
                     "retain_dropped_for_overlap": rec["dropped_for_overlap"]})
        return f_tok[:n].clone(), r_rows, info, lengths
    return f_tok[:n].clone(), r_tok[:n].clone(), info


# ---------------------------------------------------------------------------------------------
# Chat-formatted MCQ retain corpus for Cyber-chatretain (decision 3, user decision 2):
# MMLU auxiliary_train (ARC / RACE / OBQA / MCTest), no MMLU-test subjects or items, seeded,
# filtered against every evaluation question (WMDP + all 57 MMLU test subjects).
# ---------------------------------------------------------------------------------------------
CHAT_PRE = "The following are multiple choice questions (with answers).\n"
CHAT_RETAIN = "mmlu-aux-chat"


def _ngram_set(text, n=8):
    import re

    w = re.findall(r"[a-z0-9]+", text.lower())
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def build_chat_retain(seed: int = 0, n: int = 400, max_chars: int = 3000) -> dict:
    """Sample n auxiliary_train items as chat MCQ prompts; drop any that overlap an eval question."""
    from datasets import load_dataset

    from dsgx import paths
    from dsgx.data.mcq import GEMMA_INST_FORMAT, load_mcq
    from dsgx.data.splits import ALL_DATASETS
    from dsgx.util import atomic_write_json

    out = paths.cache_dir() / "corpora" / f"{CHAT_RETAIN}_s{seed}_n{n}.json"
    if out.exists():
        return json.loads(out.read_text())
    aux = load_dataset("cais/mmlu", "auxiliary_train", split="train")
    eval_grams, eval_q = set(), set()
    for d in ALL_DATASETS:
        for it in load_mcq(d):
            eval_grams |= _ngram_set(it.question)
            eval_q.add(" ".join(it.question.lower().split()))
    rng = random.Random(seed)
    order = list(range(len(aux)))
    rng.shuffle(order)
    keep, dropped = [], 0
    for i in order:
        x = aux[i]
        x = x.get("train", x)  # some versions nest the record under "train"
        q, ch = x["question"], list(x["choices"])
        if len(ch) != 4 or len(q) > max_chars:
            continue
        g = _ngram_set(q + " " + " ".join(ch))
        if " ".join(q.lower().split()) in eval_q or (g and len(g & eval_grams) / len(g) >= 0.5):
            dropped += 1
            continue
        body = CHAT_PRE + q + "".join(f"\n{l}. {c}" for l, c in zip("ABCD", ch))
        keep.append({"aux_index": i, "prompt": GEMMA_INST_FORMAT.format(prompt=body) + "Answer: ("})
        if len(keep) == n:
            break
    rec = {"name": CHAT_RETAIN, "seed": seed, "n": len(keep), "dropped_for_overlap": dropped,
           "source": "cais/mmlu auxiliary_train", "items": keep}
    atomic_write_json(out, rec)
    return rec


def prompt_rows(tokenizer, prompts: list[str]):
    """Tokenise prompts separately (they already start with <bos>); right-pad. Returns (tokens, lengths)."""
    toks = [tokenizer(p, add_special_tokens=False, return_tensors="pt")["input_ids"][0] for p in prompts]
    L = max(len(t) for t in toks)
    out = torch.full((len(toks), L), tokenizer.pad_token_id, dtype=torch.long)
    for i, t in enumerate(toks):
        out[i, : len(t)] = t
    return out, torch.tensor([len(t) for t in toks])
