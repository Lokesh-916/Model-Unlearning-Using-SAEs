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
    else:
        raise ValueError(f"unknown forget corpus {name!r}")
    return docs


def load_retain_docs(name: str = "wikitext", min_len: int = 50) -> list[str]:
    from datasets import load_dataset

    if name == "wikitext":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        return [str(x["text"]) for x in ds if len(x["text"]) > min_len]
    raise ValueError(f"unknown retain corpus {name!r} (chat-retain is built by exp/A1)")


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
    """Return (forget_tokens [N, L], retain_tokens [N, L], info) exactly as the DSG code builds them."""
    forget_docs = load_forget_docs(forget)
    retain_docs = load_retain_docs(retain)
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
    return f_tok[:n].clone(), r_tok[:n].clone(), info
