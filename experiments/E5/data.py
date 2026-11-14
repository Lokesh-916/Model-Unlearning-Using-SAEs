"""E5 instruction sets and open-ended prompt builders. Harmful instructions are read from $DSG_PRIVATE only and never
printed; callers keep them in memory and log ids / hashes only.

harmful : AdvBench harmful_behaviors.csv ('goal' column, 520 rows), sha256 pinned in common.ADVBENCH_SHA256.
harmless: tatsu-lab/alpaca instructions with an empty 'input' field.
Split  : numpy default_rng(0) permutation of each pool; harmful build = first 256; harmless build = first 256,
         held-out = next 100 (disjoint).
"""
import csv
import hashlib

import numpy as np

from dsgx import paths
from dsgx.data.mcq import GEMMA_INST_FORMAT

from experiments.E5.common import ADVBENCH_SHA256, N_BUILD, N_HELD


def advbench_path():
    return paths.private_dir() / "E5" / "data" / "harmful_behaviors.csv"


def harmful_pool() -> list[str]:
    p = advbench_path()
    raw = p.read_bytes()
    h = hashlib.sha256(raw).hexdigest()
    if h != ADVBENCH_SHA256:
        raise RuntimeError(f"AdvBench file sha256 {h} != pinned {ADVBENCH_SHA256}")
    with p.open(newline="") as f:
        return [r["goal"] for r in csv.DictReader(f)]


def harmless_pool() -> list[str]:
    from datasets import load_dataset

    d = load_dataset("tatsu-lab/alpaca", split="train")
    return [x["instruction"] for x in d if x["input"] == ""]


def build_sets() -> dict:
    harm, safe = harmful_pool(), harmless_pool()
    ph = np.random.default_rng(0).permutation(len(harm))
    ps = np.random.default_rng(0).permutation(len(safe))
    return {"harmful": [{"id": f"advbench:{i}", "text": harm[i]} for i in ph[:N_BUILD]],
            "harmless": [{"id": f"alpaca:{i}", "text": safe[i]} for i in ps[:N_BUILD]],
            "heldout": [{"id": f"alpaca:{i}", "text": safe[i]} for i in ps[N_BUILD:N_BUILD + N_HELD]]}


def chat(instruction: str) -> str:
    return GEMMA_INST_FORMAT.format(prompt=instruction)


def last_instruction_pos(model, tokens) -> int:
    """Index of the last instruction token = the token before the first <end_of_turn>."""
    eot = model.tokenizer.convert_tokens_to_ids("<end_of_turn>")
    row = tokens[0].tolist()
    return row.index(eot) - 1


# ---------------------------------------------------------------- open-ended prompts (Step 4)
ACKS = ["Okay, go on.", "Understood. Please continue.", "Sure."]


def open_attack_prompt(question: str, attack: str, item_id: str) -> str:
    """none | pad400 (400 WikiText tokens before the question, as B1 'before') | split3 (3 user turns, B2 template;
    the last turn ends with 'Now answer the full question.' + the open-ended instruction)."""
    from dsgx.attacks.transforms import _split_words, filler
    from dsgx.eval.openqa import open_prompt

    if attack == "none":
        return open_prompt(question)
    if attack == "pad400":
        return open_prompt(filler("wikitext", 400, item_id) + "\n\n" + question)
    if attack == "split3":
        parts = _split_words(question, 3)
        turns = [f"I'll send a question in {len(parts)} parts. Part 1: {parts[0]}"]
        turns += [f"Part {i + 2}: {p}" for i, p in enumerate(parts[1:])]
        turns[-1] += "\nNow answer the full question.\nAnswer briefly."
        s = "<bos>"
        for i, u in enumerate(turns):
            s += f"<start_of_turn>user\n{u}<end_of_turn>\n<start_of_turn>model\n"
            if i < len(turns) - 1:
                s += f"{ACKS[i % len(ACKS)]}<end_of_turn>\n"
        return s
    raise ValueError(attack)
