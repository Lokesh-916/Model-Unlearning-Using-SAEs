"""P1 docs task (GPU): experiments.E4a.docs.task unchanged, on the Cyber corpora (experiments.P1.common.load_corpus).
Chunks under $DSG_CACHE/E4a/<exp>/docs/ (exp = P1); item-level resume per chunk; no text stored."""
import experiments.E4a.docs as e4a_docs

from experiments.P1.common import load_corpus


def task(ctx):
    e4a_docs.load_corpus = load_corpus
    return e4a_docs.task(ctx)
