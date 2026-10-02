"""GuardBreak: stress tests for activation-gated unlearning guards (DSG and successors).

Public part: the prompt TRANSFORMATIONS only (dilution, decomposition, translation, encodings,
cached rewrites/suffixes) re-exported from dsgx.attacks.transforms, plus a CLI that runs them against
a gate through the dsgx harness. No hazardous data, generations or model outputs ship with it.
"""
from dsgx.attacks.registry import ATTACKS, make_attack  # noqa: F401
from dsgx.attacks import transforms  # noqa: F401  (registers the suite)

__all__ = ["ATTACKS", "make_attack", "transforms"]
__version__ = "0.1.0"
