"""Labels that every results table must carry.

User decision (P2, 2026-10-02): the third-party RMU checkpoint stays in the queue but is reported
only as an "unverified reference", never as a main comparison.
"""

UNVERIFIED_WEIGHTS = {
    "AMindToThink/gemma-2-2b-it_RMU_s200_a300_layer3": "unverified reference (third-party RMU)",
}


def reference_label(cfg: dict | None) -> str | None:
    """The reference label of a run config (the `config` block of config.json), or None."""
    weights = ((cfg or {}).get("model") or {}).get("weights")
    return UNVERIFIED_WEIGHTS.get(weights)
