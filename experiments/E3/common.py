"""E3 "detect here, erase there": shared constants. DEV split only (never TEST).

SAEs: Gemma Scope 2B residual 16k, canonical L0 per layer (sae_lens release gemma-scope-2b-pt-res-canonical,
resolved to the concrete average_l0 paths below), plus DSG's own layer-3 SAE (average_l0_142) so that DSG's 20
detection features can be compared in their own dictionary.
"""
import json

from dsgx import paths

RELEASE = "gemma-scope-2b-pt-res"
CANONICAL_L0 = {3: 59, 4: 124, 5: 68, 6: 70, 7: 69, 8: 71, 9: 73, 10: 77, 11: 80, 12: 82, 13: 84, 14: 84}
SAES = [(f"L{l:02d}", f"layer_{l}/width_16k/average_l0_{v}") for l, v in CANONICAL_L0.items()]
SAES.append(("L03dsg", "layer_3/width_16k/average_l0_142"))
BENIGN_BIO = ["college_biology", "high_school_biology", "anatomy", "virology", "medical_genetics"]
TOPK = (10, 20, 50)
N_RANDOM = 50


def features_dir(exp_id: str):
    d = paths.cache_dir() / "features" / "E3" / exp_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_features(exp_id: str, name: str) -> list[int]:
    return [int(x) for x in json.loads((features_dir(exp_id) / f"{name}.json").read_text())["features"]]


def fit_heldout(dev_ids):
    """Deterministic halves of the WMDP-Bio dev ids: features are ranked on `fit`, accuracy is also reported on
    `heldout` (items never used for ranking)."""
    dev = list(dev_ids)
    return dev[0::2], dev[1::2]
