"""E2f-L12 (reduced, EXPLORATORY; DEVIATIONS 2026-11-12): TOFU-model ignorance direction at layer 12.

Layer 12 at c 1 is outside the original grid (c 2 / 4 / 8); it was chosen from the Step-1 safe-entity check only
(IDK 0.32 vs base 0.00, gibberish 0.02, n 100 held-out real-entity questions). Step 1 saved the TOFU-model direction
only at the selected layer (6), so this task applies the same recipe (experiments.E2f.ignorance.direction: mean
last-token resid_post made-up − real on the 600 safe build questions, unit-normalised; norm = mean last-token
residual norm) to the TOFU model at layer 12. No evaluation, no tuning. The base-model layer-12 direction is
Step 1's own (dsg_cache/E2f/E2f/ignorance_base.npz).
"""
from experiments.E2f.common import dir_path
from experiments.E2f.entities import build_sets
from experiments.E2f.ignorance import direction, save_dirs


def task(ctx):
    from dsgx.models.loader import get_bundle
    from dsgx.run import resolve_weights

    layer = int(ctx.args.get("layer", 12))
    w = resolve_weights(ctx.args.get("tofu_weights", "ckpt:E2f/tofu_full"), ctx.exp_id)
    b = get_bundle(weights=w)
    d = direction(b.model, build_sets(), [layer])
    save_dirs(dir_path(ctx.exp_id, "tofu"), d, {"model": w, "selected_from": "E2f-L12 override (exploratory)"})
    ctx.write_config(layer=layer, tofu_weights=w)
    met = {"layer": layer, "norm": d[layer]["norm"], "raw_diff_norm": d[layer]["raw_diff_norm"]}
    ctx.write_metrics(met)
    ctx.finish({"view": "e2f-l12-tofu-dir", "forget": None, "layer": layer})
    return met
