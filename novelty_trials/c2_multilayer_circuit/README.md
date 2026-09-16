# C2: Multi-layer circuit intervention (preliminary)

**Status: preliminary version only, not the full C2 spec.** The full spec calls
for semantic (mid-layer) detection plus a minimal-cut feature set found via an
attribution graph across layers (built on C1's attribution scoring). C1 hasn't
been built yet, so this is a simpler precursor experiment: does intervening at
a *second* layer at all, using the same Fisher-info-style feature selection as
the paper's original method, help -- independent of whether the layer/feature
choice is principled via attribution.

## Setup

Layer 3 (paper's main config, width_16k, average_l0=142) + layer 8 (width_16k,
average_l0=142 -- chosen to match layer 3's sparsity for a fair comparison).
Each layer gets its own independently-selected 20-feature set and its own
retain-calibrated threshold (same method as the paper, applied twice). Both
layers' gates run together in one forward pass.

## Result

Sanity check: layer-3-only reproduces the cached baseline exactly (confirms
the eval pipeline), so the combined-layer number below is trustworthy.

| Config | WMDP-bio | MMLU_avg |
|---|---|---|
| Paper/original baseline | 29.368% | 99.412% |
| Layer 3 only (this run) | 29.368% | 99.412% |
| Layer 3 + layer 8 combined | 27.695% | 97.941% |

Layer 3 threshold: 0.5458. Layer 8 threshold: 0.4289.

**Real, non-degenerate movement in both directions**: forgetting improves
(WMDP down 1.67pp) but utility drops (MMLU down 1.47pp). Diagnosed cause: each
layer's gate independently has its own ~5% retain false-positive rate; running
two independent gates together compounds that risk (a retain question now gets
wrongly flagged if *either* layer's gate misfires), which is the expected cost
of an OR-combination of two independent detectors, not a bug.

## Next step for a full C2 (not yet done)

1. Build C1 (attribution-based feature selection) first -- it directly extends
   into this: run attribution scoring at ~4 layers (e.g. 3, 8, 13, 18) instead
   of the current Fisher-info-style ratio, build a feature attribution graph
   between layers, and select a minimal cross-layer feature set via ablation
   impact rather than "just pick two layers and union their gates."
2. Test an AND-combination of the two layers' gates (require both to fire,
   not either) to see if it recovers some of the MMLU cost while keeping the
   forgetting gain -- proposed but not yet tested.
3. Test against paraphrased WMDP questions (the spec's actual win condition
   for C2, addressing the jailbreak/rephrasing gap) -- not yet done.
