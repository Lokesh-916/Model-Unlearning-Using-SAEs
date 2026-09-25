# Idea 1 (round 2): chi-squared co-occurrence feature selection (negative result)

User's idea: "can we select the features with any other alternative which
improves the values rather than fisher information." Round 1 already tried
attribution-patching (`imp-c1-attribution-selection`, negative). This tries a
statistically different alternative: instead of DSG's mean-squared-activation
ratio (a Fisher-info-style continuous magnitude statistic), rank features by a
chi-squared test of independence between "feature fires anywhere in this
sequence" and "sequence is forget-corpus vs retain-corpus" -- a document-level,
non-negative, magnitude-robust association statistic, closer in spirit to
DSG's own method than C1's signed single-token attribution was.

Reuses the cached layer-3 activations from the main run (`act_fgt.pkl`,
`act_ret.pkl`, `feature_sparsity_{forget,retain}.txt`) -- no new forward or
backward passes over the forget/retain corpora, only the final WMDP-bio + MMLU
evaluation touches the model.

## Two variants tried

**v1** (`imp2_1_results_v2.json` is v2; v1's raw log is `run_v1.log`): rank all
16384 features by `chi2_signed` (chi-squared statistic, zeroed out for
features that fire more on retain than forget), no separate percentile filter
beyond that. Top 20 taken directly.

**v2** (`imp2_1_results_v2.json`): tried to mirror DSG's actual two-constraint
filter more literally (forget-rate >= 95th percentile AND
forget/retain-rate-ratio >= 90th percentile, both computed on firing rates)
before ranking by chi2. This collapsed to only **2 eligible features out of
16384** -- firing-rate ratios are degenerate for sparse binary rates (most
features never fire on any retain sequence, so the ratio is undefined/huge for
a large fraction of features, and intersecting two independent top-percentile
sets on top of that leaves almost nothing). The resulting 2-feature
intervention produced 100%/100% WMDP-bio/MMLU -- uninterpretable, most likely
the threshold calibration degenerating with so few features. **v2 is not a
valid result, only recorded to show why the filter needed to be dropped.**

## Result (v1, the valid comparison, N=20 features)

| Config | WMDP-bio ↓ | MMLU avg ↑ |
|---|---|---|
| DSG original (Fisher-info-style, cached) | 29.368% | 99.412% |
| Chi-squared co-occurrence-selected | 69.517% | 81.270% |

Feature overlap with DSG's original top-20: **3/20**.

**This is a clean negative result, worse on both axes simultaneously** --
unlike C1 (which traded away almost all forgetting for a negligible utility
gain), chi-squared selection is both *worse at forgetting* (69.5% vs 29.4%,
barely better than no intervention) *and worse for utility* (81.3% vs 99.4%,
a real MMLU hit). The calibrated whole-sequence threshold for the new set came
out an order of magnitude lower than DSG's original (0.012 vs 0.546), meaning
the gate fires on almost every input -- the chi-squared-selected features are
common enough in ordinary (retain) text that the "dynamic" gate stops being
selective, explaining the collateral MMLU damage.

## Why, honestly

Document-level firing (binary presence/absence per sequence) throws away the
magnitude information DSG's squared-activation-ratio keeps. A feature that
fires once, weakly, in most forget sequences and never on retain sequences
gets a strong chi-squared score, but that same weak, ubiquitous firing pattern
also makes it a bad "specific knowledge carrier" to clamp -- it is
distinctive but not concentrated, so its calibrated trigger threshold ends up
low, and the resulting gate over-fires on ordinary text. DSG's magnitude-based
ratio implicitly favors features whose *activation strength* (not just
presence) differs between forget and retain -- a real signal that the
binarized chi-squared statistic discards.

## Status
Done. Two independent alternatives to DSG's Fisher-info-style ranking (C1's
attribution patching, this chi-squared co-occurrence test) have now both come
back negative or worse. Both preserve DSG's non-negative, sequence-level
character but still lose to the original mean-squared-activation-ratio
approach -- suggests the specific choice of *statistic* is not the low-hanging
fruit; DSG's original heuristic, despite being called "crude" in critiques, is
doing real, hard-to-replace work here too (consistent with C3/A2's finding
that the clamp *value* is likewise not an arbitrary crude hack).
