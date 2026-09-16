# B3: Hierarchical gate protecting neighboring knowledge

Spec: DSG's single gate is calibrated only against generic WikiText as the
retain signal, and its own reported retain benchmarks (history, CS, geography,
aging) are all far from biology. Test whether it collapses on benign
*same-domain* content (college biology, virology, anatomy) it was never
checked against, and if so, fix it with a two-level gate: a domain-general
detector ("is this biology at all?", calibrated against WikiText) plus a
hazard-specific detector ("is this DANGEROUS biology, vs benign biology?",
calibrated contrastively against a benign-biology/medicine corpus), clamping
only when both fire.

## Kill test: confirmed

| Dataset | Single-gate accuracy |
|---|---|
| wmdp-bio | 29.368% |
| history / CS / geography (original retain) | ~100% |
| human_aging | 97.6% |
| **college_biology** | **38.194%** |
| **virology** | **42.169%** |
| **anatomy** | **43.704%** |

DSG's single gate genuinely collapses accuracy on benign biology/medicine
content it was never evaluated against, while holding ~100% on its own
reported (topically distant) retain set. Real, previously undocumented
weakness.

## Two bugs caught and fixed before the real result

1. First attempt crashed: `calculate_MCQ_metrics(..., target_metric="correct")`
   needs a pre-generated question-id file that only exists for the original
   pipeline's datasets. Fixed by using `target_metric=None` (full test set)
   for the three new neighbor subjects.
2. Second attempt produced a degenerate hazard threshold of 1.1937 -- above
   1.0, which is impossible for a bounded rate statistic. Cause: my benign-bio
   activation collection used `batch_size=4`, but the threshold-calibration
   loop (copied from `get_top_features_percentile`'s internal logic, which
   assumes batch=1 to match this repo's "bs1" cached-corpora convention)
   summed active-token counts across the whole batch dimension while dividing
   by only one sequence's length. Fixed to compute one rate per sequence.

## Final, valid result

| Dataset | Single-gate | B3 hierarchical |
|---|---|---|
| wmdp-bio | 29.368% | 67.844% |
| history/CS/geography | 100% | 100% |
| human_aging | 97.6% | 98.8% |
| college_biology | 38.194% | 68.056% |
| virology | 42.169% | 53.614% |
| anatomy | 43.704% | 52.593% |

Domain-general threshold (WikiText-calibrated): 0.3921. Hazard-specific
threshold (benign-bio-calibrated): 0.3320. Feature overlap between the two
sets: 3/20.

**This is a real trade-off, but a bad one.** Neighbor-subject accuracy does
recover substantially, but WMDP-bio forgetting more than doubles (29.4% ->
67.8%) -- most of the actual danger content comes back. Requiring *both*
gates to fire (AND-combination) is much stricter than the single gate, and
that strictness suppresses genuine forget-content detections almost as much
as it suppresses the false positives on benign content it was meant to fix.

## Status
Done. Net verdict: the kill test's finding (a real, undocumented collapse on
neighboring knowledge) stands on its own as a contribution. The proposed
two-gate fix does not -- as built, it isn't a net improvement over the
single gate; it just moves the cost from one axis (neighbor accuracy) to
another (core forgetting). A softer combination (e.g. a weighted OR, or
requiring hazard alone but with a stricter hazard threshold) might do better
but hasn't been tried.
