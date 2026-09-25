# Idea 3 (round 2): learned logistic-regression gate (mixed / negative result)

User's idea: "can we improve input classifier." DSG's own gate isn't really a
trained classifier: it's a hand-designed statistic -- a position counts as
"active" if ANY of the top-k selected features fires there (unweighted OR
across features), then the whole-sequence average of that is thresholded at
a fixed percentile calibrated on WikiText retain data. All 20 features are
treated as equally informative. This replaces that rule with an actual
learned classifier: logistic regression on each sequence's mean activation of
the same 20 DSG-selected features (so the feature *set* is identical between
the two gates -- only the gate itself changes), trained on the same cached
layer-3 activations (275 forget + 275 retain sequences) DSG's own gate is
calibrated on, and calibrated to the same target false-positive rate
convention (threshold = 95th percentile of the retain-side score).

## Calibration-set behavior (looked excellent)

- Training accuracy: 98.9%
- At the calibrated threshold: **100% of forget sequences trigger**, only
  **5.09% of retain sequences trigger** (matches the 5% FPR target almost
  exactly) -- a much cleaner separation on this data than a simple OR-rule
  typically gets, since weighting features by learned coefficients instead of
  treating all 20 equally lets the more-discriminative features dominate the
  score.

## Real evaluation (WMDP-bio + MMLU) result

| Config | WMDP-bio ↓ | MMLU avg ↑ |
|---|---|---|
| DSG original gate (cached) | 29.368% | 99.412% |
| Learned logistic-regression gate | **26.952%** | 93.207% |

Per-subject MMLU: high_school_us_history 100%, college_computer_science
88.9%, high_school_geography 98.1%, human_aging 85.9%.

**Mixed result: better forgetting, worse utility -- a Pareto trade-off, not a
clean win.** The gate does suppress WMDP-bio slightly more than DSG's
original (26.95% vs 29.37%), but MMLU drops more than 6 points (99.4% ->
93.2%), driven almost entirely by two subjects (college_computer_science and
human_aging) even though the *same calibration data* predicted only a 5%
false-positive rate.

## Why, honestly

This is the same failure mode the project's very first diagnosed bug already
identified (the original WMDP-Cyber MMLU collapse, and later A3's FDR-gating
result): **the gate is calibrated on raw WikiText prose but evaluated on
MCQ-formatted prompts**, and that calibration/eval-distribution mismatch does
not show up in the calibration numbers themselves. A logistic regression is
more expressive than DSG's simple OR-rule, which lets it fit the WikiText
calibration distribution's forget/retain boundary more precisely (98.9%
train accuracy, near-exact 5% target FPR) -- but that extra expressiveness
also means it fits WikiText-specific quirks more tightly, so it transfers
*worse*, not better, to the differently-formatted MCQ retain questions at eval
time. DSG's cruder OR-across-features rule turns out to be more robust to
this distribution shift precisely because it has fewer degrees of freedom to
overfit with.

## Status
Done, mixed result. A genuine improvement to the gate's *own* calibration
metrics does not translate to a genuine improvement at eval time, because the
calibration data itself (WikiText) does not represent the eval-time input
distribution (MCQ prompts) well enough for a more expressive classifier to
generalize. A natural follow-up (not attempted here, would need new
calibration data) is fitting the same logistic-regression gate on
MCQ-formatted calibration sequences instead of raw WikiText, which the
project's own earlier Cyber-corpus fix suggests should close most of this
gap.
