# A3: Many-concept guardrails with statistical error control (scaled down; surfaces a deeper problem)

Spec: replace a single merged gate with a bank of per-concept detectors, using
Bonferroni-style multiple-testing control so the combined retain
false-positive rate stays bounded as more concepts are added, instead of
compounding (DSG-union's expected failure mode at scale). Scaled down from the
spec's 50 concepts to 4, given this hardware and available data: wmdp-bio,
wmdp-cyber (both already had cached forget/retain activations from the main
reproduction), plus two MMLU-subject concepts built the same way as B3's
benign-bio corpus (high_school_us_history, college_computer_science).
high_school_geography and human_aging serve as untouched holdout retain
checks -- never targeted, purely to measure collateral damage.

10 features per concept (vs the main config's 20, to keep 4 concepts'
combined feature budget reasonable). Naive calibration: each detector
independently at the plain 95th percentile (DSG's original per-detector
convention). Bonferroni calibration: each at the 100*(1-alpha/n) = 98.75th
percentile (alpha=0.05, n=4 concepts).

## Result

| Dataset | Naive union | Bonferroni union |
|---|---|---|
| wmdp-bio (forget target) | 25.09% | 25.46% |
| wmdp-cyber (forget target) | 26.55% | 28.73% |
| high_school_us_history (forget target) | 25.98% | 25.49% |
| college_computer_science (forget target) | 14.00% | 15.00% |
| **high_school_geography (untouched holdout)** | **24.27%** | **28.16%** |
| **human_aging (untouched holdout)** | **37.65%** | **36.47%** |

Per-concept naive/Bonferroni thresholds: wmdp-bio 0.486/0.803, wmdp-cyber
0.030/0.059, high_school_us_history 0.286/0.358, college_computer_science
0.079/0.190.

## This did not produce a clean "FDR control helps" result -- it surfaced a real, different problem

The four forget targets all show the intended effect (accuracy near or below
the 25% chance level -- college_computer_science actually goes notably below
chance, a known DSG "over-unlearning" pattern also seen elsewhere this
session). But the two **untouched holdout subjects are damaged almost as
badly as the intentional forget targets** (24-38% accuracy, vs the ~97-100%
these exact same subjects show under the paper's original single-gate
config, evaluated repeatedly earlier this session). Bonferroni correction
raised every individual threshold substantially (e.g. wmdp-cyber's by 2x) but
barely moved the holdout numbers (geography 24.3%->28.2%, human_aging
37.6%->36.5%) -- essentially no protective effect, despite the stricter
per-detector thresholds.

## Diagnosed cause

Investigated wmdp-cyber's detector specifically (its threshold, 0.0296, was
suspiciously low compared to the other three concepts' 0.08-0.49 range).
Its 10 selected features do fire genuinely rarely on WikiText (median
per-sequence rate 0.012, 95th percentile 0.030) -- narrow, low-base-rate
features, which is normal. The problem is what they get calibrated against
vs. what they get evaluated against: **the threshold is calibrated on raw
WikiText prose, but applied at eval time to MCQ-formatted prompts** (question
+ lettered choices), a stylistically different distribution. This is the
same root-cause pattern as the very first fix of this whole reproduction
project (the original WMDP-Cyber MMLU collapse, caused by exactly this kind
of calibration/eval-distribution mismatch) -- it turns out to resurface here
at smaller per-concept feature budgets (10 vs the main config's validated 20),
where the selected features are apparently less robust to this format shift.
Raising the threshold via Bonferroni correction doesn't fix a mismatch between
*what the threshold was calibrated against* and *what it's being applied to*
-- it just raises the bar on the same miscalibrated statistic, which is why it
barely helped the holdout subjects here.

## Verdict

A3 as specified (statistical multiple-testing control across independently
calibrated detectors) is a reasonable idea, but this test shows it isn't
sufficient on its own: the FDR-control layer can't compensate for a
calibration/eval-distribution mismatch happening one level below it. The
real fix suggested by this result (not attempted, out of scope for tonight)
would be calibrating each concept's threshold against MCQ-formatted retain
text in the same style as eval time, not generic WikiText prose -- consistent
with the project's very first diagnosed lesson, now re-confirmed at a
different (multi-concept, reduced-feature-budget) scale.

## Status
Done, honestly reported as a negative/diagnostic result rather than the
clean scaling-curve win the spec envisioned. Not re-run with MCQ-formatted
calibration data due to time constraints, but the fix direction is clear and
documented above.
