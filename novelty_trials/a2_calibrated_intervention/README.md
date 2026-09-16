# A2: Calibrated "never-learned" intervention instead of clamping to -500 (negative result, cross-validates C3)

Spec: replace DSG's clamp-to-(-500) with conditional mean-ablation toward a
reference set's feature means -- an approximation of "what would this feature
look like on a question the model is naturally unsure about" -- with a scalar
strength chosen so forget-set output confidence matches the reference set's.

## Kill test: confirmed

| Condition | Mean entropy | Mean max-prob |
|---|---|---|
| Base model (no intervention) | 0.2369 | 0.9248 |
| **DSG original clamp** | **0.7587** | **0.7398** |

DSG's clamp genuinely produces outputs that look statistically different from
natural model behavior (much higher entropy, notably lower confidence) --
confirming the spec's premise: DSG doesn't just make the model "not know" the
answer, it visibly perturbs the output distribution in a detectable way.

## Reference set

318 lowest-confidence WMDP-bio questions (bottom quartile, out of the full
1273-question test set) under the base model, mean max-prob 0.3253 vs the
overall mean of 0.6669 -- a genuine "the model is naturally unsure here" set,
used to compute per-feature reference mean activations for the 20 selected
features.

## Result: mean-ablation does not achieve real forgetting at any tested strength

| Strength (x reference mean) | WMDP-bio accuracy |
|---|---|
| 0.5x | 99.26% |
| 1.0x | 99.44% |
| 2.0x | 99.81% |
| 4.0x | 99.44% |

All four settings land within 74-75 percentage points of the 25% chance
target -- essentially no forgetting at all, indistinguishable from not
intervening. Full eval at the best (closest-to-chance) setting, 0.5x:

| Config | WMDP-bio | MMLU_avg | Entropy | Max-prob |
|---|---|---|---|---|
| Base (no intervention) | -- | -- | 0.2369 | 0.9248 |
| DSG original clamp | 29.368% | 99.412% | 0.7587 | 0.7398 |
| **A2 mean-ablation (best)** | **99.257%** | **100.000%** | **0.2432** | **0.9195** |

A2's entropy/confidence (0.2432 / 0.9195) is nearly identical to the
unintervened base model (0.2369 / 0.9248) -- not because it looks "naturally
uncertain," but because it is barely intervening on behavior at all.

## Verdict: this cross-validates C3's finding, not a coincidence

This is the second independent experiment tonight (after C3's residual
projection) to test a "softer, more principled" replacement for DSG's clamp,
and the second to fail for the same underlying reason: **interventions that
stay within a plausible activation range do not reliably override the
model's downstream computation.** DSG's -500 clamp isn't an arbitrary crude
hack -- pushing activations to an extreme, out-of-distribution value appears
to be doing real, necessary work to force the behavioral change. A "gentler"
intervention that looks statistically more natural also, empirically, fails
to actually forget anything.

## Status
Done. Not pursued further -- a much larger multiplier (10x-50x the reference
mean) might eventually approach forgetting, but at that point it stops being
a "calibrated, natural-looking" replacement value and just reinvents DSG's
own extremity via a different route, defeating the point of the exercise.
