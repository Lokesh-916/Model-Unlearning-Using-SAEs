# C1: Attribution-based feature selection (negative result)

Spec: replace DSG's Fisher-info-style feature ranking (average squared
activation, forget/retain ratio) with attribution-patching scores
(activation x gradient of the correct-answer log-probability w.r.t. each
feature), then greedy-select a redundancy-aware feature set penalizing
decoder-vector cosine overlap. Compare against DSG's original selection at
equal feature count (20).

## Bug caught before the real result (fixed, documented in the script)

First attempt never froze the model's parameters before calling `.backward()`,
so every backward pass allocated gradient tensors for all ~2.6B model
parameters (not just the one activation tensor actually needed), OOMing the
GPU almost immediately. Only 1/150 forget samples and 0/150 retain samples
succeeded before that run's numbers were pure noise (reported MMLU collapse to
44.95% off a single sample -- discarded, not a real finding). Fixed by
`model.requires_grad_(False)` / `sae.requires_grad_(False)` plus explicitly
marking only the captured SAE feature-activation tensor as requiring grad.
Verified via a 20-iteration smoke test that memory stays flat before the real
run (150/150 forget, 148/150 retain succeeded, zero OOMs).

## Result (valid run)

| Config | WMDP-bio | MMLU_avg |
|---|---|---|
| DSG original (Fisher-info-style) | 29.368% | 99.412% |
| C1 attribution-selected | 65.613% | 99.757% |

Feature overlap with DSG's original top-20: **0/20**.

**This is a clean negative result, not an improvement.** Attribution-selected
features are dramatically worse at forgetting (65.6% vs 29.4% -- more than
double) for a negligible utility gain (99.76% vs 99.41%, near the ceiling
either way).

## Why, honestly

Attribution as implemented here measures something narrower than what DSG's
cruder method measures: "how much does this feature affect the log-probability
of the correct answer *letter* at the final token position," averaged (with
sign) across 150 different questions. That's a different signal from "does
this feature fire densely across forget-domain *content* in general" (DSG's
non-negative activation-rate proxy). A feature can be causally important for
picking the right multiple-choice letter on specific questions without
encoding the broad topical knowledge that needs suppressing -- and averaging
signed attribution across many different questions can cancel out real
per-question importance in a way a non-negative activation-rate measure never
does. Empirically, DSG's blunter method turns out to be the better proxy for
"this feature carries the knowledge that needs to be clamped."

## Status
Done. This directly tests (and empirically refutes, in this setting) the
paper's own stated critique of its Fisher-info shortcut. Not pursued further --
C2's dependency on this (attribution-graph-based layer selection) should be
reconsidered given this result, rather than built on top of it uncritically.
