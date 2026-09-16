# C3: Residual/error-term projection + linear-probe leak measurement (negative result, but informative)

Spec: replace "clamp these 20 latents to -500" with "project the whole
residual stream (SAE reconstruction error included) out of the subspace
spanned by an expanded, redundancy-clustered feature family's decoder
directions" (h <- h - U U^T(h - mu_ref)). Measure with linear probes whether
either intervention actually erases forget-relevant information from
activations -- a metric DSG never reports.

## Setup

Expanded the original 20 seed features to 100 via decoder-cosine similarity
(threshold 0.15), built an orthonormal basis via QR of those 100 decoder
vectors, and computed a reference mean residual (mu_ref) from ~17k WikiText
tokens. The projection intervention is gated by the exact same whole-sequence
rho(x) threshold as DSG's original clamp (0.5458) -- only the *replacement
value* differs (a subspace projection toward mu_ref, instead of an extreme
constant).

## Two issues found and fixed

1. `sklearn`'s `LogisticRegression(multi_class=...)` argument was removed in
   the installed version (1.9) -- multinomial is now the default. Trivial fix.
2. First probing pass measured residuals at layer 3 (the intervention layer
   itself). Even the **unintervened base model's** probe accuracy came out at
   chance (24.44% vs 25% chance) at that layer -- not evidence of "no leak,"
   evidence that layer 3 is too early in the network to have linearly decided
   the answer yet, making the whole comparison meaningless. Fixed by probing
   at layer 22 instead (intervention still happens at layer 3, upstream);
   verified via a smoke test that the unintervened base model's probe accuracy
   there is clearly above chance (75% on a small sample) before trusting the
   full run.

## Result

| Config | WMDP-bio | MMLU_avg |
|---|---|---|
| DSG original clamp | 29.368% | 99.412% |
| C3 residual projection | **94.610%** | 100.000% |

Linear-probe accuracy predicting the correct WMDP-bio answer from layer-22
residual activations (4-way, chance = 25%):

| Condition | Probe accuracy |
|---|---|
| No intervention (upper bound) | 63.33% |
| **DSG original clamp** | **17.78%** |
| **C3 residual projection** | **56.67%** |

## Verdict: a clean negative result for C3, but a genuinely informative one

Both the behavioral metric (WMDP-bio accuracy) and the representational metric
(probe accuracy) agree, independently, on the same conclusion: **DSG's blunt
clamp to an extreme, out-of-distribution value is not wasteful -- it is
substantially more effective at actually erasing the forget-relevant
information than the more "principled" subspace projection.** DSG's clamp
drops probe accuracy from 63.33% to 17.78% (*below* chance -- the clamp
doesn't just scramble the signal, it consistently steers it toward
wrong-answer directions a probe can exploit). C3's projection barely moves
probe accuracy (63.33% -> 56.67%) and barely moves behavior (WMDP-bio stays
at 94.6%, near the unintervened base rate).

This directly speaks to the paper's own implicit critique (that clamping to
-500 is a crude, out-of-distribution hack): the crudeness may be doing real
work. Forcing an extreme value reliably overrides whatever redundant pathways
carry the information forward, whereas surgically projecting out a specific
(even fairly broad, 100-feature) decoder-derived subspace leaves other paths
untouched. C3 as specified is not a better replacement for DSG's clamp on
either axis tested here.

## Status
Done. Both bugs (sklearn API, chance-level probe layer) documented and fixed
before trusting the result above. Not pursued further -- a natural follow-up
(not attempted) would be to test whether a MUCH larger expanded feature family
(more of d_model's dimensions) or a stronger push past mu_ref (not just
projecting to it) closes the gap, but that starts to resemble reintroducing
DSG's own extremity by another route.
