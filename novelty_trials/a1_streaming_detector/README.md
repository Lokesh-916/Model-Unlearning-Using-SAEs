# A1: Streaming detector that survives dilution

Spec: replace DSG's whole-sequence-average trigger rho(x) with a token-level
streaming detector (per-token forget-vs-retain log-likelihood ratio, accumulated
via CUSUM, gated once accumulated evidence crosses a boundary calibrated on
retain data).

## 01: the vulnerability (why this is needed)

DSG's trigger is `rho(x) = mean over all tokens of "does a forget feature fire
here"`. Padding a question DSG currently blocks with unrelated benign text
dilutes this average below threshold without changing the dangerous content.

Threshold tau = 0.5458 (main Bio config, 20 features, layer 3).

| Padding | Sequence length | Mean rho(x) | Bypass rate |
|---|---|---|---|
| None | 115 tokens | 0.80 | 0% |
| ~150 words | 285 tokens | 0.29 | 98.3% |
| ~400 words | 563 tokens | 0.16 | 95.0% |
| ~800 words | 1004 tokens | 0.11 | 96.7% |

## 02: windowed gate (intermediate attempt, not the final A1 design)

Replace the whole-sequence average with a local sliding-window rate. Two
calibration bugs found and fixed (edge-shrinking inflated rate to 1.0; then
calibrating against the per-sequence MAX local rate hit the same max-of-many-
samples statistical trap) before landing on a working pooled-percentile
calibration (`02_windowed_gate_defense_final.py`, the fixed version).

Window=24 results, utility WMDP-bio=26.95%/MMLU=97.65% (baseline 29.37%/99.41%):

| Padding | Bypass rate |
|---|---|
| 0 | 1.7% |
| 150 | 20.0% |
| 400 | 30.0% |
| 800 | 35.0% |

Window=8 gave essentially the same curve (1.7%/23.3%/28.3%/35.0%), showing the
residual bypass isn't from window width -- it's attention-based dilution of the
model's own representations, one layer below where any aggregation-level fix
can reach.

## 03: CUSUM streaming detector (the actual A1 spec)

Per-feature zero-inflated Gaussian fit on forget vs retain token activations,
per-token LLR, CUSUM accumulation, boundary calibrated at 95th percentile of
per-sequence max S_t on retain data (h=38.279). One real coding bug caught and
fixed (gate never checked whether S crossed h at all) before the result below.

Utility: WMDP-bio=30.86%, MMLU=98.24% (baseline 29.37%/99.41%).

| Padding | Bypass rate (CUSUM) | Bypass rate (windowed, w=24) | Bypass rate (whole-sequence) |
|---|---|---|---|
| 0 | 35.0% | 1.7% | 0% |
| 150 | 36.7% | 20.0% | 98.3% |
| 400 | 36.7% | 30.0% | 95.0% |
| 800 | 36.7% | 35.0% | 96.7% |

**Key finding**: CUSUM's bypass rate is flat regardless of padding amount --
genuinely dilution-invariant, unlike both other gates, which degrade as padding
grows. Trade-off: weaker baseline sensitivity (35% vs windowed's 1.7% at zero
padding), traced to a heavy-tailed retain calibration distribution (median
max_S=2.87, max=2843.19) forcing a conservative boundary.

## Not yet done for A1
- SSPU's four jailbreak-rewrite styles (spec's second win condition) --
  untested here, only the dilution axis has been run.
- Tightening the retain calibration (trimming outlier retain sequences, or a
  more robust per-feature model) to try to close the baseline-sensitivity gap
  without losing dilution-invariance.
