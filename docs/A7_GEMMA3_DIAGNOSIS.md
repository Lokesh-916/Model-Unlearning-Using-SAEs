# A7 diagnosis: DSG fires on Gemma 3 but changes almost no answers (session 17, 2026-10-06)

**Verdict: a real finding, not a bug.** DSG's clamp value (−500 per selected feature, unit-norm decoder rows) is an
absolute constant. It fits Gemma-2-2B's residual stream at layer 3 (median norm ≈ 92) but Gemma 3's residual stream is
75–340× larger at the Gemma Scope 2 layers. The same edit that overwrites Gemma 2's residual by 34× is only
0.33× (1B) or 0.10× (4B) of Gemma 3's residual. When the gate fires on only a few tokens, the edit is too weak to reach
the answer position. No re-run was queued.

## Checks (CPU, benign sentences, harness loader; `scripts/diag_gemma3_clamp.py`)

| check | Gemma-2-2B L3 | Gemma-3-1B L13 | Gemma-3-4B L17 | conclusion |
|---|---|---|---|---|
| hook site | `blocks.3.hook_resid_post` | `blocks.13.hook_resid_post` | `blocks.17.hook_resid_post` | SAE metadata names a valid TL hook; JumpReLU, no input normalisation, `apply_b_dec_to_input` false |
| ‖W_dec[f]‖ (selected) | 1.000 | 1.000 | 1.000 | same decoder scale |
| median ‖resid‖ | 91.7 | 6,834 | 30,931 | **the difference** |
| ‖edit‖ (20 features at −500) | 3,078 | 2,221 | 3,065 | |
| ‖edit‖ / ‖resid‖ | **33.6** | **0.33** | **0.10** | |
| ‖edit‖ = ‖(a′−a)·W_dec‖ | yes | yes (fp32; bf16 within 0.2 %) | yes (bf16 within 0.2 %) | error term does not cancel the edit |
| next layer changes (all tokens clamped) | 32.7× | 0.35× | 0.18× | hooked output feeds the next layer |
| last-token KL, all tokens clamped | 24.8 | 6.95 | 0.26 | |
| last-token KL, one middle token clamped | 0.021 | 0.0024 (bf16 0.0028) | 0.0029 | ~9× weaker per fired token on Gemma 3 |

## Measured run behaviour (gpuws A7 TEST, items.parquet, ids and probabilities only)

| size | condition | fired WMDP / utility | preds changed (of 848) | WMDP acc base → gated | median max |Δp| on fired items |
|---|---|---|---|---|---|
| 1B | DSG | 0.157 / 0.014 | 6 | 0.4647 → 0.4647 | 3.5e-5 |
| 4B | DSG | 0.323 / 0.081 | 0 | 0.6107 → 0.6107 | 2.4e-7 |
| 12B | DSG | 0.187 / 0.014 | 0 | 0.7143 → 0.7143 | 3.6e-7 |
| 1B | best-fix (window w16) | 0.002 / 0.000 | 1 | 0.4647 → 0.4631 | 9.7e-1 (1 item) |
| 4B | best-fix | 0.022 / 0.009 | 0 | 0.6107 → 0.6107 | 6.6e-7 |
| 12B | best-fix | 0.049 / 0.000 | 0 | 0.7143 → 0.7143 | 1.4e-7 |

Correction to the session-16 wording: on 1B, DSG does change 6 answers (they cancel in accuracy). On 4B/12B it changes none.
Non-fired items are bit-identical, which also shows the hook is a no-op when the gate is off. τ calibrates to ≈ 0
(0.00098 on 1B, 0.0 on 4B/12B) because the selected features almost never fire on the retain cache, so any single firing
token opens the gate.

## Implication and a possible follow-up (not run; would need a DEVIATIONS row)

DSG's recipe assumes an SAE layer whose residual scale makes −500 overwhelming. A norm-relative clamp, e.g. multiplier =
500 × (median ‖resid‖ at the hook / 91.7), about 37,000 on 1B and 169,000 on 4B, would test whether the selected Gemma
Scope 2 features are on the answer path at all. That is a new method variant, not a fix. In the paper, A7 reads:
"DSG's fixed clamp does not transfer across model families; its strength must be scaled to the residual norm."
