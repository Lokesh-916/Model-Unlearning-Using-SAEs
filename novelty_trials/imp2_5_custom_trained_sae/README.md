# Idea 5 (round 2): custom-trained SAE instead of the pretrained release (negative result)

User's idea: "can we make our own sae instead of using the existing one."
Round 1 (`imp-b2-sae-facts-tofu`) scoped this to a TOFU-based delta-latent SAE
extension and marked it out of scope for lack of training/eval infra. This
attempt scopes the same underlying question down to something tractable in
one sitting: train a small SAE from scratch on layer-3 residual-stream
activations collected from *this project's own* forget/retain corpora
(bio-forget-corpus + wikitext -- the same corpora DSG's own feature selection
is calibrated on), instead of using the pretrained Gemma Scope release
(trained on Pile-uncopyrighted, d_sae=16384). Everything downstream --
feature selection (`get_top_features_percentile`), gate calibration, and the
clamp intervention itself -- is DSG's own unchanged code; only the SAE is
swapped.

## Two failed attempts before a valid result (both caught via the L0 diagnostic)

**v1** (plain ReLU + L1, `L1_COEFF=8e-4`, d_sae=4096): converged to a very
low reconstruction error (MSE ~0.001-0.003) but **L0 ~4048/4096 -- essentially
every feature fires on every token.** This is not a sparse autoencoder, just a
dense one with a ReLU. At eval time the whole-sequence gate threshold
calibrated to ~1.0 (since almost every token trips almost every feature), so
the intervention **never fired** (`active=0/1` on every single batch in the
log) -- the run would have reported near-baseline (no-intervention) numbers
for both WMDP-bio and MMLU, which would look deceptively "safe" without the
L0 diagnostic catching the real problem first. Killed before finishing the
full (slow, print-heavy) evaluation once this was diagnosed.

**v2** (same architecture, `L1_COEFF` raised 60x to 0.05): L0 barely moved
(~3778/4096) even at 60x the penalty -- the reconstruction gradient dominated
regardless of the L1 coefficient's magnitude. Killed for the same reason
without finishing evaluation.

**v3** (this run's result): switched to a **TopK SAE** (hard-zero every
activation outside the top K=32 per token, chosen to roughly match the
pretrained release's L0=142/16384 ratio at our smaller width) instead of
tuning an L1 coefficient. This guarantees exact sparsity by construction and
immediately gave L0=32.0 (exactly K, every step) -- no coefficient search
needed.

## Result (v3, the valid comparison)

| Config | WMDP-bio ↓ | MMLU avg ↑ |
|---|---|---|
| Pretrained Gemma Scope SAE (cached) | 29.368% | 99.412% |
| Our custom-trained SAE (TopK, d_sae=4096, 3000 steps, ~307k tokens) | 46.840% | 93.632% |

**Negative result, but a genuine and informative one** (unlike v1/v2's
degenerate never-fires failure): our SAE is worse at both forgetting
(46.8% vs 29.4%) and utility (93.6% vs 99.4%) than the pretrained release.

## Why, honestly

Scale. The pretrained Gemma Scope SAE was trained at d_sae=16384 on
presumably a very large slice of Pile-uncopyrighted; ours is d_sae=4096,
trained for 3000 steps on ~307k tokens drawn from only 150+150 sequences.
Reconstruction MSE plateaued around 0.53 (vs the small residuals a
production-scale SAE achieves) -- our SAE's 32 active features per token are
a much noisier, less disentangled proxy for "this content is
forget-domain-relevant" than the pretrained release's features, so both
DSG's selection step (picking the wrong/noisier features) and its gate
(calibrating a threshold on noisier activation statistics) inherit that
noise. Directly training on the target domain's own corpora did not
compensate for the difference in scale and training budget.

## Status
Done, negative result, but with a real methodological lesson for any future
attempt: **always check L0 immediately after training an SAE, before
evaluating anything downstream.** Both failed attempts (v1, v2) would have
silently produced misleadingly "safe-looking" WMDP-bio/MMLU numbers (both
near baseline, since the never-firing gate means no intervention actually
happens) if the L0 diagnostic hadn't caught the real problem first --
exactly the kind of quiet, undetected failure mode this project's own
"negative results" log elsewhere warns about. A follow-up with a larger
d_sae, more training tokens, and more steps might close the gap to the
pretrained release, but was out of scope for this sitting.
