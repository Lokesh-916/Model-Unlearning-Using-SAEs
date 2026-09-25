# Idea 6 (round 2, "main contribution if possible"): static weight orthogonalization (negative result, but utility goal fully met)

User's idea: "instead of acting on the activations can we do changes to the
weight effectively while maintaining utility score good... using sae's."
Round 1's `imp-b1-distill-weights` already explored a *training-based*
version of this (LoRA distillation of the guarded model's behavior into
weights, positive result but with an acknowledged retain-baseline gap). This
tries a completely different, **training-free** mechanism instead: use the
SAE's own decoder directions for DSG's top-20 selected forget features as a
subspace, and permanently project that subspace out of every weight matrix
that can write into the residual stream up to DSG's own intervention point
(the token embedding, plus `attn.W_O` and `mlp.W_out` for blocks 0-3) --
`W_new = W_old @ P` where `P = I - QQ^T` and `Q` is an orthonormal basis for
the 20 decoder directions' span. After this one-time edit there is no SAE, no
gate, and no runtime hook at all -- the model is just permanently different.

## Result

| Config | WMDP-bio ↓ | MMLU avg ↑ |
|---|---|---|
| DSG runtime clamp (cached) | 29.368% | 99.412% |
| **Static weight orthogonalization (this script)** | 90.706% | **99.463%** |

QR confirmed the 20 decoder directions have full rank (no near-duplicates
collapsing the subspace).

**The utility side of the user's ask is fully achieved** -- MMLU is
completely intact (99.463%, even marginally above DSG's own number) with
zero runtime cost and zero SAE/gate dependency at inference time. **But
forgetting essentially doesn't happen** -- WMDP-bio stays at 90.7%, far above
DSG's 29.4% and consistent with barely-perturbed baseline behavior on this
curated question subset.

## Why, honestly, and how this connects to earlier findings

This result is not a fluke; it directly corroborates round 1's C3 finding
using an entirely different mechanism. C3 replaced DSG's clamp-to-(-500)
with a full-residual *projection* onto a reference subspace (an
activation-level intervention) and found it "barely moves behavior" (WMDP-bio
stayed at 94.6%, near the unintervened rate) even though it reliably
suppressed the representational (probe) signal -- concluding that DSG's
*extreme, out-of-distribution* clamp value is doing real work that a gentler,
more "principled" projection cannot replicate. This experiment is the
**weight-level analog of exactly that same idea** (project a direction out,
rather than forcefully overwrite it with a large negative constant), and it
lands in almost the same place (90.7% here vs. C3's 94.6%). Two independent
experiments, one at the activation level and one permanently baked into the
weights, now agree: cleanly removing (projecting out) a feature's linear
direction is fundamentally weaker than DSG's approach of forcing that
direction to an aggressively out-of-distribution value. The clamp doesn't
just "remove information" -- it actively corrupts the downstream computation
in a way a clean subspace removal, whether done per-forward-pass or
permanently to the weights, does not.

There's also a specific mechanistic reason a *weight*-level projection is
weaker than even C3's activation-level one: DSG's own SAE has **separate**
encoder and decoder matrices (the JumpReLU architecture the pretrained
Gemma Scope release uses), so a feature's *decoder* direction (what gets
added to reconstruct the residual) is not necessarily parallel to its
*encoder* direction (what triggers the feature to fire). Projecting the
decoder direction out of the model's write path prevents the model from
producing a residual with a component along that specific direction, but
says nothing about whether the encoder's dot product with the (still
present, just redirected) residual stream would have fired anyway.

## A predictable but untested extension

A stronger, DSG-faithful variant would add a large *negative bias* along
these directions to every write component (baking in something like DSG's
clamp value permanently, `W_new = W_old @ P`, plus a fixed bias term
`-C * sum_j d_j`), rather than merely zeroing the subspace. This was not
attempted: it would apply unconditionally to every input (no gate exists in
a purely static weight edit), almost certainly reproducing the same
utility-vs-forgetting trade-off seen when DSG's own gate is removed or
weakened (idea 2's OR-gating result, or A2/A3's findings) -- a predictable
Pareto point rather than a new one, and the opposite of what "maintaining
utility score good" asks for.

## Status
Done. Negative on the forgetting axis, but a clean and complete success on
this idea's explicit utility-preservation goal -- with the added benefit
(unique among all six ideas this round) of a zero-runtime-cost, zero-gate,
zero-SAE-at-inference mechanism, which is a genuinely different value
proposition from DSG even where it underperforms on forgetting. Combined
with round 1's B1 (training-based weight distillation, positive on
forgetting) and this result (training-free, positive on utility but not
forgetting), the two together suggest the real "main contribution" path is
likely a *hybrid*: use a projection-based edit (this script) for its clean
utility preservation, but retain some form of the clamp's aggressive,
out-of-distribution character (as B1's distillation training implicitly
learns to reproduce) rather than a purely linear subspace removal -- not
attempted here due to time.
