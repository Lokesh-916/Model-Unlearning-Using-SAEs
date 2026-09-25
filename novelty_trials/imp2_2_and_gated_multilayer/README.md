# Idea 2 (round 2): AND-gated multi-layer intervention (positive result)

User's idea: "currently we are using sae at layer 3 can we use that any other
layer or multiple layers so that improvements will come." Round 1
(`imp-c2-multilayer-circuit`) already tried a second layer (layer 8, matching
average_l0=142) but gated each layer's clamp **independently** (OR
semantics): clamp fires at layer 3 whenever layer 3's own whole-sequence
firing rate exceeds its threshold, regardless of layer 8, and vice versa.
That compounded each layer's own retain false-positive rate into a real
Pareto trade-off (WMDP-bio 29.368% -> 27.695%, but MMLU 99.412% -> 97.941%).

This tries the fix that diagnosis directly suggests: gate on the
**conjunction** of both layers' triggers (AND semantics) instead of the
union. Mechanically, `calculate_MCQ_metrics` is called with
`mcq_batch_size=1` so it scores one question per forward pass with a
deterministic hook order (layer 3 fires before layer 8). Pass 1 installs
read-only hooks that record each layer's own per-question trigger without
modifying the residual stream; pass 2 installs clamp hooks that read back the
precomputed joint (AND) decision per question and clamp both layers only
where both agreed. Reuses round 1's cached layer-3/layer-8 feature-sparsity
files -- no new sparsity computation, just two evaluation passes per dataset.

## Result

| Config | WMDP-bio ↓ | MMLU avg ↑ |
|---|---|---|
| Single-layer-3 (paper's original, cached) | 29.368% | 99.412% |
| OR-gated layer 3 + layer 8 (imp-c2, round 1) | 27.695% | 97.941% |
| **AND-gated layer 3 + layer 8 (this script)** | 31.227% | **99.706%** |

Per-dataset trigger rates for the AND-gate: wmdp-bio 476/538 (88.5% of
forget-domain questions still trigger both layers), vs 0/108, 0/9, 0/103, and
only 4/85 on the four retain-domain MMLU subjects.

**This is a positive result relative to OR-gating.** Requiring both layers to
agree recovers almost all of the utility that independent (OR) gating lost --
MMLU goes from 97.941% back up to 99.706%, actually *above* the single-layer
baseline -- while still triggering on the large majority of forget-domain
questions (88.5%) and suppressing WMDP-bio accuracy to 31.2%, only 1.9
points worse than the single-layer baseline's 29.4% and far better than no
intervention (base model ~65-67% per prior branches' notes). The multi-layer
idea genuinely helps *if* the two layers' gates are combined conjunctively
rather than independently: OR-gating's problem wasn't "using two layers,"
it was letting each layer's own false-positive rate leak through unfiltered.

## Trade-off, honestly

AND-gating does not beat the single-layer baseline outright -- WMDP-bio is
slightly worse (31.2% vs 29.4%). Requiring agreement between two layers is
inherently more conservative than either single-layer gate, so some
forget-domain questions that only strongly trigger one layer's features slip
through un-clamped. The gain here is specifically in *utility preservation
when adding a second layer's gate*, not in improving forgetting beyond what
layer 3 alone already achieves. Whether this trade is worth it depends on
whether the eventual goal is "add more layers/coverage without utility
regressing" (this result supports that) vs. "add more layers to forget
strictly more" (this result does not support that, at least for this
layer-3+layer-8 pair at n=20 features per layer).

## Status
Done. First positive multi-layer result in this line of investigation:
demonstrates that the OR-gating Pareto trade-off diagnosed in round 1 is
fixable by changing the *gate combination logic*, not by abandoning
multi-layer intervention. A natural follow-up (not attempted here) is
whether a 3rd layer, or a soft/weighted combination (e.g. require agreement
from at least 2 of N layers) shifts the WMDP-bio/MMLU trade-off further.
