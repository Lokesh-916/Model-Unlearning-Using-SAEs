# Idea 4 (round 2): per-feature variable clamp values (null result)

User's idea: "we are currently clamping feature values with -500... is there
any other way (variable value clamping for each feature instead of one
single value clamping to every feature)." Prior branches (C3's residual
projection, A2's calibrated mean-ablation) already showed that making the
clamp *gentler* collapses forgetting almost entirely -- the extreme magnitude
does real work. This experiment respects that finding: it keeps the same
*total* suppression budget (per-feature values average out to exactly -500,
matching the baseline) but redistributes it proportionally to each feature's
own natural activation range on the forget corpus:

```
clamp_j = -500 * (max_forget_activation_j / mean_j(max_forget_activation_j))
```

Motivation: the 20 selected features' natural max forget-corpus activations
range from ~10 to ~48 (a ~5x spread), so a flat -500 is proportionally far
more extreme for the smallest-range feature (~50x its own natural max) than
for the largest-range one (~10x). This tests whether removing that
proportional disparity changes anything, using the same feature set and gate
threshold as the baseline (only the clamp magnitude vector changes).

One real bug hit and fixed along the way: computing each feature's
per-sequence max with `.max(axis=1)` instead of `.max(axis=0)` maxed over the
wrong axis (features instead of positions), producing a 1024-length
"per-feature" vector instead of a 20-length one and crashing at broadcast
time; fixed and reverified (per-feature multipliers now correctly span
207-958 with mean exactly 500, matching the design).

## Result

| Config | WMDP-bio ↓ | MMLU avg ↑ |
|---|---|---|
| DSG flat -500 clamp (cached) | 29.368% | 99.412% |
| Per-feature variable clamp (same average magnitude) | 29.182% | 99.118% |

**Null result: no meaningful difference in either direction** (0.19pp better
forgetting, 0.29pp worse utility -- both well within the kind of run-to-run
noise seen elsewhere in this project's single-seed evaluations).

## Why, honestly

This is consistent with, and adds evidence to, the finding from A2/C3: once a
clamp value is already far outside a feature's natural activation range (here,
10x-50x for every one of the 20 features, even after redistributing the
budget), the specific allocation of that "how far outside" ratio across
features doesn't matter -- what matters is crossing some sufficiency
threshold at all. The clamp's job (per those two branches' interpretation) is
to push the reconstructed residual stream toward a value the decoder was
never trained to produce for that feature, and once you are decisively past
that boundary for every feature, making the "how far past" number bigger or
smaller by 2x in either direction for any individual feature doesn't change
the qualitative effect.

## Status
Done, null result. Along with A2 and C3 (gentler values fail) and this
(reallocating the same aggressive budget differently doesn't help either),
three independent experiments now agree that DSG's flat -500 constant is not
low-hanging fruit for improvement via the clamp *value*, in either the
"softer" or the "differently distributed" direction. Combined with idea 1's
and C1's negative results on feature *selection*, this round's two positive
findings (idea 2's AND-gated multi-layer, idea 3's learned gate, the latter
a trade-off) both improve DSG by changing *when/where* the intervention
fires, not the intervention's own magnitude -- a pattern worth keeping in
mind for the remaining ideas in this round.
