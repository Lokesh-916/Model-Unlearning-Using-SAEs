# B2: Extend SAEs to individual facts (TOFU/MUSE-style deletion) -- not attempted

Spec: keep the pretrained Gemma Scope SAE frozen and train a small trainable
"delta dictionary" of new latents (roughly 256-1024) on a fine-tuned model's
residual reconstruction error, so entity-specific knowledge (a fictitious
author in TOFU, a single news article in MUSE) gets its own addressable
features that a pretrained SAE has no feature for. Then gate those new
features with A1/A2, or bake them in with B1.

## Why this was not attempted tonight

This is a substantially larger undertaking than the other 8 improvements
attempted in this session, for three compounding reasons:

1. **New dataset**: TOFU (or a MUSE fold) isn't part of this reproduction at
   all. WMDP-bio/cyber and a handful of MMLU subjects are the only forget/
   retain corpora set up here. Pulling in TOFU means new data loading,
   understanding its fictitious-author question format, and its
   forget/retain split conventions.
2. **New SAE training loop**: every other experiment this session (including
   B1, C1, C3) reused a *pretrained, frozen* Gemma Scope SAE and only ever
   changed feature selection or the intervention mechanism around it. B2
   requires training NEW latents from scratch (even a small delta dictionary
   is a real SAE training run: reconstruction loss, a sparsity penalty,
   convergence monitoring) -- a different category of task from anything else
   attempted tonight, and one this reproduction's codebase has no existing
   scaffolding for.
3. **New evaluation infrastructure**: the spec's win condition (TOFU forget
   quality via a KS-test p-value against the retain model, model utility,
   Truth Ratio; MUSE VerbMem/KnowMem/PrivLeak) uses metrics that don't exist
   anywhere in this reproduction's `evals/` code. Building even a minimal,
   honest version of these would be its own multi-hour task before any
   result could be trusted.

Given the night's time budget was already spent getting A1-A3, B1, B3, and
C1-C3 to valid (positive or negative) results -- several requiring 2-3 bug-fix
cycles each -- attempting B2 as well risked producing a rushed, likely-buggy
result in an area with no existing safety net (no cached baseline to sanity-
check against, unlike every other experiment tonight, which could smoke-test
against an established number before trusting a full run).

## What a real attempt would need, if picked up later

1. Load a TOFU forget-split fine-tuned checkpoint (or fine-tune one -- this
   itself may need the LoRA feasibility confirmed tonight, since TOFU's
   standard setup fine-tunes on the fictitious-author data first).
2. Compute the frozen SAE's residual reconstruction error on TOFU forget/
   retain data.
3. Train a small (256-1024 latent) sparse dictionary on that residual error
   with a standard SAE training loop (reconstruction + L1 sparsity), which
   this repo does not currently have -- SAE Lens itself may provide a
   training entry point worth checking first, rather than writing one from
   scratch.
4. Select delta latents by forget/retain importance (same ratio-based
   selection method used everywhere else this session, directly reusable).
5. Gate with the A1 CUSUM detector or bake in via the B1 pipeline (both
   already built and validated tonight on `imp-a1-streaming-detector` and
   `imp-b1-distill-weights`).
6. Build minimal TOFU evaluation (Truth Ratio, a KS-test comparing forget-set
   answer distributions to the retain model's) before trusting any result.

## Status
Not attempted. This is an honest, scoped "out of budget for tonight," not a
failed attempt -- distinct from A2/A3/C1/C3's genuine negative results, which
were run and diagnosed.
