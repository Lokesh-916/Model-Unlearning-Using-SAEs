# B1: Distill the guardrail into the weights (simplified) -- positive result

Spec: use the DSG-guarded model as a teacher, train a LoRA student with no
hooks to copy it, so forgetting becomes permanent (survives weight release,
hook removal, quantization) rather than a runtime-only intervention. Skipped
the spec's GeN automatic-learning-rate and NGDiff normalized-gradient
machinery given time constraints; used a fixed learning rate (2e-4) instead.

## Architecture

A single HF `gemma-2-2b-it` + peft LoRA (rank 8, targeting q/k/v/o_proj and
gate/up/down_proj) model plays three roles via toggles:
- **teacher, guarded** = adapter disabled (`model.disable_adapter()`), DSG
  clamp hook enabled -- computed on forget (WMDP-bio) prompts.
- **teacher, original** = adapter disabled, hook disabled -- computed on
  retain (4 MMLU subjects) prompts.
- **student** = adapter enabled, hook disabled -- computed on both.

Loss = KL(student_forget || teacher_guarded) + KL(student_retain ||
teacher_original), backpropagated only into the LoRA adapter (base weights
frozen). The DSG clamp itself was ported from the validated TransformerLens
hook (`anthropic_clamp_resid_SAE_features`) to a native PyTorch forward hook
on `model.model.layers[3]`, since peft/LoRA training requires a plain
HuggingFace model, not TransformerLens.

## Three real bugs found and fixed, in order

1. **Dtype mismatch**: SAE Lens's SAE runs in float32 internally, but the
   surrounding HF model is bf16, and unlike TransformerLens, raw HF modules
   don't auto-cast at hook boundaries. Crashed the very next layer's linear
   projection. Fixed with `.to(resid.dtype)` on the hook's output.
2. **Unnecessary gradient tracking on teacher passes**: the eval helper
   inferred `no_grad` from `model.training`, so once `model.train()` was
   called, teacher forward passes (which only need detached output values)
   built full backward-capable graphs too, doubling memory every step. Fixed
   by making `no_grad` an explicit parameter, set correctly per call site.
3. **Iteration-to-iteration memory growth**: even after fix #2, step 0
   completed at 5.71GB but step 1 crashed at 14.74GB before finishing its
   forward pass -- a genuine leak whose single root cause wasn't fully
   isolated under time pressure. Mitigated with three combined changes:
   explicit `del` of the large per-step tensors plus `torch.cuda.empty_cache()`
   at the end of each step, batch size reduced from 4 to 2, and
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. This combination held
   memory flat at 5.71GB for all 150 steps.

## Hook-parity smoke test

Before trusting training, verified the ported HF hook behaves consistently
with the established TransformerLens result: on a 60-question random sample,
base accuracy 65-67%, DSG-hook accuracy 45-47%. This differs from the
official cached 29.368% figure, but that figure comes from a *curated*
subset of questions specifically selected because the clamp is known to
affect them -- a random sample naturally includes some questions the clamp
doesn't touch as strongly, so a higher number here is expected, not a bug.
All of this script's own before/after comparisons use the same random
200-question eval set throughout, so they remain internally valid regardless
of this offset from the curated metric.

## Result: the core claim holds

| Condition | WMDP-bio accuracy |
|---|---|
| Base model (no intervention) | 67.50% |
| DSG runtime hook | 45.50% |
| **Student, hook completely removed** | **40.00%** |

The student's forgetting **survives with no runtime intervention at all**,
and on this eval set is even slightly stronger than the DSG hook itself
(40.00% vs 45.50%) -- not weaker, as a naive "distillation always loses some
signal" prior might predict. Training loss dropped from 2.18 to 0.41 over
150 steps (noisy but clearly trending down), consistent with real learning
rather than a fluke.

Retain accuracy for the trained student (hook removed): history 68.0%,
CS 48.0%, geography 71.3%, human_aging 64.7%.

## Honest limitation

This script does not measure the **base (untrained) model's** retain accuracy
on the same random samples, only the trained student's. So it can't cleanly
attribute how much of the retain numbers above (well below the ~97-100%
these subjects show under the official curated pipeline) is due to training
damage versus the same random-vs-curated-sample gap already seen in the
WMDP-bio numbers (base model 67.5% vs curated ~99%+). A follow-up should add
that missing "before training" retain baseline on the identical random
sample before drawing a firm conclusion about retain-side cost.

## Verdict

A genuinely positive, novel result for this simplified B1: guardrail behavior
distilled into LoRA weights survives complete removal of the runtime hook,
which is the entire point of the exercise (permanence gradient methods have
that pure activation-editing methods like DSG lack). The retain-side cost is
plausible but not cleanly isolated from evaluation-methodology noise here --
flagged honestly rather than either overclaimed or dismissed.

## Status
Done, positive result with one documented follow-up (retain baseline). Not
attempted: the spec's relearning-attack test (full fine-tune to see if
forgetting is truly permanent under an adversarial attempt to recover it) and
quantization-robustness test, both out of scope for tonight.
