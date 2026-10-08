# DSG → Publication: Improvement Roadmap for the SAE Unlearning Capstone

Sep 30, 2026 · @Amaloch

## 1. Executive summary

The publishable version of this project is not "DSG with lower WMDP accuracy". It is: **SAE guardrails hide knowledge instead of erasing it — we break DSG along five axes, explain each failure with SAE features, and propose a method that moves the gate into the weights with a provable retain guarantee.**

Your mid-review has good raw material: a clean reproduction (29.37% vs 29.64%), a real bypass (dilution), a streaming detector, and a distillation result. A reviewer would still reject it today. It reads as nine loosely linked experiments, on one MCQ benchmark, one model, one seed, with no error bars.

Two of your planned directions now have published close neighbours. CRISP (ACL 2026) already makes SAE unlearning permanent with LoRA across multiple layers, which is roughly your B1 + C2. ARIA (arXiv, 14 Sep 2026) already gates SAE unlearning during generation on TOFU and WMDP, which is roughly your A1 + "QA-type" idea. Novelty has to come from what they did not do — and both left big gaps (Section 4).

**The seven must-dos, in priority order**

1. Pick one thesis (Section 5) and drop the "nine interventions" framing.
2. Build one evaluation harness and rerun DSG on TOFU, MUSE, open-ended WMDP and biology-adjacent MMLU, with 3+ seeds, confidence intervals and a held-out split for tuning.
3. Turn the dilution finding into a short theorem plus a full stress suite: dilution, question-splitting, cross-lingual and code-mixed prompts, LLM red-teaming that transfers, and white-box obfuscation.
4. Redesign B1. Distilling into a LoRA on the same base model is the setting UNDO showed is *not* robust, and it overlaps CRISP. Distill into a noised student with DSG as teacher, or do a closed-form, transcoder-guided null-space edit.
5. Measure depth of forgetting, not MCQ accuracy: activation probing, relearning, quantization and steering-recovery attacks, plus MUSE PrivLeak and TOFU Forget Quality.
6. Compare against the 2025–26 neighbours (CRISP, SSPU, ARIA-style detector, UNDO, RMU, SimNPO) and against a plain linear-probe detector.
7. Add a second model family. Gemma Scope 2 covers Gemma 3 (270M to 27B) with SAEs trained on the instruction-tuned models, plus transcoders.

**Premise check.** You were right to doubt the MT-Bench point: DSG did report MUSE (both corpora) and MT-Bench. It did not test TOFU, open-ended hazardous generation, non-English or encoded prompts, adaptive attacks, or any model besides Gemma-2-2B.

## 2. How a reviewer would read the mid-review today

Verdict as it stands: weak reject at a workshop, clear reject at a conference. The problem is framing and evidence, not effort.

**What a reviewer would like**

- A faithful reproduction within noise. This buys trust for every later number.
- The dilution bypass. It is simple, reproducible, and breaks a method from a top venue (ICML 2025).
- The negative results (A2, C3). They become useful ablations once framed that way.
- The hook-removal distillation result. It is a good seed for the "permanence" story.

**What a reviewer would attack**

| Issue | Why a reviewer cares | Where this report fixes it |
| --- | --- | --- |
| Nine interventions, most marked "Insight" | Reads like a lab notebook. No single claim to accept or reject. | Section 5 |
| Your best forgetting gain is not significant | On 522 questions at about 28% accuracy, one standard error is about 2.0 points. 29.37% vs 27.70% is inside one standard error. | Sections 6, 11 |
| Only WMDP-Bio multiple choice | Near the 25% floor, MCQ accuracy cannot separate "hidden" from "erased". | Section 6 |
| One model, one SAE, one layer, one seed | Every claim could be a Gemma-2-2B quirk. | Section 6 (A7) |
| No 2025–26 baselines | CRISP, SSPU, ARIA and conditional clamping all target the same problem. | Sections 4, 12 |
| "Attack success" is undefined | CUSUM sits at 36% attack success even at zero padding, which looks worse than plain DSG on normal inputs. No false-positive rate is shown. | Sections 7, 8 |
| B1 tested only by removing the hook | A LoRA on frozen base weights keeps the knowledge in those weights. Relearning was never tried. | Section 9 |
| Tuning on the test questions | DSG picks the config that minimises WMDP accuracy on the evaluation set itself. Inheriting this is a red flag. | Section 6 (A8) |
| Utility excludes biology | DSG's MMLU check for WMDP-Bio uses history, CS, geography and human aging only. Biology neighbours are where damage would show. | Section 6 (A3) |
| Numbers disagree across slides | See the list in Section 16. | Section 16 |

## 3. What DSG actually tested, and where its own numbers are weak

DSG was evaluated more widely than your slide suggests, but only in friendly conditions. Every test uses plain English, one model, and a defender who keeps the hook.

| Evaluation axis | Tested in DSG? | What exactly | Opening for you |
| --- | --- | --- | --- |
| WMDP-Bio / Cyber MCQ | Yes | Only questions the model gets right under all 24 option orders: 522/1,273 Bio, 275/1,987 Cyber | Report full set too; add open-ended version |
| MMLU utility | Yes, partly | 305 filtered questions: history, CS, geography, human aging. No biology for the Bio task | Biology-adjacent subjects (A3) |
| MT-Bench fluency | Yes | 7.78 vs 7.36 for the untouched model (std 0.16 over 5 runs) | Uninformative for gated methods (below) |
| MUSE News + Books | Yes | VerbMem, KnowMem, PrivLeak, utility, scalability (0.8M–3.3M tokens), 4-step sequential | Fix VerbMem and PrivLeak failures (C5, C6) |
| Relearning | Yes, one threat model | Fine-tune on WMDP-Bio test set, 10 epochs, defender keeps the hook | Open-weight setting; stronger attacks (A6) |
| Zero-shot / data efficiency | Yes | Neuronpedia labels pick features; 20–80% data | — |
| TOFU | **No** | — | A2, A5 |
| Open-ended hazardous generation | **No** | MUSE KnowMem is QA, but on news/books only | A2 |
| Non-English, code-mixed, encoded prompts | **No** | — | B3 |
| Adaptive attacks (black- or white-box) | **No** | Dismissed as needing white-box access | B4, B5 |
| Probing for leftover knowledge | **No** | — | A4 |
| Other models, SAE widths, layers | **No** | Listed as future work | A7, C3 |

**Weak spots inside DSG's own tables** (each is a measurable target for you)

- **Verbatim memorisation is only half removed.** MUSE VerbMem drops 44.2% (News) and 44.7% (Books). Plain gradient ascent drops 97.1% and 83.5%. DSG hides what a text *says*, not its exact *words*.
- **Privacy leakage fails on both corpora.** PrivLeak is +12.08 (over-unlearning) on News and −23.18 (under-unlearning) on Books. The accepted band is ±5. A membership attack can tell the gate fired.
- **MT-Bench cannot see a gated method's damage.** Prompts that do not trigger the gate produce exactly the base model's output. A score above the untouched model (7.78 vs 7.36) is judge noise, not an improvement.
- **The optimality theorem assumes away the attack.** Theorem 3.4 (Neyman–Pearson) needs the trigger score to be larger on forget inputs. Dilution is precisely an input where that stops being true.
- **Relearning resistance needs the defender's hook.** In the paper's own words, the setting is API-only. Anyone with the open weights simply skips the hook.
- **The SAE does not match the model it guards.** DSG uses an SAE trained on the *pretrained* Gemma-2-2B to guard the *instruction-tuned* model, and on MUSE a model further fine-tuned on the corpus. Whether the SAE even "sees" the fine-tuned knowledge was never checked.

## 4. The competitive landscape (as of 30 Sep 2026)

The field moved fast after DSG. Two of your ideas are now published by others, but every neighbour left a gap you can own.

| Work (venue) | What it does | What it did not test | What it means for you |
| --- | --- | --- | --- |
| [CRISP](https://aclanthology.org/2026.acl-long.82/) (ACL 2026) | LoRA fine-tuning that suppresses salient SAE features across several layers. WMDP Bio/Cyber, Harry Potter; Gemma-2-2B and Llama-3.1-8B | Relearning or adversarial extraction (listed as future work). No DSG comparison | Close to your B1 + C2. Its "persistent" claim was never stress-tested — you can |
| [ARIA](https://arxiv.org/abs/2609.16229) (arXiv, Sep 2026) | Sparse logistic head over SAE statistics (frequency, max, early-window max) gates each generation step; outputs "I don't know". TOFU, R-TOFU, WMDP-Cyber | Dilution, adaptive or transfer attacks, open weights. Its TOFU Forget Quality is 0.01% and WMDP accuracy 0%, i.e. refusal, not ignorance | Close to your A1 + QA idea. Must be a baseline. Your openings: adaptive attacks and indistinguishability |
| [SSPU](https://arxiv.org/abs/2505.24428) (ICML 2025 workshop) | SAE decoder directions define a subspace that constrains an RMU-style weight update | WMDP-Cyber only; four fixed jailbreak templates; no relearning; no DSG comparison | A weight-based SAE baseline for Section 9 |
| [PISCES](https://arxiv.org/abs/2505.22586) (2025) | Erases concepts inside MLP output weights using SAE-decomposed directions; features picked by hand | Automatic selection; robustness | Closest neighbour to the closed-form edit in D2 |
| [Yamashita et al.](https://arxiv.org/abs/2509.15631) (2025) | Pushes a target entity's SAE activations toward those of *unknown* entities | Hazard domains | Template for "plausible ignorance" (C5) |
| [UNDO](https://arxiv.org/abs/2506.06278) (NeurIPS 2025) | Unlearn, then distil into a noised copy; robust to relearning | SAE-based teachers | Distilling into the *same* reference model is not robust. Fixes B1 (D1) |
| [Mechanistic Unlearning](https://proceedings.mlr.press/v267/guo25k.html) (ICML 2025) | Editing fact-lookup MLPs gives unlearning that survives relearning and format changes | SAE or transcoder features | Motivates D2; warns that output-based attribution (your C1) gives less robust edits |
| [Obfuscated Activations](https://arxiv.org/abs/2412.09565) (ICLR 2026) | Attacks that fool SAE and probe monitors while keeping harmful output; code released | DSG specifically | Ready-made tooling for B5 |
| [Kantamneni et al.](https://proceedings.mlr.press/v267/kantamneni25a.html) (ICML 2025) | SAE probes rarely beat plain linear probes | Unlearning gates | A linear-probe detector baseline is mandatory (C2) |
| [Multilingual unlearning](https://arxiv.org/abs/2606.03291) (ICML 2026) | Unlearning acts in late layers; one steering direction recovers 50–90% of forgotten facts across languages | SAE-based methods | Motivates B3 and the steering-recovery attack (A6) |
| [OpenUnlearning](https://arxiv.org/abs/2506.12618) (NeurIPS 2025) | 13 methods, 16 metrics, TOFU/MUSE/WMDP, 450+ checkpoints | — | Your evaluation harness (A1) |
| [GROM](https://arxiv.org/abs/2608.05783), [ZeroUnlearn](https://arxiv.org/abs/2605.18879) (arXiv 2026) | Gradient-free or editing-based unlearning | SAE guidance | Read before claiming novelty for D2 |

**The white space nobody owns yet**

1. Adaptive, black-box *transfer* attacks on SAE guardrails. DSG's claim that such attacks need white-box access is untested, and Gemma plus Gemma Scope are fully public.
2. Stress-testing *persistent* SAE unlearning (CRISP, SSPU) with relearning, probing and quantization.
3. Using a gated SAE guardrail as a *teacher* for robust distillation.
4. Explaining the cross-lingual and verbatim failures by *which layer* the gate reads.

Re-run a novelty search the week before any submission. ARIA appeared only two weeks ago.

## 5. Recommended paper thesis

One claim, three parts: **gated SAE unlearning hides knowledge rather than erasing it; we show where it breaks, explain why with the SAE itself, and fix it for both deployment settings.**

**Part I — Break (stress test).** Attack DSG along five axes: dilution, question-splitting, cross-lingual and code-mixed prompts, black-box transfer red-teaming, and white-box obfuscation. Add generation-time leakage. Back the dilution result with a closed-form bound (Section 10).

**Part II — Explain (diagnosis).** Use SAE features and linear probes to tie each failure to a cause. Candidate causes to test: layer 3 is language-specific, so non-English prompts slip past; topic features are not the features that store exact wording, so verbatim memory survives; a −500 clamp is an abnormal activation, so membership attacks spot it; and the knowledge is still decodable downstream.

**Part III — Fix (two settings).**

- *API deployment:* a hardened gate. Streaming statistic, read at a semantic layer, calibrated so its output looks like ignorance rather than refusal.
- *Open weights:* baked erasure. DSG as a teacher for distillation into a noised student, and/or a closed-form transcoder-guided edit with a provable retain bound.

**Title options**

1. Hidden, Not Forgotten: Stress-Testing and Hardening Sparse Autoencoder Guardrails for LLM Unlearning
2. From Guardrail to Erasure: Baking SAE-Based Unlearning into Model Weights
3. Guardrails Are Not Erasure: A Five-Axis Audit of SAE Unlearning and a Provable Fix

**Contribution list, as it would appear in the introduction**

1. The first adaptive evaluation of SAE-based unlearning guardrails: five attack families plus a closed-form dilution bound. We show black-box transfer attacks work, contrary to DSG's stated assumption.
2. A mechanistic diagnosis linking each failure to a layer or feature property, verified with probes and ablations.
3. Two fixes: a hardened gate with a false-alarm guarantee, and a baked method with a retain-preservation bound and measured relearning resistance.
4. A unified evaluation across TOFU, MUSE and WMDP (multiple choice and open-ended), biology-adjacent utility, probing and tampering, on two model families, with released code.

**Built-in fallback.** Parts I and II alone make a solid workshop paper, even if Part III underdelivers. Parts I–III together, with the full evaluation, target a main conference or TMLR. This protects your grade whatever the method results turn out to be.

## 6. Improvements, part A — evaluation (your section)

Evaluation is the foundation: every later claim depends on it, and it is where DSG is thinnest. Build this first; it is also squarely your ownership area.

**A1 · One harness, one footing** — Owner: you · Priority: must

- *Now:* custom scripts on the 522-question WMDP-Bio subset. Baseline numbers come from papers with different protocols.
- *Change:* build on OpenUnlearning's TOFU, MUSE and WMDP metrics. Add an adapter so hook-based methods (DSG, your gates) run through the same pipeline via SAELens or TransformerLens hooks. Re-run every baseline on the same model, questions and prompt template. Report both the filtered 522 and the full 1,273.
- *Measure:* one results table where every row is on the same footing.

**A2 · Open-ended ("QA-type") evaluation** — Owner: you · Priority: must

- *Now:* multiple choice only.
- *Change:* three tracks. (i) TOFU forget01/05/10: ROUGE-L, Truth Ratio, Forget Quality. (ii) MUSE KnowMem, which is already QA. (iii) Open-ended WMDP: drop the options, ask the question, score with an LLM judge against the correct option's text, and check the judge on 100 hand-labelled items.
- *Key detail:* QA changes *when* the gate must fire — during generation, not just on the prompt. Check DSG's released code: if ρ(x) is computed once on the prompt, a benign-looking prompt whose answer drifts into the forget topic is a free finding.
- *Measure:* forget-set ROUGE / judge accuracy, and generation-time leakage rate.
- *Care:* keep hazardous generations private and report only aggregates.

**A3 · Hard negatives: the neighbours of the forget topic** — Owner: you · Priority: must

- *Now:* utility on history, CS, geography and human aging, plus MT-Bench.
- *Change:* for Bio, add MMLU college and high-school biology, virology, medical genetics, anatomy, college medicine. For Cyber, add computer security and college CS. Report the gate's false-positive rate on these, and accuracy on *triggered* benign questions only — the one place a gated method can hurt.
- *Measure:* false-positive rate on hard negatives; biology-adjacent accuracy change.

**A4 · Depth of forgetting via probing** — Owner: you · Priority: must

- *Now:* none. This is your "probing gap" observation: MUSE excludes probing, WMDP built the probe, DSG works in activation space.
- *Change:* train linear probes on the base model at layers 3, 6, 9, 12, 18 and 24 to predict the correct WMDP answer, as the WMDP paper did for RMU. Apply them to guarded and baked models. Add logit-lens readouts of the answer token.
- *Measure:* probe accuracy as a percentage of the base model's ("recoverable knowledge").

**A5 · Indistinguishability: does it look like ignorance?** — Owner: you · Priority: should

- *Now:* DSG fails MUSE PrivLeak on both corpora; you have not measured it.
- *Change:* report MUSE PrivLeak (±5 band), TOFU Forget Quality against a retain-only model, refusal rate, and a flag for below-chance accuracy. For TOFU on Gemma, fine-tune the target and the retain model yourselves; 4,000 QA pairs is a few GPU-hours.
- *Side check:* measure whether a pretrained SAE even reconstructs fine-tuned facts well (reconstruction error on TOFU text before and after fine-tuning). If it does not, briefly fine-tune the SAE and say so.
- *Measure:* PrivLeak, Forget Quality, refusal %.

**A6 · Threat models and tampering** — Owner: gradient teammate runs it, you define the protocol · Priority: must

- *Now:* only DSG's API setting where the defender keeps the hook.
- *Change:* name three threat models. T1: query-only API. T2: fine-tuning API, defender keeps the hook. T3: open weights. Under T3 attack every baked method with relearning on forget-adjacent text, relearning on unrelated benign text, 4-bit and 8-bit quantization, steering-direction recovery, and few-shot prompting.
- *Measure:* relearning curves (accuracy vs fine-tuning steps) and area under the recovery curve.

**A7 · Scale and generality** — Owner: shared · Priority: should

- *Now:* Gemma-2-2B, one 16k SAE, layer 3.
- *Change:* add Gemma-3-1B-it and Gemma-3-4B-it with Gemma Scope 2. Its SAEs are trained on the instruction-tuned models, which removes DSG's model–SAE mismatch, and it ships transcoders. Add Llama-3.1-8B with Llama Scope only if GPUs allow. Compare 16k vs 65k widths.
- *Measure:* does the method ranking hold across models?

**A8 · Reporting standards** — Owner: you · Priority: must

- *Now:* one run, no intervals, tuned on the test questions.
- *Change:* split WMDP into validation and test halves and tune only on validation. Run 3–5 seeds. Report bootstrap 95% intervals, McNemar tests for paired accuracy differences, full Pareto curves instead of single points, and compute cost (GPU-hours, latency per token).
- *Measure:* every table carries ± intervals.

## 7. Improvements, part B — stress tests and attacks ("Break")

This is your strongest and most defensible contribution, and it is where the current work already has a real result. The frame: DSG's own paper claims adaptive attacks need white-box access. You test that claim directly. Owner: the DSG-issues teammate leads; you own the evaluation protocol for every attack.

**B1 · Dilution, done properly** — Priority: must (you have this; harden it)

- *Now:* padding drives detection from 100% to about 2%; a CUSUM detector holds attack success near 36%.
- *Change:* define "attack success" precisely (forget-set accuracy recovered above the guarded baseline, on questions the guard originally caught). Report false-positive rate on benign padded prompts at every padding length, so 36% is read against its cost. Sweep padding position (before, after, interleaved) and padding topic (random vs plausibly related). Pair it with the theorem in Section 10.

**B2 · Question-splitting and decomposition** — Priority: should

- *New:* split one hazardous question into sub-questions, none of which alone crosses the trigger, then recombine answers. This attacks the prompt-averaged ρ(x) directly and needs no model access. A clean, cheap, novel attack.
- *Measure:* recovered accuracy vs number of splits.

**B3 · Cross-lingual and encoded prompts** — Priority: must (this is Gap 5, and now well-motivated)

- *New:* translate WMDP-Bio questions into several languages (including at least one non-Latin script), plus leetspeak, base64, and a simple cipher. Two effects to separate: does the *gate* still fire, and does the *model* still answer. Recent work shows unlearning lives in late, language-specific layers, so a layer-3 gate likely misses other languages — that is a mechanistic prediction you can confirm.
- *Measure:* trigger rate and recovered accuracy per language / encoding.
- *Care:* you are translating public MCQs, not generating new hazardous content. Keep it to the existing benchmark items.

**B4 · Black-box transfer red-teaming** — Priority: should

- *New:* use a separate LLM to rewrite forget questions to read as benign (the SSPU jailbreak styles: hypothetical, roleplay, instruction-override, narrative), with no access to DSG's SAE. If these transfer, you have directly refuted DSG's "adaptive attacks need white-box access" claim. That single sentence is a headline result.
- *Measure:* recovered accuracy under each rewrite style; transfer rate.

**B5 · White-box obfuscation (upper bound)** — Priority: nice-to-have

- *New:* adapt Bailey et al.'s released obfuscation code (they already break SAE monitors) to DSG specifically. This is the worst-case bound: an attacker with the public SAE optimises a suffix so ρ(x) stays low. Position as "even the strongest attacker", not the main threat.
- *Measure:* recovered accuracy at a fixed perturbation budget.

**B6 · Generation-time leakage** — Priority: must (ties to A2)

- *New:* many DSG-style triggers score the prompt once. In open-ended answering the model can walk into the forget topic mid-generation. Measure how often the answer leaks even when the prompt did not trigger. This is the concrete, measurable core of the "MCQ hides the real behaviour" argument and motivates the streaming gate in Section 8.
- *Measure:* fraction of generations that leak after a non-triggering prompt.

## 8. Improvements, part C — detector redesign (interpretability side, your build)

This is your "Build" track and your interpretability ownership. Each item is a design alternative to one specific weak point of DSG's gate, and each is a clean ablation even if it does not win.

**C1 · Causal feature selection** — Priority: must (you have C1; finish it)

- *Now:* attribution patching (activation × gradient) picks a substantially different feature set than DSG's activation-frequency ranking. That is an observation, not yet a result.
- *Change:* close the loop — show the causal set forgets more per unit of utility lost, or is harder to bypass under B1–B4. If it does not beat frequency ranking, that is still a publishable negative given DSG's Fisher-information framing rests on squared activation. Caveat from the literature: output-attribution localization can give *less* robust edits than mechanism-based localization, so test robustness, not just forget rate.

**C2 · Detector architecture: streaming and per-token** — Priority: must (you have the CUSUM seed)

- *Now:* CUSUM holds attack success near 36% under padding but starts weaker than a windowed gate at zero padding.
- *Change:* frame the prompt-average trigger, a sliding window, and CUSUM as three points on one curve (sensitivity vs robustness to dilution). Add a plain logistic probe over SAE features (ARIA-style) and, as the honest baseline, a logistic probe over *raw* activations. Kantamneni et al. show raw-activation probes usually match SAE probes — if the SAE gate does not beat the raw probe, the interpretability story is about auditability, not accuracy, and you should say so.
- *Measure:* one ROC-style plot, robustness (area under the attack-vs-padding curve) against clean-prompt false-positive rate.

**C3 · Layer and site sweep** — Priority: must

- *Now:* DSG uses layer 3 residual stream; SSPU also picked layer 3 but by a one-axis sweep. You have a multi-layer variant (C2 old) that beat DSG's forget metric at a utility cost.
- *Change:* sweep the gate's read site across depth and across residual / attention-out / MLP-out (all shipped by Gemma Scope 2). Tie it to B3: predict and then show that a semantic mid-layer gate catches cross-lingual and split attacks that layer 3 misses. This turns a hyperparameter sweep into a mechanistic claim.
- *Measure:* robustness and false-positive rate as a function of layer and site.

**C4 · Transcoders instead of SAEs** — Priority: nice-to-have (novelty flag)

- *New:* Gemma Scope 2 ships transcoders and cross-layer transcoders, which trace input-to-output computation rather than snapshotting one layer. No unlearning paper has used them for the gate. Even a small result ("transcoder features give a gate that is X") is genuinely new. Scope this as exploratory.

**C5 · Intervention that looks like ignorance** — Priority: should (ties to A5)

- *Now:* DSG clamps to −c (about −500), a value the paper itself calls behaviourally abnormal; A2 (mean-ablation) confirmed it. That abnormality is exactly what a membership attack detects (the PrivLeak failure).
- *Change:* replace the fixed clamp with a projection onto the "unknown-entity" or retain-mean direction (as Yamashita et al. do), so the guarded state resembles genuine not-knowing. Measure whether this fixes PrivLeak and lowers the membership-attack AUC while keeping forget rate.
- *Measure:* PrivLeak, membership-attack AUC, forget rate — before and after.

**C6 · Two-level gate: domain then hazard** — Priority: nice-to-have (you have B3-old)

- *Now:* your two-level gate reduced damage to knowledge next to the forget domain. Fold this into A3 as the mechanism that protects hard negatives, rather than a standalone experiment.

## 9. Improvements, part D — permanent forgetting in the weights ("Bake")

This is the highest-risk, highest-reward track and the one most exposed to CRISP. Your slide's B1 (LoRA distilled from the guarded model, hook removed, 40% accuracy) is a start, but as designed it is both beaten by CRISP and, per UNDO, not actually robust. Redesign before investing. Owner: gradient teammate leads the training; you own the evaluation and the proofs.

**The problem with B1 as it stands.** Two independent issues.

1. *Overlap:* CRISP already does "select salient SAE features, then LoRA-suppress them across layers, preserving benign contexts". A LoRA distilled from a guarded model to reproduce its behaviour is close enough that a reviewer will ask what is new.
2. *Not robust:* UNDO shows that distilling an unlearned teacher into *the same reference weights* leaves capabilities recoverable; only distilling into a *re-initialised or noised* student makes forgetting survive relearning. Your B1 keeps the base weights and adds a LoRA, so relearning (which you have not run) will likely revive the knowledge.

**D1 · Guarded-model distillation, done the robust way** — Priority: must (if you pursue Bake)

- *New framing:* the DSG-guarded model is a free, high-quality teacher that already refuses forget content while behaving normally elsewhere. Distil *that* into a noised copy of the student (UNDO's noise step), not into a clean LoRA. Novelty over UNDO: the teacher is an interpretable SAE gate, so you can report which features stop transferring and at what noise level. Novelty over CRISP: you measure relearning robustness, which CRISP never did.
- *Measure:* forget rate, retain, and the full A6 tampering suite, versus CRISP and versus plain unlearn-and-distil.

**D2 · Closed-form, transcoder-guided weight edit with a retain proof** — Priority: stretch (the strongest novelty, hardest to land)

- *New:* combine three existing ideas none of which have been combined. Use SAE or transcoder decoder directions to define the forget subspace (as SSPU and PISCES do). Then apply a *closed-form* edit to the relevant MLP output weights (as ROME/MEMIT/AlphaEdit do) that removes those directions, and project the edit onto the null space of retain-set keys (AlphaEdit) so retain outputs are provably unchanged. This gives an interpretable, training-free erase with a real preservation guarantee — the "proper proofs" you want.
- *Why it is defensible even if it underperforms:* it is the natural synthesis a reviewer will ask about, and a clean statement plus a negative result ("closed-form SAE editing does/doesn't beat LoRA") is a contribution.
- *Measure:* forget, retain, relearning, and edit locality, versus D1 and SSPU.

**D3 · Interpretability of the baked model** — Priority: should (keeps the interpretability thread through the whole paper)

- *New:* after baking, re-run the SAE over the edited model. Do the target features stop activating on forget content while surviving on benign content? This is the audit that a weight edit actually removed the concept rather than masking it, and it is what lets you claim "interpretable permanent unlearning", which neither CRISP nor UNDO frames as an audit.
- *Measure:* target-feature activation on forget vs benign, before and after; probe accuracy (A4) on the baked model.

**Honest scoping.** If the team is stretched, do D1 well and treat D2 as a framed future-work section with a small pilot. Parts I and II (Break and Explain) already carry a paper; Bake is the upside.

## 10. Theory: what you can honestly prove

Reviewers reward a small, correct theorem tied to an experiment far more than a grand one hand-waved. You asked for "proper proofs"; here is what is actually provable at your level, in rising order of difficulty.

**T1 · A dilution bound on the prompt-averaged trigger** — feasible, high value. DSG's statistic is ρ(x) = (1/T) Σ 1\[some selected feature fires at token t\]. If a hazardous core of k tokens triggers and you pad with m benign non-triggering tokens, then ρ ≤ k/(k+m). So for any fixed threshold τ, once m > k(1−τ)/τ the prompt is classified retain and the attack succeeds. This is a two-line proof that turns your empirical dilution curve into a guarantee, and it directly exposes the gap in DSG's Theorem 3.4: Neyman–Pearson optimality assumes ρ is stochastically larger on forget inputs, which dilution violates by construction.

```latex
\rho(x) \;=\; \frac{1}{T}\sum_{t=1}^{T}\mathbf{1}\!\left[\exists\, j\in S:\; f_j(h_t)>0\right]
\;\le\; \frac{k}{k+m}
\;\xrightarrow[m\to\infty]{}\; 0
```

**T2 · A streaming detector guarantee** — feasible. For a windowed or CUSUM detector, show that detection depends on the local density of trigger tokens, not the global average, so it is invariant to benign padding of any length (formally: attack success is bounded independent of m). This is the theory companion to your CUSUM result and explains why it flatlines at 36% instead of decaying.

**T3 · A false-positive / detection trade-off** — feasible, borrow from ARIA. ARIA already proves forget leakage ≤ (false-negative rate) and retain degradation ≤ (false-positive rate) for a detector-gated policy. Do not re-prove it; cite it, and use it as the frame for your ROC plots. Your contribution is empirical: showing where DSG's fixed-threshold gate sits on that trade-off and how far your hardened gate moves it.

**T4 · A retain-preservation guarantee for the baked edit** — feasible only if you do D2. If the weight edit is projected onto the null space of the retain-set key matrix, then retain outputs are unchanged to first order — this is AlphaEdit's theorem, which you would be instantiating for SAE/transcoder-defined forget directions. State it as "we inherit AlphaEdit's guarantee", do not claim it as new.

**What not to claim.** Do not claim complete erasure, an information-theoretic bound on residual knowledge, or that clamping provably removes a concept. DSG's own Fisher-information theorems already overreach (they rest on a near-deterministic-output assumption stated in the appendix); a reviewer who knows that will be primed to punish the same move from you. Prove the small things exactly.

## 11. What to drop or deprioritize

Saying no is part of a publishable plan. A tight paper with one clear claim beats a broad one with many weak ones. Here is what to stop doing or shrink, and why.

- **Pushing WMDP multiple-choice accuracy below \~25%.** It is already at the four-option floor. Lower is not more forgetting; it is noise, and it cannot tell "hidden" from "erased". You said this yourself, and you are right. Report the number once and move on.
- **The "nine interventions" framing.** Keep the experiments; drop the flat list. Each one becomes an attack (Part B), a detector variant (Part C), or an ablation, all under one thesis.
- **Training or growing the SAE (old B2, dictionary extension).** Building a new or larger SAE is a separate, GPU-heavy project and overlaps CRISP. Use the released Gemma Scope 2 SAEs and transcoders instead.
- **Full residual-stream projection at every layer (old C3).** It hurts utility broadly and is hard to interpret. The targeted layer-and-site sweep (new C3) gives the same insight at far lower cost.
- **Base64 and heavy ciphers as a headline attack.** A 2B model often cannot answer an encoded question even before unlearning, so a low recovery rate proves nothing about the gate. If you keep them, normalise recovered accuracy by the base model's accuracy on the same encoded item, and treat them as a small robustness check. Cross-lingual and leetspeak prompts (B3) are the variants that actually separate "gate missed it" from "model can't read it".
- **A brand-new theorem about complete erasure.** Section 10 lists what is provable. A small correct bound beats a large shaky one, and over-claiming invites the same criticism you are aiming at DSG.
- **MT-Bench as evidence about your gate.** It cannot see a conditional method's damage (Section 3). Keep it only as a fluency sanity check, never as a headline metric.

One rule of thumb: if an experiment does not either break DSG, explain a break, or fix one, it is probably a distraction this cycle.

## 12. Master experiment matrix

Every experiment in one place, with its owner and priority. Owners: **E** = you (evaluation and interpretability), **A** = the DSG-issues teammate (attacks), **P** = the gradient teammate (permanence). Priority: **must** = paper does not stand without it; **should** = strengthens it; **stretch** = upside if time allows. Do the must rows first; they are the workshop-paper core.

| ID | Experiment | Owner | Priority | Primary metric |
| --- | --- | --- | --- | --- |
| A1 | One evaluation harness on OpenUnlearning (TOFU + MUSE + WMDP) | E | must | one comparable results table |
| A2 | Open-ended / QA evaluation (TOFU, MUSE KnowMem, open-WMDP) | E | must | forget ROUGE / judge accuracy |
| A3 | Hard-negative biology-adjacent utility | E | must | false-positive rate on neighbours |
| A4 | Depth of forgetting via linear probes + logit lens | E | must | probe accuracy vs base model |
| A5 | Indistinguishability (PrivLeak, Forget Quality, refusal rate) | E | should | PrivLeak within ±5; FQ |
| A6 | Three threat models + tampering suite (protocol) | E defines, P runs | must | relearning curve, recovery AUC |
| A7 | Second model family (Gemma Scope 2; Gemma-3-it) | shared | should | does ranking hold? |
| A8 | Reporting standards (val/test split, seeds, intervals) | E | must | ± intervals on every table |
| B1 | Dilution, with defined attack success + false-positive cost | A | must | recovered accuracy vs padding |
| B2 | Question-splitting / decomposition | A | should | recovered accuracy vs splits |
| B3 | Cross-lingual and encoded prompts | A | must | trigger rate, recovered acc / language |
| B4 | Black-box transfer red-teaming | A | should | transfer rate by rewrite style |
| B5 | White-box obfuscation (upper bound) | A | nice | recovered acc at fixed budget |
| B6 | Generation-time leakage | A | must | leak rate after non-triggering prompt |
| C1 | Causal feature selection vs frequency ranking | E | must | forget per utility lost; robustness |
| C2 | Streaming/per-token detector + linear-probe baseline | E | must | ROC: robustness vs false positives |
| C3 | Layer and read-site sweep | E | must | robustness / FPR by layer & site |
| C4 | Transcoder-based gate | E | nice | gate quality (exploratory) |
| C5 | Ignorance-like intervention (retain-mean projection) | E | should | PrivLeak, membership AUC, forget |
| C6 | Two-level gate (domain then hazard) | E | nice | folded into A3 |
| D1 | Guarded-teacher distillation into a noised student | P | must\* | forget/retain + tamper suite |
| D2 | Closed-form transcoder edit with AlphaEdit retain proof | P + E | stretch | forget/retain/relearn/locality |
| D3 | Interpretability audit of the baked model | E | should | target-feature activation before/after |
| T1 | Dilution bound on the prompt-averaged trigger | E | must | proof + matches empirical curve |
| T2 | Streaming-detector padding-invariance guarantee | E | should | proof + matches CUSUM flatline |
| T3 | Leakage/degradation trade-off frame (cite ARIA) | E | should | ROC framing |
| T4 | Retain-preservation bound (inherit AlphaEdit) | E | stretch | proof, only if D2 runs |

\*D1 is "must" only if the team pursues the Bake track; if not, the paper still stands on Parts I and II (Sections 7, 8, 10).

**Critical path.** A1 unblocks everything, because every other row reports through it. Do A1, then A8, then the must-priority attacks (B1, B3, B6) and detector work (C1, C2, C3) in parallel, then the depth metrics (A4, A5). Bake (D1, D2) runs last and in parallel, on its own risk budget.

## 13. How to split the work across three people

The split follows the ownership you already have, so no one restarts from scratch. Each person owns one leg of the thesis, and evaluation (yours) is the shared spine everything reports through.

**You (evaluation + interpretability).** You own the harness (A1, A8), the depth-of-forgetting metrics (A2, A4, A5), hard negatives (A3), the detector redesign (all of C), the interpretability audit (D3), and the proofs (T1–T4). You also define the tampering protocol (A6) even though a teammate runs it, because it has to line up with the rest of the evaluation. This is the largest share, which fits: you own the spine.

**DSG-issues teammate (attacks / "Break").** Owns the full stress suite (all of B): dilution done properly, question-splitting, cross-lingual and encoded prompts, transfer red-teaming, obfuscation, generation-time leakage. Runs each attack through your harness so results stay comparable. This person's headline is the single sentence that refutes DSG's white-box-only claim (B4).

**Gradient teammate (permanence / "Bake").** Owns the training side of D: the robust distillation (D1) and, as a stretch, the closed-form edit (D2). Also runs the tampering suite (A6) against every baked model, since relearning and quantization attacks are where gradient-method side-effects show up. Pairs with you on the D2 proof (T4).

**Shared.** The second model family (A7) and the final write-up are shared. One person should own the released code repository end to end so it does not fragment.

### A phased timeline

Adjust the week counts to your actual calendar; the ordering is what matters.

| Phase | Rough window | Goal | Key rows |
| --- | --- | --- | --- |
| 0 · Foundation | Weeks 1–3 | Harness runs; DSG reproduced inside it; validation/test split fixed | A1, A8 |
| 1 · Break | Weeks 3–7 | Attack suite complete and comparable; dilution theorem written | B1–B6, T1, T2 |
| 2 · Explain | Weeks 6–10 | Each failure tied to a layer/feature; detector variants benchmarked | A4, C1–C3, C5, T3 |
| 3 · Depth | Weeks 8–11 | Probing, PrivLeak, Forget Quality, hard negatives, tampering | A2, A3, A5, A6 |
| 4 · Bake | Weeks 9–14 | D1 trained and stress-tested; D2 pilot; audit | D1, D3, (D2, T4) |
| 5 · Scale + write | Weeks 12–16 | Second model; all tables with intervals; paper drafted | A7, write-up |

Phases overlap on purpose. The moment Phase 0 is done, Break and Explain can run in parallel across two people while you keep extending the harness. A workshop submission is viable after Phase 2; a conference or TMLR submission needs Phases 3–5 too. Freeze results and re-run a novelty search in the last week before any deadline.

## 14. Venue strategy

Aim for two submissions from one body of work: a workshop paper first, then a fuller paper. The workshop version protects your grade; the full version is the upside.

**Tier 1 — Workshop paper (Parts I and II).** Viable after Phase 2 of the timeline. Content: the five-axis stress test, the dilution bound, and the SAE-based diagnosis of each failure. Mechanistic interpretability and unlearning workshops at the major ML conferences fit well. Mech Interp Workshop recurs at ICML, with deadlines usually in May. Check current dates on the workshop websites, since they change every year.

**Tier 2 — Full paper (Parts I to III).** Needs Phases 3 to 5: depth metrics, at least one baked method with tampering results, and a second model family. Targets are a main-track ML or NLP conference, or TMLR, which has rolling submissions and no deadline pressure. TMLR suits a thorough, evidence-heavy paper like this one.

**Tier 3 — Capstone report and viva.** Independent of any venue. The master matrix (Section 12) and the reviewer Q&A (Section 15) are written to double as your defence material.

**Practical rules**

- Put the preprint on arXiv when the workshop paper is ready. This timestamps your ideas, which matters because ARIA appeared only two weeks before this report.
- Re-run the novelty search (Section 4) one week before every submission. Search CRISP, ARIA, SSPU and "SAE unlearning" on arXiv and Google Scholar.
- Check each workshop's policy on archival versus non-archival. A non-archival workshop paper can still become a full paper later.
- Release code and configs. Reviewers reward reproducibility, and it is cheap once the harness (A1) exists.
- Discuss author order and contributions with your guide early. It avoids friction later.

**Decision point.** At the end of Phase 2, look at the results. If the attacks and diagnosis are strong, submit the workshop paper and continue. If Part III also looks strong by Phase 4, merge everything into the full paper.

## 15. Reviewer Q&A simulation

These are the questions a reviewer or viva panel is most likely to ask, with short answers you can defend once the experiments exist. Fill in the numbers from your own results.

**Q1. DSG already reports MUSE and MT-Bench. What exactly is new?** DSG tested friendly conditions: plain English, one model, a defender who keeps the hook. We test five attack families, open-ended answers, probing and open weights, and we explain each failure with SAE features.

**Q2. Is your dilution attack just a trivial trick?** It is simple, but we prove it (T1) and show it breaks the statistic behind DSG's own optimality theorem. A simple attack that defeats a top-venue method is a valid finding.

**Q3. Why should we trust gains of 1 to 2 points on 522 questions?** You should not, and we do not claim them. We report 95% bootstrap intervals, 3 to 5 seeds, McNemar tests, and tune only on a validation split.

**Q4. CRISP already makes SAE unlearning permanent. How is your Bake track different?** CRISP never tested relearning or adversarial extraction. We distil into a noised student (per UNDO), and we test it under the full tampering suite against CRISP as a baseline.

**Q5. Does an SAE detector beat a plain linear probe?** We test this directly (C2). Prior work suggests raw probes often match SAE probes. If ours does not beat the raw probe, our claim is auditability and explanation, not accuracy.

**Q6. Low WMDP accuracy only shows the model refuses. Is knowledge erased?** Agreed, and that is our thesis. We measure depth with linear probes, logit lens, relearning curves, quantization and steering recovery, not MCQ accuracy alone.

**Q7. Is this specific to Gemma-2-2B?** We add Gemma-3 models using Gemma Scope 2, whose SAEs are trained on the instruction-tuned models. We report whether the method ranking holds across models.

**Q8. Are the attacks realistic?** B1 to B4 need no model internals, only text queries. B5 is white-box and is framed as a worst-case bound, not the main threat.

**Q9. Is publishing these attacks a safety risk?** We use existing public benchmark items, report aggregates only, and do not release hazardous generations. Attacks target the guardrail, not new hazardous content.

**Q10. Does your theory claim complete erasure?** No. We prove only a dilution bound and a padding-invariance property, and inherit AlphaEdit's retain guarantee if D2 runs. We state this limit explicitly.

## 16. Quick fixes to the current slides and results

These are cheap corrections that remove easy reviewer objections. Do them before any further experiments. Owner: whoever owns each slide, reviewed by you.

- **Bar chart contradiction.** The slide shows a B1 bar at 27.7 while the text says C2 is the only variant beating DSG's forget metric. Decide which is correct and relabel.
- **Illogical sentence.** "40% though above 45.5%" cannot be true. Rewrite it so the comparison direction is clear (40% is below the 45.5% hook-active score).
- **Mislabelled baseline.** "MMLU 99.34 vs RMU 50" attaches a number to RMU that belongs to a WMDP result. Re-check the source table and label each number by metric.
- **Undefined "attack success".** Define it in one line (forget-set accuracy recovered above the guarded baseline, on questions the guard originally caught) and use it everywhere.
- **No error bars.** Add 95% intervals or at least standard errors. On 522 questions, one standard error is about 2 points, so 29.37 vs 27.7 is not a significant difference.
- **CUSUM needs its cost.** Show the false-positive rate next to the 36% attack-success figure, and explain why it starts worse than the windowed gate at zero padding.
- **Stop tuning on the test set.** Split WMDP into validation and test halves and choose configurations on validation only.
- **Reword the "nine interventions" slide.** Group results as attacks, detector variants and permanence experiments under one thesis.
- **One consistent set of numbers.** Put all final results in one table generated by one script, and have every slide read from it.

## 17. Glossary

Plain-language definitions of every term used in this report.

- **LLM (large language model).** A neural network trained to predict the next token of text.
- **Machine unlearning.** Making a model behave as if it never learned some data or topic, without retraining from scratch.
- **Residual stream.** The running vector that each transformer layer reads from and adds to. SAEs usually read it.
- **SAE (sparse autoencoder).** A small network that rewrites a layer's activations as a long list of mostly-zero features, which are easier to interpret.
- **Feature.** One entry in an SAE's list. Ideally it tracks one human-readable idea, like "virus biology".
- **Transcoder.** Like an SAE, but it predicts a layer's output from its input, so it traces computation instead of a snapshot.
- **Gemma Scope / Gemma Scope 2.** Public SAE (and transcoder) sets for Gemma models.
- **DSG (Dynamic SAE Guardrails).** The baseline method. It picks SAE features tied to the forget topic and clamps them when they fire on an input.
- **Clamp / gate.** Forcing a feature to a fixed value (clamp), triggered only when a detector fires (gate).
- **ρ(x) (rho).** DSG's trigger score: the fraction of tokens in a prompt where a selected feature is active.
- **RMU.** A weight-based unlearning method that pushes forget-topic activations toward random noise while keeping other activations stable.
- **SSPU.** An RMU-style method that restricts the weight update to an SAE-defined subspace.
- **CRISP.** A method that suppresses important SAE features using LoRA across several layers.
- **ARIA.** A method that gates generation with a probe over SAE statistics and answers "I don't know".
- **UNDO.** Unlearn, then distil into a noised copy of the model, so forgetting survives relearning.
- **LoRA.** Low-rank adapters: a small set of extra trainable weights added to a frozen model.
- **Distillation.** Training a student model to copy a teacher model's outputs.
- **Null-space edit (AlphaEdit).** A weight edit constrained so it does not change outputs on a chosen retain set.
- **WMDP.** A benchmark of multiple-choice questions probing hazardous knowledge (biology, cyber, chemistry).
- **MMLU.** A general-knowledge multiple-choice benchmark, used to check the model still works.
- **MT-Bench.** A conversation-quality benchmark scored by an LLM judge.
- **TOFU.** A benchmark of fictitious author profiles for testing unlearning of specific facts, with QA-style scoring.
- **MUSE.** A benchmark that uses news and books to test unlearning across several criteria.
- **VerbMem.** MUSE's check for verbatim memorisation: can the model recite the exact text?
- **KnowMem.** MUSE's check for knowledge memorisation: can the model answer questions about the content?
- **PrivLeak.** MUSE's privacy metric. A membership attack tries to tell if the model was trained on the forget data. Values near zero are good, and the accepted band is about ±5.
- **Forget Quality (TOFU).** A statistical test of how close the unlearned model is to a model that never saw the data.
- **Truth Ratio, ROUGE-L.** TOFU answer-quality scores: how the model rates true vs false answers, and word overlap with the reference answer.
- **Membership inference attack.** An attack that guesses whether specific data was in training or was unlearned.
- **Relearning attack.** Fine-tuning an unlearned model briefly to see if the forgotten knowledge returns quickly.
- **Tampering attack.** Any weight-level attack on an unlearned model, such as relearning or quantization.
- **Quantization.** Storing weights in fewer bits (for example 4-bit). It can sometimes bring back suppressed knowledge.
- **Linear probe.** A simple classifier trained on internal activations to test whether some information is still decodable.
- **Logit lens.** Reading a layer's activations as if they were the final output, to see what the model is "about to say".
- **Attribution patching.** A fast way to estimate how much each feature matters to an output, using activation times gradient.
- **Fisher information.** A measure of how sensitive a model's output is to a parameter or feature. DSG uses it in its theory.
- **Neyman–Pearson.** Statistical result on the best possible detector for a given false-alarm rate. DSG's Theorem 3.4 builds on it.
- **CUSUM.** A streaming change detector that sums evidence over time, so it reacts to a local burst and ignores long stretches of padding.
- **Dilution attack.** Padding a hazardous prompt with harmless text so the prompt-average trigger score falls below threshold.
- **False-positive rate (FPR).** How often the gate fires on harmless inputs. A high FPR harms normal users.
- **False-negative rate (FNR).** How often the gate misses hazardous inputs.
- **ROC / Pareto curve.** Plots showing the trade-off between two goals, such as forgetting and utility.
- **Bootstrap interval, McNemar test.** Statistics tools: a range of plausible values from resampling, and a significance test for paired accuracy differences.
- **OpenUnlearning.** A shared code framework that runs many unlearning methods and metrics on TOFU, MUSE and WMDP.

## 18. References and links

**Your project papers**

- Muhamed et al., Dynamic SAE Guardrails (DSG), ICML 2025. The baseline.
- Farrell et al., SAEs and unlearning (project file p1).
- SSPU, SAE-subspace-constrained unlearning, ICML 2025 workshop: https://arxiv.org/abs/2505.24428
- Li et al., WMDP benchmark and RMU (project file wmdp).
- Maini et al., TOFU (project file tofu).
- Shi et al., MUSE (project file muse).
- SAEBench (project file saebench).

**Competing and related work (2025 to 2026)**

- CRISP, ACL 2026: https://aclanthology.org/2026.acl-long.82/ (arXiv 2508.13650)
- ARIA, arXiv, Sep 2026: https://arxiv.org/abs/2609.16229
- PISCES: https://arxiv.org/abs/2505.22586
- Yamashita et al., unknown-entity steering: https://arxiv.org/abs/2509.15631
- UNDO, NeurIPS 2025: https://arxiv.org/abs/2506.06278
- Guo et al., Mechanistic Unlearning, ICML 2025: https://proceedings.mlr.press/v267/guo25k.html
- Bailey et al., Obfuscated Activations, ICLR 2026: https://arxiv.org/abs/2412.09565
- Kantamneni et al., SAE probing, ICML 2025: https://proceedings.mlr.press/v267/kantamneni25a.html
- Xiang et al., Multilingual unlearning, ICML 2026: https://arxiv.org/abs/2606.03291
- OpenUnlearning, NeurIPS 2025: https://arxiv.org/abs/2506.12618
- GROM: https://arxiv.org/abs/2608.05783
- ZeroUnlearn: https://arxiv.org/abs/2605.18879
- AlphaEdit (null-space model editing), for the retain guarantee in T4.
- Gemma Scope 2 (Jan 2026): SAEs and transcoders for Gemma 3.

**Note.** Links were gathered on 30 Sep 2026. Open each one and confirm title, authors and venue before citing it in your paper.

## 19. New additions (30 Sep update): more ideas, nothing dropped

**Note on this update.** Sections 1 to 18 are unchanged. This section only adds. Your decision is that nothing is dropped, so every item in Section 11 moves to a "backlog" and can be picked up when time allows. The ideas below make the paper bigger and more useful to outsiders. Owners: E = you, A = DSG-issues teammate, P = gradient teammate. Priority: must, should, stretch.

**Hard rule.** Whatever is not finished by the Oct 8 freeze moves to "later work" and is reported as future work, never as a result.

**N1 · Public attack suite and audit harness ("GuardBreak")** — Owner: A + E · Priority: must

- *New:* package Part I (dilution, splitting, cross-lingual, rewrite attacks) as one pip-installable toolkit with config files, so anyone can run it on their own SAE guardrail. Release scripts that transform public benchmark items, not new hazardous text.
- *Why it helps:* reviewers and researchers can reproduce and cite it. A reusable tool gets feedback far more easily than a PDF.

**N2 · OpenUnlearning adapter and pull request** — Owner: E · Priority: should

- *New:* contribute the hook-based adapter from A1 to the OpenUnlearning codebase, or at least open a pull request or issue with a working example. Even an open PR is outside visibility.

**N3 · Interactive demo of the gate** — Owner: E · Priority: should

- *New:* a web demo (Hugging Face Space) where a visitor pastes a prompt and sees the per-token trigger score, which features fire, and what dilution does to the score. It makes the abstract finding visible in 30 seconds.

**N4 · Public red-team challenge on a safe proxy topic** — Owner: A · Priority: should

- *New:* guard a harmless fictional topic (for example an invented fictional domain, not a hazard) and let classmates, other colleges and online researchers try to get the model to reveal it. Record which attack styles win. This gives human-vs-automated attack data, plus real outside participation.
- *Care:* use only a proxy topic. Never run the challenge on real hazardous knowledge.

**N5 · Adaptive hardening loop (attacker vs defender)** — Owner: A + E · Priority: should

- *New:* let an LLM attacker rewrite prompts, retrain or recalibrate the gate on the attacks it finds, then attack again for several rounds. Report whether robustness improves or whether the attacker always finds a new hole. Either answer is a finding.

**N6 · Conformal threshold with a false-positive guarantee** — Owner: E · Priority: must (theory)

- *New:* set the gate threshold using conformal calibration on held-out benign prompts, which gives a distribution-free guarantee on the false-positive rate under exchangeability. This replaces DSG's hand-picked threshold with a provable one and extends T3.
- *Scope:* the guarantee covers benign prompts from the calibration distribution only. Say so.

**N7 · Multi-layer and multi-gate voting** — Owner: E · Priority: should

- *New:* combine gates at an early, middle and late layer and fire when any or most agree. Test against every attack in Part I. This extends C3 from a sweep into a design.

**N8 · Reasoning-trace gating** — Owner: E + A · Priority: stretch

- *New:* test whether the gate fires on the model's chain-of-thought tokens, and whether hazardous facts can leak in reasoning text even when the final answer is gated. ARIA evaluated a reasoning variant of TOFU, so cite it and compare.

**N9 · SAE-quality explanation of failures** — Owner: E · Priority: must (Part II)

- *New:* use the SAEBench metrics in your project files to ask whether SAE quality on hazardous text (reconstruction, sparsity, feature splitting and absorption) predicts which concepts the gate misses. Feature absorption is a known SAE failure mode, so it is a testable cause of leaks.

**N10 · Unlearning Audit Card** — Owner: E · Priority: should

- *New:* a one-page reporting template for any unlearning method. It lists the threat model, benchmarks, attack families, seeds, intervals, utility neighbours and compute cost. You fill one in for DSG and for your gate. A checklist that other people can reuse is a real contribution and a good discussion point.

**T5 · Detection-delay optimality for CUSUM** — Owner: E · Priority: should (theory)

- *New:* the classic result for CUSUM (Lorden, Moustakides) says it minimises worst-case expected detection delay for a fixed false-alarm rate in the standard setting. Instantiate it for the trigger stream to give T2 a solid reference. State its assumptions (independent, known distributions) and check them empirically, because real token streams violate them.

**Extra models and domains (add to A7, no new section needed)**

- Add a multi-topic run: guard several forget topics at once and measure gate interference. DSG tested four sequential MUSE steps, so this extends it.

**Updated must-list for the Oct 8 freeze.** A1, A8, B1, B3, B6, T1, N6, N9 (plus A4 and C2 if time allows). Everything else is best effort and reported honestly as done, partial, or later work.

## 20. After Oct 15: taking the work to the outside world

The goal is external attention and real feedback, not only an acceptance. Every step below is something you can honestly report at the end-of-November review. None of the outcomes is guaranteed, so report what actually happened.

**Step 0 · Courtesy before publishing attacks (Oct 8–15).** Email the DSG authors a short heads-up and summary before posting an attack that breaks their method. It is the normal research courtesy, it often gets a reply, and it protects you. Publish without waiting for permission, but give them a few days.

**Step 1 · Make the artifacts (Oct 10–20)**

- arXiv preprint (cs.CL or cs.LG). First-time authors usually need an endorser: ask your guide early. Moderation can take a few days.
- GitHub repo with README, one-command reproduction, license and results tables.
- Hugging Face Space (N3) and a short page of results.
- A plain-language blog post and a 2-minute demo video.
- One-page poster and a short slide deck for talks.

**Step 2 · Share where researchers actually look (Oct 20–31)**

- Blog: the Alignment Forum or LessWrong (strong interpretability audience) and a Hugging Face community article.
- Social: a short thread on X, Bluesky or LinkedIn that tags the authors of DSG, CRISP, SSPU, UNDO and the OpenUnlearning maintainers. Ask one specific question, not "please review".
- Communities: the open-source mechanistic interpretability Slack, EleutherAI and Hugging Face Discords, and the machine learning subreddit (check each community's rules and tag posts correctly).
- Hugging Face Papers: once the arXiv paper is live, add your code and demo to its page.

**Step 3 · Ask experts directly (Oct 20–Nov 10)**

- Send 10 to 15 short emails or messages to PhD students and researchers who wrote the closest papers. Offer a 15-minute call. Use the template in Section 21.
- Ask your guide for introductions to people he knows, and ask your department for alumni working in ML.
- Ask for one thing each: a critique of one figure, a missed baseline, or a pointer to a venue.

**Step 4 · Talks and seminars (Nov)**

- Offer a 20-minute talk at reading groups (university, AI-safety and interpretability groups), the department seminar, and student chapters.
- Record the talk. Questions you receive there are evidence of engagement.

**Step 5 · Open-source contributions (Oct–Nov)**

- Open the OpenUnlearning pull request (N2).
- Report issues or improvements you found to SAELens or TransformerLens if they apply.

**Step 6 · Programs and shared tasks**

- Apart Research runs sprints and a Lab Fellowship that turns prototypes into publishable research, per its own pages. Check its events page for an interpretability or unlearning sprint.
- Look for unlearning shared tasks. SemEval ran an LLM unlearning task in 2025, so check whether a 2027 edition or similar shared tasks exist. This is the same route your friends took.
- Other research programs (for example MATS or SPAR) exist. I did not verify their eligibility or dates, so check before relying on them.

**Step 7 · Formal submissions (from earlier sections)**

- EACL SRW pre-submission mentorship draft by Nov 6, then full submission on Dec 15 (Section 14).
- EACL 2027 workshops, ICLR 2027 workshops, and ICML 2027 Mech Interp workshop as deadlines appear. Re-check each date.

**Step 8 · Keep going after November**

- Turn every piece of feedback into a change log and a "v2" of the preprint.
- Extend the work with the N-items you did not finish and aim the fuller paper at TMLR or a main venue.

**Safety and ethics for outreach**

- Release transformation scripts and aggregate numbers, not hazardous generations or new hazardous prompts.
- Do not claim acceptance, endorsement or results you do not have.
- Credit everyone whose work you built on, and be polite when you disagree with a published claim.

## 21. Outreach templates and a progress scoreboard

**Email template to a researcher (keep it under 120 words)**

> Subject: Quick question about your SAE unlearning work
>
> Hi Dr. \[Name\], I'm a final-year undergraduate at IIIT DM Kurnool working with Dr. M. Naresh Babu. We stress-tested dynamic SAE guardrails (DSG) and found \[one sentence result, for example that padding drops detection from 100% to about 2%\]. Our preprint and code are here: \[links\]. Could you spare two minutes to tell us whether \[one specific question\] matches your experience, or point us to a baseline we missed? Thank you for your time.
>
> &#91;Names, institute, links\]

**Short social post template**

> We tested whether SAE-based unlearning guardrails actually erase knowledge. Short answer: they hide it. \[One result with a number.\] Preprint, code and a live demo: \[links\]. We'd love critique from people working on unlearning and interpretability.

**Heads-up email to the DSG authors (Step 0)**

> We reproduced DSG within noise and found some failure cases we plan to post on \[date\]. We'd appreciate any comments before then, and we are happy to share the details and code now.

**Progress scoreboard for the end-of-November review**

Keep one table and update it weekly. Report only what is true on the day.

| Milestone | Status on the day | Evidence to show |
| --- | --- | --- |
| arXiv preprint live | yes / pending | link and date |
| Code released | yes / pending | repo link, stars, forks |
| Live demo | yes / pending | Space link |
| Blog post published | yes / pending | link, views |
| Emails sent to researchers | number sent | sent list |
| Replies or feedback received | number and summary | quotes (with permission) |
| Talks or seminars given | number | date, audience |
| Open-source PR or issue | opened / merged | link |
| SRW mentorship draft submitted | yes / pending | receipt |
| Formal submissions | venue and date | receipt |
| Changes made because of feedback | number | change log |

**How to describe this at the review.** Lead with what exists (paper, code, demo), then outreach actions, then the responses, then the next venue with its date. "We have a public preprint, N people tried the demo, and we are submitting to X on date Y" is honest and strong.

## 22. New additions 2 (2 Oct): full benchmark coverage

**Note on this update.** Sections 1 to 21 are unchanged. This adds benchmark runs for the baseline AND for the final method, so the paper can show results on every standard benchmark. Report every benchmark honestly, including the ones where you do not win.

**Where things stand now**

| Benchmark | Baseline DSG | Final method | Where it runs | Status |
| --- | --- | --- | --- | --- |
| WMDP-Bio / Cyber + MMLU | A1 | best C gate and D models | lab PC | running in the queue |
| TOFU (forget quality, model utility, truth ratio, ROUGE) | A2 (LoRA fine-tune) | add the best gate and D2 | lab PC; full fine-tune on GPU server | queued; full fine-tune deferred |
| MUSE News and Books (VerbMem, KnowMem, PrivLeak) | new | add the best gate | GPU server (needs target models) | **new: add to the server job list** |
| MT-Bench (general chat quality) | new | add the best gate | GPU server | **new: replace the current judge** |

**BM1 · MUSE re-run** — Owner: E · Priority: must

- Fine-tune gemma-2-2b on MUSE News and Books to create target models, as the DSG paper did. Run DSG and your best fix, using MUSE's official metric code.
- Why: DSG fails PrivLeak on both MUSE corpora. Showing whether your fix repairs this is a strong quantitative point.

**BM2 · TOFU at full scale** — Owner: E · Priority: must

- Keep the lab PC LoRA version. Add a full fine-tune version on the server so numbers are comparable with other TOFU papers.
- The TOFU retain-only model is a "never learned it" reference. It is reused in the qualitative track (Q3).

**BM3 · MT-Bench with a reproducible judge** — Owner: A · Priority: should

- **Problem with the current setup:** a Claude Code session acting as the judge is not reproducible. It is not a fixed model with a fixed prompt, and it changes between sessions. Reviewers would reject it as the main evidence. Keep the current numbers as an internal sanity check only.
- **Fix without an API key:** run an open judge model on the GPU server with a fixed prompt and fixed settings. Options include an open evaluator model trained to judge (for example the Prometheus-2 family) or a strong open instruct model that fits in 48 GB. Report the judge used, and compare methods relative to each other rather than to the paper's GPT-4 numbers.
- **Caveat:** MT-Bench prompts rarely trigger the gate, so it measures little for DSG. Report it, but rely on targeted utility (A3, full MMLU) for the main claims.

**Rule for the paper.** Every benchmark is run for: base model, DSG, the unverified third-party RMU and your own RMU (reference), the best C gate and the best D model. One table generator produces all of it.

## 23. New additions 2: qualitative interpretability track (making unlearning visibly real)

Numbers show *that* something changed. This track shows *what* changed inside the model and what the model now does. It makes the "hiding vs erasing" thesis something a reader can see. Competitors mostly stop at numbers: CRISP reports feature suppression, ARIA shows "I don't know" outputs, and Guo et al. localise facts. Combining causal pictures, a "never learned" reference and a small human study is the gap.

**Safety rule for this whole track.** Public figures and human annotation use the fictional TOFU domain only. Any analysis on hazardous text stays automatic and aggregate, and nobody reads raw hazardous generations.

**Q1 · Feature cards** — Owner: E · Priority: must · Lab PC

- For each selected DSG feature: what it responds to (Neuronpedia explanation for Gemma Scope features, fetched on the lab PC), how often it fires on forget, benign-bio and general text, and which tokens trigger it. Use redacted or TOFU examples in figures.
- Add a causal check: switch the feature on in a neutral prompt and see whether the output drifts toward the topic.

**Q2 · Attribution graphs before and after** — Owner: E · Priority: should · GPU server

- Use open circuit-tracing tools with transcoders (check that transcoders for gemma-2-2b are available; otherwise use Gemma 3 with Gemma Scope 2) to draw the path from question to answer.
- Show three pictures for the same TOFU-style fact: base model, DSG (the path still exists but is clamped at layer 3), and a baked model (D2), where the path should be gone.
- Add one picture for a successful attack, for example a cross-lingual prompt that routes around the gated features. This is the clearest possible visual of "hidden, not erased".

**Q3 · "Never learned" reference comparison** — Owner: E · Priority: must · Lab PC

- The ideal unlearned model behaves like a model that never saw the data. TOFU's retain-only model is exactly that.
- Compare DSG, the best gate and the D models against it on answer confidence, entropy, calibration, fluency, and how often the model says it does not know. Show side-by-side example answers.
- A method that matches the reference "feels" truly unlearned. A method that produces gibberish or refusals is visibly hiding.

**Q4 · Output-behaviour taxonomy with human labels** — Owner: all three · Priority: should · No GPU

- Label 200 or more outputs per method into: correct leak, plausible but wrong, "I don't know", refusal, gibberish, and off-topic.
- Each of the three of you labels the same items independently. Report agreement (Cohen's kappa), then the distribution per method.
- TOFU outputs only for human labels. An automatic classifier applies the same categories to hazardous outputs.

**Q5 · Representation geometry** — Owner: E · Priority: should · Lab PC

- Project mid-layer activations of forget vs retain prompts into 2D (PCA or UMAP) for base, DSG and the D models.
- Pair the picture with the A4 probe numbers, because 2D plots can mislead on their own. Expected story: for DSG the forget cluster stays separate; for a good baked model it merges with retain.

**Q6 · Layer-by-layer knowledge trajectory** — Owner: E · Priority: must · Lab PC (from A4)

- One figure per method: probability of the correct answer read at every layer (logit lens), plus probe accuracy per layer.
- For DSG the knowledge should appear before the clamp layer and remain partly decodable after it. This is the single most convincing "hidden" figure.

**Q7 · Gate decision explanations and a small user study** — Owner: A · Priority: stretch · No GPU

- For each gate decision, show which features fired and why (an explanation card).
- Small study with 10 to 20 classmates: from the explanation card, can they predict the gate's decision on new prompts? This measures whether the gate is really interpretable, not just claimed to be.

**Q8 · Case-study gallery** — Owner: all three · Priority: must · No GPU

- Pick 5 to 8 representative TOFU examples and show, for each, the base answer, DSG answer, attacked-DSG answer, best-fix answer and the never-learned answer, with the gate trace and top features.
- This becomes the qualitative figure in the paper and the clearest slide for the panel.

**How it fits the timeline**

- Q1, Q3, Q5 and Q6 reuse data already logged by the queue (A2, A4 traces and items files), so they need analysis jobs only. They can be added as a new wave after the Wave 1 review.
- Q2 runs on the GPU server after the RMU job.
- Q4, Q7 and Q8 need people, not GPUs. The team can start the labelling guide and case-study template now.
