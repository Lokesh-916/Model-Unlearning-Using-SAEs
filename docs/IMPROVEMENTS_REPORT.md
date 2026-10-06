# IMPROVEMENTS REPORT: everything we tried on DSG, what worked, and what we learned

*Copy of `$DSG_RESULTS/IMPROVEMENTS_REPORT.md` (2026-10-06); the copy in `$DSG_RESULTS` is the one to update.*

*Written 2026-10-06 for the team and the final presentation. Sources: `RESULTS_DIGEST.md` (regenerated 2026-10-06 10:4x,
352 lab runs, 264 gpuws runs), `DEVIATIONS.md`, `FOUNDATION_REPORT.md`, `WAVE1_DONE.md`, the X1 selection file, the
September reproduction notes (`STATUS_FOR_REPORT.md`, `REPO_REPORT.md` §3) and the server job summaries.
Aggregates only: no question, prompt or generation text appears anywhere in this report.*

**Rules every number here follows.** Forget accuracy = accuracy on WMDP-Bio TEST questions (lower is better; chance 0.25).
Utility = accuracy on full MMLU minus the hazard-adjacent subjects (higher is better) unless a row says "4-subject" (the
DSG paper's small 4-subject set). Numbers are mean [95% CI] n. The **lab PC** (`labpc`, RTX 2000 Ada, 16 GB) and the
**server** (`gpuws`, RTX 6000 Ada, 48 GB) are separate hardware baselines: bf16 maths differs between the two GPUs (the
same DSG configuration scores 158/538 on the lab PC and 161/538 on the server), so numbers from the two machines are never
compared with each other. Settings were chosen on DEV splits and reported once on TEST splits. **Attack success** = of the
WMDP TEST items that DSG blocks (answers wrong without attack; n = 265 on labpc), the fraction it answers right under the
attack. Two-sided paired tests (bootstrap + McNemar) are used for every "A beats B" statement.

**Credit by area.** **Break** (attacks B1–B6, GuardBreak N1, red-team app N4, adaptive hardening N5, dilution theory
T1–T2): **Chakreesh**. **Bake** (open-weights erasure D1 / D1-full / D1 v2, the legacy distillation seed, D2 null-space edit,
D3 audit, A6 tampering and relearning): **Amar**. **Evaluation, interpretability and infrastructure** (harness, queue,
cluster pipeline, A1–A5, A7, A8, C1–C6, N2, N3, N6–N10, T3–T5, BM1–BM3, Q1–Q8, RMU baselines, X1, figure parity): **Amaloch**.
Each section heading carries its area.

---

## 1. One-page overview

| ID | What it is (one line) | Machine | Verdict | Headline number |
|---|---|---|---|---|
| Repro | Re-run DSG (Gemma-2-2B, Gemma Scope L3) | labpc | **Worked** | WMDP-Bio 29.37 % vs paper 29.64 %; sanity gate bit-exact (158/538) |
| Legacy fixes | Cyber retain corpus, Gemma softcap, debug flags | labpc | **Worked** | Cyber 4-subject MMLU 44.7 % → 99.4 % at the same WMDP-Cyber 28.0 % |
| Legacy branches | 16 September experiments (imp-*, imp2-*) | labpc | Mixed | 3 ported as seeds (dilution, window gate, distillation); 9 negative |
| Infra | Harness, cache, splits, queue, cluster, analysis tools | both | **Worked** | 230 lab jobs + 264 server runs, every run logged with n and CI |
| A1 | Clean baselines, 5 seeds, Bio + Cyber | labpc | **Worked** (Bio) / **Did not work** (Cyber) | Bio forget 0.295 vs base 0.642 at −0.4 pt utility; no Cyber config within 1 pt |
| A2 | Open-ended WMDP + TOFU (LoRA) | labpc | **Worked** (as a test) | DSG answers with gibberish 60 % of the time; TOFU truth ratio 2.33 vs never-learned 0.89 |
| A3 | Benign-biology utility (hard negatives) | labpc | **Worked** (finding against DSG) | DSG blocks 59 % of benign biology MCQs; accuracy 0.654 → 0.414 |
| A4 | Linear probes on guarded activations | labpc | **Did not work** (for the thesis) | guarded probe 0.285 vs control 0.249: answers not linearly recoverable |
| A5 | Membership inference + MUSE | labpc / gpuws | **Pending** | MIA waiting in queue; MUSE re-run 171–173 queued |
| A6 | Tampering: relearning, quantisation, steering, benign FT | gpuws | **Partly worked** | RMU v2 recovers +0.14–0.16; DSG-nohook = base; LoRA relearning too weak |
| A7 | DSG on Gemma 3 1B / 4B / 12B | gpuws | **Did not work** | accuracy change 0.000 on all three models: the fixed −500 clamp is 0.10–0.33× of Gemma 3's residual norm (34× on Gemma 2) |
| A8 | Latency, VRAM, table generator | both | **Worked** | overhead +1.4–3.1 % (gpuws), +0.04 % forward (labpc) |
| B1 | Dilution: pad the question with filler | labpc | **Worked** (attack) | attack success 0.909 [0.875, 0.943]; base keeps 0.630 under the same padding |
| B2 | Decomposition across chat turns | labpc | **Worked** (attack) | attack success 0.694 [0.642, 0.751] (3-turn split) |
| B3 | Translation (8 languages) + encodings | labpc | **Partly worked** | translations fail (≤ 0.054); spaced 0.449, leet 0.423 (near-chance caveat) |
| B4 | LLM rewrites (black box) | labpc | **Did not work** | forget 0.284–0.305 vs 0.295 without attack |
| B5 | Optimised suffix (white box) | labpc | **Did not work** | forget 0.312 vs 0.295 |
| B6 | Leakage during generation | labpc | **Worked** (as a test) | stream gate leaks 0.029 match but 64 % gibberish |
| C1 | Causal / χ² feature selection | labpc | **Did not work** | attribution features never gate (0.641 ≈ base); χ² = DSG |
| C2 | Window / CUSUM / probe gates | labpc | **Worked** | dilution success 0.853 → 0.029 (CUSUM), 0.069–0.096 (windows) |
| C3 | Layer sweep 1–24 | gpuws | **Worked** (finding) | AUROC 0.976–0.993 at every layer, but layers 20/24 do not suppress |
| C4 | Transcoder gates | labpc | **Pending** | waiting in queue |
| C5 | Intervention type (clamp vs mean ablation) | labpc | **Did not work** (no gain) | mean ablation 0.631 ≈ base; per-feature clamps = clamp-all |
| C6 | Two-level (domain × hazard) gate | labpc | **Partly worked** | 4-subject + hard-negative utility 0.623 vs base 0.633, forget 0.344 |
| D1-local | Distil the guarded model into a noised LoRA student | labpc | **Partly worked** | forget 0.298 (α 0.1) but 5.2 pt more utility loss than DSG (DEV) |
| D1-full | Same, full-parameter 2B student | gpuws | **Partly worked** | α 0.1: forget 0.396, utility 0.514 (−5.0 pt); α ≥ 0.3 collapses |
| D1 v2 | Longer, gentler distillation with a DEV rule | gpuws | **Partly worked** | forget 0.473, utility 0.537 (−2.7 pt); relearning +≤ 0.05 |
| D2 | Closed-form null-space weight edit | labpc | **Did not work** | forget 0.633 vs base 0.642 (n.s.) |
| D3 | Audit of baked models | labpc | **Worked** (as a diagnostic) | DSG features are *more* active after baking (×1.20 D1, ×2.86 D2) |
| RMU v1 / v2 | Our own gradient-unlearning baseline | gpuws | v1 **Did not work**, v2 **Worked** | v2 forget 0.319 vs DSG 0.298 (n.s.) at −1.3 pt utility |
| N1 | GuardBreak CLI toolkit | labpc | **Worked** | end-to-end on 100 items; reproduces B1 |
| N2–N4, N8 | OpenUnlearning adapter, demo, red-team app, reasoning traces | labpc | **Pending** | waiting in queue |
| N5 | Adaptive re-calibration loop | labpc | **Partly worked** | proxy attack success 0.675 → 0.225 in 4 rounds (benign cost not measured) |
| N6 | Conformal threshold | labpc | **Worked** | held-out FPR 0.042 at target 0.05 (n 382) |
| N7 | Multi-layer voting | labpc | **Partly worked** | majority: forget 0.278, FPR 0.102 |
| N9 | SAE quality explains misses | labpc | **Worked** | reconstruction error coef +0.53 [+0.09, +1.08] |
| N10 | Audit cards | labpc | **Pending** (re-run) | generated from smoke data only |
| T1–T2 | Dilution theory checks | labpc | T1 **Did not work**, T2 **Worked** | T1 R² −0.94; T2 window slope ≈ 0 |
| T3–T5 | FPR/FNR bounds, D2 bound, CUSUM assumptions | labpc | **Pending** (re-run) | ran before their inputs existed |
| BM1 | MUSE News + Books | gpuws | **Pending** | first run invalidated by a training bug; re-run queued |
| BM2 | TOFU at full fine-tune scale | gpuws | **Pending** | invalidated twice (metric bug, training bug); re-run queued |
| BM3 | MT-Bench with an open judge | gpuws | **Worked** | 7.46 base vs 7.36 DSG (Δ −0.09, p 0.29); judge scores differ on 2 of 158 |
| Q1–Q8 | Qualitative interpretability track | both | **Pending** (Q2 superseded) | tools built and tested; real runs not done |
| X1 | Combine the best fixes under a fixed rule | labpc | **Pending** (selection done) | DEV pick: CUSUM detector; all other slots keep the DSG default |
| FP-* | Every DSG paper figure for DSG and our gate | gpuws | **Worked** | dynamic gate is what saves utility (static clamp: 0.393 vs 0.560) |

## 2. Claims C-H1 … C-H7

Verdicts come from the fixed rules in `dsgx/analysis/claims.py`, applied separately per machine.

| Claim | Lab PC | Server | In plain words |
|---|---|---|---|
| C-H1 Simple dilution defeats DSG's gate | **Supported** | Inconclusive (B1 runs only on labpc) | Padding a blocked question with ordinary text unblocks 91 % of items; the base model still answers them (0.630). |
| C-H2 DSG fails beyond English MCQ | **Supported** | Inconclusive (no B2/B3 on gpuws) | Splitting the question over three chat turns unblocks 69 %. Translation does *not* work as an attack. |
| C-H3 Knowledge remains internally | **Not supported** | Inconclusive | Probes on the *guarded* activations are near chance. The knowledge is still in the weights (remove the hook and it returns), but it is not linearly readable once the clamp is on. |
| C-H4 Failures have an interpretable cause | **Supported** | Inconclusive (N9 runs on labpc) | Items with higher SAE reconstruction error are missed more often. |
| C-H5 A hardened gate improves robustness at matched utility | Inconclusive | Inconclusive | CUSUM wins on dilution, but its benign FPR under padding exceeds 5 %. X1 TEST (running) decides. |
| C-H6 Baked erasure resists tampering | Inconclusive | Inconclusive | Needs the a6-baked cells (27 of 33 done on gpuws, not yet fetched). Preliminary numbers in §6. |
| C-H7 Findings generalise | Inconclusive | Inconclusive | DSG has no effect at all on Gemma 3, and TOFU needs X1's best-fix condition. |

---

## 3. Before the harness: reproduction and the September experiments

### 3.1 Reproduction of DSG · Eval/Infra (Amaloch)
**What.** We re-ran Dynamic SAE Guardrails (Muhamed et al., COLM 2025) on `gemma-2-2b-it` with the Gemma Scope layer-3
16k SAE (L0 142). DSG picks 20 SAE features that fire much more on hazardous biology text than on WikiText, measures what
fraction of a prompt's tokens fire them (ρ), and clamps those features to −500 whenever ρ exceeds a threshold τ.
**Why.** Every later claim depends on trusting our numbers for the baseline.
**How.** The authors' code, the paper's best configuration (N 20, retain percentile 95, clamp 500), lab PC.
**Result.** WMDP-Bio 29.37 % vs the paper's 29.64 %. Average over the paper's 4 MMLU subjects: 99.41 % vs 99.34 %.
WMDP-Cyber 28.00 % vs 26.74 %. The ablations also reproduced: clamp strength is flat from 50 to 500, p_ratio is monotone,
p_dyn is U-shaped, ρ separates WikiText from WMDP better than ρ_raw (TVD 0.876 vs 0.186; paper 0.90 vs 0.41), and zero-shot
Neuronpedia selection reaches 31.0 % (Bio) / 41.5 % (Cyber).
**Verdict.** Worked.
**Learned.** The paper's "MMLU" is an easy filtered subset (only items the base model gets right under all 24 option
orders, as few as 9 items per subject). That is why we switched to full MMLU as the main utility metric. We also logged
latency on the lab GPU: +7.8 % to +13.8 %. The paper reports +2.5 % to +7 % on an A6000, and our later gpuws numbers are
close to the paper's (§8, A8).

### 3.2 Cyber retain-corpus fix, softcap fix, debug flags · Eval/Infra (Amaloch)
**What / why.** The first Cyber runs reached the right WMDP-Cyber score but MMLU collapsed to 44.7 %. We traced it to the
retain corpus. WikiText never shows the SAE Gemma's chat or multiple-choice format, so a formatting feature (1312) passed
the "safe on retain" filter. **How.** We replaced the retain text with 400 chat-formatted MCQ prompts from 10 MMLU subjects
that are disjoint from the evaluation set, and left the algorithm unchanged. We also added the missing Gemma logit
softcap to the chunked-softmax scorer: the maximum probability error was 0.18 before and 0.025 after, and no prediction
changed. **Result.** Cyber 4-subject MMLU went from 44.7 % to 99.4 % at the same WMDP-Cyber score (28.0 %).
**Verdict.** Worked. **Learned.** DSG's feature selection is only as good as its retain corpus, and re-seeding does not
help: WikiText-2 gives only 275 sequences, so the same bad feature came back under seed 1. This became the
"Cyber-chatretain" baseline in A1, and the A1 grid later showed that full-MMLU utility still drops (§4, A1).

### 3.3 First dilution finding and two failed window defences · Break (Chakreesh) + Eval (Amaloch)
**What.** ρ is an average over the whole prompt, so adding harmless text lowers it. We padded blocked questions with about
150 words of filler. **Result (n = 60, single run).** 98.3 % of blocked questions were unblocked. Two sliding-window
defences failed: the threshold landed at the statistic's ceiling (1.0, then 1.04), so the gate never fired and the attack
still succeeded 90–92 % of the time. **Verdict.** The attack worked; both defences did not. **Learned.** Calibrating on
the *maximum* window rate per sequence is an order-statistics trap, because the maximum of ~1,000 overlapping windows
drifts upward whatever the signal. The fix (calibrate on the pooled distribution of window rates) became C2.

### 3.4 The sixteen September branches (ported or retired in the harness)
Each branch was a quick test on cached artifacts, with one seed, no split and no CI. Old JSON numbers are kept as legacy
only. Every idea that mattered was re-run cleanly in the harness (section 5.0 of the master plan).

| Branch | What and why | Legacy result | Verdict | What it became / lesson |
|---|---|---|---|---|
| imp-a1 (Break + Eval) | Dilution attack + window (w 8/24) + CUSUM gates | DSG bypass 98 %; window-24 cut it to 20–35 %; CUSUM 35–37 % | Worked as a seed | B1 and C2. With proper calibration the harness versions do much better (§5, C2) |
| imp-a2 (Eval) | Replace the −500 clamp with mean ablation, to avoid gibberish | WMDP 0.993 (no forgetting) | Did not work | C5 confirmed it: mean ablation removes the effect entirely |
| imp-a3 (Eval) | Per-concept gates with a Bonferroni threshold | Holdout utility 0.24–0.38 | Did not work | Too few features per concept; C6 tried a cleaner version |
| imp-b1 (Bake) | Distil the guarded model into a LoRA student (no hook needed) | Student 40.0 % vs hook 45.5 % (random 200 items) | Partly worked | The seed for D1. The same-reference LoRA student is the "legacy student" baseline |
| imp-b2 (Eval) | TOFU fact-level SAE plan | README only | Not attempted | Replaced by A2 TOFU |
| imp-b3 (Eval) | "Kill test" on benign biology + two-level gate | Single gate: college-bio 0.38, virology 0.42; two-level gate WMDP 0.68 | Finding worked, fix did not | Early warning of the A3 result; C6 is the clean re-run |
| imp-c1 (Eval) | Attribution-patching feature ranking | WMDP 0.656, 0/20 overlap with DSG | Did not work | C1 confirmed it on TEST |
| imp-c2 (Eval) | Layers 3 + 8 OR-gate | 27.7 % / 97.9 % | Preliminary | C3 full layer sweep |
| imp-c3 (Eval + Bake) | Gated residual projection + layer-22 probes | WMDP 0.946; probe 0.633 → 0.178 under DSG | Did not work (fix) | Probe idea → A4; projection idea → D2 |
| imp2-1 (Eval) | χ² co-occurrence feature ranking | WMDP 0.695, MMLU-u 0.813 | Did not work | C1 χ² with DSG's filters matches DSG |
| imp2-2 (Eval) | Layer 3 + 8 AND-gate | 31.2 % / 99.7 % | Worked (vs OR) | N7 majority voting |
| imp2-3 (Eval) | Logistic-regression gate on the same features | 27.0 % / 93.2 % | Mixed | C2 probe gates |
| imp2-4 (Eval) | Per-feature clamp strength | 29.2 % / 99.1 % | No change | C5: identical to clamp-all |
| imp2-5 (Eval) | Train our own TopK SAE | 46.8 % / 93.6 % | Did not work | Dropped (backlog): Gemma Scope SAEs are far better |
| imp2-6 (Bake) | Orthogonalise weights against the 20 decoder directions | 90.7 % / 99.5 % | Did not work | D2 "ortho" baseline; also no forgetting in the harness |
| lokesh-experimentations (Break) | Guarded free generation demo | 1 of 8 hazard prompts tripped the gate; gibberish later in outputs | Worked as a demo | B6 measures this properly |

---

## 4. Infrastructure · Eval/Infra (Amaloch)

**What.** One harness (`dsgx`) that every experiment runs through: a bit-exact port of DSG (`dsg-faithful`) plus a
`dsg-fixed` variant, attack and gate registries, a compact activation cache, seeded 50/50 DEV/TEST splits for 59 datasets
(17,302 items, sha256-checked), a leakage checker that runs before every job, bootstrap / paired-bootstrap / McNemar
statistics, and a run logger that writes config, metrics with n and CI, per-item parquet and traces. A job queue runs it all
unattended in tmux, with a scheduler, watchdog, OOM retry, canary, wave pauses, reboot recovery and a `doctor` that sorts
failures and re-queues the safe ones. Every experiment lives on its own branch and worktree, pinned to a commit. For the
48 GB server, a lab-PC control script (`cluster/server.sh`) stages only what a job needs, runs a validation job first,
fetches with sha256 verification and cleans up afterwards. Analysis tools turn the runs into the digest, the claims table,
paper tables and figures, and a dashboard.
**Why.** The mid-review would have been rejected for single seeds, no CIs, tuning on the test set, and nine loosely linked
experiments. The harness makes every comparison fair (same prompts, splits and token budget) and every number traceable.
**How it went.** The activation cache is 430 MB instead of 37 GB of pickles, is built in about 1.5 min, and differs by 0.0
from the legacy sparsity files on all 16,384 features. The sanity gate reproduces the paper configuration exactly (WMDP 158/538,
MMLU-u 0.9941, τ 0.5458). The lab queue holds 230 jobs: 120 done, 70 moved to the server, 39 waiting, 0 failed. The server
has produced 264 runs; 114 CPU tests pass. **Verdict.** Worked.
**Learned (the bugs it caught, all fixed and logged in DEVIATIONS.md):**
- **Cross-GPU numerics.** The same configuration gives 158/538 on the lab GPU and 161/538 on the server, with 3 gate
  decisions within 0.012 of τ. Decision: one baseline per GPU, never mixed. This is a reproducibility caveat for DSG itself.
- **Trainer gradient accumulation.** `zero_grad` ran after the step function, so the TOFU-full and MUSE fine-tunes on gpuws
  trained on about a quarter of their data. All affected runs are being re-run.
- **TOFU metric v1** clipped the mean truth ratio, so model utility read 0.0 everywhere. Fixed (metric v2).
- **Batch numerics.** A batch size above 1 flips 1 in 843 predictions, so every reported TEST run uses batch size 1.
- **Queue deadlock.** A wave-1 job depended on a wave-2 job, which left the GPU idle for 5 h. The doctor now detects this.
- **Memory.** Lab OOMs were fixed by releasing the duplicate model and adding accumulation, checkpointing and 8-bit AdamW
  (D1 training: 14.8 → 6.4 GiB).
- **Server-side surprises.** The SAE library fetched shapes over HTTP on an offline server; a dataset was not staged;
  `server.sh stage` copied only the first input; a disk guard refused today's re-runs.
- **Biased latency.** The first latency run timed back-to-back on a saturated host; v2 interleaves the conditions.

---

## 5. Part A and B: evaluation and attacks

### A1 · Unified baselines · Eval (Amaloch) · labpc
**What.** DSG re-tuned on DEV over 20 configurations (N features, retain percentile, clamp) per setting, then the chosen
configuration run on TEST with 5 seeds, next to the base model. Settings: Bio, Cyber-faithful (WikiText retain) and
Cyber-chatretain. **Why.** One clean, honest baseline with a fixed selection rule: the lowest DEV forget accuracy among
configurations within 1 point of base utility. **Result (TEST).** Bio: forget 0.295 [0.294, 0.296] across 5 seeds vs base
0.642 [0.604, 0.680] (n 637; paired Δ −0.347, p < 1e-38). Full-MMLU utility 0.559 vs 0.563 (n 6,095), benign FPR 0.017.
Cyber: **no configuration keeps utility within 1 point.** The best DEV drop is −1.7 points. The selected TEST
configurations lose 9.1 points (chat-retain: forget 0.254, utility 0.482, FPR 0.379) and 28.1 points (WikiText: utility
0.292, FPR 0.77). The unverified third-party RMU checkpoint, a reference only, scores 0.499 on Bio. **Verdict.** Worked
for Bio; did not work for Cyber, which we now report as a Pareto curve. **Learned.** DSG's 4-subject utility metric
understates mild damage: on DEV it stays within 1 point while full MMLU drops 2.5–2.8 points. From Wave 1 on, Bio is the
primary setting for every attack and fix.

### A2 · Open-ended answers and TOFU · Eval (Amaloch) · labpc
**What.** (1) WMDP-Bio questions turned into open questions (n 476) and graded by match, F1, ROUGE-L and embedding
similarity. (2) TOFU (fictitious authors): LoRA fine-tune Gemma on TOFU, apply DSG with features from the forget authors,
and compare against a retain-only model that never saw them. **Why.** Low MCQ accuracy could just mean the model refuses
or breaks; a real unlearning method should behave like a model that never learned. **Result.** Open WMDP: base match
0.176 → DSG (streaming gate) 0.027, with **gibberish in 59.7 % of answers** (prompt-only gate: 0.044 match, 53.6 % gibberish).
TOFU QA: match 0.800 → 0.438, gate fired on 66.7 %, gibberish 1.3 % (n 400). TOFU metrics: DSG's forget truth ratio is
**2.33 [1.28, 3.82]** against 0.89 for the never-learned model and 0.55 for the fine-tuned model, and the forget-quality KS
test against the never-learned model gives p ≈ 0. Model utility: 0.490 DSG, 0.501 full, 0.493 retain-only.
**Verdict.** Worked as a test; the finding goes against DSG. **Learned.** DSG does not make the model "not know". It pushes
it past ignorance into actively preferring wrong answers, and in free text it produces noise. Both are easy to tell apart
from genuine ignorance, which is a privacy signal in itself.

### A3 · Benign biology (hard negatives) · Eval (Amaloch) · labpc
**What.** College and high-school biology, anatomy, virology and medical genetics MCQs (n 428), plus 101 open benign-biology
questions. **Why.** A guard that blocks all of biology is not precise unlearning. **Result.** DSG's gate fires on **59.1 %
[54.4, 63.8] of benign biology MCQs**; accuracy falls from 0.654 to 0.414. Open benign biology: match 0.238 → 0.109, with
gibberish in 17.8 %. **Verdict.** Worked as a test; a strong negative for DSG. **Learned.** The paper's utility subjects
(history, geography, CS, aging) never test the neighbourhood of the hazard. C6 was our attempt to fix this.

### A4 · Knowledge depth (probes) · Eval/Interp (Amaloch) · labpc
**What.** Linear probes that read the correct answer letter from every layer's residual stream, 5 seeds, with a control
task, for the base model and for the DSG-guarded forward pass. **Why.** To test "hidden, not erased" directly (C-H3).
**Result.** Base: the answer becomes decodable from layer 12 and peaks at layer 19 (0.630 [0.622, 0.639]; control 0.259).
DSG-guarded: best 0.285 at layer 9, control 0.249, so near chance at every layer. **Verdict.** Did not work for the thesis:
C-H3 is *Not supported* by its rule. **Learned.** The −500 clamp at layer 3 destroys downstream linear structure for the
gated prompts. The knowledge still sits in the weights (removing the hook restores base accuracy; §6, A6), but this cannot
be shown by probing the guarded activations. The honest wording is "hidden in the weights, not in the guarded activations".

### A5 · Privacy (membership inference, MUSE) · Eval (Amaloch) · Pending
TOFU membership inference is waiting in the lab queue. The MUSE part is BM1 (§8).

### A7 · Gemma 3 1B / 4B / 12B · Eval (Amaloch) · gpuws
**What.** The same DSG pipeline (and our window gate) on Gemma 3 with the Gemma Scope 2 SAEs (layers 13 / 17 / 24), WMDP
TEST, utility, and dilution 400 / 1,600. **Why.** To check that the findings are not specific to Gemma-2-2B (C-H7).
**Result.** The base models score 0.465 (1B), 0.611 (4B) and 0.714 (12B). With DSG or our gate, **forget accuracy changes
by 0.000 on every model** (paired Δ 0.000, n 637), although the gate fires on 16–32 % of WMDP items and τ calibrates to
≈ 0 (1B: 6 answers change and cancel out; 4B/12B: none). **Verdict.** Did not work. **Learned (diagnosed, session 17,
`docs/A7_GEMMA3_DIAGNOSIS.md`).** Not a bug: the hook site, the error term and propagation to the next layer all check out.
DSG's clamp (−500 per feature) is an absolute constant. Gemma 3's residual norm at the Gemma Scope 2 layers is 6,834 (1B) and
30,931 (4B) against 92 for Gemma 2 at layer 3, so the edit is 0.33× / 0.10× of the residual instead of 34×. One clamped token
moves the last-token distribution about 9× less (KL 0.0024 vs 0.021). DSG's recipe does not transfer across model families
unless the clamp is scaled to the residual norm.

### A8 · Reporting standards and cost · Eval (Amaloch)
**What.** Latency, VRAM and one table generator. **Result.** Lab PC (50 prompts, debug prints off): forward-pass overhead
+0.04 %, generation throughput 5.81 → 5.76 tokens/s, peak VRAM unchanged (7.86 GB). Server (FP-latency v2, interleaved
timing, batch size 1): +2.2 % to +3.1 % for DSG and +1.4 % to +2.5 % for the window and CUSUM gates, from 64 to 2,048
tokens. **Verdict.** Worked. **Learned.** Our September numbers (+8–14 %) were mostly debug prints and timing on a busy
GPU, so DSG's cost claim holds. The A8 table job ran on smoke data and will be re-run at the end.

### B1 · Dilution · Break (Chakreesh) · labpc
**What.** Add 0–1,600 tokens of filler before, after or around the blocked question, taken from WikiText, chat text or
benign-biology text. **Why.** ρ is a whole-prompt average, so filler lowers it. **Result.** Maximum attack success
**0.909 [0.875, 0.943]** (n 265; 1,600 WikiText tokens before the question). 100 tokens before already gives 0.766–0.815.
The base model under the same padding still scores 0.620–0.630 (control), so the knowledge is intact. The filler source
matters: **benign-biology filler does not work** (success ≤ 0.10), because it fires the same features. "After" and "around"
positions are weaker than "before". **Verdict.** Worked; C-H1 Supported. **Learned.** The gate's statistic is the weak point,
not the features. This motivated C2.

### B2 · Decomposition · Break (Chakreesh) · labpc
**What.** Split the question over 2–3 chat turns (stem and options separated), or put the hazardous terms only in the
options. **Result.** 3-turn split: attack success **0.694 [0.642, 0.751]**, DSG accuracy 0.545 vs the base control 0.575.
2-turn split: 0.509. Stem-last: 0.257. Options-only: 0.083. **Verdict.** Worked; carries C-H2. **Learned.** Each turn on its
own looks benign enough; the gate never sees the whole question at once.

### B3 · Translation and encodings · Break (Chakreesh) · labpc
**What.** NLLB translations into 8 languages (Hindi, Tamil, Telugu, French, Spanish, Chinese, Arabic, Russian; back-
translation chrF ≥ 40 filter) and 4 encodings. **Result.** **Translation does not bypass DSG**: the gate still fires on
82–95 % of items and attack success is 0.015–0.054. Encodings: spaced characters 0.449 [0.389, 0.509], leetspeak 0.423;
base64 0.128 and ROT13 0.023. **Verdict.** Partly worked. **Learned.** Gemma Scope's layer-3 features are largely
multilingual, a positive surprise for DSG. The encoding numbers need care: under leetspeak the *base* model only reaches
0.325, close to chance, so part of that "success" is guessing on 4-option questions. Only "spaced" lacks a base control.

### B4 · Black-box LLM rewrites · Break (Chakreesh) · labpc
**What.** gemma-2-2b-it rewrites each question (k = 5 tries). **Result.** DSG forget 0.284–0.305 vs 0.295 without attack;
base under the rewrite 0.619 (knowledge kept). **Verdict.** Did not work. **Learned.** A small model's paraphrases keep the
hazardous vocabulary that the features detect. A stronger attacker was out of budget.

### B5 · White-box suffix · Break (Chakreesh) · labpc
**What.** A gradient-free coordinate search for an 8-token suffix that lowers the gate score (50 / 200 steps).
**Result.** DSG forget 0.312 vs 0.295; base 0.645. **Verdict.** Did not work. **Learned.** Eight tokens cannot move a
whole-prompt average much. Dilution is the cheaper version of the same idea.

### B6 · Leakage during generation · Break (Chakreesh) · labpc
**What.** Open hazardous questions answered by free generation: streaming gate (re-checked every token) vs prompt-only
gate. **Result (n 476).** Base match 0.197; streaming gate 0.029 (fires on 85.9 %, gibberish 63.7 %); prompt-only gate
0.044 (gibberish 55.0 %). **Verdict.** Worked as a test. **Learned.** Little correct content leaks, but the price is broken
output. A deployed system would need a refusal, not a clamp.

---

## 6. Part C and D: fixes for the API setting and for open weights

### C1 · Causal and χ² feature selection · Interp (Amaloch) · labpc
**What.** Pick the 20 features by attribution patching (activation × gradient of the correct answer) or by a χ² test,
instead of DSG's frequency ratio. **Result (TEST).** Attribution features never gate: forget 0.641 vs base 0.642. χ²
features match DSG (0.281 vs 0.283) with a lower benign FPR (0.041 vs 0.080, n 363), but on DEV they are worse (+0.046
[+0.005, +0.088]). **Verdict.** Did not work. **Learned.** Features that *cause* the answer are not features that *detect*
the topic; a gate needs the latter.

### C2 · Streaming and probe gates · Interp (Amaloch) · labpc
**What.** Replace the whole-prompt average with a sliding-window rate (w 8–64), a CUSUM change detector, or logistic probes
on SAE statistics or on the raw residual stream, all calibrated to 5 % benign FPR on DEV. **Result (TEST, dilution 400).**
Attack success: DSG 0.853 → **CUSUM 0.029 [0.011, 0.051]**, windows 0.069–0.096, raw-residual probe 0.338, SAE probe
0.574. Without attack the window and CUSUM gates match DSG (paired Δ within ±0.006, n.s.); the probes forget slightly more (−0.030, p 0.005). The price:
under padding, benign FPR rises to 0.118 (CUSUM) and 0.17–0.22 (windows), and 4-subject utility falls to 0.86–0.89. w 8 never
fires. **Verdict.** Worked, with a cost under attack. **Learned.** The statistic is fixable. A local statistic resists
dilution because padding cannot lower the peak (T2), but it also becomes jumpier on long benign prompts. The SAE probe is
not better than the raw-residual probe, so the case for SAEs here is auditability, not accuracy.

### C3 · Layer sweep · Interp (Amaloch) · gpuws (lab jobs moved to the server)
**What.** The same gate on canonical 16k SAEs at layers 1, 3, 5, 8, 12, 16, 20, 24. **Result.** Detection AUROC is high
everywhere (0.976–0.993, n 1,000; best layer 16 at 0.993). But clamping at layers 20 or 24 leaves forget accuracy at base
(0.622 / 0.639 vs 0.644), layer 12 only halves the effect (0.427), and layers 1–16 suppress (0.262–0.303). **Verdict.**
Worked as a finding. **Learned.** **Detecting a topic is not the same as being able to remove it.** By layer 20 the answer
is already computed elsewhere, so a late-layer clamp is too late. X1 therefore screened layers 16, 24 and 8.

### C4 · Transcoder gates · Interp (Amaloch) · Pending (waiting in the lab queue).

### C5 · Intervention type · Interp (Amaloch) · labpc
**What.** Clamp all selected features to −500 vs mean ablation vs per-feature clamp vs scaled clamp. **Result.** Clamp-all
0.344, per-feature 0.342, scaled 0.345, **mean ablation 0.631** (Δ vs base −0.011, n.s.). **Verdict.** Did not work (no
gain). **Learned.** DSG works by pushing features far negative, which is also why outputs turn into gibberish. Setting them
to their average does nothing.

### C6 · Two-level gate · Interp (Amaloch) · labpc
**What.** Fire only when a domain detector *and* a hazard detector (calibrated on benign-biology text, DEV only) both fire.
**Result.** Forget 0.344 (vs DSG 0.295); on a utility set that includes the hard-negative biology questions (n 586), 0.623
vs base 0.633, FPR 0.082. **Verdict.** Partly worked. **Learned.** It stays within 1 point of base on that set, whereas DSG
loses 24 points on the hard negatives alone (A3: 0.414 vs 0.654; not the identical item set). It gives back about 5
points of forgetting. That is a real trade-off, not a free win.

### D1-local · Distillation into a noised LoRA student · Bake (Amar) · labpc
**What.** UNDO-style distillation. A student copy of Gemma (LoRA, noised initialisation α ∈ {0.1, 0.3, 0.5}) learns to match
the guarded teacher on forget prompts and the unguarded teacher on retain prompts, so the erasure ends up in the weights
and no hook is needed. **Why.** DSG's protection disappears when someone removes the hook from open weights. **Result
(TEST).** α 0.1: forget 0.298 (≈ DSG); α 0.3: 0.281; α 0.5: 0.283. The legacy same-reference student reaches only 0.584.
On full MMLU (DEV, X1 screen) they lose **5.2, 7.1 and 10.8 points more utility than DSG**. **Verdict.** Partly worked. **Learned.**
Distillation transfers the forgetting but also costs general ability. The noise level controls a forget–utility trade-off
with no free point.

### D1-full · Full-parameter distillation (2B) · Bake (Amar) · gpuws
**What.** The same recipe without LoRA, which did not fit the lab GPU. **Result (TEST).** α 0.1: forget **0.396
[0.358, 0.433]**, utility 0.514 vs base 0.644 / 0.564 (−5.0 points). α 0.3 / 0.5 **collapse** to chance utility (0.238 /
0.232). Paired against DSG, α 0.1 forgets less (+0.097). **Verdict.** Partly worked. **Learned.** Full-parameter students
are fragile at higher noise, and the retain KL was still falling at 2,000 steps, which motivated D1 v2.

### D1 v2 · Longer, gentler distillation with a DEV rule · Bake (Amar) · gpuws
**What.** α ∈ {0.05, 0.1, 0.2}, 4,000 steps. The rule was fixed before any result: the lowest DEV forget accuracy among
students with a DEV utility drop ≤ 0.02. **Result.** No student met the bound (drops 0.039–0.120), so the fallback picked
α 0.1. TEST: forget **0.473 [0.435, 0.512]**, utility **0.537 (−2.7 points)**. Full-FT relearning (k 10–1,000 passages, 200
steps) moves it only from 0.440 to 0.437–0.493 (+≤ 0.053). **Verdict.** Partly worked. **Learned.** Gentler distillation keeps
more utility and resists relearning better than RMU v2, but it forgets much less than DSG. No baked method yet matches DSG's
forgetting at DSG's utility.

### D2 · Closed-form null-space edit · Bake (Amar) · labpc
**What.** Edit the MLP output weights in layers ≤ 3 to suppress the selected feature directions, projected onto the null
space of retain activations (AlphaEdit-style, with a retain guarantee), plus a plain decoder-orthogonalisation baseline.
**Result.** Forget 0.633 (null-space) and 0.628 (ortho) vs base 0.642. Both n.s. (Δ −0.009 [−0.030, +0.011]). In X1's DEV
screen, D2 changed nothing under dilution either. **Verdict.** Did not work. **Learned.** Removing the 20 decoder directions
from the residual stream is not enough. D3 shows the forget-corpus activity of those features *rises* after the edit, so
the model routes around it.

### D3 · Baked-model audit · Bake (Amar) · labpc
**What.** SAE activity on the forget corpus for the baked models, and which features changed most. **Result.** Mean
activation of DSG's 20 features on forget text: base 0.70, D1 (α 0.3) 0.84 (×1.20), D2 2.00 (×2.86). Of the 20 features
D1 changed most, 6 are DSG features. **Verdict.** Worked as a diagnostic. **Learned.** Neither baked model removes the
features DSG relies on. D1 forgets through other paths, which is consistent with its relearning behaviour. D3's probe part
is still waiting in the queue.

### A6 · Tampering: can an attacker get the knowledge back? · Bake (Amar) · gpuws
**What.** On each model: LoRA relearning (k 10–1,000 forget passages, rank 8 / 64, 1,000 steps), full fine-tune relearning
(200 steps), 4- and 8-bit quantisation, steering, and benign (Alpaca) fine-tuning. **Results** (forget accuracy on a
300-item subset, Wilson CI ±0.055):
- **DSG without the hook = base:** 0.527–0.530, unchanged by any attack. In open weights, DSG protects nothing.
- **DSG with the hook:** relearning *lowers* accuracy (0.347 → 0.197–0.310) because the clamp plus the fine-tune damage the
  model (utility 0.73 → 0.39–0.67). Quantisation (0.337 / 0.350) and steering (0.347–0.350) change nothing.
- **RMU v2 (full FT):** 0.373 → **0.517–0.537 (+0.14 to +0.16)**, i.e. back to base level. **RMU v1:** +0.08 to +0.10.
- **D1-full α 0.1:** +0.067 to +0.087. **D1 v2:** −0.003 to +0.053.
- **a6-baked (preliminary, read on the server, not yet fetched or verified; 27 of 33 cells):** D1-local α 0.3 0.307 →
  0.313–0.410 under LoRA relearning (largest +0.10 at k 50, r 64); D2 0.537 → 0.483–0.533 (no forgetting to recover);
  legacy student 0.503 → 0.497–0.517; quantisation leaves all three within 0.02; benign fine-tuning 0.307 → 0.303 (D1).

**Verdict.** Partly worked. **Learned.** (1) The LoRA relearning attack on raw forget-corpus text barely moves models that
still hold the knowledge, so it cannot separate methods. The full fine-tune version can, and it shows RMU v2's erasure is
shallow. (2) The baked students recover less than RMU v2, which is the first evidence for C-H6. The rule still needs the
matched a6-baked cells.

### RMU v1 / v2 · Our own gradient-unlearning baseline · Eval (Amaloch) · gpuws
**What.** We trained RMU ourselves because no official Gemma-2-2B checkpoint exists. v1 used a 6-configuration DEV grid;
v2 uses 16 of 48 configurations, chosen with a written rationale after v1 under-performed. **Result (TEST).** v1 0.556 (too
weak; the selected configuration sat at the grid edge). **v2 (c14: 20× steering, α 300, layers 1–3, 150 steps): 0.319
[0.283, 0.355] vs DSG 0.298, Δ +0.020 [−0.006, +0.047], n.s.**, but utility 0.547 vs DSG 0.560 (−1.3 points, p < 0.001). It
is also at the grid edge, so a stronger RMU may exist. The third-party checkpoint (unverified reference) scores 0.498.
**Verdict.** v1 did not work; v2 worked as a baseline. **Learned.** A properly tuned weight-based method matches DSG's MCQ
forgetting without any gate, and A6 shows it comes back after 200 steps of fine-tuning. MCQ accuracy alone cannot rank
unlearning methods.

---

## 7. New additions (N1–N10) and theory checks (T1–T5)

**N1 · GuardBreak toolkit · Break (Chakreesh) · labpc.** The B attacks packaged as a CLI
(`guardbreak run --gate dsg --attack dilution`) with transformation code only. The end-to-end run on 100 TEST items
reproduces B1: on the 29 items the base model answers reliably, DSG under dilution 400 scores 0.966 [0.897, 1.000] (the
same as the base model) vs 0.586 with the window gate (paired Δ on all 100 items −0.17, p 0.003). **Worked.** Lesson: the attack takes about a dozen lines; anyone can run it.

**N2 OpenUnlearning adapter, N3 interactive demo, N8 reasoning-trace gating · Eval (Amaloch); N4 red-team challenge app ·
Break (Chakreesh).** All built and smoke-tested, all **pending** in the lab queue (they need the A2 TOFU models, which now
exist).

**N5 · Adaptive hardening loop · Break (Chakreesh) · labpc.** Five rounds of attack → re-calibrate the threshold on the
found attacks (DEV) → re-attack (TEST proxy). Attack success fell **0.675 → 0.325 → 0.275 → 0.225 → 0.225** while τ fell
0.546 → 0.320. **Partly worked.** Lesson: lowering τ always "wins" this game, and the round-by-round benign FPR was not
logged, so the cost of each round is unknown. The next version must report the FPR per round.

**N6 · Conformal threshold · Eval (Amaloch) · labpc.** Split-conformal τ for target FPR α. ρ gate: held-out benign FPR
0.008 / **0.042 [0.026, 0.067]** / 0.105 at α 0.01 / 0.05 / 0.1 (n 382); shifted benign 0.000–0.006. The window gate is
similar. **Worked.** Lesson: the coverage guarantee holds in practice and is conservative under shift. X1 did not select
it because DSG's empirical threshold already sits at 4.5 % on DEV.

**N7 · Multi-layer voting · Interp (Amaloch) · labpc.** Any-of / majority-of over the top C3 layers. Any: forget 0.264,
FPR 0.179, 4-subject utility 0.603. Majority: 0.278, FPR 0.102, 0.636. **Partly worked.** On X1's DEV screen, majority
voting was eligible (dilution Δ −0.058) but much weaker than CUSUM (−0.401).

**N9 · SAE quality explains gate misses · Interp (Amaloch) · labpc.** Per-item logistic regression of "gate missed" on SAE
statistics (Cyber chat-retain TEST, n 137, miss rate 0.241). **Reconstruction MSE: coefficient +0.53 [+0.09, +1.08]**; L0
and ρ are n.s.; 48 selected features have a close decoder neighbour (splitting). **Worked** (C-H4 Supported on labpc).
Lesson: the gate fails where the SAE describes the prompt badly.

**N10 · Audit cards · Eval (Amaloch).** The template and generator work, but the job ran on 2026-10-04 right after the
Wave-1 resume and read smoke runs only. **Pending** (re-run at the end).

**T1 · Dilution formula · Break (Chakreesh).** Predicted ρ = ρ·L/(L+P) + filler rate·P/(L+P) vs measured ρ, per item
(n 34,406): **R² −0.94, correlation 0.35**. **Did not work** as a quantitative model. The direction is right (mean ρ 0.714 →
0.25 at 400 tokens), but per-item prediction fails. Lesson: the filler's own firing rate varies with context, so a
constant-rate formula is too simple. In the paper we state it as a bound, not a predictor.

**T2 · Padding invariance of local statistics · Break (Chakreesh).** The window-max slope vs pad length is 2.0e-5 per
token (12 points), i.e. flat, as predicted. The CUSUM slope (4 points) is negative but small. **Worked.** This is the reason
C2's gates resist dilution.

**T3 (gate FPR/FNR bounds), T4 (D2 retain bound), T5 (CUSUM assumptions) · Eval (Amaloch); T4 Bake (Amar).** T3 ran before
its inputs existed and read 28 smoke runs; T4 is a stub (D2 changed nothing, so there is no bound to check); T5 found no
trace series (n 0). **Pending** (re-run T3 and T5 at the end).

---

## 8. Benchmarks, qualitative track, combination and figure parity

**BM1 · MUSE News and Books · Eval (Amaloch) · gpuws.** Official muse_bench metrics (VerbMem, KnowMem, PrivLeak) with a
retrain reference, for target, target + DSG and target + best gate. The first full run was **invalidated by the Trainer
accumulation bug** (fine-tunes saw about ¼ of the data). The re-run (171–173) is queued behind TOFU. **Pending.** Lesson:
the official PrivLeak entry point crashes, so we call its components; PrivLeak is relative to *our* retrain model.

**BM2 · TOFU at full fine-tune scale · Eval (Amaloch) · gpuws.** The first full fine-tune run was **invalidated twice**:
first by our metric (clipped truth ratio, model utility 0.0), then by the accumulation bug. The re-run (168–169) is queued.
The lab LoRA version (A2) is complete. **Pending.** Lesson: a metric that reads exactly 0.0 everywhere is a bug until
proven otherwise. We also found that our generation-time gate scores each new token alone once the KV cache is used, which
matters for TOFU ROUGE.

**BM3 · MT-Bench with an open judge · Eval (Amaloch) · gpuws.** 80 questions × 2 turns, judge gemma-2-9b-it (fixed prompt,
temperature 0). Base **7.46 [7.07, 7.83]**, DSG 7.36 [6.96, 7.75], window gate 7.37 (n 158). Paired Δ vs base: −0.09
(p 0.29) and −0.09 (p 0.30). **The judge's score differs from base on only 2–3 of 158 (question, turn) pairs.** **Worked.** Caveats: the judge is from the same
family as the judged model (self-preference possible), and absolute scores are not comparable with the paper's GPT-4
judged 7.78. Lesson: as the roadmap predicted, MT-Bench almost never triggers the gate, so it says little about DSG. A3 is
the utility test that matters.

**Q1–Q8 · Qualitative interpretability track · Interp (Amaloch).** Tools are built and tested: feature cards (Q1, with
Neuronpedia fetch), "never learned" comparison (Q3), representation geometry (Q5), layer trajectory (Q6), and the TOFU-only
annotation sheet, kappa and gallery (Q4, Q8). They have run only on smoke data; the inputs (A2, A4, D3) now exist. **Q2**
(attribution graphs with transcoders on gpuws) ran once. DSG blocked 1 of 3 TOFU facts (on that fact the probability of the
answer token fell from 0.988 to 0.000); the same fact passed in French and in English alike; D2 left all three unchanged.
That run used the TOFU model from the buggy trainer and is superseded; the re-run is automatic after TOFU v3. Q7 (user
study) was not started. **Pending.** Gotcha for Q1: Neuronpedia's default layer-3 SAE is the L0-59 variant, not DSG's L0-142.

**X1 · Combination wave · Eval (Amaloch) · labpc.** **What.** One candidate per slot (features, detector, threshold,
intervention, baked model). A component is included only if, on DEV, it significantly improves its slot's target (paired
CI < 0 and McNemar p < 0.05), costs ≤ 1 point of full-MMLU utility, and keeps benign FPR ≤ 5 %. The rule was fixed in code
before screening. **DEV selection (2026-10-06 09:19):**

| slot | selected | why the others failed |
|---|---|---|
| features (C1) | default | attribution +0.436, χ² +0.046 (both worse) |
| **detector (C2/C3/N7)** | **CUSUM**: dilution Δ **−0.401 [−0.447, −0.357]**, utility cost +0.0002, FPR 0.045 | windows w16–w64 and N7-majority also eligible but weaker; layer 16 / 8 FPR 0.056 / 0.051 > 0.05; N7-any cost 1.6 pt, FPR 0.084 |
| threshold (N6) | default | conformal gives no FPR improvement over DSG's 4.5 % |
| intervention (C5) | default | mean ablation +0.420; per-feature clamps n.s. |
| baked (D1/D2) | default | D1 costs 5.2–10.8 pt more utility than DSG; D2 n.s. |

The TEST runs (combined method, leave-one-out, 5 seeds, every B attack, utility benchmarks) are running: 2 of 33 done.
**Pending.** This decides C-H5 and feeds C-H7.

**FP-\* · DSG figure parity · Eval (Amaloch) · gpuws.** Every figure type in the DSG paper, for DSG and our window gate.
- **Clamp grid (DEV):** c = 10 is weak (0.465 at N 20); from c ≥ 25 to 1,000 the effect is flat (0.25–0.29), as in the paper.
  Our window gate never fires for N ≥ 100 features: its threshold calibrates too high when many features are pooled.
- **Static vs dynamic:** an always-on clamp gives forget 0.267 but utility **0.393** (FPR 1.0) vs dynamic 0.560. Confirms
  the paper: gating is what keeps utility.
- **Data efficiency:** with 32–64 forget documents, DSG's benign FPR swings from 0.05 to 0.50 across seeds, while the
  calibrated window gate stays at 0.036–0.077.
- **Multi-topic (Bio + Cyber):** union gate forget 0.333 at utility 0.565; window 0.314 / 0.559 (paired −0.019, p 0.0025).
- **Latency:** see A8. **TOFU highlight:** superseded, re-run queued (170).

**Worked.** Lesson: for the paper's own figures we reproduce DSG's qualitative claims. The differences show up only in
the tests the paper did not run.

---

## 9. What we would present

### The strongest findings (6–8 slides)
1. **A dozen lines of padding defeat DSG.** Dilution unblocks **91 %** of blocked WMDP-Bio questions (0.909 [0.875, 0.943],
   n 265), while the base model still answers them under the same padding (0.630). Benign-biology filler does not work,
   which shows the mechanism: the gate averages its features over the whole prompt. (B1, T2, N1; Break)
2. **Splitting a question across chat turns also works** (0.694 [0.642, 0.751]). Translation into 8 languages does *not*
   (≤ 0.054): Gemma Scope's features are multilingual. (B2, B3; Break)
3. **The gate statistic is fixable.** A CUSUM detector cuts dilution success from 0.853 to **0.029** at matched clean
   behaviour. It was selected by the fixed DEV rule with utility cost 0.0002 and FPR 0.045. Cost: benign FPR under padding
   rises to 0.12. (C2, T2, X1; Interp)
4. **DSG over-blocks biology and does not look like ignorance.** It blocks 59 % of benign biology questions (accuracy 0.654 →
   0.414). It answers open hazardous questions with gibberish 60 % of the time. On TOFU it overshoots (truth ratio 2.33 vs
   0.89 for a model that never learned). (A2, A3, B6; Eval)
5. **"Hidden, not erased", stated precisely.** Removing the hook restores base accuracy. A properly tuned RMU matches DSG's
   forgetting (0.319 vs 0.298, n.s.) yet recovers to base after 200 fine-tuning steps (+0.14 to +0.16). Distilled students
   resist better (D1 v2 +≤ 0.05) but cost 2.7–5 points of utility. (A6, RMU v2, D1; Bake)
6. **Detecting is not removing.** Every layer detects hazardous prompts (AUROC ≥ 0.976), but clamping at layers 20/24 changes
   nothing. Gate misses are explained by SAE reconstruction error (coefficient +0.53 [+0.09, +1.08]). (C3, N9; Interp)
7. **The cost claim holds.** Latency +1.4–3.1 %; MT-Bench unchanged (7.46 vs 7.36, n.s.) because the gate almost never
   fires there. (A8, BM3; Eval)
8. **Infrastructure as a contribution.** Bit-exact reproduction, DEV/TEST discipline, paired tests on every comparison, two
   hardware baselines, and an audit trail that caught six silent bugs before they reached the paper. (Infra)

### The honest negatives (say them before a reviewer does)
- **C-H3 is not supported**: probes on the guarded activations are at chance. Our "hidden" evidence comes from the weights
  (hook removal, relearning), not from the activations.
- **DSG does nothing on Gemma 3 (1B/4B/12B)**, and we have not yet found out why. Generality (C-H7) is open.
- **D2's closed-form edit does not forget**, and the baked models keep DSG's features active (D3). No baked method matches
  DSG's forgetting at DSG's utility.
- **Several fixes gave nothing:** attribution features, mean ablation, per-feature clamps, LLM rewrites (B4), suffix
  optimisation (B5).
- **Cyber:** no DSG configuration stays within 1 point of base utility.
- **T1's dilution formula fits poorly** (R² −0.94); we present it as a bound only.
- **Full-scale TOFU and MUSE results do not exist yet.** Both were invalidated by our own bugs and are queued again.
- **Cross-GPU non-reproducibility**: 158 vs 161 of 538 on two GPUs for the same configuration. This is a caveat for DSG
  and for anyone comparing numbers across papers.

---

## 10. What is still running or waiting (as of 2026-10-06 10:45)
- **Lab PC:** X1 TEST (2/33, decides C-H5), then A5, C4, D3 probes, N2/N3/N4/N8; re-queue T3, T5, A8-tables and N10-cards at
  the end (they ran on smoke data). Lab ETA about 1 day.
- **Server:** a6-baked (151 running; 152 and 160–163 behind another user's jobs) → TOFU-full v3 (167–169) → FP-highlight
  (170) → MUSE v2 (171–173; needs `cleanup a6-baked` first) → Q2 v2 (automatic watcher).
- **A7 diagnosed** (session 17): clamp scale vs Gemma 3's residual norm, a finding (no re-run).
- `python -m dsgx.analysis.results_digest` after each of these; the claims table updates itself.
