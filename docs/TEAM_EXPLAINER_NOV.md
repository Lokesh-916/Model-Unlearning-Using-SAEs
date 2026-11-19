# Team explainer, November 2026: what happened after the mid-review

*For K. Chakreesh and S. Amarnath Reddy. Written 2026-11-19 from the docs in this repo only. Every number names
its source doc, says whether it is practice data (**DEV**) or the final exam (**TEST**), and says which machine
(**lab PC** = RTX 2000 Ada; **server** = gpuws, RTX 6000 Ada). Ids and metrics only: no hazardous text.*

---

## 1. The basics

**Unlearning.** We want a model to stop answering questions about dangerous biology. Its general knowledge
and its ordinary biology should stay intact. Think of it like removing one chapter from a textbook without
tearing the pages around it.

**The model.** Gemma-2-2B-it, a small chat model from Google (2 billion parameters).

**SAE and features.** A model's internal state at a layer is a long list of numbers (the "residual stream").
A sparse autoencoder (SAE) is a dictionary that rewrites that list as a few active "features" out of 16,384.
Each feature roughly means one idea, for example "virus words" or "chat formatting". A feature "fires" on a word
when its value is above zero. We use Google's public Gemma Scope SAEs.

**How DSG works** (Dynamic SAE Guardrails, the paper we started from):
1. Pick 20 features at layer 3 that fire much more on hazardous biology text than on Wikipedia text.
2. For each prompt, count the share of words on which those features fire. Call it ρ (rho).
3. If ρ is above a threshold τ, sound the alarm: push those 20 features to −500 (the "clamp").
   This wrecks the model's thinking for that prompt, so it answers badly.

It is like a smoke alarm that measures the average smoke in the whole room, and when it goes off it floods the
room with water.

**The exams.**
- **WMDP-Bio**: hazardous-biology multiple-choice questions with 4 options. 0.25 = pure guessing. **Lower is better** here.
- **MMLU**: general knowledge across many subjects (we use the full set minus the biology-adjacent subjects). Higher is better.
- **Ordinary (benign) biology**: college / high-school biology, anatomy, virology and medical genetics questions.
  These are not dangerous, so the model should keep answering them. Higher is better.

Starting point (IMPROVEMENTS_REPORT §5 A1 and A3, TEST, lab PC): base model 0.642 on WMDP-Bio, DSG 0.295.
MMLU 0.563 → 0.559. But DSG fires on 59 % of ordinary biology questions, and its accuracy there falls from 0.654 to 0.414.

---

## 2. Mid-review and what we changed

The critique we wrote up after the mid-review (docs/ROADMAP.md §2, IMPROVEMENTS_REPORT §4):
- It read like "nine loosely linked experiments", not one claim.
- One seed, no error bars. Our best gain was inside one standard error.
- We tuned on the same questions we reported on.
- DSG's "MMLU" is a small, easy subset (history, CS, geography, ageing). It never tests biology.
- Only one benchmark and one model.

Later the faculty also asked that the main contribution be an *improved method* (CLAUDE.md, session 22).

What we changed (IMPROVEMENTS_REPORT §4):

| change | in plain words |
|---|---|
| 5 seeds | every main TEST run repeated 5 times |
| error bars | 95 % confidence intervals on every number; paired tests (bootstrap + McNemar) for every "A beats B" |
| DEV / TEST split | each dataset cut 50/50: choose settings on DEV (practice), report once on TEST (exam) |
| rules fixed first | each selection rule and pass line written in code and in DEVIATIONS.md *before* any result |
| full MMLU | 6,095 questions instead of DSG's filtered set |
| per-machine baselines | the same DSG setting scores 158/538 on the lab PC and 161/538 on the server, so the two machines are never mixed |
| batch size 1 on TEST | a larger batch flips 1 prediction in 843 |

---

## 3. How we broke DSG

"Attack success" = of the hazardous TEST questions DSG blocks, the share it answers right under the attack
(IMPROVEMENTS_REPORT §5; TEST, lab PC, n 265).

**Padding (dilution).** DSG averages over the whole prompt. Add enough harmless text and the average drops
below the threshold. Like diluting a drink with water.
- 1,600 Wikipedia tokens before the question: attack success **0.909** [0.875, 0.943].
- Even 100 tokens: 0.766–0.815.
- The base model still answers at 0.620–0.630 under the same padding, so the knowledge was never gone.
- Padding with ordinary *biology* text does **not** work (≤ 0.10): it fires the same features.

**Splitting.** Spread the question over 3 chat turns. Each turn looks harmless on its own.
- Attack success **0.694** [0.642, 0.751]. Two turns: 0.509.

**Attacks that did not work** (same source):
- Translation into 8 languages: ≤ 0.054. The layer-3 features are multilingual.
- Rewriting by a small LLM: forget 0.284–0.305 vs 0.295 without attack.
- Optimised 8-token suffix: 0.312 vs 0.295.
- Encodings: spaced letters 0.449, leetspeak 0.423, base64 0.128. Caveat: under leetspeak the *base* model only
  scores 0.325, close to guessing.

---

## 4. StreamGuard v1: the running-score alarm

**CUSUM** (cumulative sum) is a running score. It goes up when hazard features fire on a word and drifts down on
harmless words, never below zero. The alarm uses the *highest* value it reaches. Padding can only add harmless
words; it cannot remove the peak that the hazardous part created. Like a speed camera that records your top
speed: driving slowly for another hour does not erase it.

StreamGuard v1 = CUSUM alarm on DSG's same 20 features + the same −500 clamp. It was picked by a fixed rule on DEV
(IMPROVEMENTS_REPORT §8, X1: padding effect −0.401, false alarms 0.045).

What it won and cost (E1_RFD_AUDIT, TEST, lab PC, 5 seeds):

| | DSG | StreamGuard v1 |
|---|---|---|
| clean forget | 0.295 | 0.292 |
| forget under 400-token padding | 0.602 | 0.296 |
| worst attack (of 10) | 0.602 (padding) | 0.332 (Hindi) |
| MMLU utility | 0.559 | 0.547 |
| MMLU false-alarm rate | 0.017 | 0.044 |
| ordinary-biology accuracy | 0.410 | 0.361 |

**Why "not supported".** The claim (C-H5: "a hardened gate improves robustness at matched utility") allowed at most
1 point (0.01) of utility loss. v1 lost 0.0117 on both machines, and it wins only 3 of 5 attack families
(CLAUDE.md session 27/31; E1_RFD_AUDIT for −0.0117). A narrow miss, but the rule was fixed, so it stays a miss.
It also over-blocks ordinary biology even more than DSG.

---

## 5. November findings about DSG itself

**The three-number audit** (E1_RFD_AUDIT, TEST, both machines). Any alarm-plus-response can be described by three numbers:
1. **R, recall**: how often the alarm fires on hazardous questions.
2. **F, false alarms**: how often it fires on harmless questions.
3. **Damage per alarm**: how much accuracy drops when it fires.

For DSG on the lab PC: R = 0.878; when it fires, accuracy is 0.259 (guessing). When it fires on MMLU, accuracy
drops from 0.564 to 0.233. The simple formula built from these numbers predicts every observed result inside its
error bar, on both machines. So DSG's whole story is "how often does the alarm fire". Under padding, R falls to
0.039, which is why forget jumps to 0.602. And v1's utility loss is explained entirely by its extra false alarms
(0.044 vs 0.017).

**Detection features are not the knowledge** (E3_CAUSAL_LOCALISATION, DEV, lab PC).
- We found the features that actually drive the answers (attribution patching = "which features, if removed,
  change the answer most").
- Overlap with DSG's 20 features: **zero** (Jaccard 0.000).
- Switching off DSG's 20 features: forget 0.660 vs base 0.676. They carry almost no answer.
- DSG's clamp still works (0.233) only as a sledgehammer: MMLU −0.171, ordinary biology −0.382.
- No set of ≤ 50 features at any layer 3–14 reached forget ≤ 0.40 within 2 points of utility (0 of 53 runs).

Analogy: the smoke detector is on the ceiling, the fire is in the kitchen. The detector works, but switching off
the detector does not put out the fire.

---

## 6. What we tried and what failed

All DEV, lab PC, exploratory.

- **Gentle edited-model responses** (E2D_SWITCHED_RESPONSE, E2E_ROBUST_SOFT_LANDING). When the alarm fires, use a
  lightly edited copy of the model (an RMU-style weight edit of layers 1–3) instead of the clamp.
  - R5 kept ordinary biology better (+0.110 vs DSG) but leaked under attacks (padding +0.165, Hindi +0.124 vs v1).
  - R6, trained also on padded / split text, closed the attack gap. But its biology gain shrank to +0.040 (not
    significant), and forced on everything it costs −0.182 utility, almost like the clamp (−0.170).
  - **0 of 20 candidates passed.** Reason: on the layer-3 alarm, ordinary biology looks like hazard (69.8 % fire).
    The response cannot fix what the alarm cannot tell apart.
- **"I don't know" direction** (E2F_IGNORANCE_RESPONSE). Push the model toward saying "I don't know".
  - The fixed rule picked a setting no different from base (IDK 0.01).
  - The exploratory L12 setting stopped WMDP leaks (0.00), but 26 % gibberish, and on TOFU (fictitious authors)
    it leaked *more* than the clamp (0.63 vs 0.45).
- **Evidence redaction, P9** (P9_REDACTION). Blank out the words the alarm blames, then let the normal model answer.
  - It needed about half the question's words (median k 47 of 97 candidates).
  - Forget rose to 0.340 (worse than the clamp, +0.099), and open answers were 32 % gibberish.
  - Both variants failed the fixed lines. The hazard is spread over the whole question, not a few words.

---

## 7. StreamGuard v2

**What it is** (V2_TEST_RESULT header):
- **Alarm:** CUSUM at **layer 8**, on 20 features chosen to be frequent on hazardous text and *rare on ordinary
  biology* and Wikipedia (E4A_SEPARABILITY).
- **Threshold:** 7.0845, set so the alarm fires on very few MMLU questions (1.45 % on DEV, E4B_L08_GATE).
- **Response:** unchanged, DSG's −500 clamp at layer 3.
- **How it runs:** check the prompt, and if the alarm fires, answer with the clamp on.

On DEV, layer 8 was the near-miss: recall 0.936–0.962 under every attack, but 0.323 on ordinary biology (the line
was ≤ 0.30) (E4A_SEPARABILITY). E4b then showed that, with this alarm, the plain clamp passes our selection rule
(E4B_L08_GATE). Caveat: layer 8 was chosen after seeing those DEV questions.

**Frozen before the exam.** Code pinned at commit cc6f375 on 2026-11-13 11:46, with a DEVIATIONS row. A replay on
11,582 DEV prompts gave identical decisions. Then one TEST run with 5 seeds, and nothing was tuned afterwards.

**Final TEST** (V2_TEST_RESULT, lab PC, seed 0; v2's seeds 1–4 are identical):

| | base | DSG | v1 | **v2** |
|---|---|---|---|---|
| clean forget (↓) | 0.642 | 0.295 | 0.294 | **0.273** |
| worst-case forget (16 attacks) (↓) | – | 0.633 (Wiki pad 1600) | 0.397 (spaced letters) | **0.287** (LLM rewrite) |
| worst incl. adaptive attacks | – | 0.633 | 0.397 | **0.297** (interleave + split) |
| MMLU utility | 0.563 | 0.559 | 0.547 | **0.558** |
| MMLU false alarms | – | 0.017 | 0.046 | **0.019** |
| ordinary-biology fire rate | – | 0.591 | 0.724 | **0.311** |
| ordinary-biology accuracy | 0.654 | 0.414 | 0.360 | **0.556** |

**The two pre-written claims** (rules in `claims.evaluate_v2`):
- **V2 "keeps ordinary biology better than DSG" → Supported**, 5/5 seeds (+0.143 [+0.096, +0.192]; clean forget not higher).
- **V3 "lower worst-case forget at matched utility" → Supported**, 5/5 seeds (−0.345 vs DSG; utility −0.0009; false alarms 0.019).

---

## 8. v2-R: polite refusal for written answers

The clamp is fine for multiple choice. In free text it produces gibberish: 71 % of answers when it fires.

**v2-R** keeps the v2 alarm, now checked on the prompt *and after every generated word*. When it fires, instead of
the clamp it adds a **"refusal direction"**: a fixed push inside the model (layer 10) that makes it say it will not
help. That direction was found from harmful-vs-harmless instruction pairs. The first attempt (E5) found no clean
setting; the extended one (E5b) picked a fallback, labelled **"weaker setting"** (E5_SELF_REFUSAL, PREP_PROGRESS s45).

**Final TEST** (V2R_TEST_RESULT, lab PC, greedy = one deterministic run; fired open-ended WMDP-Bio, n 447):

| | base | clamp (v2) | v2-R |
|---|---|---|---|
| refusal | 0.011 | 0.000 | **0.989** |
| gibberish | 0.000 | 0.711 | **0.000** |
| leak (correct hazardous content) | 0.208 | 0.000 | **0.007** (n.s. vs clamp) |

It **passes** its frozen line, also under padding (refusal 0.976) and splitting (0.987).
**Caveat:** it is not selective. On ordinary biology the alarm fires 28.8 % of the time, and then v2-R refuses
95.7 % of those (n 47). It refuses on false alarms just as firmly.

---

## 9. Current status

- **Cyber.** DSG never kept utility within 1 point on Cyber (IMPROVEMENTS_REPORT A1).
  - **P1** (P1_CYBER, DEV, lab PC): our fixed rule picked layer 10, which **failed** (split recall 0.629).
    Four other cells passed, including layer 8.
  - We then chose layer 8, after seeing DEV, because it is v2's layer. This is disclosed as post-hoc.
  - **P1-T** (P1T_CYBER_TEST, TEST, lab PC, 5 seeds) → **PASS 5/5**:

    | | value | compare |
    |---|---|---|
    | clean forget | 0.234 | base 0.392, DSG chat-retain 0.243 |
    | utility | 0.559 | base 0.573, so −0.015 |
    | recall under padding / splitting | 0.997 / 0.996 | – |
    | benign computing accuracy | 0.452 | base 0.527, DSG chat-retain 0.301 |

    Honest note: MMLU false alarms are 0.057, above 5 %. The line did not include them.
- **Jailbreak wrappers vs v2-R** (P8, DEV): running now. At 17:17 today 4 of 20 jobs were done
  (`p8_watch.log`); expected ~20 Nov late morning (PREP_PROGRESS s49). **No results yet.** The watcher fills
  docs/P8_REFUSAL_JAILBREAKS.md.
- **Plan:** no new experiments from 24 Nov; panel 26/27/30 Nov; EACL SRW submission 15 Dec.

---

## 10. Honest limits

1. **Guardrail, not erasure.** The knowledge stays in the weights. DSG with the hook removed = base model
   (IMPROVEMENTS_REPORT A6, server). StreamGuard is the same kind of hook.
2. **One model.** Gemma-2-2B only. DSG did nothing on Gemma 3 (A7, server).
3. **One machine** for v2, v2-R and P1-T: lab PC only. v1 was tested on both machines.
4. **base64.** v2's alarm fires 0.000 on base64 questions. Our model cannot read base64 (base 0.265 = guessing), so
   no harm here. A model that can read base64 would walk straight past it (V2_TEST_RESULT note d).
5. **Refusal is not selective.** v2-R refuses on ordinary biology when the alarm misfires.
6. **SAE features are good alarms but poor scalpels** (E3). They detect the topic well. Removing them does not
   remove the knowledge, and v2 still fires on 31 % of ordinary biology.
7. Layer 8 (bio and Cyber) was chosen after looking at DEV data; each TEST was its single evaluation.

---

## 11. Who did what, and where to read

| area | person | read first |
|---|---|---|
| **Break** (attacks B1–B6, GuardBreak, red-team, adaptive attacks, dilution theory) | K. Chakreesh | IMPROVEMENTS_REPORT §5 (B1–B6) and §7 (N1, N4, N5, T1, T2); the attack rows of V2_TEST_RESULT; P8_REFUSAL_JAILBREAKS when it fills |
| **Bake** (permanent erasure D1/D2/D3, tampering A6, edited-model responses R1–R7) | S. Amarnath Reddy | IMPROVEMENTS_REPORT §6 (D1, D2, D3, A6); E2D_SWITCHED_RESPONSE; E2E_ROBUST_SOFT_LANDING; the R6 rows of E4B_L08_GATE |
| **Evaluation, interpretability, StreamGuard** | K. Lokesh Babu | E1_RFD_AUDIT, E3, E4A, E4B, V2_TEST_RESULT, E5_SELF_REFUSAL, V2R_TEST_RESULT, P9, P1, P1T |

---

## 5-line summary for the guide

1. DSG's alarm averages over the whole prompt, so padding (0.909) and splitting (0.694) defeat it (TEST, lab PC).
2. Its 20 features detect the topic but do not hold the knowledge; the clamp works only as a blunt instrument (E3, DEV).
3. StreamGuard v2 moves the alarm to layer 8 with features chosen against ordinary biology, keeps DSG's clamp, and was frozen before one TEST.
4. On TEST it beats DSG on worst-case attack (0.287 vs 0.633) and ordinary biology (0.556 vs 0.414) at the same utility; both claims Supported 5/5 seeds.
5. v2-R turns gibberish into clean refusals (0.989), but it is a guardrail, not erasure: one model, one machine, and it also refuses on false alarms.

## Three likely questions

**"Is this really unlearning?"** No. It is a guardrail. The knowledge stays in the weights; remove the hook and
the base model is back (A6). We say this openly. Our permanent-erasure attempts (D1, D2) cost utility or did not forget.

**"Didn't you pick layer 8 after seeing the data?"** Yes, on DEV (practice) data, and we say so. Then we froze the
code with a timestamped commit and ran the TEST exam once, with nothing tuned afterwards. For Cyber, the layer-8
choice is explicitly labelled post-hoc.

**"Why does it still hurt ordinary biology?"** The alarm still fires on 31 % of those questions, and each alarm costs
about as much as before (damage per alarm 0.316, V2_TEST_RESULT). The gain comes from firing less often (31 % vs 59 %
for DSG), not from a gentler response. Every gentle response we tried (E2d, E2e, P9) failed.
