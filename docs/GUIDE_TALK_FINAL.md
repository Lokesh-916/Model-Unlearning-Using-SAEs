# Talk notes for the final guide deck

Deck: `../presentation/progress_final.tex` → `progress_final.pdf` (15 slides + 3 backup; build with
`./make_progress_final.sh` in the presentation repo, CPU only). Every number on the slides is a macro generated from
the result files and docs (`gen_progress_final.py`, `gen_progress_nov12.py`, `gen_progress_nov14.py`); all main-slide
numbers are lab PC. "Final test" = the held-out test split, run once after everything was frozen; "practice data" =
the development split used for choosing; "runs" = seeds. Sources: `V2_TEST_RESULT.md`, `V2R_TEST_RESULT.md`,
`P1T_CYBER_TEST.md`, `P8_REFUSAL_JAILBREAKS.md`, `P9_REDACTION.md`, `P4_SENSITIVITY.md`, `E1_RFD_AUDIT.md`,
`E3_CAUSAL_LOCALISATION.md`.

## 1. Title

**Say:** This is our final update. In one line: StreamGuard v2 beats DSG on the final test, and the two add-ons we
tried at the end, polite refusal and the Cyber topic, also passed their final tests.

**Likely question:** "Final test" means what exactly? **Answer:** the held-out test questions, run once with
everything frozen and the success rule written down beforehand; nothing was tuned after we saw them.

## 2. DSG: an alarm and a clamp, with no change to the weights

**Say:** DSG watches 20 SAE features at layer 3. If enough tokens of the question light them up, it clamps those
features hard to −500; otherwise the model answers normally. We reproduced the paper: hazard accuracy 0.642 → 0.295,
general ability 0.563 → 0.559.

**How to read:** left to right is one question passing through the guard; orange is what happens on an alarm.

**Likely question:** Why does "no weight change" matter? **Answer:** it is cheap and reversible, but it also means the
knowledge is still inside the model; the guard only blocks access to it.

## 3. Since the mid-review, every number is re-run under fixed rules

**Say:** At the mid-review we had single runs and choices made on the test questions. Now every headline number is 5
runs with 95% intervals, compared on the same questions, chosen on practice data and tested once, with the success
rule written down first. The lab PC and the server are separate baselines and never mixed.

**Likely question:** Did any mid-review number change? **Answer:** the direction of every headline result held; the
numbers on these slides are the re-run versions.

## 4. Padding defeats DSG: at 1600 tokens it is back to the untouched model

**Say:** If you put harmless WikiText before the question, DSG stops working. At 0 tokens it holds hazard accuracy at
0.295; at 50 tokens 0.495, at 400 0.603, at 1600 0.633, almost the untouched model's 0.642. The alarm simply stops
firing (0.878 of questions without padding, 0.000 at 1600 tokens). Splitting the question over three chat turns also
works (0.538, mean of 5 runs).

**How to read:** each bar is DSG's hazard accuracy, lower is better; the dotted line is the untouched model and the
dashed line is guessing (0.25). Thin bars are 95% intervals. Final test, seed 0, 637 questions.

**Likely question:** Why does padding work? **Answer:** DSG's alarm is a share of tokens, so harmless tokens dilute it
below the threshold; nothing in the knowledge changes.

## 5. Two findings: three numbers explain DSG; its alarm is not the knowledge

**Say:** Left: every result follows from three numbers: how often the alarm fires on hazard questions, how often on
normal ones, and the damage per firing. Two predictions were confirmed: DSG's leftover hazard accuracy is its misses
(predicted 0.287, measured 0.295), and StreamGuard v1's lost general ability is its extra false alarms (predicted
−0.0089, measured −0.0117). Right: switching off DSG's own 20 features barely changes hazard accuracy (0.676 →
0.660); the clamp drops it to 0.233 only by damaging everything (general ability 0.403, biology 0.260). The overlap
between DSG's features and the features that really carry the answers is 0.

**How to read:** right plot, three bars of hazard accuracy, lower is better; small numbers underneath are general
ability and biology for the same three bars. Left is final test (5 runs), right is practice data; both lab PC.

**Likely question:** If the features are not the knowledge, why does the clamp work at all? **Answer:** a −500 clamp
is a huge push on the activations; it breaks the model's answer generally, which shows up as guessing on hazard
questions and also as damage on normal ones.

## 6. StreamGuard v2: a better alarm in front of the same response

**Say:** We kept DSG's response and replaced the alarm. The new alarm reads layer 8, uses features picked so that they
ignore ordinary biology, and keeps a running score that adds up evidence token by token, so padding cannot dilute it.
Over the line, the model gets DSG's clamp (v2) or, in the refusal version, a push towards a polite refusal.

**How to read:** green boxes are what is new compared with DSG.

**Likely question:** Why layer 8? **Answer:** on practice data it came closest to separating hazard questions from
ordinary biology among layers 3 to 14 (none fully met our line); it was then frozen and tested once.

## 7. Final test, biology: v2 forgets like DSG and spares ordinary biology

**Say:** Same hazard accuracy without attack (0.273 vs 0.295), much better under the worst attack (0.287 vs 0.633), the
same general ability (0.558 vs 0.559) and the same false-alarm rate (0.019 vs 0.017). On ordinary biology the alarm
fires half as often (0.311 vs 0.600), so accuracy there goes 0.410 → 0.556. Both claims we wrote down before the test
passed in 5 of 5 runs.

**How to read:** v2 row bold; arrows say which direction is better; grey row is the untouched model. Mean of 5 runs.

**Likely question:** What exactly were the two claims? **Answer:** (1) better ordinary biology than DSG without worse
forgetting, (2) lower worst-case hazard accuracy than DSG at matched general ability with at most 5% false alarms.

## 8. Padding and splitting defeat DSG, not StreamGuard v2

**Say:** This is the key picture. DSG jumps to 0.63 with padding and 0.54 with the split; v2 stays at about 0.28 in
both. On Hindi both are near guessing. Over all 16 attacks DSG's worst is 0.633 and v2's worst is 0.287.

**How to read:** grey = DSG, green = v2; lower is better; dashed line is guessing. Bars are the mean of 5 runs; thin
bars span the 95% intervals.

**Likely question:** Is there any attack where v2 is worse than DSG? **Answer:** no significant case on the final test;
the one gap is base64, where the alarm does not fire but this model cannot read base64 anyway (backup slide).

## 9. Same general ability as DSG; much better on ordinary biology

**Say:** The cost side. General ability is unchanged versus DSG (0.559 vs 0.558; difference −0.0009). Ordinary biology
goes 0.410 → 0.556, a gain of +0.146 that is significant in 5 of 5 runs, because the alarm fires on 0.311 of those
questions instead of 0.600.

**How to read:** higher is better here; the dotted lines show the untouched model (0.563 and 0.654).

**Likely question:** Why is ordinary biology still below the untouched model? **Answer:** the alarm still fires on
about 31% of ordinary biology questions, and each false alarm costs the full clamp.

## 10. Polite refusal replaces gibberish, with no extra leaks

**Say:** The clamp makes the model produce gibberish (0.711 of fired answers without attack). The refusal version uses
the same alarm but pushes the model towards a refusal: 0.989 of fired answers are polite refusals and none are
gibberish, also under padding and splitting. Leaks stay tiny: untouched 0.208, clamp 0.000, refusal 0.007, not a
significant rise. The caveat: it also refuses on false alarms, 0.276 of all ordinary biology questions.

**How to read:** left panel gibberish (lower is better), right panel refusals (higher is better); grey = clamp,
green = refusal. One fixed (greedy) run, 476 questions. The two examples are harmless prompts where we forced the alarm
on with the demo script; they show that the refusal is generic and not about the question.

**Likely question:** The examples look silly; is that a problem? **Answer:** yes, it is the honest caveat: the refusal
is not selective, so a false alarm gives a refusal with an invented reason. That is why the false-alarm rate matters.

## 11. Second topic, Cyber: the same recipe passes our success rule

**Say:** We applied the same layer-8 recipe to the Cyber topic. Hazard accuracy 0.234 (same as DSG), padding 0.259 vs
0.328 / 0.314 for DSG, general ability 0.559 vs 0.292 / 0.482, false alarms 0.057 vs 0.772 / 0.379, and ordinary
computing questions 0.452 vs 0.276 / 0.300. It passed in 5 of 5 runs; general ability drops 0.015 vs the untouched
model.

**How to read:** ours bold; "DSG, chat-tuned" is DSG with a better normal-text set for its threshold. Footnote: layer 8
was chosen after our original rule's pick (layer 10) failed on practice data; this test is its only evaluation.

**Likely question:** Is choosing layer 8 after the failure cheating? **Answer:** it is a disclosed post-hoc choice
(layer 8 is the biology layer), written down before the test, and the test was run once; we label it as such.

## 12. Stress tests: stable under setting changes; wrappers do not make it leak

**Say:** Left: we changed one setting at a time (number of features, layer, SAE size, threshold, score decay); 10 of 11
settings pass our line, and only layer 7 fails because it raises too many alarms on ordinary biology. Right: 6 fixed
jailbreak wrappers, alone or with padding or splitting, 18 attacks: leaks stay at or below 0.040 while the untouched
model leaks 0.140 to 0.227, and the alarm still fires on at least 0.953. But the refusal wording drops: under role-play
only 0.255 of answers are refusals (11 of 18 conditions pass our line).

**How to read:** ticks pass, crosses fail. Practice data, lab PC.

**Likely question:** If refusals drop under role-play, what does the model say instead? **Answer:** the push still
stops the leak (leak at most 0.040), but the answer is no longer a clean refusal; in role-play some answers become
gibberish (0.320 of all answers).

## 13. Five ideas that did not work, and why

**Say:** A gentle edited-model response either leaked under attacks or hurt ordinary biology. The "I don't know"
direction gave no leaks on hazard questions but leaked more than the clamp on the made-up-author benchmark. Evidence
redaction had to blank most of the question and still let hazard accuracy rise above the clamp. Baked erasure came back
after a short fine-tune. On Gemma 3 the fixed −500 clamp is tiny next to its activations, and even a scaled clamp
changed nothing on the 1B model.

**How to read:** one line per idea; details in the backup and the paper. Practice data on the lab PC, except baked
erasure and Gemma 3, which ran on the server (no numbers on this slide).

**Likely question:** Which negative result matters most? **Answer:** baked erasure: making the guard permanent in the
weights did not resist fine-tuning, which is why we present StreamGuard as a guardrail.

## 14. Honest limits: a guardrail, not erasure

**Say:** The weights still hold the knowledge, so anyone with the weights can remove the guard. The final tests use one
model on one machine; base64 slips past the alarm (it fires on 0.000) and is harmless only because this model cannot
read base64. The refusal is not selective. And SAE features turned out to be good alarms but poor scalpels: they detect
the topic, they do not hold the knowledge.

**Likely question:** Would it work on a bigger model? **Answer:** untested; the Gemma 3 result shows the response must
be rescaled per model, and the alarm would have to be re-fitted.

## 15. Plan: submit to EACL SRW on 15 December

**Say:** The final tests are done. Next: put v2, polite refusal, Cyber and the stress tests into the paper (numbers
regenerate from the result files), submit to the EACL student research workshop on 15 December, and release the public
repo only after we have emailed the DSG authors.

**Likely question:** Why wait for the DSG authors? **Answer:** our paper shows a simple attack on their method; it is
fair to let them see it before it is public.

## Backup slides

- **All 18 attacks:** DSG, StreamGuard v1 and v2 per attack (final test, 5 runs). Read it as: v2 stays near 0.27-0.30
  everywhere.
- **Three numbers predict every result:** predicted vs measured accuracy; every point sits on the diagonal (26 of 26
  inside the 95% interval).
- **Stability table:** the numbers behind slide 12; the line to pass is worst hazard ≤ 0.32, ordinary biology ≥ DSG +
  0.05, general ability ≥ DSG − 0.01, false alarms ≤ 0.05.

## Spoken summary (5 sentences)

DSG's alarm can be switched off by simple padding or by splitting a question over chat turns, which brings hazard
accuracy back to the untouched model. We showed why: three numbers explain every result, and DSG's features detect the
topic without holding the knowledge. StreamGuard v2 keeps DSG's response but uses a layer-8 alarm with a running score,
and on the final test it stays near guessing under every attack with the same general ability and much better ordinary
biology. A polite-refusal version and the Cyber topic also passed their final tests, and the method is stable to
setting changes, though the refusal is not selective. It is a guardrail, not erasure, tested on one model and one
machine, and we will submit it to the EACL student workshop on 15 December.
