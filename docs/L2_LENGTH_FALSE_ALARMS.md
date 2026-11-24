# L2: false alarms against prompt length, frozen StreamGuard v2 and DSG (measurement only)

**Summary.** On harmless BBC news text, v2 fires on 15.7 % of 64-token prompts, 52.1 % at 1,024 tokens and 74.3 % at 2,048 tokens; DSG fires on at most 0.6 % at every length.
The rise is far above v2's 1.9 % MMLU false-alarm rate, already at 64 tokens, so length is not the only factor (text and prompt format also differ from MMLU): the MMLU rate does not carry over to long free text.
The CUSUM drift delta = 0 is the frozen setting and stays; any fix (drift, length-aware threshold, calibration on long text) would be a new version and is future work.

This is a descriptive measurement, not a pass/fail test. No v2 or DSG setting, threshold or feature was changed, and nothing
is tuned on it. It does not change any earlier result or verdict; the reported 0.019 remains the rate on plain MMLU questions.
One RTX 2000 Ada, batch size 1, bfloat16, gate decisions only, harmless text only. Reproduce: `scripts/length_false_alarms.py run` then
`report` (about 20 min). Design logged before any score.

![fire rate against length](L2_LENGTH_FALSE_ALARMS.png)

## 1. Which text is unused by v2 (check)

| text | used by v2? | where |
|---|---|---|
| WikiText-2 test | **yes** | feature selection and CUSUM weights of the layer-8 gate (fit d4ff0a1b) and DSG's features and threshold |
| WikiText-103 test | **yes, in effect** | not on disk, but the same 60 articles and token count as WikiText-2 test (Merity et al., 2016; not compared byte by byte here) |
| WikiText-2 train | **yes** | filler inside the padded DEV prompts among the 11,582 whose scores were searched for the threshold midpoint (the binding score is an MMLU prompt); also the latency run's filler |
| WikiText-2 validation | not by v2 | used to augment training of other response models; too small (0.25 M tokens, about 120 passages of 2,048) |
| benign-biology documents (bio-retain-corpus) | **yes** | feature selection and CUSUM weights |
| MMLU DEV questions, benign-biology MMLU subjects (DEV) | **yes** | threshold calibration (960 prompts) / DEV gate checks and midpoint |
| **MUSE-News, BBC articles (raw retain1, retain2, holdout)** | **no** | chosen: the only large general-text set on disk that no v2 step touched |

MUSE-News articles are ordinary news ("retain" and "holdout" are that benchmark's copyright-unlearning names). 13-word overlaps between
the 5,889 articles and v2's harmless corpora: WikiText-2 test 0, train 1, validation 0, benign-biology pool 1, all MMLU questions 0.
The forget corpus and other private material were not opened for this check.

## 2. Prompts

- 6,598 articles, 5,889 after dropping exact duplicates and shared first 200 characters; shuffled (seed 0), packed whole, joined by a blank line, into 1,621 passages of at
  least 2,048 tokens. No article is in two passages. 1,500 passages used = 5 disjoint random draws ("seeds") x 300.
- Each passage is cut to exactly 64 / 256 / 512 / 1,024 / 2,048 tokens **including** 9 template tokens
  (`<bos><start_of_turn>user\n … <end_of_turn>\n<start_of_turn>model\n`, the latency run's written-answer template, no question
  and no `Answer: (`). The same passage is used at every length. The longer cuts usually span several articles.
- The gates are deterministic, so the 5 draws vary the prompts, not the method.

## 3. Results

Fire rate = mean of the 5 draws with a 95 % bootstrap CI (10,000 resamples of passages within each draw). Paired test: exact McNemar over the
1,500 passages, and per draw. A CI of [0, 0] means no fires; the exact 95 % upper bound for 0 of 1,500 is 0.0025.

| tokens | n (5 × 300) | v2 fire rate | DSG fire rate | v2 − DSG | only v2 / only DSG | exact McNemar p (pooled) | p per draw |
|---|---|---|---|---|---|---|---|
| 64 | 1500 | 0.157 [0.139, 0.175] | 0.002 [0.000, 0.005] | +0.155 [+0.136, +0.173] | 234 / 2 | 5.1e-67 | 9.1e-13, 2.6e-12, 7.1e-15, 4.1e-16, 1.1e-13 |
| 256 | 1500 | 0.254 [0.231, 0.276] | 0.006 [0.003, 0.010] | +0.248 [+0.226, +0.270] | 379 / 7 | 3.1e-102 | 7.2e-18, 1.8e-18, 5.3e-23, 1.2e-25, 4e-21 |
| 512 | 1500 | 0.358 [0.335, 0.382] | 0.005 [0.002, 0.009] | +0.353 [+0.328, +0.377] | 534 / 5 | 4.2e-151 | 2.7e-28, 1.2e-27, 3.1e-33, 3.7e-34, 2e-31 |
| 1024 | 1500 | 0.521 [0.495, 0.545] | 0.000 [0.000, 0.000] | +0.521 [+0.495, +0.546] | 781 / 0 | 1.6e-235 | 4.5e-44, 1.1e-44, 6.8e-49, 5.3e-51, 1.4e-48 |
| 2048 | 1500 | 0.743 [0.721, 0.765] | 0.000 [0.000, 0.000] | +0.743 [+0.721, +0.765] | 1115 / 0 | < 1e-300 | 2.4e-66, 1.2e-63, 1.9e-68, 1.4e-70, 9.3e-69 |

| tokens | v2 score s(x), mean | s(x) 95th percentile | mean s / threshold | 95th pct. s / threshold | median s(x) | DSG rho, mean |
|---|---|---|---|---|---|---|
| 64 | 8.62 [8.17, 9.14] | 13.03 [12.52, 14.79] | 1.22 [1.15, 1.29] | 1.84 [1.77, 2.09] | 7.06 | 0.094 |
| 256 | 11.71 [10.01, 13.84] | 18.09 [14.72, 20.69] | 1.65 [1.41, 1.95] | 2.55 [2.08, 2.92] | 7.06 | 0.096 |
| 512 | 14.88 [12.26, 18.02] | 26.87 [23.21, 35.14] | 2.10 [1.73, 2.54] | 3.79 [3.28, 4.96] | 7.06 | 0.091 |
| 1024 | 21.53 [17.55, 26.18] | 47.74 [36.54, 62.19] | 3.04 [2.48, 3.70] | 6.74 [5.16, 8.78] | 8.24 | 0.085 |
| 2048 | 34.88 [28.13, 42.88] | 86.20 [64.85, 177.36] | 4.92 [3.97, 6.05] | 12.17 [9.15, 25.03] | 10.79 | 0.079 |

| tokens | v2 fire rate in each draw | DSG fire rate in each draw | prompts without health / biology words | v2 / DSG fire rate on those |
|---|---|---|---|---|
| 64 | 0.137, 0.147, 0.160, 0.190, 0.150 | 0.000, 0.003, 0.000, 0.003, 0.003 | 1375 of 1500 | 0.145 / 0.001 |
| 256 | 0.237, 0.217, 0.250, 0.317, 0.250 | 0.010, 0.003, 0.000, 0.010, 0.007 | 1139 of 1500 | 0.218 / 0.000 |
| 512 | 0.343, 0.323, 0.363, 0.410, 0.350 | 0.007, 0.007, 0.000, 0.007, 0.007 | 957 of 1500 | 0.299 / 0.000 |
| 1024 | 0.483, 0.490, 0.537, 0.560, 0.533 | 0.000, 0.000, 0.000, 0.000, 0.000 | 689 of 1500 | 0.438 / 0.000 |
| 2048 | 0.730, 0.700, 0.753, 0.777, 0.757 | 0.000, 0.000, 0.000, 0.000, 0.000 | 373 of 1500 | 0.635 / 0.000 |

Score reading: the frozen threshold is 7.0845 (tied floor 7.0620 + 0.0225); fire iff s(x) > threshold. Ratios are s(x) / 7.0845.

- **The floor.** The lowest score is 7.0620 at every length, the same value that 98.6 % of MMLU DEV questions share. 84 % of the 64-token
  prompts sit exactly on it, 26 % of the 2,048-token prompts. The threshold lies 0.02 above it, so a small extra excursion fires.
- **Running maximum.** v2's score is the maximum of a CUSUM with drift 0: longer text adds chances to exceed the floor. The
  mean score grows 4-fold from 64 to 2,048 tokens and the 95th percentile 6.6-fold (nested prefixes, so fires are almost
  always kept as the text grows; 11 of 1,500 passages un-fire when cut longer). DSG's score is a token *fraction* and falls instead (0.094 to 0.079).
- **Not only biology words.** On prompts with none of a fixed list of health / biology words (fixed before any score) v2 still fires
  on 14.5 % (64 tokens) to 63.5 % (2,048); on the others 28 % to 78 %. News that does mention health topics is still harmless, so those fires are false alarms too.

Checks (first 10 passages x 5 lengths): the scorer equals the frozen method object, scores repeat bit for bit, and DSG's gate with an early stop equals the full
forward: 0 mismatches of 50 each. DSG's threshold reproduces 0.5458; the fit sha256 and all features are the frozen ones.

## 4. Cross-check with the latency run (existing results, no new runs)

`docs/LATENCY_V2.md`, natural v2 decisions on 2,048-token harmless prompts: multiple choice 15 / 60 = 0.25 (exact 95 % CI
[0.15, 0.38]), written answer 16 / 60 = 0.27 ([0.16, 0.40]); 64 tokens: 2 / 60 and 1 / 60. The same direction (rising with length),
but L2's 0.743 is about three times higher. The two sets differ in text: the latency prompts are WikiText-2 train filler around an MMLU question,
while L2 is news with no question. WikiText is the source of v2's "rare feature" background corpus, so a lower rate there is plausible; this explanation was not tested.
The 60 latency prompts are partly shared between the two workloads.

## 5. Limits

One text source (British news), one model, one GPU, prompt-only gate, no check of what the clamp does to the answers; the rate on other text can differ in either direction.
Attack recall was not measured here. The delta = 0 setting was not varied; nothing here recommends a new threshold.
