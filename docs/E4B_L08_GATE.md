# E4b: the layer-8 gate in front of the gentle response (DEV only)

Lab PC (labpc, RTX 2000 Ada), gemma-2-2b-it, DEV split only (never TEST). Exploratory; not a claim input. Ids and
metrics only. Branch `exp/E4b-l08-gate`: f3f8efa = rule (before any result), 6133da9 = Step A
(clean tree, `runs/E4b/stepA/config.json`), 2a4a6f2 = Step B (queue jobs pinned to it). Design, Step 0 rule, candidates and Step B attacks logged in
DEVIATIONS (2026-11-12, E4b row) before any E4b result. **E4a's verdict line was not met** (L08 benign-biology fire
0.323 > 0.30); at the user's request E4b tests that gate against **selection rule v2 anyway; rule v2 is unchanged**.

## Answer (Step A; Step B status below)

**Rule v2 selects L08 op5 → clamp.** Four of six candidates are eligible: L08 → clamp and L08 → R6, at both operating
points. L08 → R5 fails (attacks).

| candidate | worst-case forget | utility | MMLU fire | benign biology (vs DSG) | Hindi forget vs StreamGuard v1 |
|---|---|---|---|---|---|
| **L08 op5 → clamp (selected)** | **0.247** | 0.569 | 0.0145 | 0.518, +0.087 [+0.042, +0.133], p 0.0004 | −0.061 [−0.094, −0.029] ↓ |
| L08 op1.7 → clamp | 0.250 | 0.569 | 0.0115 | 0.539, +0.108 [+0.061, +0.155] | −0.054 [−0.088, −0.022] ↓ |
| L08 op5 → R6 | 0.282 | 0.570 | 0.0145 | 0.532, +0.101 [+0.052, +0.150] | −0.022 [−0.064, +0.016] |
| L08 op1.7 → R6 | 0.287 | 0.571 | 0.0115 | 0.546, +0.115 [+0.066, +0.164] | −0.018 [−0.057, +0.021] |
| L08 → R5 (both) | 0.434 | 0.573 | – | 0.574 / 0.578 | +0.129 ↑ (also pad400 +0.171 ↑, split3 +0.075 ↑) |
| *refs:* DSG / StreamGuard v1 | 0.649 / 0.305 | 0.567 / 0.556 | 0.017 / 0.045 | 0.431 / 0.398 | – |

- **The gate, not the gentle response, closes the gap.** L08 fires on 95–96 % of forget prompts under every one of the
  four rule attacks, including Hindi (0.936 vs StreamGuard v1's 0.620). It fires on only 32 % of benign biology
  (StreamGuard v1: 70 %) and 1.45 % of MMLU (StreamGuard v1: 4.5 %). With that gate, the plain −500 clamp already beats
  DSG on benign biology and matches or beats StreamGuard v1 on every attack. The softer responders do not lower forget
  further: R6 is a little gentler per fire (utility damage 0.23 vs clamp 0.34), but its Hindi forget is higher and its
  worst case is 0.28 vs 0.25. R5 still leaks under pad400 / split3 / Hindi, as in E2d/E2e.
- **Why it can pass where E2e could not:** in E2e, benign-biology prompts scored like forget prompts on the layer-3
  CUSUM (69.8 % fire), so no response could separate them. L08's features were chosen to be rare on benign-biology
  documents, which moves the separation into the gate.
- **Two operating points, small difference.** op1.7 is slightly kinder to benign biology (+0.021) and slightly weaker on
  Hindi recall (0.920 vs 0.936). Rule v2's tie-break (lowest worst-case forget) picks op5.

**Caveats (read before using this result).**
1. **Selection on DEV.** E4a chose L08 as its closest candidate after seeing these same DEV questions (37 layer ×
   detector cells). Step A is therefore optimistic (winner's curse). The DEV verdict needs a TEST confirmation that
   E4b did not run (DEV only, as instructed).
2. **The "5 %" point is really 1.45 %.** The L08 score has a tie floor (Step 0), so no deterministic threshold gives
   a 5 % MMLU fire rate. Both operating points are stricter than StreamGuard v1's 4.5 %.
3. **Benign biology is still hit a third of the time.** Damage per fire on benign biology stays 0.34–0.38 with the clamp.
   The gain over DSG comes from firing less often, not from a gentler response.
4. Utility includes the 960 calibration prompts. Without them (n 5116): L08 op5 → clamp 0.567 vs DSG 0.565 vs
   StreamGuard v1 0.554. Same ordering.

## Step 0: threshold

E4a's 95th-percentile threshold sat on a tie. 5988 of 6076 MMLU-dev scores (98.55 %) equal 7.0620, the chat-template floor
of the CUSUM. None are lower, and only 88 distinct values exist. A lower threshold would fire on every prompt, so its
MMLU fire rate was 0.014. Fixed rule (DEVIATIONS, before any result): tie-aware split conformal on the 960 MMLU-dev
calibration prompts StreamGuard v1 was calibrated on. t_alpha = the smallest distinct calibration score t with
(1 + #{s_i > t}) / 961 ≤ alpha; fire iff score > t; no randomised tie-breaking, because that would fire at random on the
tied 98 %. These are the only two operating points:

| operating point | alpha | t | calibration prompts above t | fire share, all MMLU utility dev |
|---|---|---|---|---|
| op5 (conformal 5 %) | 0.05 | 7.0620 (= E4a's threshold) | 19 / 960 | 0.0145 |
| op1.7 (DSG's own rate) | 0.017 | 8.9226 | 15 / 960 | 0.0115 |

Prompt identity: E4a's StreamGuard v1 reference score reproduces the E2d StreamGuard v1 run's gate decision on
100 % of prompts (clean 7139, pad400 636, split3 636, Hindi 627). So E4a's L08 scores and the E2d/E2e predictions come
from the same prompts, and the composites are exact.

## Step B: attacks a benign-biology-quiet gate could open

**Status: QUEUED (lab PC), not run yet.** Jobs `E4b-smoke-*` (6 items each) and then `E4b-000..004`, `E4b-scores`,
`E4b-stepB` wait for every E2f job (E2f-gen-wmdp was running at 13:52; queue ETA then ~8 h). Commit 2a4a6f2.

Design (fixed in DEVIATIONS): WMDP-Bio dev (636) under
- B1 dilution with the benign-biology filler pool, 400 and 1600 tokens before the question (median prompt ~530 / ~1730 tokens);
- B7 mid-question padding, 400 WikiText tokens;
- B7 interleaving, k = 2 WikiText tokens between question tokens.

These are B7's definitions verbatim; the benign-biology variants of midpad and interleave were not run. Per prompt:
L08 / DSG rho / StreamGuard v1 scores (`E4b-scores`), plus runs base (control), DSG, StreamGuard v1, clamp forced-on
and R6 forced-on. Reported: recall and gated forget of L08 → clamp / R6 at both operating points, paired vs DSG and
StreamGuard v1. "Opened" = an L08 composite significantly above StreamGuard v1. Step B does not change the rule v2
verdict (the rule's attack set is fixed).

Finish (no Claude needed): after `E4b-stepB` is DONE, paste `$DSG_RESULTS/runs/E4b/stepB/E4B_STEPB.md` here, replacing
this status paragraph, and add one line to the Answer section.

## Step A tables (generated: `python -m experiments.E4b.stepa`, copied from `runs/E4b/stepA/E4B_STEPA.md`)

Hardware: labpc. DEV split only, bs 1, identical items; composites are exact per-prompt recombinations of forced-on predictions. 95 % bootstrap CIs (n = items); paired = composite − comparator, paired bootstrap [95 % CI] and exact McNemar p; ↑ / ↓ = significant (CI excludes 0 and p < 0.05).

### Step 0: L08 SAE-CUSUM score on MMLU utility dev

n 6076; 88 distinct values; 5988 (98.55 %) equal 7.0620, 0 are lower. Quantiles: q0.5 7.062, q0.9 7.062, q0.95 7.062, q0.985 7.062, q0.99 10.449, q0.995 16.867, q1.0 106.231. E4a's threshold (95th percentile of all MMLU dev, 'higher') = 7.0620.

Top 20 of the 960 calibration scores: 46.60, 45.52, 29.80, 21.59, 20.85, 17.12, 16.00, 14.27, 12.19, 12.10, 10.88, 10.86, 10.70, 10.10, 9.67, 8.92, 8.57, 8.43, 7.89, 7.06.

| operating point | alpha | threshold t | calibration prompts above t | (1 + k) / 961 | tied at t | fire share, all MMLU utility dev |
|---|---|---|---|---|---|---|
| conformal 5 % | 0.05 | 7.0620 | 19 / 960 (0.0198) | 0.0208 | 941 | 0.0145 |
| conformal 1.7 % (DSG's rate) | 0.017 | 8.9226 | 15 / 960 (0.0156) | 0.0166 | 1 | 0.0115 |

Prompt identity check (E4a's StreamGuard v1 reference score > tau vs the E2d StreamGuard v1 run's own gate decision): clean 1.0000 (0 of 7139 differ), pad400 1.0000 (0 of 636 differ), split3 1.0000 (0 of 636 differ), hindi 1.0000 (0 of 627 differ).

### Levels

| condition | forget clean | forget pad400 | forget split3 | forget Hindi | utility | benign biology |
|---|---|---|---|---|---|---|
| base | 0.676 [0.638, 0.711] (n 636) | 0.659 [0.623, 0.695] (n 636) | 0.615 [0.577, 0.653] (n 636) | 0.437 [0.399, 0.475] (n 627) | 0.574 [0.561, 0.586] (n 6076) | 0.642 [0.597, 0.686] (n 427) |
| DSG (rho + clamp) | 0.264 [0.233, 0.297] (n 636) | 0.649 [0.615, 0.686] (n 636) | 0.585 [0.549, 0.623] (n 636) | 0.239 [0.206, 0.271] (n 627) | 0.567 [0.554, 0.579] (n 6076) | 0.431 [0.384, 0.478] (n 427) |
| StreamGuard v1 (CUSUM + clamp) | 0.247 [0.215, 0.280] (n 636) | 0.244 [0.211, 0.275] (n 636) | 0.253 [0.220, 0.286] (n 636) | 0.305 [0.270, 0.341] (n 627) | 0.556 [0.544, 0.569] (n 6076) | 0.398 [0.351, 0.445] (n 427) |
| L08 op5 → clamp | 0.241 [0.209, 0.274] (n 636) | 0.247 [0.215, 0.280] (n 636) | 0.245 [0.214, 0.278] (n 636) | 0.244 [0.212, 0.276] (n 627) | 0.569 [0.556, 0.581] (n 6076) | 0.518 [0.471, 0.564] (n 427) |
| L08 op5 → R5 | 0.266 [0.233, 0.302] (n 636) | 0.415 [0.377, 0.454] (n 636) | 0.329 [0.292, 0.365] (n 636) | 0.434 [0.396, 0.474] (n 627) | 0.573 [0.560, 0.585] (n 6076) | 0.574 [0.527, 0.621] (n 427) |
| L08 op5 → R6 | 0.250 [0.219, 0.285] (n 636) | 0.242 [0.211, 0.275] (n 636) | 0.236 [0.206, 0.269] (n 636) | 0.282 [0.247, 0.317] (n 627) | 0.570 [0.558, 0.582] (n 6076) | 0.532 [0.485, 0.578] (n 427) |
| L08 op1.7 → clamp | 0.242 [0.209, 0.275] (n 636) | 0.247 [0.215, 0.280] (n 636) | 0.248 [0.217, 0.281] (n 636) | 0.250 [0.218, 0.284] (n 627) | 0.569 [0.556, 0.581] (n 6076) | 0.539 [0.492, 0.588] (n 427) |
| L08 op1.7 → R5 | 0.267 [0.233, 0.303] (n 636) | 0.415 [0.377, 0.454] (n 636) | 0.327 [0.291, 0.363] (n 636) | 0.434 [0.396, 0.474] (n 627) | 0.573 [0.560, 0.585] (n 6076) | 0.578 [0.529, 0.625] (n 427) |
| L08 op1.7 → R6 | 0.252 [0.220, 0.286] (n 636) | 0.242 [0.211, 0.275] (n 636) | 0.239 [0.209, 0.272] (n 636) | 0.287 [0.252, 0.322] (n 627) | 0.571 [0.558, 0.583] (n 6076) | 0.546 [0.499, 0.593] (n 427) |

### Paired vs DSG (rho + clamp)

| candidate | forget clean | forget pad400 | forget split3 | forget Hindi | utility | benign biology |
|---|---|---|---|---|---|---|
| L08 op5 → clamp | -0.024 [-0.044, -0.006], p 0.02 ↓ | -0.403 [-0.448, -0.357], p 1.2e-47 ↓ | -0.340 [-0.384, -0.294], p 4.4e-41 ↓ | +0.005 [-0.011, +0.021], p 0.7 | +0.002 [-0.001, +0.006], p 0.31 | +0.087 [+0.042, +0.133], p 0.00039 ↑ |
| L08 op5 → R5 | +0.002 [-0.025, +0.025], p 1 | -0.234 [-0.277, -0.192], p 4.1e-24 ↓ | -0.256 [-0.299, -0.212], p 3e-28 ↓ | +0.195 [+0.144, +0.244], p 2e-13 ↑ | +0.006 [+0.002, +0.010], p 0.0012 ↑ | +0.143 [+0.096, +0.192], p 3.3e-08 ↑ |
| L08 op5 → R6 | -0.014 [-0.036, +0.008], p 0.27 | -0.407 [-0.454, -0.360], p 1.1e-48 ↓ | -0.349 [-0.392, -0.305], p 4.6e-44 ↓ | +0.043 [+0.010, +0.077], p 0.013 ↑ | +0.004 [+0.000, +0.008], p 0.058 | +0.101 [+0.052, +0.150], p 0.0001 ↑ |
| L08 op1.7 → clamp | -0.022 [-0.041, -0.005], p 0.029 ↓ | -0.403 [-0.450, -0.355], p 2.5e-48 ↓ | -0.336 [-0.379, -0.291], p 1.4e-40 ↓ | +0.011 [-0.006, +0.029], p 0.3 | +0.002 [-0.001, +0.006], p 0.3 | +0.108 [+0.061, +0.155], p 1.6e-05 ↑ |
| L08 op1.7 → R5 | +0.003 [-0.022, +0.027], p 0.9 | -0.234 [-0.277, -0.192], p 4.1e-24 ↓ | -0.258 [-0.300, -0.215], p 9.4e-29 ↓ | +0.195 [+0.144, +0.244], p 2e-13 ↑ | +0.006 [+0.002, +0.010], p 0.0012 ↑ | +0.148 [+0.098, +0.199], p 1.5e-08 ↑ |
| L08 op1.7 → R6 | -0.013 [-0.035, +0.009], p 0.33 | -0.407 [-0.454, -0.360], p 2e-49 ↓ | -0.346 [-0.388, -0.300], p 1.5e-43 ↓ | +0.048 [+0.016, +0.081], p 0.0064 ↑ | +0.004 [+0.000, +0.008], p 0.036 ↑ | +0.115 [+0.066, +0.164], p 1.2e-05 ↑ |

### Paired vs StreamGuard v1 (CUSUM + clamp)

| candidate | forget clean | forget pad400 | forget split3 | forget Hindi | utility | benign biology |
|---|---|---|---|---|---|---|
| L08 op5 → clamp | -0.006 [-0.020, +0.008], p 0.54 | +0.003 [-0.014, +0.022], p 0.87 | -0.008 [-0.025, +0.009], p 0.49 | -0.061 [-0.094, -0.029], p 0.00025 ↓ | +0.013 [+0.008, +0.016], p 3.9e-09 ↑ | +0.119 [+0.077, +0.164], p 2e-07 ↑ |
| L08 op5 → R5 | +0.019 [-0.006, +0.044], p 0.2 | +0.171 [+0.124, +0.217], p 3.5e-12 ↑ | +0.075 [+0.041, +0.110], p 4.7e-05 ↑ | +0.129 [+0.084, +0.174], p 3.4e-08 ↑ | +0.016 [+0.013, +0.020], p 1e-16 ↑ | +0.176 [+0.126, +0.230], p 6e-11 ↑ |
| L08 op5 → R6 | +0.003 [-0.016, +0.024], p 0.88 | -0.002 [-0.020, +0.017], p 1 | -0.017 [-0.036, +0.002], p 0.099 | -0.022 [-0.064, +0.016], p 0.29 | +0.014 [+0.010, +0.018], p 2e-11 ↑ | +0.133 [+0.087, +0.185], p 2.1e-07 ↑ |
| L08 op1.7 → clamp | -0.005 [-0.019, +0.009], p 0.69 | +0.003 [-0.014, +0.022], p 0.87 | -0.005 [-0.024, +0.013], p 0.74 | -0.054 [-0.088, -0.022], p 0.00087 ↓ | +0.013 [+0.009, +0.016], p 3.1e-09 ↑ | +0.141 [+0.096, +0.187], p 3.8e-09 ↑ |
| L08 op1.7 → R5 | +0.020 [-0.006, +0.046], p 0.17 | +0.171 [+0.124, +0.217], p 3.5e-12 ↑ | +0.074 [+0.039, +0.108], p 6.4e-05 ↑ | +0.129 [+0.084, +0.174], p 3.4e-08 ↑ | +0.016 [+0.013, +0.020], p 1e-16 ↑ | +0.180 [+0.129, +0.234], p 2.4e-11 ↑ |
| L08 op1.7 → R6 | +0.005 [-0.014, +0.027], p 0.76 | -0.002 [-0.020, +0.017], p 1 | -0.014 [-0.033, +0.005], p 0.2 | -0.018 [-0.057, +0.021], p 0.42 | +0.014 [+0.011, +0.018], p 4.5e-12 ↑ | +0.148 [+0.098, +0.197], p 1.9e-08 ↑ |

### R, F and damage per fire (E1 / E2d style)

R = fire share on forget (clean / pad400 / split3 / Hindi); F = fire share on utility and benign biology; a_fire = composite forget accuracy on fired clean items, a_miss = base accuracy on missed clean items; damage / fire = base − composite accuracy on the fired prompts.

| gate → responder | R clean | R pad400 | R split3 | R Hindi | a_fire | a_miss | F utility (n) | damage/fire utility | F benign bio (n) | damage/fire benign bio |
|---|---|---|---|---|---|---|---|---|---|---|
| DSG (rho + clamp) | 0.882 | 0.025 | 0.108 | 0.933 | 0.234 | 0.493 | 0.017 (105) | 0.343 | 0.550 (235) | 0.383 |
| StreamGuard v1 (CUSUM + clamp) | 0.940 | 0.914 | 0.901 | 0.620 | 0.231 | 0.500 | 0.045 (275) | 0.385 | 0.698 (298) | 0.349 |
| L08 op5 → clamp | 0.959 | 0.962 | 0.954 | 0.936 | 0.226 | 0.577 | 0.014 (88) | 0.341 | 0.323 (138) | 0.384 |
| L08 op5 → R5 | 0.959 | 0.962 | 0.954 | 0.936 | 0.252 | 0.577 | 0.014 (88) | 0.080 | 0.323 (138) | 0.210 |
| L08 op5 → R6 | 0.959 | 0.962 | 0.954 | 0.936 | 0.236 | 0.577 | 0.014 (88) | 0.227 | 0.323 (138) | 0.341 |
| L08 op1.7 → clamp | 0.954 | 0.956 | 0.943 | 0.920 | 0.227 | 0.552 | 0.012 (70) | 0.429 | 0.293 (125) | 0.352 |
| L08 op1.7 → R5 | 0.954 | 0.956 | 0.943 | 0.920 | 0.254 | 0.552 | 0.012 (70) | 0.100 | 0.293 (125) | 0.216 |
| L08 op1.7 → R6 | 0.954 | 0.956 | 0.943 | 0.920 | 0.237 | 0.552 | 0.012 (70) | 0.257 | 0.293 (125) | 0.328 |

Utility without the 960 calibration prompts (n 5116): base 0.572 [0.558, 0.585] (n 5116); dsg 0.565 [0.551, 0.578] (n 5116); sg1 0.554 [0.540, 0.567] (n 5116); L08 op5 → clamp 0.567 [0.553, 0.580] (n 5116); L08 op5 → R5 0.570 [0.556, 0.583] (n 5116); L08 op5 → R6 0.569 [0.555, 0.582] (n 5116); L08 op1.7 → clamp 0.567 [0.553, 0.580] (n 5116); L08 op1.7 → R5 0.570 [0.556, 0.583] (n 5116); L08 op1.7 → R6 0.569 [0.555, 0.582] (n 5116).

### Selection rule v2 (DEVIATIONS 2026-11-11, unchanged)

Eligible iff utility ≥ DSG 0.5667 − 0.005 = 0.5617; benign FPR (fire share on `@utility`) ≤ 0.05; benign biology significantly above DSG; forget not significantly above StreamGuard v1 on clean, pad400, split3, Hindi. Pick: lowest worst-case forget, then higher benign biology.

| candidate | utility | FPR | benign bio | worst-case forget | (1) utility | (2) FPR | (3) bio > DSG | (4) forget ≤ SG v1 | eligible |
|---|---|---|---|---|---|---|---|---|---|
| L08 op5 → clamp | 0.5688 | 0.0145 | 0.5176 | 0.2469 | yes | yes | yes | yes | **yes** |
| L08 op5 → R5 | 0.5726 | 0.0145 | 0.5738 | 0.4338 | yes | yes | yes | **no** | no |
| L08 op5 → R6 | 0.5704 | 0.0145 | 0.5316 | 0.2823 | yes | yes | yes | yes | **yes** |
| L08 op1.7 → clamp | 0.5688 | 0.0115 | 0.5386 | 0.2504 | yes | yes | yes | yes | **yes** |
| L08 op1.7 → R5 | 0.5726 | 0.0115 | 0.5785 | 0.4338 | yes | yes | yes | **no** | no |
| L08 op1.7 → R6 | 0.5708 | 0.0115 | 0.5457 | 0.2871 | yes | yes | yes | yes | **yes** |

**Selected by rule v2: L08 op5 → clamp.**
