# E2f: ignorance response for free-text answers

**Status: FINAL. Step 1 final; Steps 2-3 with the selected cell (L6 c2) STOPPED as uninformative (2026-11-12); reduced exploratory re-run E2f-L12 (layer 12, c 1) DONE 2026-11-13: both lines NOT MET for every response.** Lab PC (RTX 2000 Ada), DEV only, never
TEST. Exploratory; not a claim input. Ids and metrics only.
Code: branch `exp/E2f-ignorance-response` f81e84b (clean). Design, the Step-1 selection rule and the two lines were
logged in DEVIATIONS (2026-11-11, E2f row) before any result. Run dirs: `$DSG_RESULTS/runs/E2f/*`; generations only
under `$DSG_PRIVATE`.

## Step 1: ignorance direction (final)

Built from 300 questions about made-up entities and 300 matched questions about real ones (people, places and
books; no hazardous topics; `experiments/E2f/entities.py`). Direction = mean last-token resid_post (made-up − real),
unit-normalised, at layers 6, 9 and 12. Steering adds c × the mean residual norm × the direction at every position.
It is checked forced-on, on 100 held-out real-entity questions with the base model.

Base (no steering): IDK 0.00 [0.00, 0.00], refusal 0.00 [0.00, 0.00], gibberish 0.00 [0.00, 0.00]; n 100 held-out real-entity questions.
Mean last-token residual norm: L6 89.4, L9 134.0, L12 182.8; raw (made-up − real) mean-difference norm: L6 3.8, L9 14.1, L12 28.5.

| layer | c (x norm) | IDK | IDK vs base (McNemar p) | refusal | gibberish |
|---|---|---|---|---|---|
| 6 | 2 | 0.01 [0.00, 0.03] | +0.01 [+0.00, +0.03] (p 1) | 0.00 [0.00, 0.00] | 0.04 [0.01, 0.08] |
| 6 | 4 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| 6 | 8 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| 6 | 0.5 (diag.) | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| 6 | 1 (diag.) | 0.01 [0.00, 0.03] | +0.01 [+0.00, +0.03] (p 1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| 9 | 2 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 0.80 [0.72, 0.88] |
| 9 | 4 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| 9 | 8 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| 9 | 0.5 (diag.) | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| 9 | 1 (diag.) | 0.04 [0.01, 0.08] | +0.04 [+0.01, +0.08] (p 0.12) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| 12 | 2 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.01 [0.00, 0.03] | 0.57 [0.47, 0.67] |
| 12 | 4 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| 12 | 8 | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 1.00 [1.00, 1.00] |
| 12 | 0.5 (diag.) | 0.00 [0.00, 0.00] | +0.00 [+0.00, +0.00] (p 1) | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] |
| 12 | 1 (diag.) | 0.32 [0.23, 0.41] | +0.32 [+0.23, +0.41] (p 4.7e-10) | 0.16 [0.09, 0.23] | 0.02 [0.00, 0.05] |

**Selection (rule fixed in advance):** among requested-grid cells (c 2, 4, 8) with gibberish <= 0.05, take the one
with the highest IDK rate. Only one cell is eligible: **layer 6, c 2** (gibberish 0.04). Its IDK rate is 0.01, not
significantly above base (`works: False`). The diagnostic grid (c 0.5 and 1) is used only when no requested cell
is eligible, so it is not used here. The same recipe on the TOFU model: layer 6, norm 82.6.

**Reading:** the direction does not produce "I don't know" at the requested strengths; it breaks the text.
- **Why:** a steering vector 2–8× the residual norm overwhelms the residual. The natural made-up − real
  difference is only 4–16 % of the norm (3.8 / 14.1 / 28.5 vs 89 / 134 / 183).
- **The one cell that works:** layer 12 at c 1 raises IDK to 0.32 [0.23, 0.41] (McNemar p 5e-10), plus refusals
  0.16, with gibberish 0.02. It is outside the requested grid, and the fixed rule does not reach it.
- **Consequence for Steps 2-3:** they use layer 6, c 2, as instructed ("best layer and coefficient from step 1
  only"). That ignorance response is close to a no-op in content: it should behave roughly like base after the
  gate fires.
- **Possible follow-up (not run):** the same Steps 2-3 with L12 c 1. This needs a new DEVIATIONS row before any
  result.

## Steps 2-3 with L6 c2: stopped (2026-11-12, DEVIATIONS row)

Stopped at the user's request, because Step 1 showed the selected cell does not differ from base (IDK 0.01 vs 0.00,
McNemar p 1). `E2f-gen-wmdp` was killed at 239 / 459 items (5 h 33 min); its item checkpoint keeps 200 items
(`$DSG_PRIVATE/E2f/gen-wmdp/partial/`, kept, not analysed; the 38 after the last 200-item chunk are lost).
`E2f-gen-benign`, `-gen-tofu-forget`, `-gen-tofu-retain` and `-summary` never started and were removed from the queue
(archive `queue/_archive/manual-2026-11-12-e2f-stop/`). No Steps 2-3 result exists for L6 c2.

## E2f-L12: reduced Steps 2-3 with layer 12, c 1 (EXPLORATORY; DONE)

DEVIATIONS row (2026-11-12, E2f-L12) logged before any result: layer 12 at c 1 is **outside the original grid** and was
chosen from the Step-1 safe-entity check only (IDK 0.32, gibberish 0.02). Every number below is exploratory.
Code: exp/E2f-ignorance-response 818a379 (`configs/experiments/E2f-L12.yaml`, `experiments/E2f/l12.py`; override
args `ign`, `n_items`, `subset_seed` in `gen.py`). Run dirs `$DSG_RESULTS/runs/E2f-L12/*`.

- Same as E2f: StreamGuard v1 streaming CUSUM gate (StreamGuard v1 = the CUSUM gate on DSG's 20 features at a 5 %
  MMLU-dev false-fire rate), responses clamp / R6 / ign / R6+ign / R5+ign, graders, and the two lines (gibberish
  <= 0.05 on every set; leak on WMDP-Bio-open and TOFU forget not significantly above the clamp).
- Direction: base model = Step 1's L12 direction (norm 182.8). TOFU model = the same recipe at L12 (task `tofu-dir`).
- **Reduced to fit ~6 h:** **100 items per set** (seeded random subset, numpy `default_rng(0)`, original order kept,
  from WMDP-Bio-open dev 459, benign biology open dev 180, TOFU forget10 400, retain_perturbed 400) and **48 new
  tokens on every set** (E2f: 96 / 48 / 64 / 64); 10 items per set re-checked against `dsgx.gen.stream` (E2f: 20).
- Jobs: `E2f-L12-smoke-*` (3 items, 16 tokens) then `E2f-L12-tofu-dir`, `-gen-wmdp`, `-gen-benign`,
  `-gen-tofu-forget`, `-gen-tofu-retain`, `-summary`; one GPU job at a time; item-level resume (chunks in the
  private dir). Expected ~5 h (gen-wmdp ran at 84 s per item at 96 tokens; ~42 s at 48 tokens x 400 items + setup).

### Results (EXPLORATORY, DEV only, labpc; n 100 per set, 48 new tokens)

All 12 E2f-L12 jobs DONE on their first attempt (smoke 6/6, then `tofu-dir`, 4 generation jobs, `summary`).
TOFU-model L12 direction (`tofu-dir`): residual norm 184.0, raw made-up − real difference norm 19.8 (base model: 182.8 / 28.5).
StreamGuard v1 stream check: switch generator = StreamGuard v1 stream on 10/10 checked items in every set.
Gate fire rates: WMDP-Bio open 0.910 and benign biology 0.580 (threshold 16.82); TOFU forget10 0.600 and retain 0.040
(TOFU-model threshold 5.475).

**Lines verdict: NOT MET.** No response meets both lines (gibberish <= 0.05 on every set; leak not significantly
above the clamp on WMDP-Bio-open and TOFU forget).
- **The ignorance response does what Step 1 promised on WMDP:** ign alone gives leak 0.00 (= clamp), IDK 0.36 and
  gibberish 0.26 (clamp: gibberish 0.59, IDK 0.00). R6+ign raises IDK to 0.66. Gibberish stays far above 0.05 for
  every response, the clamp included, so the gibberish line fails on WMDP-Bio-open alone.
- **TOFU forget: ign leaks more than the clamp.** 0.63 vs 0.45 (+0.18 [+0.10, +0.27], p 0.0003). R6+ign 0.60 and R5+ign 0.62
  are also significantly above the clamp. R6 alone (0.79) ≈ base (0.80). The L12 direction barely produces "I don't
  know" on TOFU (IDK 0.03 for every condition, = base).
- **Benign biology:** ign 0.06 correct = clamp 0.06 (base 0.17; R6 0.12, +0.06 vs clamp, p 0.031). ign adds IDK 0.09 and gibberish 0.20.
  **TOFU retain** is unaffected by every response (0.85–0.86; the gate fires on 4 %).
- The IDK and gibberish labels are not exclusive (R6+ign on WMDP: gibberish 0.73, IDK 0.66), so the columns do not sum to 1.
- Limits: 100-item seeded subsets and 48 new tokens (shorter than E2f's 96 / 64 on WMDP / TOFU); L12 c1 was chosen
  outside the fixed grid from the safe-entity check only (DEVIATIONS 2026-11-12). Not a claim input.

### E2f-L12 tables (generated: `E2f-L12-summary`, copied from `runs/E2f-L12/summary/SUMMARY.md`, from "WMDP-Bio open")

#### WMDP-Bio open (dev)

Gate fire rate 0.910, threshold 16.82; switch generator = StreamGuard v1 stream on 10/10 checked items.

| condition | leak (match) | vs clamp | gibberish | IDK or refusal | IDK |
|---|---|---|---|---|---|
| base | 0.130 [0.070, 0.200] (n 100) | +0.130 [+0.070, +0.200], p 0.00024 ↑ | 0.000 [0.000, 0.000] | 0.010 [0.000, 0.030] | 0.000 [0.000, 0.000] |
| clamp | 0.000 [0.000, 0.000] (n 100) | - | 0.590 [0.490, 0.680] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R6 | 0.010 [0.000, 0.030] (n 100) | +0.010 [+0.000, +0.030], p 1 | 0.590 [0.490, 0.690] | 0.010 [0.000, 0.030] | 0.000 [0.000, 0.000] |
| ign | 0.000 [0.000, 0.000] (n 100) | +0.000 [+0.000, +0.000], p 1 | 0.260 [0.170, 0.350] | 0.370 [0.280, 0.460] | 0.360 [0.260, 0.450] |
| R6+ign | 0.000 [0.000, 0.000] (n 100) | +0.000 [+0.000, +0.000], p 1 | 0.730 [0.640, 0.810] | 0.660 [0.570, 0.750] | 0.660 [0.570, 0.750] |
| R5+ign | 0.000 [0.000, 0.000] (n 100) | +0.000 [+0.000, +0.000], p 1 | 0.730 [0.640, 0.820] | 0.510 [0.410, 0.610] | 0.510 [0.410, 0.610] |

#### benign biology open (dev)

Gate fire rate 0.580, threshold 16.82; switch generator = StreamGuard v1 stream on 10/10 checked items.

| condition | correct (match) | vs clamp | gibberish | IDK or refusal | IDK |
|---|---|---|---|---|---|
| base | 0.170 [0.100, 0.240] (n 100) | +0.110 [+0.050, +0.180], p 0.00098 ↑ | 0.010 [0.000, 0.030] | 0.010 [0.000, 0.030] | 0.010 [0.000, 0.030] |
| clamp | 0.060 [0.020, 0.110] (n 100) | - | 0.110 [0.050, 0.170] | 0.020 [0.000, 0.050] | 0.020 [0.000, 0.050] |
| R6 | 0.120 [0.060, 0.190] (n 100) | +0.060 [+0.020, +0.110], p 0.031 ↑ | 0.230 [0.150, 0.310] | 0.010 [0.000, 0.030] | 0.010 [0.000, 0.030] |
| ign | 0.060 [0.020, 0.110] (n 100) | +0.000 [+0.000, +0.000], p 1 | 0.200 [0.120, 0.280] | 0.090 [0.040, 0.150] | 0.090 [0.040, 0.150] |
| R6+ign | 0.060 [0.020, 0.110] (n 100) | +0.000 [+0.000, +0.000], p 1 | 0.360 [0.270, 0.450] | 0.470 [0.380, 0.570] | 0.470 [0.380, 0.570] |
| R5+ign | 0.060 [0.020, 0.110] (n 100) | +0.000 [+0.000, +0.000], p 1 | 0.310 [0.220, 0.400] | 0.270 [0.180, 0.360] | 0.250 [0.170, 0.340] |

#### TOFU forget10

Gate fire rate 0.600, threshold 5.475; switch generator = StreamGuard v1 stream on 10/10 checked items.

| condition | leak (match) | vs clamp | gibberish | IDK or refusal | IDK |
|---|---|---|---|---|---|
| base | 0.800 [0.720, 0.870] (n 100) | +0.350 [+0.260, +0.440], p 5.8e-11 ↑ | 0.000 [0.000, 0.000] | 0.030 [0.000, 0.070] | 0.030 [0.000, 0.070] |
| clamp | 0.450 [0.350, 0.550] (n 100) | - | 0.040 [0.010, 0.080] | 0.030 [0.000, 0.060] | 0.030 [0.000, 0.060] |
| R6 | 0.790 [0.710, 0.870] (n 100) | +0.340 [+0.250, +0.430], p 1.2e-10 ↑ | 0.000 [0.000, 0.000] | 0.030 [0.000, 0.070] | 0.030 [0.000, 0.070] |
| ign | 0.630 [0.540, 0.720] (n 100) | +0.180 [+0.100, +0.270], p 0.00028 ↑ | 0.010 [0.000, 0.030] | 0.030 [0.000, 0.070] | 0.030 [0.000, 0.070] |
| R6+ign | 0.600 [0.510, 0.700] (n 100) | +0.150 [+0.070, +0.230], p 0.0015 ↑ | 0.000 [0.000, 0.000] | 0.030 [0.000, 0.070] | 0.030 [0.000, 0.070] |
| R5+ign | 0.620 [0.520, 0.720] (n 100) | +0.170 [+0.080, +0.260], p 0.00049 ↑ | 0.010 [0.000, 0.030] | 0.030 [0.000, 0.070] | 0.030 [0.000, 0.070] |

#### TOFU retain_perturbed

Gate fire rate 0.040, threshold 5.475; switch generator = StreamGuard v1 stream on 10/10 checked items.

| condition | correct (match) | vs clamp | gibberish | IDK or refusal | IDK |
|---|---|---|---|---|---|
| base | 0.860 [0.790, 0.920] (n 100) | +0.010 [-0.020, +0.040], p 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| clamp | 0.850 [0.780, 0.910] (n 100) | - | 0.000 [0.000, 0.000] | 0.010 [0.000, 0.030] | 0.010 [0.000, 0.030] |
| R6 | 0.860 [0.790, 0.920] (n 100) | +0.010 [-0.020, +0.040], p 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| ign | 0.860 [0.790, 0.920] (n 100) | +0.010 [-0.020, +0.040], p 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R6+ign | 0.850 [0.780, 0.920] (n 100) | +0.000 [-0.030, +0.030], p 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| R5+ign | 0.850 [0.780, 0.920] (n 100) | +0.000 [-0.030, +0.030], p 1 | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

#### Lines (fixed before any result)

| response | gibberish <= 0.05 on every set | leak not above clamp | both |
|---|---|---|---|
| clamp | False | True | **False** |
| R6 | False | False | **False** |
| ign | False | False | **False** |
| R6+ign | False | False | **False** |
| R5+ign | False | False | **False** |
