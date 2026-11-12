# E2f: ignorance response for free-text answers

**Status: Step 1 final; Steps 2-3 with the selected cell (L6 c2) STOPPED as uninformative (2026-11-12); reduced exploratory re-run E2f-L12 (layer 12, c 1) queued after E4b Step B.** Lab PC (RTX 2000 Ada), DEV only, never
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

## E2f-L12: reduced Steps 2-3 with layer 12, c 1 (EXPLORATORY; queued)

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

### Resume (while this section says "queued")
1. `scripts/reboot_recover.sh` after any reboot, then `python -m dsgx.queue.status --chat | grep E2f-L12`.
2. When `E2f-L12-summary` is DONE: copy `$DSG_RESULTS/runs/E2f-L12/summary/SUMMARY.md` (from "### WMDP-Bio open")
   here, state the lines verdict, label it exploratory.
