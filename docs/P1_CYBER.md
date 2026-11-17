# P1: the hard-negative gate for WMDP-Cyber, with DSG's Cyber clamp as the response (DEV only)

**P1 status: QUEUED** (lab PC queue after every P9 job; `P1-summary` fills the block below when every P1 job is DONE).

DEV split only (never TEST), lab PC (RTX 2000 Ada), exploratory, not a claim input. Rules and lines: DEVIATIONS row
2026-11-17 (P1), logged before any P1 result. Code: branch `exp/P1-cyber` (`experiments/P1`, `configs/experiments/P1.yaml`
from `scripts/make_p1_config.py`, `tests/test_p1.py`); E4a docs / fit selection and E4b's conformal rule reused unchanged.
Ids, hashes and metrics only.

**Step 1.** Hard negatives = WMDP cyber-retain-corpus (general programming / computer-science text), leakage-filtered
against every eval item (word 8-grams: WMDP-Cyber, WMDP-Bio, all MMLU). E4a procedure: per layer 3-14 (+ DSG's layer-3
SAE), SAE features frequent on the Cyber forget corpus and rare on benign computing text and WikiText, top 20, CUSUM gate;
thresholds by the tie-aware conformal rule at DSG's MMLU fire rate (opDSG) and at 5 % (op5).

**Step 2.** Response = DSG's Cyber clamp (paper recipe as in A1: N 100, pct 90, x1000). On DEV: WMDP-Cyber clean / pad400 /
split3, full-MMLU utility (Cyber set), false fires, fire rate and accuracy on benign computing questions (5 MMLU computing
subjects); DSG paper recipe and DSG chat-retain on identical items.

**Lines:** (a) utility loss vs base <= 0.02; (b) clean forget not significantly above DSG chat-retain; (c) recall >= 0.85
under pad400 and split3. One (layer, op) picked by lowest worst-case forget subject to (a), **chosen on DEV**.

**Queue:** P1-smoke → (after all P9 jobs) `P1-docs` (~25 min) → `P1-fit` (CPU) → `P1-questions` (~45) → `P1-000..003` MCQ runs
(~40 each) → `P1-summary` (CPU). When the summary has run, commit this file; nothing else by hand.

## Results

<!-- P1-SUMMARY -->
(filled by P1-summary)
<!-- /P1-SUMMARY -->
