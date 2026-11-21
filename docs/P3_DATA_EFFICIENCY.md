# P3: data efficiency of the StreamGuard v2 gate (DEV only)

**P3 status: QUEUED** (lab PC queue; the watcher `scripts/p23_when_done.sh` fills the block below when every P3 job is DONE).

Lab PC (RTX 2000 Ada), DEV split only (never TEST), exploratory, not a claim input. Ids and metrics only. Rules and lines:
DEVIATIONS row 2026-11-21 (P3), logged 2026-11-21T17:00:05+05:30 before any P3 result. Branch `exp/P3-data-efficiency`
(from exp/P4-sensitivity 772ab8b; `experiments/P3`, `configs/experiments/P3.yaml`, `tests/test_p3.py`).
Frozen StreamGuard v2 and its TEST result never change, whatever P3 shows.

**What is re-built.** Only the gate's fitted parts: the 20 layer-8 features (E4a's rule), their Bernoulli-LLR CUSUM
weights and the threshold (E4b's tie-aware conformal op5 on the 960 calibration prompts, then v2's midpoint to the next DEV
score). They are fitted on random 20 / 40 / 60 / 80 / 100 % subsets of the forget documents and, independently, of the
benign-biology documents (WikiText whole), 3 seeds each. SAE, layer, response (DSG -500 clamp) and calibration prompts
are fixed. **Stop rule:** the 100 % re-build must reproduce frozen v2 exactly (features, weights, every DEV score bit for
bit, threshold 7.084529876708984, every decision).

**Jobs:** `P3-docs` (GPU: per-document L8 fire counts; each chunk must sum to E4a's cached counts) → `P3-fit` (CPU) →
`P3-questions` (GPU: every fit's score on every DEV question) → `P3-summary` (CPU: composites with E2e sgF / E2d base).

**Pass at 20 % (means of 3 seeds):** recall >= 0.90 clean, >= 0.85 pad400 / split3 / Hindi; benign-biology fire <= 0.40;
composed forget on each condition <= frozen v2's DEV value + 0.02.

## Results

<!-- P3-SUMMARY -->
Pending.
<!-- /P3-SUMMARY -->
