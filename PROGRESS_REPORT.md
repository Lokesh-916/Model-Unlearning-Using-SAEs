# Progress Report: Dynamic SAE Guardrails (DSG) — Reproduction + 9 Improvement Attempts

Prepared from a read-only inspection of the git repository at `baselines_DSG/` (all
branches), its cached result files (`.json`, `.pkl`), the project-root markdown notes in
`mechunlearn-project/`, `DSG_Claude_Code_Full_Handoff.docx`, `STATUS_FOR_REPORT.md` (an
earlier repo-inspection note, cross-checked against source files below), and the paper PDF
(`research_papers/baseline.pdf`). No code was run, modified, or installed to produce this
report. Every number below is quoted directly from a file; where a value could not be
independently re-verified, it is marked accordingly.

**Paper identity note (flag it before anything else):** the PDF's own footer reads
*"Proceedings of the 42nd International Conference on Machine Learning (ICML), Vancouver,
Canada. PMLR 267, 2025"* (`research_papers/baseline.pdf`, page 1). The repo's own
`README.md` and the commit message `7f30128` ("Reduce figure size in README") both label
this **COLM 2025**, and `f7a3c35` is titled *"Polish codebase for COLM 2025 camera-ready
submission"*. These two attributions conflict inside the repo itself — see Section 8.

---

## 1. TIMELINE (day 1 to now)

**Structural fact first:** the top-level `mechunlearn-project/` directory is not a git
repository (only markdown/docx notes live there). The only git history is inside
`baselines_DSG/`. Within it, `main` and the 9 improvement branches diverge at commit
`19578fd` — **none of the 9 improvement branches are merged into `main`**, and `main` does
not contain any of their work. This is confirmed via `git merge-base main <branch>` for all
9 branches, which returns `19578fd` in every case.

| # | Date (author date) | Branch | Commit | Author | What was attempted (1-2 lines) |
|---|---|---|---|---|---|
| 1 | 2025-03-31 | main | `9795ca8` "Initial commit" | Aashiq Muhamed | Upstream paper authors' original DSG codebase. |
| 2 | 2025-10-12 | main | `f7a3c35` "Polish codebase for COLM 2025 camera-ready submission" | aashiqmuhamed | Upstream cleanup before camera-ready. |
| 3 | 2025-10-12 | main | `7f30128` "Reduce figure size in README" | aashiqmuhamed | Cosmetic README fix. |
| 4 | 2026-09-09 01:38 | main | `7807d72` "Fix SAE-Lens API compatibility and add DSG intervention debug instrumentation" | Lokesh-916 | Ported the upstream code to a newer SAE-Lens API; added permanent debug prints to the clamp hook (still present, unremoved). |
| 5 | 2026-09-09 02:05 | main | `12f0525` "Add Gemma logit softcap to custom chunked softmax evaluation" | Lokesh-916 | Fixed a missing Gemma-2 logit softcap in the custom vocab-chunked softmax used for MCQ scoring. |
| 6 | 2026-09-09 15:53 | main | `4781bf9` "Add zero-shot interpretable unlearning reproduction (paper Section 4.4)" | Lokesh-916 | Reproduced the paper's Neuronpedia-feature-search zero-shot result. |
| 7 | 2026-09-10 11:06 | main | `23e490d` "Add data efficiency reproduction (paper Section 4.4, Figure 6A)" | Lokesh-916 | Reproduced the forget/retain-data-fraction sweep. |
| 8 | 2026-09-10 11:30 | main | `db5c439` "Add inference latency benchmark (paper Appendix L, Table 15)" | Lokesh-916 | Measured DSG's wall-clock overhead vs. base model on this hardware. |
| 9 | 2026-09-10 12:07 | main | `ae51863` "Add clamp-strength ablation (paper Section 4.5, Figure 11)" | Lokesh-916 | Swept clamp multiplier c. |
| 10 | 2026-09-10 14:28 | main | `4374d9a` "Add p_dyn and p_ratio ablations (paper Section 4.5)" | Lokesh-916 | Swept the two threshold percentiles. |
| 11 | 2026-09-10 14:30 | main | `19578fd` "Add rho vs rho_raw activation-statistic comparison (paper Section 4.5/K.1)" | Lokesh-916 | Compared the paper's ρ statistic against a raw-count alternative via TVD. **This is the branch point for all 9 improvement branches.** |
| 12 | 2026-09-16 16:16 | `imp-a1-streaming-detector` | `02eedff` | Lokesh-916 | Found + reproduced a guardrail-bypass vulnerability (padding dilution); built and tested 2 defenses (windowed gate, CUSUM detector). |
| 13 | 2026-09-16 16:16 | `imp-c2-multilayer-circuit` | `1cb45fe` | Lokesh-916 | Preliminary 2-layer (layer 3 + layer 8) intervention test. |
| 14 | 2026-09-16 16:25 | main | `fd4d16c` "Add reproduction figures for the paper's key plots/tables" | Lokesh-916 | Generated 7 PNGs from the already-committed reproduction/ablation JSON files. **Not present on any improvement branch.** |
| 15 | 2026-09-17 00:06 | `imp-c1-attribution-selection` | `237b66e` | Lokesh-916 | Attribution-patching-based feature selection, tested against DSG's original selection. |
| 16 | 2026-09-17 00:45 | `imp-b3-hierarchical-gate` | `885f42b` | Lokesh-916 | Two-level (domain + hazard) gate targeting neighboring-knowledge collateral damage. |
| 17 | 2026-09-17 01:01 | `imp-c3-residual-projection-probing` | `148fad9` | Lokesh-916 | Full-residual subspace projection instead of clamping; added a linear-probe leak metric. |
| 18 | 2026-09-17 01:24 | `imp-a2-calibrated-intervention` | `de2d20c` | Lokesh-916 | Conditional mean-ablation toward a low-confidence reference set, instead of clamping to −500. |
| 19 | 2026-09-17 01:40 | `imp-a3-fdr-concept-gating` | `2896ab1` | Lokesh-916 | 4-concept detector bank with Bonferroni-corrected thresholds. |
| 20 | 2026-09-17 01:56 | `imp-b1-distill-weights` | `e47f3e1` | Lokesh-916 | Distilled the guardrail's behavior into LoRA weights via KL distillation. |
| 21 | 2026-09-17 01:57 | `imp-b2-sae-facts-tofu` | `c2e5293` | Lokesh-916 | Not attempted; README documents scope/why. |

**Gaps / abandoned directories (evident from repo state, cause noted where evident):**

- **`novelty-trials` branch**: created at the same commit (`19578fd`) as the 9 improvement
  branches, has **zero commits beyond `main`** (`git log main..novelty-trials` returns
  nothing). Superseded by the per-improvement branch structure before any work was added to
  it. Cause: evident from conversation record — a deliberate restructuring decision, not a
  failure.
- **`eval_results/unlearning_dynamic_bs1/` is an empty directory** (`git ls-tree` /
  `find` both confirm zero files) despite `main`'s own `README.md` documenting it as
  where the library's own final unlearning-score JSON should be written. The reproduction
  never produced this file; all reported numbers instead come from the cached `.pkl` metric
  files directly. Cause: unclear from the repo — the library's own summary-writing step was
  apparently never invoked or its output was never committed.
- **`src/{absorption,common,robustness,unlearning_ext}/`, `scripts/`, `checkpoints/`,
  `data/`, `docs/`, `results/baseline/`** (top-level, outside `baselines_DSG/`): all
  confirmed empty. Whatever these were scaffolded for was never started. Cause: unclear.
- **Backup files preserved in place** (not deleted, not reverted via git — a manual
  before/after preservation pattern used throughout): e.g.
  `artifacts_dynamic_bs1_cyber/.../metrics/clamp_feature_activation_multiplier500_nfeatures30_layer3_retainthres90_seed0_ORIGINAL_with_feature1312.pkl`
  and `*_PRESOFTCAPFIX.pkl` files sit alongside the current versions. These are evidence of
  in-place fixes to cached data files, not git reverts — see Section 3 mod row 1.

---

## 2. BASELINE REPRODUCTION

### Exact setup used

| Item | Value | Source |
|---|---|---|
| Model | `gemma-2-2b-it` (`google/gemma-2-2b-it`, ~2.6B params, 26 layers, d_model=2304) | `README.md`; live-verified `torch`/tokenizer load this session |
| SAE release | `gemma-scope-2b-pt-res` | `README.md` usage example |
| SAE block (Bio + Cyber main config) | `layer_3/width_16k/average_l0_142` → hook `blocks.3.hook_resid_post`, d_sae=16,384 | `README.md`; confirmed via `ablate_params` in every cached `.pkl` |
| SAE used for zero-shot (Sec. 4.4) | Same layer/width but **ℓ0=59**, deliberately different from ℓ0=142 | paper page 7 ("gemma-scope-2b-pt-res SAE (width 16k) at layer 3 (ℓ0 59)"); corroborated by `STATUS_FOR_REPORT.md` |
| n_features (Bio) | 20 | `ablate_params['features_to_ablate']` length in every Bio `.pkl` |
| n_features (Cyber) | 30 | same, Cyber `.pkl` files |
| p_ratio (importance-ratio percentile) | 95 (Bio), 90 (Cyber) | `ablations/dsg_threshold_ablations_results.json` (Bio "95" row = main config); `STATUS_FOR_REPORT.md` for Cyber (not independently re-verified against a Cyber-side ablation JSON — none exists in this repo) |
| p_dyn (dynamic threshold percentile) | 95 (both) | `ablations/dsg_threshold_ablations_results.json`, "95" row matches main config exactly (threshold 0.5458007812500021) |
| Clamp multiplier c | 500 | `ablate_params['multiplier']` in every `.pkl`; also the only value in `ablations/dsg_clamp_ablation_results.json`'s "500" key matching main |
| Retain corpus D_retain (Bio) | WikiText-2 raw test split, unmodified, as specified in the paper | `CYBER_DIAGNOSTIC_2026-09-08.md` line 3 |
| Retain corpus D_retain (Cyber, current/fixed) | 400 Gemma-chat-formatted MCQ prompts from 10 MMLU subjects (anatomy, astronomy, elementary_mathematics, formal_logic, global_facts, jurisprudence, philosophy, world_religions, nutrition, marketing) — **deliberately disjoint from Cyber's own eval subjects** | `CYBER_DIAGNOSTIC_2026-09-08.md` line 122 |
| Seeds | `--random_seed 0` for all main results; one confirmatory `--seed 1` rerun of Cyber feature selection only (not a multi-seed report) | `CYBER_DIAGNOSTIC_2026-09-08.md` §8c (per `STATUS_FOR_REPORT.md`; not independently reopened in this pass) |
| LLM dtype | bfloat16 | live-verified model load; `STATUS_FOR_REPORT.md` |
| GPU | NVIDIA RTX 2000 Ada Generation, 16,380 MiB | `nvidia-smi --query-gpu`, run live this session (read-only) |
| Driver / CUDA (driver-reported) | 580.173.02 / CUDA 13.0 | `nvidia-smi`, run live this session |
| PyTorch (installed) | **2.11.0+cu128** | `pip show torch`, run live this session — **this corrects `STATUS_FOR_REPORT.md`'s claim of "2.13.0+cu130," which does not match the live environment** |
| transformer-lens (installed) | **3.8.1** | `pip show transformer-lens`, run live this session — corrects `STATUS_FOR_REPORT.md`, which only quoted the `pyproject.toml` minimum pin (`>=2.0.0`), not the installed version |
| sae_lens (installed) | 6.50.0 | live `import sae_lens; sae_lens.__version__`; matches `STATUS_FOR_REPORT.md` |

### RAW vs. NORMALIZED — WMDP-Bio subset size

**The WMDP-Bio and WMDP-Cyber numbers below are NOT computed on the raw/full dataset.**
`calculate_MCQ_metrics(..., target_metric="correct")` loads a pre-filtered question-ID list
(`data/question_ids/all/<dataset>_correct.csv`) and scores **only that filtered subset** —
this filtering is inherited unmodified from the upstream DSG/SAEBench code, not introduced
in this reproduction. Confirmed item counts (from `is_correct` array lengths inside the
cached `.pkl` files, read directly this session):

| Dataset | Scored subset size | Source |
|---|---|---|
| WMDP-Bio | 538 | `artifacts_dynamic_bs1_bio/.../metrics/clamp_..._nfeatures20_..._seed0.pkl`, key `wmdp-bio['is_correct'].shape` |
| WMDP-Cyber | 275 | `artifacts_dynamic_bs1_cyber/.../metrics/clamp_..._nfeatures30_..._seed0.pkl` |
| MMLU human_aging (Bio config) | 85 | same Bio `.pkl` |
| MMLU high_school_us_history (Bio config) | 108 | same |
| MMLU high_school_geography (Bio config) | 103 | same |
| MMLU college_computer_science (Bio config) | **9** | same — **this is a very small sample; treat any single-question flip on this subject as an ~11-point swing** |
| MMLU college_biology (Cyber config) | 75 | Cyber `.pkl` |

The full WMDP-Bio test set is 1,273 questions per prior repo notes (`STATUS_FOR_REPORT.md`
§3) — **not independently re-counted from the raw HF dataset in this pass**; flagged in
Section 8. Whether this 538/1273 filtering matches the paper's own "raw vs. normalized"
convention could not be determined from any file in this repo — the paper does not specify
its exact filtering procedure in the pages reviewed for this report (see Section 8).

### Main WMDP-Bio/Cyber result: paper vs. our reproduction

| Metric | DSG paper's number | Our reproduced number | Absolute gap | Notes |
|---|---|---|---|---|
| WMDP-Bio accuracy ↓ | **29.64%** (Table 1, paper p.5, quoted in body text: *"reducing accuracy to 29.64%"*) | **29.368%** (0.2936802804470062) | −0.27 pts | `artifacts_dynamic_bs1_bio/.../metrics/clamp_feature_activation_multiplier500_nfeatures20_layer3_retainthres95_seed0.pkl`, key `wmdp-bio.mean_correct` |
| MMLU avg (Bio config, 4 subjects) ↑ | 99.34% ("All" column, Table 1) — **scope of paper's "All" column is unclear, see Section 8** | 99.412% (mean of 4 subjects: 100.0, 100.0, 100.0, 0.9764706) | +0.07 pts (if scopes match) | same `.pkl`, 4 dict keys averaged manually |
| MT-Bench | 7.78 (Table 1) | **NOT RUN** | N/A | No MT-Bench result file exists anywhere in this repo; `STATUS_FOR_REPORT.md` states it requires paid OpenAI API access not authorized — this specific claim could not be independently re-verified in any plain-text log or the handoff docx in this pass (see Section 8) |
| WMDP-Cyber accuracy ↓ | 26.74% (per repo notes; **page not re-located in the PDF during this pass**, see Section 8) | **28.00%** (0.28) | +1.26 pts | `artifacts_dynamic_bs1_cyber/.../metrics/clamp_feature_activation_multiplier500_nfeatures30_layer3_retainthres90_seed0.pkl`, key `wmdp-cyber.mean_correct` |
| MMLU avg (Cyber config, 4 subjects) ↑ | 99.73% (per repo notes, page not re-located) | 99.435% (mean of 0.990741, 0.986667, 1.0, 1.0) | −0.29 pts | same Cyber `.pkl` |

**Zero-shot (paper Section 4.4) — comparison to paper's own RMU/Farrell numbers (those two
rows are copied from the paper, not run in this environment):**

| Method | WMDP-Bio ↓ | WMDP-Cyber ↓ | Source |
|---|---|---|---|
| RMU (paper) | 50.00% (Table 1, p.5, confirmed) | 88.00% | `figures/reproduction/make_figures.py` line 250 (Cyber value not independently re-located in the PDF this pass) |
| Farrell et al. (paper) | 59.22% (Table 1, p.5, confirmed) | 52.73% | same, Cyber value not re-located |
| DSG full, data-driven (paper) | 29.64% (confirmed, Table 1) | 26.74% (not re-located) | Table 1 (Bio); repo notes (Cyber) |
| DSG zero-shot (**this reproduction**) | 31.04% (τ=0.6), MMLU=96.29% | 41.45% (τ=0.2), MMLU=98.40% | `zeroshot_reproduction/dsg_zeroshot_bio_results.json` key `"0.6"`; `dsg_zeroshot_cyber_results.json` key `"0.2"` |

**Reproduction status per metric, stated explicitly:**

| Metric | Status |
|---|---|
| WMDP-Bio | Reproduced (full-data config) |
| WMDP-Cyber | Reproduced, but only after diagnosing and fixing a retain-corpus bug (see Section 3, mod 1) |
| MMLU (Bio, Cyber configs) | Reproduced, but only on a 4-subject subset (9-108 items each), not full MMLU |
| MT-Bench | **NOT RUN** — no result file exists |
| Zero-shot (Bio, Cyber) | Reproduced |
| Data efficiency (Bio) | Reproduced |
| Inference latency | Reproduced (different GPU) |
| MUSE (NEWS/BOOKS full eval) | **NOT RUN** — feature-identification config classes exist in `dynamic_sae_guardrails/evals/unlearning/eval_config.py` (`UnlearningEvalConfigBooks`/`News`) but zero output files exist anywhere in the repo |
| Sequential unlearning, relearning-attack resistance, TOFU, jailbreak/paraphrase testing | **NOT RUN** (one padding-dilution attack was run instead — see Section 3, Improvement A1) |

---

## 3. ALL 9 IMPROVEMENT IDEAS ATTEMPTED

Numbering below uses the branch names actually used in the repo (`imp-<id>-<slug>`), which
map to an internal Break/Bake/Build (Person 1/2/3, A/B/C) planning scheme referenced in the
branch names' first letter: **A = "Break" (attack DSG's own gate mechanism), B =
"Bake" (permanence via weight-editing), C = "Build" (interpretability/mechanism improvements)**.

### A1 — `imp-a1-streaming-detector`

- **What changed vs. baseline**: (i) demonstrated an attack on DSG's gate, (ii) replaced the
  whole-sequence-average trigger ρ(x) with two alternative detectors.
- **Hypothesis**: ρ(x) is a simple average over the whole sequence; padding a
  forget-flagged prompt with unrelated benign text should dilute this average below
  threshold without changing the dangerous content, defeating the gate. A local-window or
  streaming statistic should resist this.
- **Implementation status**: fully coded and evaluated (attack + 2 full defense variants:
  windowed gate at 2 window sizes, CUSUM streaming detector).
- **OUTCOME**:
  - **Attack: POSITIVE (as a finding)** — the vulnerability is real and reproducible.
  - **Windowed-gate defense: PARTIAL** — cuts bypass substantially but does not eliminate
    it, and degrades with more padding.
  - **CUSUM defense: PARTIAL, different trade-off** — bypass rate becomes flat regardless
    of padding amount, but baseline (no-padding) sensitivity is worse than the windowed
    gate.
- **Numbers** (all from `git show imp-a1-streaming-detector:novelty_trials/a1_streaming_detector/<file>`):

  Attack (`01_dilution_attack_results.json`), threshold=0.5458007812500021:
  | Padding (words) | Mean token len | Mean ρ | Attack success rate | n |
  |---|---|---|---|---|
  | 0 | 115.0 | 0.8014580862217396 | 0.0 | 60 |
  | 150 | 284.517 | 0.2862844896002161 | 0.9833333333333333 | 60 |
  | 400 | 563.233 | 0.16179208582846563 | 0.95 | 60 |
  | 800 | 1003.867 | 0.1088090895437756 | 0.9666666666666667 | 60 |

  Windowed gate, window=24 (`02_windowed_gate_w24_results.json`), threshold=0.6666666865348816:
  utility: wmdp-bio=0.2695167362689972, mmlu_avg=0.9764705896377563.
  | Padding | Mean ρ | Attack success rate |
  |---|---|---|
  | 0 | 1.0381945192813873 | 0.016666666666666666 |
  | 150 | 0.960416708389918 | 0.2 |
  | 400 | 0.9326389153798421 | 0.3 |
  | 800 | 0.9291666934887568 | 0.35 |

  Windowed gate, window=8 (`02_windowed_gate_w8_results.json`), threshold=0.75: utility
  wmdp-bio=0.271375447511673, mmlu_avg=0.9647058844566345; attack success 0.0167 / 0.2333 /
  0.2833 / 0.35 at padding 0/150/400/800 — **essentially unchanged from window=24**,
  indicating the residual bypass is not caused by window width.

  CUSUM detector (`03_cusum_detector_results.json`), boundary h=38.27937648296357: utility
  wmdp-bio=0.3085501790046692, mmlu_avg=0.9823529422283173; attack success 0.35 / 0.3667 /
  0.3667 / 0.3667 at padding 0/150/400/800 — **flat regardless of padding, but the
  no-padding baseline (0.35) is worse than the windowed gate's (0.017)**.

### A2 — `imp-a2-calibrated-intervention`

- **What changed**: replaced clamping selected features to a fixed constant (−500) with
  conditional mean-ablation toward the mean feature activations of a reference set (the 318
  lowest-confidence WMDP-Bio questions under the unintervened base model, out of 1,273
  total), at a searched scalar strength.
- **Hypothesis**: −500 is an out-of-distribution constant; the model may be "broken" rather
  than "doesn't know," and this should be visible as abnormal output entropy/confidence
  relative to the base model. A milder, reference-calibrated replacement should reduce
  WMDP accuracy toward chance (25%) while looking more "naturally uncertain."
- **Implementation status**: fully coded and evaluated (kill test + 4-point strength grid +
  full eval at best strength).
- **OUTCOME: NEGATIVE.** Kill test confirmed the premise (DSG's clamp does look abnormal:
  entropy 0.7586766481399536 vs. base 0.23689204454421997, max-prob 0.7398324012756348 vs.
  base 0.9247715473175049). But **the fix does not work**: all 4 tested strengths give
  99.26–99.81% WMDP-Bio accuracy — indistinguishable from no intervention at all.
- **Numbers** (`git show imp-a2-calibrated-intervention:novelty_trials/a2_calibrated_intervention/a2_results.json`):
  - Reference set: 318 questions, mean confidence 0.3252807557582855 (vs. overall mean
    0.6669321060180664).
  - Strength grid: 0.5→0.9925650358200073, 1.0→0.9944238066673279, 2.0→0.9981412291526794,
    4.0→0.9944238066673279 (all WMDP-Bio accuracy).
  - Best (0.5x) full eval: wmdp-bio=0.9925650358200073, mmlu_avg=1.0 (all 4 subjects
    exactly 1.0).
  - Entropy/max-prob at best strength: 0.2431631088256836 / 0.9194891452789307 — nearly
    identical to the base model's own numbers, **not because it looks naturally uncertain,
    but because it is not intervening on behavior at all.**

### A3 — `imp-a3-fdr-concept-gating`

- **What changed**: scaled the spec's 50-concept design down to 4 (wmdp-bio, wmdp-cyber,
  high_school_us_history, college_computer_science, 10 features each); compared naive
  independent 95th-percentile calibration against Bonferroni-corrected calibration
  (98.75th percentile, n=4, α=0.05).
- **Hypothesis**: a single merged gate's false-positive rate compounds as more concepts are
  added; per-concept detectors with multiple-testing correction should keep the combined
  false-positive rate bounded.
- **Implementation status**: fully coded and evaluated.
- **OUTCOME: NEGATIVE / surfaced a different, deeper problem.** All 4 intended forget
  targets show suppression (near or below chance), but two **untouched holdout subjects**
  (high_school_geography, human_aging — never targeted by any detector) are damaged almost
  as badly as the real targets, and Bonferroni correction barely helps.
- **Numbers** (`git show imp-a3-fdr-concept-gating:novelty_trials/a3_fdr_concept_gating/a3_results.json`):
  - Thresholds: wmdp-bio naive=0.4861328125000016/bonferroni=0.8031982421874999;
    wmdp-cyber naive=0.02958984375000001/bonferroni=0.05930175781249997;
    high_school_us_history naive=0.2861328125/bonferroni=0.3584228515624999;
    college_computer_science naive=0.07890625000000007/bonferroni=0.1895507812499997.
  - Naive results: wmdp-bio=0.2509293556213379, wmdp-cyber=0.26545456051826477,
    high_school_us_history=0.2598039209842682, college_computer_science=0.14000000059604645,
    **high_school_geography=0.24271844327449799**, **human_aging=0.3764705955982208**.
  - Bonferroni results: wmdp-bio=0.25464683771133423, wmdp-cyber=0.28727272152900696,
    high_school_us_history=0.2549019753932953, college_computer_science=0.14999999105930328,
    **high_school_geography=0.28155338764190674**, **human_aging=0.364705890417099**.
  - Diagnosed cause (per branch README, traced to wmdp-cyber's suspiciously low threshold):
    thresholds calibrated on raw WikiText prose but applied to MCQ-formatted eval prompts —
    the identical root-cause pattern as the original Cyber MMLU collapse (Section 3, mod 1),
    resurfacing at a reduced 10-feature-per-concept budget. Not re-run with MCQ-formatted
    calibration data due to time constraints.

### B1 — `imp-b1-distill-weights`

- **What changed**: distilled the DSG-guarded model's output distribution into a LoRA
  adapter (rank 8, q/k/v/o_proj + gate/up/down_proj) via KL loss (student vs. guarded
  teacher on forget prompts, student vs. original teacher on retain prompts), then tested
  the student with the runtime hook **completely removed**.
- **Hypothesis**: DSG's forgetting is a runtime-only, reversible intervention (removable by
  dropping the hook, releasing the weights, or quantizing). If the guarded model's behavior
  can be distilled into permanent weight changes, forgetting survives without any runtime
  mechanism.
- **Implementation status**: fully coded and evaluated, after fixing 3 real bugs in
  sequence (dtype mismatch between the float32 SAE and the bf16 HF model; unnecessary
  gradient tracking on teacher forward passes, inferred from `model.training` instead of an
  explicit flag, which doubled memory every step; and a genuine iteration-to-iteration
  memory leak, mitigated by explicit `del`+`empty_cache()`, batch size 4→2, and
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`).
- **OUTCOME: POSITIVE.** WMDP-Bio accuracy for the student with the hook removed (40.00%)
  is *lower* than the DSG runtime hook itself on the same 200-question random eval sample
  (45.50%), both far below the base model (67.50%).
- **Numbers** (`git show imp-b1-distill-weights:novelty_trials/b1_distill_weights/b1_results.json`):
  - `acc_base_no_intervention`: 0.675
  - `acc_dsg_hook`: 0.455
  - `acc_student_hook_removed`: 0.4
  - `retain_acc_student`: high_school_us_history=0.68, college_computer_science=0.48,
    high_school_geography=0.7133333333333334, human_aging=0.6466666666666666
  - `training_losses`: first=2.181845188140869, last=0.4138040542602539
  - **Honest gap, stated in the branch's own README**: no "before training" retain baseline
    was collected on the same random sample, so the retain numbers above cannot be cleanly
    attributed to training damage vs. the same random-vs-curated-sample gap already visible
    in the 67.50% WMDP-Bio base number (vs. the curated-subset ~99%+ figure elsewhere in
    this repo).

### B2 — `imp-b2-sae-facts-tofu`

- **What changed**: nothing — **not attempted**.
- **Hypothesis** (from spec): train a small trainable "delta dictionary" of new SAE latents
  on residual reconstruction error, to give entity-specific facts (TOFU-style) their own
  addressable features that a pretrained SAE has none for.
- **Implementation status**: idea only. Zero code.
- **OUTCOME: NOT EVALUATED.** Reason recorded in the branch's own README: requires (1) a
  new dataset (TOFU, not present anywhere in this repo), (2) a new SAE-training loop (every
  other experiment in this project reused a frozen, pretrained SAE — none trained new
  latents), and (3) new evaluation infrastructure (Truth Ratio, KS-test) that does not exist
  in `dynamic_sae_guardrails/evals/`. Time budget was spent on the other 8 items instead.

### B3 — `imp-b3-hierarchical-gate`

- **What changed**: added a "kill test" (does the single gate hurt neighboring, non-eval
  MMLU subjects?), then a two-level gate: a domain-general detector (calibrated on
  WikiText) AND a hazard-specific detector (calibrated contrastively against a
  benign-biology/medicine corpus), clamping only when both fire.
- **Hypothesis**: DSG's retain calibration only uses WikiText and 4 topically-distant MMLU
  subjects; it may collapse on benign content that is topically *close* to the forget
  domain (e.g., college biology, virology) without this ever showing up in the paper's own
  reported retain set.
- **Implementation status**: fully coded and evaluated, after fixing 2 bugs (a missing
  question-ID file for the new MMLU subjects, and a degenerate hazard threshold of 1.1937 —
  above the impossible 1.0 ceiling for a bounded rate statistic — caused by a
  batch-dimension aggregation bug in the calibration loop).
- **OUTCOME: kill test POSITIVE (real finding); proposed fix NEGATIVE (overcorrects).**
- **Numbers** (`git show imp-b3-hierarchical-gate:novelty_trials/b3_hierarchical_gate/b3_results.json`):

  Kill test (single gate): wmdp-bio=0.2936802804470062, high_school_us_history=1.0,
  college_computer_science=1.0, high_school_geography=1.0, human_aging=0.9764705896377563,
  **college_biology=0.3819444477558136, virology=0.42168673872947693,
  anatomy=0.43703702092170715** (vs. ~100% on the paper's own reported retain set — a real,
  previously-undocumented weakness).

  B3 hierarchical gate: wmdp-bio=**0.678438663482666** (more than doubled from baseline —
  most real forgetting destroyed), high_school_us_history=1.0, college_computer_science=1.0,
  high_school_geography=1.0, human_aging=0.9882352948188782, college_biology=0.6805555820465088,
  virology=0.5361445546150208, anatomy=0.5259259343147278 (neighbor subjects recover, but at
  the cost above).

  Domain threshold=0.3920898437500015 (20 features), hazard threshold=0.33203125 (20
  features), domain/hazard feature overlap=3/20.

### C1 — `imp-c1-attribution-selection`

- **What changed**: replaced DSG's Fisher-info-style feature ranking (squared mean
  activation, forget/retain ratio) with attribution-patching scores (activation × gradient
  of the correct-answer log-probability), greedy-selected with a decoder-cosine redundancy
  penalty.
- **Hypothesis**: the paper's own Appendix (per the planning note handed to this session)
  argues its Fisher-info shortcut ignores each feature's actual causal effect on the
  output; a gradient-based, causally-grounded score should find a better feature set at
  equal count.
- **Implementation status**: fully coded and evaluated, after fixing a real memory leak
  (backward() was allocating gradients for all ~2.6B model parameters, not just the target
  activation tensor, because the model's parameters were never frozen) that corrupted the
  first attempt (only 1/150 forget and 0/150 retain samples succeeded before that run's
  numbers were discarded as noise).
- **OUTCOME: NEGATIVE.** Zero feature overlap with DSG's original selection, and far worse
  forgetting for a negligible utility difference.
- **Numbers** (`git show imp-c1-attribution-selection:novelty_trials/c1_attribution_selection/c1_attribution_results.json`):
  - `threshold_c1`=0.3851562500000001 vs. `threshold_dsg_original`=0.5458007812500021.
  - `overlap_count`: 0 (of 20).
  - `c1_results`: wmdp-bio=**0.6561338305473328** (vs. baseline 0.2936802804470062),
    high_school_us_history=1.0, college_computer_science=1.0,
    high_school_geography=0.9902912974357605, human_aging=1.0.
  - `mmlu_c1`=0.9975728243589401 (vs. baseline 0.994118 — a negligible +0.36pp gain that
    does not offset the forgetting loss).

### C2 — `imp-c2-multilayer-circuit`

- **What changed**: applied DSG's clamp at **two** layers simultaneously (layer 3, the
  paper's config, plus layer 8 — chosen because a Gemma Scope SAE checkpoint at that layer
  has a matching ℓ0=142), each with its own independently-selected 20-feature set and
  independently-calibrated threshold. **Explicitly labeled "preliminary" in its own
  README** — the full C2 spec (attribution-graph-based cross-layer feature selection)
  depends on C1, which came back negative.
- **Hypothesis**: the clamp-strength ablation showed a flat WMDP plateau across c=50–500,
  suggesting the bottleneck is feature *recall* (which tokens get caught at all), not clamp
  *strength*. A second layer might catch forget-relevant content the first layer misses.
- **Implementation status**: fully coded and evaluated (this simpler 2-layer union test
  only — the attribution-graph-based full version was not built).
- **OUTCOME: PARTIAL / real trade-off, not a clean win.** WMDP-Bio does improve, but MMLU
  drops measurably — diagnosed as the expected cost of an OR-combination of two
  independent gates (each detector's own ~5% retain false-positive rate compounds).
- **Numbers** (`git show imp-c2-multilayer-circuit:novelty_trials/c2_multilayer_circuit/01_multilayer_prelim_results.txt`):
  | Config | WMDP-Bio | MMLU_avg |
  |---|---|---|
  | Paper/original cached | 29.368% | 99.412% |
  | Layer-3-only (sanity check, this run) | 29.368% | 99.412% (exact match — confirms the eval pipeline is working) |
  | Layer-3 + layer-8 (multi) | **27.695%** | **97.941%** |

  Layer 3: 20 features, threshold=0.5458. Layer 8: 20 features, threshold=0.4289.

### C3 — `imp-c3-residual-projection-probing`

- **What changed**: replaced "clamp 20 latents to −500" with "project the whole residual
  stream (SAE reconstruction error included) out of the subspace spanned by an expanded
  (100-feature, decoder-cosine>0.15) family's decoder directions, toward a WikiText-derived
  reference mean." Added a **linear-probe leak metric**: train a logistic-regression probe
  on post-intervention residual activations to predict the correct WMDP-Bio answer.
- **Hypothesis**: clamping only 20 SAE latents leaves the SAE's own reconstruction error
  term untouched, so forget-relevant information the SAE didn't capture could still leak
  through; a full-residual projection should close this gap and should be measurably better
  by the (previously unreported) probe-accuracy metric.
- **Implementation status**: fully coded and evaluated, after fixing 2 issues: a removed
  `sklearn` API argument (`LogisticRegression(multi_class=...)`, no longer accepted in the
  installed sklearn 1.9), and a probe-layer choice bug — probing at layer 3 (the
  intervention layer itself) gave a chance-level probe accuracy (24.44%) even for the
  **unintervened** base model, which is not evidence of "no leak," just evidence that layer
  3 is too early in the network to have linearly decided the answer yet. Fixed by moving
  the probe to layer 22 (verified above-chance base-model accuracy there first).
- **OUTCOME: NEGATIVE, but informative.** Both the behavioral (WMDP-Bio accuracy) and
  representational (probe accuracy) metrics agree: DSG's crude clamp is *more* effective at
  genuinely erasing information than the "more principled" projection.
- **Numbers** (`git show imp-c3-residual-projection-probing:novelty_trials/c3_residual_projection_probing/c3_results.json`):
  - `c3_results`: wmdp-bio=**0.946096658706665** (vs. baseline 0.2936802804470062 — almost
    no forgetting), high_school_us_history=1.0, college_computer_science=1.0,
    high_school_geography=1.0, human_aging=1.0; `mmlu_c3`=1.0.
  - `probe_accuracy`: no_intervention=**0.6333333333333333**,
    dsg_clamp=**0.17777777777777778** (below the 0.25 chance level — DSG's clamp doesn't
    just scramble the signal, it reliably steers toward wrong answers),
    c3_projection=**0.5666666666666667** (barely moved from the unintervened baseline).
  - `expanded_feature_count`=100.

---

## 4. COMPARISON TABLE: BASELINE vs. EACH IMPROVEMENT

All values are point estimates from a single run (`--random_seed 0` where applicable — see
Section 2 for the one exception). "Ours DSG" = the reproduced main config
(`clamp_..._nfeatures20_layer3_retainthres95_seed0.pkl` for Bio,
`..._nfeatures30_..._retainthres90_seed0.pkl` for Cyber). Bold = beats **both** the paper's
number and our own reproduced-DSG number on that row. "NOT RUN" = no result file exists for
that method/metric combination.

| Metric | DSG paper | Ours: DSG (Bio) | Ours: DSG (Cyber) | A1 (best variant = CUSUM) | A2 | A3 (Bonferroni) | B1 (hook removed) | B2 | B3 (hierarchical) | C1 | C2 | C3 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| WMDP-Bio ↓ | 29.64% | 29.368% | NOT RUN | 30.855%¹ | 99.257% | 25.465%¹ | 40.00%² | NOT RUN | 67.844% | 65.613% | **27.695%** | 94.610% |
| WMDP-Cyber ↓ | 26.74%³ | NOT RUN | 28.00% | NOT RUN | NOT RUN | 28.727% | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| MMLU avg (4-subject Bio config) ↑ | 99.34%³ | 99.412% | — | 98.235% | 100.0% | — | 64.7%² | NOT RUN | 87.7%⁴ | 99.757% | 97.941% | 100.0% |
| MMLU avg (4-subject Cyber config) ↑ | 99.73%³ | — | 99.435% | NOT RUN | NOT RUN | — | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| MT-Bench ↑ | 7.78 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| TOFU (any metric) | — | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| MUSE (any metric) | see Table 2, paper p.6 | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN | NOT RUN |
| Guardrail-bypass resistance (attack success at 150-word padding) ↓ | not evaluated by paper | 98.3% (no defense) | — | **20.0%** (windowed w=24) / 36.7% (CUSUM) | not applicable | not applicable | not applicable | — | not applicable | not applicable | not applicable | not applicable |
| Probe-recoverable leak (accuracy predicting correct answer post-intervention) ↓ | not reported by paper | not measured for plain clamp elsewhere | — | not applicable | not applicable | not applicable | not applicable | — | not applicable | not applicable | not applicable | 56.67% (vs. DSG clamp's own 17.78%, measured in the *same* C3 run) |

¹ A1's "WMDP-Bio" and A3's Bonferroni row use their own recalibrated feature sets/thresholds
evaluated on the full unfiltered WMDP-Bio question set used by each script, not the same
538-item subset as the main config — see each script's own README for exact methodology;
figures are taken as-run, not renormalized to match.
² B1's numbers come from a 200-question **random** sample of the full 1,273-item WMDP-Bio
set (not the 538-item curated "correct" subset used for the main DSG numbers) — see B1's
own README for why this offset from the headline 29.368%/67.5% base rate is expected, not a
bug. MMLU figure is the mean of B1's own 4 retain-subject numbers (0.68, 0.48, 0.7133,
0.6467); B1 has no "before training" baseline on the same sample (flagged in Section 3).
³ Not independently re-located in the PDF during this pass (see Section 2/8) — quoted from
`figures/reproduction/make_figures.py` and prior repo notes.
⁴ Mean of B3's own 4 original-retain-subject numbers under the hierarchical gate (1.0, 1.0,
1.0, 0.9882352948188782); the three *neighbor* subjects (college_biology, virology,
anatomy) are reported separately in Section 3 and are not folded into this cell.

**No cell in this table beats both the paper's number and our own reproduced-DSG number on
the same metric with a matched evaluation protocol.** C2's 27.695% WMDP-Bio beats both
29.64% (paper) and 29.368% (ours) in isolation, but its own MMLU (97.941%) is below both
99.34% (paper) and 99.412% (ours) on the same run — a Pareto trade, not a strict win (see
Section 5 framing note).

---

## 5. THE 2 POSITIVE RESULTS — DEEP DIVE

Two branches are marked POSITIVE in Section 3: **B1** (distillation into weights) and
**A1's CUSUM detector** (dilution-invariant defense). Neither is a strict, no-cost win — both
sections below say so plainly.

### B1 — Distill DSG guardrail into LoRA weights

**Full metric table** (all from `novelty_trials/b1_distill_weights/b1_results.json` on
`imp-b1-distill-weights`; single run, no seeds/std reported):

| Condition | WMDP-Bio accuracy | Retain: HS US History | Retain: College CS | Retain: HS Geography | Retain: Human Aging |
|---|---|---|---|---|---|
| Base model, no intervention | 0.675 | not measured | not measured | not measured | not measured |
| DSG runtime hook (ported to HF) | 0.455 | not measured | not measured | not measured | not measured |
| **Student, hook removed** | **0.400** | 0.68 | 0.48 | 0.7133333333333334 | 0.6466666666666666 |

Training loss: first step 2.181845188140869, last step 0.4138040542602539 (150 steps,
batch size 2, fixed LR 2e-4, noisy but trending down per the branch's own log excerpt).

**Mechanism, in plain terms**: a LoRA adapter (rank 8, ~10.4M trainable params of ~2.6B
total, per the branch's `print_trainable_parameters()` output referenced in its script) is
trained so that, on forget-domain prompts, its output distribution matches what the model
produces *with the DSG clamp hook active*; on retain-domain prompts, it's trained to match
the plain, unclamped model. Because this is a weight update (not a runtime check), the
resulting behavior no longer depends on the hook being present at inference time.

**Cost vs. vanilla DSG**:
- Compute: one training run (150 steps, batch 2) — DSG itself needs zero gradient-based
  training. This is a categorical cost DSG did not have.
- Complexity: requires porting the clamp hook to plain PyTorch/HuggingFace (DSG normally
  needs TransformerLens), a KL-distillation training loop, and — per the branch's own
  README — took 3 debugging cycles (dtype mismatch, an unnecessary-gradient memory bug,
  and a genuine cross-iteration memory leak) before it ran stably.
- Permanence gained: the entire point — DSG's forgetting is removable by dropping the hook;
  B1's is not (within the scope tested).

**Robustness**: tested under exactly one condition (150 training steps, one LR, one batch
size, one random 200-question eval sample). **Not tested**: different forget-set sizes,
different MMLU subjects beyond the 4 already in the main config, a relearning/fine-tuning
attack against the distilled student, or quantization. This is a first positive signal, not
a validated robust result.

**Clean win or trade-off?** Not a clean win: the retain numbers above (48–71%) are well
below the ~97–100% the same subjects show under the official curated-subset pipeline
elsewhere in this repo, **but B1's own README explicitly flags that it never collected a
"before training" retain baseline on this same random sample**, so it cannot separate
training-induced retain damage from the same random-vs-curated-sample gap already visible
in the 67.5% WMDP-Bio base-model number (vs. ~99%+ on the curated subset). This is an open
gap, not a resolved trade-off — state it exactly this way if asked.

### A1 (CUSUM streaming detector) — Dilution-invariant guardrail defense

**Full metric table** (`novelty_trials/a1_streaming_detector/03_cusum_detector_results.json`
on `imp-a1-streaming-detector`; single run, no seeds/std reported):

| Padding | Mean max CUSUM statistic S | Attack success rate (n=60) | WMDP-Bio (utility eval) | MMLU avg (utility eval) |
|---|---|---|---|---|
| 0 words | 1317.3150615821282 | 0.35 | 0.3085501790046692 | 0.9823529422283173 |
| 150 words | 945.77834020257 | 0.36666666666666664 | (same run) | (same run) |
| 400 words | 920.0864077051481 | 0.36666666666666664 | (same run) | (same run) |
| 800 words | 913.7738584041596 | 0.36666666666666664 | (same run) | (same run) |

Per-subject utility breakdown (same file): wmdp-bio=0.3085501790046692,
high_school_us_history=1.0, college_computer_science=1.0, high_school_geography=1.0,
human_aging=0.929411768913269.

**Mechanism, in plain terms**: instead of one average over the whole sequence, a per-token
log-likelihood-ratio score (forget-feature-distribution vs. retain-feature-distribution) is
accumulated token-by-token with a CUSUM (cumulative-sum) statistic that resets toward zero
whenever evidence stops accumulating. Because it resets on negative evidence rather than
being smoothed by a fixed-size average, a long run of unrelated padding tokens doesn't drag
the statistic down the way a whole-sequence or windowed average does.

**Cost vs. vanilla DSG**: requires fitting per-feature forget/retain activation
distributions (a one-time offline step) and running a sequential (token-by-token, not
vectorizable in the same way as a single threshold check) accumulation at inference —
concretely slower per-token than a single whole-sequence comparison, though no wall-clock
benchmark of this specific detector exists in the repo (the latency benchmark in Section 2
measures the original DSG clamp only, not this CUSUM variant — NOT RUN for CUSUM).

**Robustness**: tested at 4 padding amounts (0/150/400/800 words) on a fixed 60-question
sample; not tested against non-padding attack styles (paraphrase, jailbreak rewrites — both
explicitly named in the original spec handed to this session), and not tested at padding
amounts beyond 800 words.

**Clean win or trade-off?** Not a clean win: the CUSUM detector's own no-padding bypass
rate (35%) is *worse* than the windowed-gate alternative's no-padding rate (1.7%, from
`02_windowed_gate_w24_results.json`, "0" key). CUSUM trades baseline sensitivity for
padding-invariance; it does not dominate the windowed gate on every axis. State this
explicitly if presenting CUSUM as "the" defense — it is one of two partial defenses with
different failure profiles, not a strictly better replacement for the windowed gate.

---

## 6. ALL FIGURES AND PLOTS AVAILABLE

All in `baselines_DSG/figures/` on `main` (none exist on any of the 9 improvement
branches — confirmed via `git ls-tree -r <branch> --name-only | grep figures` returning
nothing for all 9).

| Path | What it shows | Bars/lines | Quality |
|---|---|---|---|
| `figures/SAE_plot.png` | Upstream paper's own DSG architecture overview diagram (not generated by this reproduction) | N/A (diagram) | Publication-quality (paper's own figure) |
| `figures/reproduction/fig1_data_efficiency.png` | WMDP-Bio and MMLU-avg vs. % of forget/retain data used, our reproduction only | 2 lines × 7 points | Clean, single-panel, legible at quarter-page |
| `figures/reproduction/fig2_zeroshot_tau_sweep.png` | Zero-shot τ sweep, Bio (left) and Cyber (right) panels | 2 panels × 2 lines × 9 points | Two-panel — legible at half-page, tight at quarter-page (small annotation text) |
| `figures/reproduction/fig3_clamp_ablation.png` | Clamp-strength ablation, log-x axis | 2 lines × 8 points | Clean, single-panel |
| `figures/reproduction/fig4_threshold_ablations.png` | p_ratio sweep (left) and p_dyn sweep (right) | 2 panels × 2 lines × 5-6 points | Two-panel, same legibility note as fig2 |
| `figures/reproduction/fig5_rho_vs_rho_raw.png` | ρ vs. ρ_raw total-variation-distance bar chart, 2 comparisons | 2 groups × 2 bars, with error bars | Clean, single-panel |
| `figures/reproduction/fig6_latency_overhead.png` | Inference overhead %, ours (RTX 2000 Ada) vs. paper (A6000), 3 sequence lengths | 3 groups × 2 bars | Clean, single-panel |
| `figures/reproduction/fig7_main_comparison.png` | RMU / Farrell et al. / DSG-zero-shot / DSG-full bar chart, Bio (left) and Cyber (right) | 2 panels × 4 bars each | **Caution: "DSG (full)" bars use the paper's OWN 29.64%/26.74% numbers, not our reproduced 29.368%/28.00% — this figure does not plot "our reproduction," despite its title suggesting a reproduction comparison. Verify this framing before presenting it as "our result."** (`figures/reproduction/make_figures.py`, lines 249-250) |

**No figure exists for any of the 9 improvement branches** — none of the branches contain a
`figures/` directory or any `.png`/plotting script (confirmed via `git ls-tree -r <branch>`
for all 9). Any chart showing A1, A2, A3, B1, B3, C1, C2, or C3 numbers would need to be
built from scratch from the JSON files cited in Sections 3–4; none currently exists in the
repo.

### Top picks for a 2-page, space-limited report

1. **`fig7_main_comparison.png`** — the single best "does this method beat prior work"
   figure, IF the caption is corrected to state that the "DSG (full)" bar is the paper's own
   number, not this reproduction's. With that caveat added, it's the clearest way to place
   DSG (and, by extension, why it was worth reproducing) against RMU/Farrell for a panel
   audience. Reads fine shrunk — bars with printed value labels stay legible.
2. **`fig1_data_efficiency.png`** — the cleanest single-panel line chart, directly supports
   the "DSG is data-efficient" claim, and is the *only* figure in the repo that shows a
   result which continues to hold at reduced data fractions (data efficiency), a distinct
   claim from the headline number. Two lines only, shrinks well.
3. **`fig3_clamp_ablation.png`** — single-panel, directly supports the robustness-to-clamp-
   strength claim, and pairs naturally with a spoken statement of the WMDP-Bio ablation
   numbers in Section 3 of the paper (Figure 11 in the original). Legible at quarter-page.
4. **`fig5_rho_vs_rho_raw.png`** — the most "mechanistic" figure (why ρ rather than a raw
   count), useful if the panel is likely to ask "why this specific statistic." Bar charts
   with error bars read cleanly even small.

**Explicitly not recommended for a space-limited slide**: `fig2_zeroshot_tau_sweep.png` and
`fig4_threshold_ablations.png` — both are two-panel figures with small in-plot annotation
text ("paper-optimal τ=0.6", "paper's sweet spot") that will not survive shrinking to
quarter-page; use them only if a full page or two half-pages are available.

**A gap worth stating directly to the panel**: there is no figure anywhere in this repo for
either of the two positive results (B1, A1/CUSUM) — Sections 5's tables above are the only
place those numbers currently exist in comparable form.

---

## 7. RAW NUMBERS APPENDIX

Every result file's key contents, verbatim from source. All values re-read directly from
the cited file during this report's preparation (via `cat` for JSON, a read-only `pickle.load`
for `.pkl`, and `git show <branch>:<path>` for files that only exist on an improvement
branch — no file was written to or modified to produce these reads).

### `ablations/dsg_clamp_ablation_results.json` (main)
features (20): `[8459, 10229, 9953, 12260, 794, 6481, 8908, 9398, 11392, 9292, 6020, 8786, 9986, 6687, 14821, 1676, 8802, 4235, 12407, 6673]`
threshold: `0.5458007812500021`
| c | wmdp-bio | mmlu_avg |
|---|---|---|
| 10 | 0.6412639617919922 | 0.9970588237047195 |
| 25 | 0.3085501790046692 | 0.9941176474094391 |
| 50 | 0.29925650358200073 | 0.9941176474094391 |
| 100 | 0.29925650358200073 | 0.9941176474094391 |
| 200 | 0.3011152446269989 | 0.9941176474094391 |
| 300 | 0.2955390214920044 | 0.9941176474094391 |
| 400 | 0.29739776253700256 | 0.9941176474094391 |
| 500 | 0.2936802804470062 | 0.9941176474094391 |

### `ablations/dsg_threshold_ablations_results.json` (main)
p_ratio_sweep: 75→wmdp=0.5855018496513367,mmlu=1.0,thr=0.7736328125000004;
80→wmdp=0.6022304892539978,mmlu=1.0,thr=0.7712890625000004;
85→wmdp=0.43308550119400024,mmlu=1.0,thr=0.6559570312500016;
90→wmdp=0.37546467781066895,mmlu=1.0,thr=0.7335937500000014;
95→wmdp=0.2936802804470062,mmlu=0.9941176474094391,thr=0.5458007812500021.
p_dyn_sweep: 60→wmdp=0.23605947196483612,mmlu=0.6187316328287125,thr=0.107421875;
70→wmdp=0.23605947196483612,mmlu=0.6187316328287125,thr=0.1201171875;
80→wmdp=0.23605947196483612,mmlu=0.6233612596988678,thr=0.158203125;
90→wmdp=0.24535316228866577,mmlu=0.8769845962524414,thr=0.3158203125;
95→wmdp=0.2936802804470062,mmlu=0.9941176474094391,thr=0.5458007812500021;
97→wmdp=0.8289963006973267,mmlu=1.0,thr=0.8125195312499998.

### `ablations/dsg_rho_comparison_results.json` (main)
n_wikitext=275, n_mmlu=160, n_wmdp_bio=160.
wikitext_vs_mmlu: rho mean=0.4663102207857963,std=0.03890225775640721;
rho_raw mean=0.8293635741081632,std=0.014970278022744084.
wikitext_vs_wmdp_bio: rho mean=0.8755800073564993,std=0.020789599542691585;
rho_raw mean=0.18630523511744804,std=0.041414957950377564.

### `zeroshot_reproduction/dsg_zeroshot_bio_results.json` (main)
τ=0.1: wmdp=0.25836431980133057,mmlu_avg=0.24194323271512985
τ=0.2: wmdp=0.25836431980133057,mmlu_avg=0.24194323271512985
τ=0.3: wmdp=0.26022303104400635,mmlu_avg=0.25944751873612404
τ=0.4: wmdp=0.2639405131340027,mmlu_avg=0.39584894105792046
τ=0.5: wmdp=0.2769516706466675,mmlu_avg=0.7697062194347382
τ=0.6: wmdp=0.31040892004966736,mmlu_avg=0.9628709554672241
τ=0.7: wmdp=0.4553903341293335,mmlu_avg=0.9970588237047195
τ=0.8: wmdp=0.8513011336326599,mmlu_avg=1.0
τ=0.9: wmdp=1.0,mmlu_avg=1.0

### `zeroshot_reproduction/dsg_zeroshot_cyber_results.json` (main)
τ=0.1: wmdp=0.37090909481048584,mmlu_avg=0.7153932750225067
τ=0.2: wmdp=0.4145454466342926,mmlu_avg=0.9840476363897324
τ=0.3: wmdp=0.6436363458633423,mmlu_avg=1.0
τ=0.4: wmdp=0.9599999785423279,mmlu_avg=1.0
τ=0.5–0.9: wmdp=1.0,mmlu_avg=1.0 (all five)

### `data_efficiency_reproduction/dsg_dataefficiency_bio_results.json` (main)
100%: wmdp=0.29368,mmlu=0.99412
80%: wmdp=0.2769516706466675,mmlu=0.9705882370471954
60%: wmdp=0.3475836515426636,mmlu=1.0
40%: wmdp=0.321561336517334,mmlu=0.9911764711141586
20%: wmdp=0.33643123507499695,mmlu=0.967647060751915
10%: wmdp=0.24907062947750092,mmlu=0.8208737894892693
5%: wmdp=0.9330855011940002,mmlu=0.9488218426704407

### `latency_benchmark/dsg_latency_results.json` (main)
seq=256: original 0.13697422103883583±0.00018669941460214222, dsg 0.14762326257943642±0.00037736958632405746, overhead 7.7744859286926%
seq=512: original 0.1789338603305805±0.0006144078093406075, dsg 0.20160625518939923±0.000990928676337389, overhead 12.670824189972455%
seq=1024: original 0.29568043660969123±0.0014295252880841556, dsg 0.33651115919070435±0.0009247055914102332, overhead 13.809071391121872%

### Main Bio/Cyber `.pkl` metric files (read via `pickle.load`, not modified)
`artifacts_dynamic_bs1_bio/.../metrics/clamp_feature_activation_multiplier500_nfeatures20_layer3_retainthres95_seed0.pkl`:
wmdp-bio mean_correct=0.2936802804470062 (n=538, total_correct=158);
high_school_us_history=1.0 (n=108); college_computer_science=1.0 (n=9);
high_school_geography=1.0 (n=103); human_aging=0.9764705896377563 (n=85, total_correct=83).
features=[8459,10229,9953,12260,794,6481,8908,9398,11392,9292,6020,8786,9986,6687,14821,1676,8802,4235,12407,6673];
multiplier=500; threshold=0.5458007812500021.

n_features ablation rows, same directory, same threshold-selection method (n_features=50/100/200
`.pkl` files, full 4-subject MMLU average computed directly from source in this pass — see
Section 2 correction of `STATUS_FOR_REPORT.md`'s human-aging-only figures):
| n_features | wmdp-bio | mmlu_avg (4-subject, computed this pass) | history | college_cs | geography | human_aging |
|---|---|---|---|---|---|---|
| 20 | 0.293680 | 0.994118 | 1.0 | 1.0 | 1.0 | 0.9765 |
| 50 | 0.327138 | 0.958824 | 1.0 | 1.0 | 1.0 | 0.8353 |
| 100 | 0.314126 | 0.952941 | 1.0 | 1.0 | 1.0 | 0.8118 |
| 200 | 0.258364 | 0.770452 | 1.0 | 0.8889 | 0.9223 | 0.2706 |

`_PRESOFTCAPFIX.pkl` variants of all 4 rows above: **identical accuracy values to the
current files in every case** — confirms the softcap fix changed probability precision
only, never any accuracy number, across the full n_features sweep (not just n=20).

`artifacts_dynamic_bs1_cyber/.../metrics/clamp_feature_activation_multiplier500_nfeatures30_layer3_retainthres90_seed0.pkl`
(current/fixed): wmdp-cyber=0.28 (n=275); high_school_us_history=0.990741 (n=108);
college_biology=0.986667 (n=75); high_school_geography=1.0 (n=104); human_aging=1.0 (n=84).
threshold=0.1855339403973509, n_features=30.

`..._ORIGINAL_with_feature1312.pkl` (broken, pre-fix): wmdp-cyber=0.28 (unchanged);
high_school_us_history=0.953704; **college_biology=0.266667; high_school_geography=0.221154;
human_aging=0.345238**. threshold=0.1416015625000001.
Exact feature-set diff (computed this pass, not previously quantified in any note read):
removed {1312, 6545, 5765, 6175}, added {10721, 467, 6761, 5149} — **4 of 30 features
differ, not the single feature the filename suggests.**

`..._PRESOFTCAPFIX.pkl` (Cyber): identical to the current/fixed file in every field.

### 9 improvement-branch result files
All reproduced verbatim in Section 3 above, each tagged with its exact `git show
<branch>:<path>` source. Not re-duplicated here to avoid a second, easier-to-desync copy —
Section 3 is the canonical location for these numbers.

---

## 8. OPEN QUESTIONS / WHAT'S STILL UNCLEAR

1. **Venue conflict**: PDF footer says ICML 2025 (PMLR 267); repo's own `README.md` and two
   commit messages say COLM 2025. Confirm which is correct before citing the venue in the
   report.
2. **Paper's Cyber-side numbers (26.74% WMDP-Cyber, 99.73% MMLU) and its RMU/Farrell
   WMDP-Cyber numbers (88.00%, 52.73%)** were not re-located in the 33-page PDF during this
   pass (only the Bio row of Table 1, page 5, was visually confirmed). They are used
   throughout `figures/reproduction/make_figures.py` and prior notes — verify against
   Appendix G.2 of the paper directly before defending them as paper-verified.
3. **Scope of the paper's Table 1 "All" MMLU column (99.34%) is unclear**: it is not
   obviously a plain average of the 4 displayed subject columns (which show 100.00 for
   DSG on all 4) — it may be a full-MMLU (57-subject) average rather than the same
   4-subject scope our reproduction uses. If so, our 99.412% and the paper's 99.34% are
   **not measuring the same thing** despite looking comparable in Section 2/4's tables.
   Could not resolve this from the pages read.
4. **Whether the 538/1273 WMDP-Bio filtering matches the paper's own definition of "raw" or
   "normalized" accuracy** could not be determined — the paper's exact filtering procedure
   was not located in the pages reviewed, and the filtering logic in this repo is inherited,
   unmodified, unexplained-in-comments code (`calculate_MCQ_metrics`,
   `target_metric="correct"`).
5. **The claim that a real OOM test blocked full-parameter fine-tuning** (cited in
   `STATUS_FOR_REPORT.md` as the reason MUSE/sequential-unlearning/relearning-attack work was
   never attempted) **could not be independently located** in any markdown file, script
   output, or the `DSG_Claude_Code_Full_Handoff.docx` paragraph/table text searched during
   this pass. It may exist in a part of the docx not captured by a plain paragraph/table
   text search (e.g., inside an image), or may only exist in prior session context not
   written to any file. Treat as **unverified** until the docx is checked directly by a
   human, or re-run the test yourself.
6. **MT-Bench "not authorized" claim**: same status as #5 — asserted in
   `STATUS_FOR_REPORT.md`, not independently located in a file during this pass.
7. **Cyber's p_ratio=90 (vs. Bio's 95) and the exact retain corpus construction seed**:
   sourced only from `STATUS_FOR_REPORT.md`; no Cyber-side ablation JSON exists in this repo
   to cross-check the p_ratio value the way Bio's was cross-checked against
   `dsg_threshold_ablations_results.json`.
8. **No aggregate GPU-hour total exists anywhere** for either the baseline reproduction or
   the 9 improvement attempts. Individual run durations are recorded ad hoc for some Cyber
   runs only (per `STATUS_FOR_REPORT.md`, itself sourced from the docx); Bio's full-run
   duration and all 9 improvement branches' run durations are **not recorded in any file**.
9. **`eval_results/unlearning_dynamic_bs1/` is empty** — the library's own designated
   summary-JSON output was apparently never produced or never committed; every number in
   this report instead comes from the lower-level `.pkl` files directly, which is a real
   deviation from the documented output structure in `README.md`.
10. **B1's retain-accuracy numbers (48-71%) have no matched "before training" baseline** on
    the same random sample — flagged already in Sections 3 and 5, repeated here because it
    is the single most important caveat on the project's one clean positive weight-editing
    result.
11. **No figure exists for either positive result (B1 or A1/CUSUM)** — if the report needs a
    visual for these, one must be built new; nothing in the repo can be reused as-is.
12. **Full WMDP-Bio dataset size (1,273)** is quoted from `STATUS_FOR_REPORT.md` only, not
    independently re-counted from the raw HuggingFace dataset in this pass.
