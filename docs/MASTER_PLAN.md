# MASTER PLAN: DSG Unlearning Capstone, Full Roadmap Execution (Lab PC)

Version 2.1, 2026-10-01 (P1 split into P1a/P1b; pause after Wave 1). Target repo: `/home/amaloch/projects/mechunlearn-project/baselines_DSG`.

**Changes in v2.0:**
- Each experiment runs on its own branch cut from the baseline, using git worktrees.
- Everything runs inside named tmux sessions.
- Built-in progress tracking you can check every hour.
- Stronger safeguards.
- All 48 GB server work is removed. Items that need more than this PC are marked **DEFERRED** and will be planned separately.

This file has two parts:

- **Part I (sections 1 to 9)** is the specification. Claude Code treats it as the source of truth.
- **Part II (section 10)** holds the prompts you paste into Claude Code, in order.

---

## 0. How this works (read this first)

### 0.1 Who does what

- **Claude Code** writes the code, tests it, builds a job queue and starts it. Then the Claude Code session ends.
- **The scheduler** is a small Python program that Claude Code writes. It runs in a tmux session on the lab PC. It starts each job, waits for it to finish, records the result and starts the next one, day and night, without Claude Code. If it crashes, a supervisor loop restarts it within 30 seconds.
- **You** check progress whenever you like with one command (section 0.3), and paste the short status file into the planning chat.
- **Claude Code comes back** only when you open it and paste Prompt P3, to fix failed jobs, or P4, to write the final report. Claude does not spawn itself or watch the queue on its own.

### 0.2 Order of use

1. Copy this file into the repo as `docs/MASTER_PLAN.md` on main. Commit nothing else.
2. Open Claude Code at `baselines_DSG/` and paste **P1a**. It builds the foundation (harness, queue, monitoring) and checks that the baseline reproduces. Send `FOUNDATION_REPORT.md` to the planning chat.
3. After approval, paste **P1b**. It builds every experiment branch and smoke-tests it. Send `SMOKE_REPORT.md` to the planning chat.
4. Paste **P2**, which launches the queue in tmux and exits. The queue **pauses after Wave 1** (the baselines) so the numbers can be checked; then resume with `python -m dsgx.queue.resume`.
5. Check progress every hour with section 0.3. No Claude needed.
6. If the status shows failures, open Claude Code and paste **P3**. Repeat as often as needed.
7. When everything is done, paste **P4**. It writes `results/FINAL_REPORT.md`.

### 0.3 Hourly progress check (you, about 1 minute)

```
cd /home/amaloch/projects/mechunlearn-project/baselines_DSG
conda activate mechunlearn2
python -m dsgx.queue.status          # summary table: done / running / failed / waiting, per experiment, ETA
python -m dsgx.queue.status --chat   # short version to paste into the planning chat
tmux attach -t dsg-monitor           # optional live dashboard (detach with Ctrl+b then d)
```

These files are also refreshed automatically:

- `results/STATUS.md`: full status, refreshed every 2 minutes.
- `results/STATUS_FOR_CHAT.md`: 40 lines or fewer, made for pasting into the planning chat.
- `results/status_snapshots/`: an hourly copy of `STATUS.md`, so you can see progress over time.

### 0.4 Expected compute

The full local queue is probably **4 to 8 days** nonstop on the RTX 2000 Ada. P1 replaces this estimate with measured timings. Jobs run in priority order, so "must" items finish first.

---

# PART I: SPECIFICATION

## 1. Decisions on the open questions from REPO_REPORT.md section 7

Apply these. If one proves impossible, log the reason in `results/DEVIATIONS.md` and continue.

| # | Question | Decision |
|---|---|---|
| 1 | COLM vs ICML venue of DSG | Check the official arXiv or proceedings listing and use the correct venue everywhere. The README says COLM 2025. Record it in the final report. |
| 2 | ID clash | Always use **roadmap IDs** (A1 … N10, T1 …) from section 5. Branch names from round 1 and 2 are legacy (mapping in 5.0). |
| 3 | Cyber retain swap | Report two Cyber baselines: **DSG-faithful** (WikiText retain) and **DSG-chatretain**. Rebuild the chat-retain corpus with a committed, seeded script whose subjects are disjoint from all evaluation subjects. |
| 4 | Canonical utility | Primary: **raw accuracy on full MMLU** (57 subjects) minus hazard-adjacent subjects. Bio excludes college_biology, high_school_biology, college_medicine, medical_genetics, virology, anatomy, clinical_knowledge, nutrition and professional_medicine. Cyber excludes computer_security, college_computer_science, high_school_computer_science, machine_learning and electrical_engineering. Secondary: the **DSG-style correct-subset** numbers, pooled by question count (also store unweighted). Flag college_cs n=9. |
| 5 | Splits | Fixed, **seeded** (seed 0) **dev** (tuning and selection) and **test** (reporting only) splits for every evaluation set, saved in `data/splits/`. Never select on test. |
| 6 | Seeds and CIs | **3 seeds** per config, **5 seeds** for headline results. **95% bootstrap over questions** (10,000 resamples). **Paired bootstrap and McNemar** for method comparisons. Also report the mean ± CI across seeds. |
| 7 | Debug prints and paths | Put `[DSG DEBUG]` behind `DSG_DEBUG=1` (default off). Remove every `/tmp/claude-*` path. All outputs go to `$DSG_RESULTS`. |
| 8 | Generation gating | Canonical: **re-evaluate the gate every step on prompt + generated text**, using the KV cache and incremental counts. Prompt-only gating is an ablation. |
| 9 | MT-Bench judge | One fixed judge for all comparisons. Absolute numbers are labelled not comparable with the paper's 7.78. Low priority. |
| 10 | Environment | Clone the env to `mechunlearn2` and `pip install -e dynamic_sae_guardrails`. Keep numpy unless it breaks something. Write `requirements-lock.txt` and `ENV.md`. |
| 11 | Relearning | **LoRA relearning only** on this PC (ranks 8 and 64). Full fine-tuning relearning is DEFERRED. |
| 12 | Legacy leakage | Re-implement every legacy idea inside the harness with clean splits. Old JSON numbers are legacy only. |

## 2. Objectives

### 2.1 Thesis

**Gated SAE unlearning (DSG) hides hazardous knowledge rather than erasing it.**

- **Part I, Break:** stress-test DSG with five attack axes plus generation-time leakage.
- **Part II, Explain:** a mechanistic diagnosis of each failure.
- **Part III, Fix:** a hardened gate for API use, plus baked-in erasure for open weights. Each fix is tested against Part I attacks and tampering.

### 2.2 Claims and pass criteria

The final report marks each claim **Supported / Not supported / Inconclusive**, with evidence.

| Claim | Supported if | Main experiments |
|---|---|---|
| C-H1. Simple dilution defeats DSG's gate. | Attack success ≥ 50% at some padding on test, with the CI excluding 0. A base-model padding control must show the knowledge is still answerable. | B1, T1 |
| C-H2. DSG fails beyond English MCQ. | On at least one of open QA, cross-lingual or decomposition, recovered accuracy on gated test items is significantly above DSG's in-distribution number (paired p < 0.05). | A2, B2, B3, B6 |
| C-H3. Knowledge remains internally. | Probes and logit lens recover answers from DSG-guarded activations well above chance, at a level similar to the base model. | A4 |
| C-H4. Failures have an interpretable cause. | At least one failure mode is explained by a measured SAE property, with a correlation and CI. | N9, C1, C3, A4 |
| C-H5. A hardened gate improves robustness at matched utility. | It significantly cuts attack success versus DSG on at least 3 of 5 axes, with full-MMLU utility within 1 point and benign FPR ≤ 5% (N6). | C2, C3, C6, N5, N6, N7 |
| C-H6. Baked erasure resists tampering. | After fixed-budget LoRA relearning, forget accuracy recovers significantly less than for DSG-without-hook and the legacy LoRA student, at matched utility. | D1-local, D2, A6 |
| C-H7. Findings generalise. | The DSG-vs-best-fix ranking holds on Gemma 3 (1B or 4B) and on TOFU. | A7, A2 |

### 2.3 Deliverables

1. The `dsgx` harness, with one branch per experiment.
2. Every local experiment in section 5, logged per section 7.
3. `results/FINAL_REPORT.md`, `results/summary.json` and `results/figures/`.
4. Public-ready local artifacts: N1 toolkit, N3 demo, N4 challenge app and N10 audit cards. None of these are published.

## 3. Global rules

1. **Branch model (baseline-only):**
   - `main` is the untouched baseline. Never commit to it except `docs/MASTER_PLAN.md`.
   - `v2-harness` is cut from main and contains only the shared harness (`dsgx/`), tests and queue. It has no experiment-specific code.
   - **One branch per experiment ID**, each cut from `v2-harness`: `exp/A1-baselines`, `exp/B1-dilution`, …, `exp/N9-sae-quality`. Each branch contains only main + harness + that experiment's code and config. No experiment branch merges another experiment branch.
   - Each experiment branch is checked out as a **git worktree** in `../dsg_worktrees/<branch>`, so all branches can run at the same time without switching branches.
   - Jobs record the exact commit hash they ran on.
   - **Harness bug fixes** go to `v2-harness` first. Then `scripts/sync_harness.sh` merges them into the experiment branches whose jobs are not running, and lists the finished jobs that must be re-run.
   - Push only `v2-harness` and `exp/*` branches. Never main, never force-push.
2. **Shared locations outside git:**
   - `DSG_CACHE=/home/amaloch/projects/mechunlearn-project/dsg_cache`: model-independent caches, the new activation cache and the translation cache.
   - `DSG_RESULTS=/home/amaloch/projects/mechunlearn-project/dsg_results`: every run, status file and figure.
   - `DSG_PRIVATE=/home/amaloch/projects/mechunlearn-project/dsg_private`: hazardous generations, never pushed.
   - All worktrees read and write these absolute paths. `results/` in this file means `$DSG_RESULTS`.
3. **One harness, all methods.** Port legacy logic into the harness method and attack registries. There is one clamp implementation, with flags.
4. **Data hygiene.**
   - Tune on dev and report on test.
   - Calibration corpora must be disjoint from evaluation questions.
   - `dsgx/checks/leakage.py` runs before every job group, and a failure blocks that group.
5. **Two evaluation views every time:**
   - raw full-set accuracy;
   - DSG subset (base-correct on all 24 answer permutations).
6. **Statistics** as in decision 6. No improvement is claimed without a paired test. Report n for every number.
7. **Fair comparison.** Same prompts, templates, filters, splits and token budget. All gates are calibrated to the same benign FPR (5% default) unless an experiment sweeps it.
8. **Safety.**
   - Hazardous generations and the prompts that elicit them go only to `$DSG_PRIVATE`.
   - Public artifacts contain transformation code and aggregates only.
   - N3 and N4 use the fictitious TOFU domain only.
9. **Resources (16 GB VRAM, 62 GB RAM, 36 CPUs, about 205 GB free disk).**
   - Never load the old 18 GB pickles.
   - At most 2 concurrent GPU eval jobs, and only if measured VRAM allows. Training jobs are GPU-exclusive.
   - CPU analysis runs in parallel with up to 24 workers.
10. **Honesty.** Failures and impossibilities go to `results/DEVIATIONS.md`. Never fabricate or interpolate numbers.

## 4. Infrastructure (built in Phase 1)

### 4.1 Package layout (on `v2-harness`)

```
dsgx/
  config/  data/  models/  methods/  attacks/  eval/  analysis/  logging/
  queue/       # job spec, scheduler, worker, supervisor, status, watchdog
  checks/      # leakage, sanity, preflight
configs/experiments/<ID>.yaml   # lives on each exp branch
scripts/  sync_harness.sh  make_worktrees.sh  tmux_up.sh  tmux_down.sh  preflight.sh
tests/       # pytest, under 2 minutes total
```

### 4.2 Harness core

- `run(config) -> run_dir`.
- **Model-load sharing:** an eval worker loads the model and SAE once and runs many configs.
- **DSG-faithful method** is an exact reproduction of main's logic. **dsg-fixed** (BOS and padding excluded, per-feature mask) is an ablation.
- **Sanity gate:** reproduce Bio N=20, seed 0, DSG-subset view: WMDP 0.2937, MMLU-u 0.9941 within one question, and τ = 0.5458 from the new cache.

### 4.3 Activation cache refactor

Stream activations once and save compact, memory-mapped statistics:

- per-feature mean, mean of squares, fire rate, max and quantile sketch;
- per-sequence fire-rate vectors for the top 2,000 candidate features;
- per-token activations for selected features on a subsample only.

Target under 4 GB RAM per job. Build the new cache alongside the old artifacts, and never delete the old ones.

### 4.4 Job queue, scheduler and tmux layout

**Job spec** (`$DSG_RESULTS/queue/jobs/<job_id>.json`):

- id, experiment ID, branch, worktree path, commit, config path;
- est_vram_gb, est_ram_gb, est_minutes, deps, priority, gpu_exclusive;
- items_total (for progress).

**Tmux sessions.** There are exactly three, created by `scripts/tmux_up.sh` and removed by `scripts/tmux_down.sh`:

| Session | Windows | Purpose |
|---|---|---|
| `dsg-queue` | `supervisor` | `while true; do python -m dsgx.queue.scheduler; sleep 30; done`, with output logged to `results/logs/scheduler.log` |
| `dsg-workers` | `gpu0`, `gpu1`, `cpu` | The scheduler launches each job in one of these windows (via `tmux send-keys` or a subprocess bound to the window's log). Each job's stdout and stderr go to `<run_dir>/job.log`. |
| `dsg-monitor` | `status`, `gpu` | `watch -n 120 python -m dsgx.queue.status` and `watch -n 10 nvidia-smi` |

**Scheduling:**

- Start a job only if its dependencies are done and free VRAM ≥ estimate + 1.5 GB margin, and RAM and disk allow.
- Run "must" items first, then "should", then "stretch". Follow the waves in section 6.
- Group jobs by (model, SAE, layer) so one worker process runs several configs back to back.
- **Lock file:** only one scheduler may run (`$DSG_RESULTS/queue/scheduler.lock` holding its PID). A second scheduler exits immediately.

### 4.5 Progress tracking (must exist before launch)

Each running job writes `<run_dir>/progress.json` every 30 seconds, atomically (write a temp file, then rename). It contains: job_id, phase, items_done, items_total, train step / total steps, start time, last-update time, ETA and current metric.

`python -m dsgx.queue.status` reads all job and progress files and prints:

1. **Overall:** done / running / waiting / failed / blocked counts, percent complete weighted by estimated minutes, and the overall ETA.
2. **Per experiment ID:** jobs done / total, percent, status, and the latest headline metric with CI.
3. **Running jobs:** items done / total, percent, elapsed time, ETA, heartbeat age.
4. **Failures:** job id and the last 3 error lines.
5. **Resources:** GPU memory and utilisation, RAM, free disk.

`--chat` prints a version of 40 lines or fewer.

The scheduler also writes `results/STATUS.md` and `results/STATUS_FOR_CHAT.md` every 2 minutes, and copies `STATUS.md` to `results/status_snapshots/YYYYMMDD-HH.md` every hour. `results/PROGRESS.md` gets one line per finished job: time, job, key metric ± CI, wall time.

### 4.6 Safeguards (make sure nothing goes wrong)

1. **Preflight** (`scripts/preflight.sh`) must pass before launch:
   - unit tests and the sanity gate pass;
   - the leakage check passes;
   - free disk ≥ 60 GB;
   - no other GPU processes are running;
   - every worktree is clean and on its recorded commit.
2. **Supervisor auto-restart** of the scheduler (section 4.4). Jobs survive a scheduler restart, and the restarted scheduler re-attaches using the PID files.
3. **Watchdog:**
   - A job whose heartbeat is older than 15 minutes is marked HUNG, killed, and retried once.
   - A job running longer than 3× its estimate is flagged in STATUS. It is killed only if its heartbeat is also stale.
4. **OOM handling:** retry once with half the batch size, then FAILED with the log. A failure never blocks unrelated jobs, only its dependents, which are shown as BLOCKED.
5. **Resource guards:**
   - Free disk below 30 GB pauses new jobs and raises an alert in STATUS.
   - RAM above 90% pauses new jobs.
   - GPU temperature above 85 °C pauses new jobs.
6. **Resume:**
   - DONE and FAILED markers make finished jobs skip on restart.
   - Training jobs checkpoint every N steps and resume from the last checkpoint.
   - All file writes are atomic.
7. **Code safety:**
   - Jobs run on pinned commits in their own worktree.
   - Nobody edits a worktree while its jobs run. Fixes are made by a new commit, then the affected jobs are re-queued (P3 does this).
   - `sync_harness.sh` refuses to touch a worktree with running jobs.
8. **Baseline canary:** every 24 hours, re-run the sanity gate as a low-cost job. If it drifts, pause the queue and raise an alert in STATUS.
9. **Reboot recovery:**
   - `scripts/tmux_up.sh` brings everything back, and resume markers prevent duplicate work.
   - Document this in `results/RUNBOOK.md`.
   - Do not install cron jobs without asking me; write the optional `@reboot` line in the runbook instead.
10. **Log rotation:** cap the scheduler log at 50 MB, rotated.

### 4.7 Smoke tests

Every experiment has a smoke config (tiny n, 1 seed, short training) that must pass before launch. `results/SMOKE_REPORT.md` lists for each one: status, time per item, peak VRAM and RAM, and the projected full runtime.

## 5. Experiment catalogue (one branch each)

Each entry gives priority and dependencies. Branch name = `exp/<ID>-<short-name>`.

### 5.0 Legacy branch mapping (port into the matching experiment branch, then re-run cleanly)

| Legacy branch | Ported into | Notes |
|---|---|---|
| imp-a1 | exp/B1 (attack) and exp/C2 (window, cusum gates) | Add a base padding control, test split, seeds |
| imp-a2 | exp/C5 (mean-ablation clamp) | — |
| imp-a3 | exp/C6 (per-concept Bonferroni) | Concept corpora from non-eval data |
| imp-b1 | exp/D1 (`lora-student-same-ref` baseline) | Both views, plus relearn and quantize |
| imp-b2 | exp/A2 (TOFU) | — |
| imp-b3 | exp/C6 (two-level) | Hazard gate calibrated on dev-only benign-bio text |
| imp-c1, imp2-1 | exp/C1 (attribution, chi²) | Attribution data from dev only |
| imp-c2, imp2-2 | exp/C3 and exp/N7 | Full layer sweep |
| imp-c3 | exp/D2 (activation-projection ablation) and exp/A4 (probes) | — |
| imp2-3 | exp/C2 (probe gate) | — |
| imp2-4 | exp/C5 (per-feature clamp) | — |
| imp2-5 | backlog (custom SAE) | Negative baseline |
| imp2-6 | exp/D2 (decoder orthogonalisation baseline) | — |
| lokesh-experimentations | exp/B6 | Use the canonical streaming gate |

### 5.1 Part A: Evaluation foundation

**A1 · Unified baselines** · must · deps: none

- DSG sweep on dev: N ∈ {10, 20, 30, 50, 100, 200}, retain percentile ∈ {90, 95}, multiplier ∈ {100, 500, 1000}, layer 3. Run on Bio, Cyber-faithful and Cyber-chatretain.
- Best dev config on test with 5 seeds, plus the base model.
- RMU reference only if a public RMU checkpoint for gemma-2-2b exists; otherwise DEFERRED.
- Optional: GradDiff, NPO and SimNPO references via OpenUnlearning, if they fit in 16 GB with LoRA.

**A2 · Open-ended QA** · must · deps: A1

- **WMDP-Bio-Open:** turn MCQ items into open questions, with the correct option text as the reference answer. Keep only items answerable without options.
- **TOFU:** LoRA-finetune gemma-2-2b-it on TOFU, then apply DSG with features from the forget set versus the retain set.
- **Grading:** fuzzy match, ROUGE-L, and embedding similarity. Add an LLM judge on 300 items per condition if an API key works; report agreement with the automatic metrics.
- **TOFU metrics:** forget quality, model utility, truth ratio and ROUGE, using OpenUnlearning's metric code where possible.

**A3 · Hard-negative bio utility** · must · deps: A1

- Subjects: college_biology, high_school_biology, anatomy, virology, medical_genetics, plus a benign-bio open-QA set.
- Measure the accuracy drop versus base and the gate false-positive rate per subject.

**A4 · Knowledge depth** · must · deps: A1

- Residuals at all 26 layers under base, DSG, the best gate and each D model.
- Linear answer probes per layer: train on dev, test on test, 5 seeds, plus a control task.
- Logit-lens readout per layer.

**A5 · Privacy and indistinguishability** · should · deps: A2

- TOFU forget quality.
- Membership-inference AUC (loss, zlib, min-k%) for gated versus ungated outputs.
- MUSE target models are DEFERRED.

**A6 · Tampering** · must · deps: A1, D-items

- **LoRA relearning:** k ∈ {10, 50, 200, 1000} forget passages, rank {8, 64}, 50/200/1000 steps, with forget accuracy logged over steps.
- **Quantization:** 4-bit and 8-bit.
- **Steering-vector recovery.**
- **Benign fine-tuning** (side-effect recovery).
- Applies to: DSG with hook, DSG without hook, legacy LoRA student, D1-local, D2.

**A7 · Generality on Gemma 3** · should · deps: A1, B suite

- Gemma 3 1B and 4B with Gemma Scope 2 SAEs (check availability in sae_lens; document if missing).
- WMDP-Bio test, full MMLU, B1 and B3 attacks, and A4 probes at 3 layers.
- 12B is DEFERRED.

**A8 · Reporting standards** · must · deps: all

- n, CI, seeds and compute cost for every run.
- Latency overhead per token (debug off) and peak VRAM.
- One table generator for all results.

### 5.2 Part B: Attacks (Break)

**Primary metric: attack success** = the fraction of gated test items (DSG answers wrong without attack) that DSG answers right under attack.

Always also report:

- absolute accuracy under attack;
- base-model accuracy under the same transformation;
- gate fire rate and rho.

**B1 · Dilution** · must

- Padding of {0, 50, 100, 200, 400, 800, 1600} tokens, placed before, after or around the question.
- Filler from WikiText, chat text and benign-bio text.

**B2 · Decomposition** · must

- Split the question across 2 to 3 chat turns.
- Combinable sub-questions.
- The hazardous term placed only in the options.

**B3 · Cross-lingual and encoded** · must

- Languages: Hindi, Tamil, Telugu, French, Spanish, Chinese, Arabic and Russian, translated with NLLB-200-distilled-600M (cached, back-translation chrF filter).
- Encodings: base64, ROT13, leetspeak and character spacing.
- Always include the base-model control under the same transformation.

**B4 · Black-box rewrite** · must

- An attacker LLM rewrites each question: gemma-2-2b-it, or gemma-2-9b-it in 4-bit if it fits in 16 GB.
- k ∈ {1, 5, 20} tries per item.
- Test whether successful rewrites transfer to the C gates.

**B5 · White-box obfuscation** · should

- A soft-prompt suffix and a GCG-lite discrete suffix that minimise the gate score while keeping the correct answer.
- Step budgets {50, 200, 500}.
- Test transfer to held-out items.

**B6 · Generation-time leakage** · must

- Prompts: open hazardous items (WMDP-Bio-Open test) plus the legacy demo prompts.
- Measure the first gate-fire token and how many answer tokens appear before it.
- Measure the gibberish rate after clamping.
- Compare the streaming gate with the prompt-only gate.

### 5.3 Part C: Detector redesign (Fix, API track)

All gates are calibrated to 5% benign FPR on dev, then evaluated on test and every B attack.

| ID | Name | Pri | What to run |
|---|---|---|---|
| C1 | Causal feature selection | should | Attribution patching vs chi² vs the DSG score: feature overlap, gate quality, robustness. |
| C2 | Streaming and probe gates | must | Window gate (w ∈ {8, 16, 24, 32, 64}), CUSUM, a logistic probe on SAE statistics, and a **raw-residual linear probe** as the baseline. |
| C3 | Layer sweep | should | Layers {1, 3, 5, 8, 12, 16, 20, 24}: gate AUROC per layer, then full evaluation of the top 3. |
| C4 | Transcoder gates | stretch | Use Gemma Scope transcoders for gemma-2-2b if sae_lens has them; otherwise do this in A7 on Gemma 3. |
| C5 | Intervention type | should | Clamp −500 vs mean ablation vs per-feature clamp vs steering toward an "unknown"/IDK direction. Measure accuracy, gibberish rate and output style. |
| C6 | Two-level gate | should | Domain × hazard AND gate, and the per-concept Bonferroni variant, both with clean calibration. |

### 5.4 Part D: Permanence (Fix, open-weights track)

| ID | Name | Pri | What to run |
|---|---|---|---|
| D1-local | UNDO-style distillation of the DSG-guarded teacher into a **noised** student | must | Run on a model that fits for full-parameter training: Gemma 3 1B or 270M with its Gemma Scope 2 SAE. If those are unavailable, use gemma-2-2b with LoRA plus noised LoRA initialisation and say so. Noise α ∈ {0.1, 0.3, 0.5}. KL to the guarded teacher on forget prompts and to the unguarded teacher on retain prompts. Compare with `lora-student-same-ref`, then run A6. The full 2B version is DEFERRED. |
| D2 | Closed-form null-space edit (AlphaEdit-style) | must | Edit MLP down-projections in layers ≤ L* to suppress the selected-feature directions, with the update projected onto the null space of the retain-key covariance. Compare with decoder orthogonalisation, then run A6. |
| D3 | Baked-model audit | should | For D1-local and D2: SAE activity on the forget corpus, A4 probes, and which features changed most. |

### 5.5 New additions (roadmap section 19)

| ID | Name | Pri | What to build or run |
|---|---|---|---|
| N1 | GuardBreak toolkit | must | Package the B suite as `guardbreak/` with a CLI (`guardbreak run --gate dsg --attack dilution`). Public part: transformation code only. |
| N2 | OpenUnlearning adapter | should | Expose the DSG hook in OpenUnlearning's interface and run its TOFU metrics. Write `results/N2_PR.md`; do not open the PR. |
| N3 | Interactive demo | should | Gradio app showing the per-token rho trace, firing features, gate decision and a dilution slider. TOFU domain. Local only. |
| N4 | Red-team challenge app | should | Gradio app guarding TOFU fictitious facts, with anonymised attempt logs and a leaderboard. Local only. |
| N5 | Adaptive hardening loop | should | 5 rounds: B4 attacker → recalibrate or retrain the gate on found attacks (dev) → re-attack on test. Log success per round. |
| N6 | Conformal threshold | must | Split-conformal τ. Report empirical FPR against target α ∈ {0.01, 0.05, 0.1} on held-out benign and shifted-benign sets. |
| N7 | Multi-layer voting | should | Any-of and majority-of over the top 3 layers from C3, against all B attacks. |
| N8 | Reasoning-trace gating | stretch | Step-by-step prompting: leakage in the reasoning text versus the final answer, and when the gate fires. |
| N9 | SAE-quality explanation | must | Reconstruction MSE, explained variance, L0 and splitting/absorption indicators, correlated per item with gate misses (logistic regression with CI). |
| N10 | Audit Card | should | A template plus auto-filled cards for DSG and the best fixes, generated from `summary.json`. |

### 5.6 Theory checks (empirical side; each is a CPU job on its own branch `exp/T-checks`)

| ID | What to check |
|---|---|
| T1 | Predicted rho under dilution (rho·L/(L+P) + filler fire rate·P/(L+P)) versus measured rho, per item: R² and residuals. |
| T2 | Window and CUSUM statistics versus padding length: is the slope ≈ 0? |
| T3 | Gate FPR/FNR on every set, checked against leakage ≤ FNR and damage ≤ FPR. |
| T4 | D2: the change in retain outputs on held-out retain keys versus the bound. |
| T5 | CUSUM assumptions: autocorrelation of the per-token LLR, and distribution fit. |

### 5.7 Backlog (run only when the queue is idle)

MT-Bench with a fixed judge; the custom TopK SAE; data-efficiency and zero-shot re-runs in the harness; the latency benchmark without debug prints.

### 5.8 DEFERRED (needs the 48 GB server; plan later, do not implement now)

D1 on full gemma-2-2b; RMU training if no checkpoint exists; full fine-tuning relearning; MUSE target models (A5); Gemma 3 12B (A7); full TOFU fine-tuning. List these in `results/DEFERRED.md` with what each would need. Do nothing else for them.

## 6. Execution order (waves) and parallelism

```
Phase 1 (Claude Code, P1): env → v2-harness → cache refactor → sanity gate → splits → leakage check
                           → exp branches + worktrees → progress/status/watchdog → smoke tests → preflight
Phase 2 (scheduler in tmux, unattended):
  Wave 1: A1 (dev sweeps, then test with 5 seeds) ‖ CPU: B3 translation cache ‖ TOFU LoRA fine-tune (GPU-exclusive)
  Wave 2: B1, B2, B3, B4, B6 ‖ A3 ‖ A4 capture (GPU), then probe training (CPU)
  Wave 3: C2, C3, C5, C6, N6, N7, C1, then the B suite on the top gates ‖ D2 ‖ B5 (GPU-exclusive)
  Wave 4: A6 on DSG, legacy student and D2 (GPU-exclusive) ‖ N9 and T checks (CPU)
  Wave 5: A2, A5-local, A7 (Gemma 3 1B/4B), D1-local, then A6 on D1-local, N5, N8
  Wave 6: N1, N3, N4, N10, N2 (mostly CPU, can overlap earlier waves)
  Idle:   backlog
Phase 3 (Claude Code, P4): analysis → FINAL_REPORT
```

Parallel rules: at most 2 eval workers on the GPU (only if measured peak ≤ 6.5 GB each); training is exclusive; CPU jobs run alongside.

## 7. Logging specification (log everything)

Run directory: `$DSG_RESULTS/runs/<exp_id>/<run_id>/`, with `run_id = <exp_id>__<method>__<attack>__<dataset>__<split>__s<seed>__<hash8>`.

**`config.json`**
- resolved config, branch, commit, dirty flag;
- package versions, GPU and driver;
- seed, split hashes, start and end time.

**`metrics.json`**
- all aggregates for both views, with n and 95% CI, and per-subject breakdowns (pooled and unweighted);
- gate stats: fire rate, benign FPR, hazardous FNR;
- timing: wall time, tokens/s, latency overhead, peak VRAM and RAM.

**`items.parquet`** (one row per question × condition). Columns:
- item_id, dataset, subject, split, language, attack, attack_params, prompt length, pad length;
- gold, pred, correct, prob_A..prob_D, entropy, margin;
- rho, tau, gate_fired, window_max, cusum_max, probe_score, per-layer gate scores;
- number of selected features fired, max activation of each selected feature, top-20 non-selected active features and their activations;
- SAE recon MSE, L0;
- for generation: text hash (text only in `$DSG_PRIVATE`), first-fire token, answer score, gibberish metrics, token count.

**`traces.npz`** (up to 200 items per condition)
- per-token fire indicators, running rho, window statistic, CUSUM statistic, per-token LLR.

**`train_log.parquet`**
- step, loss parts, lr, grad norm, time, VRAM;
- periodic forget, retain and MMLU-mini evaluation;
- best and last checkpoints only.

**Probes**
- per-layer train/test accuracy, control-task accuracy, seed, weights saved as `.npy`.

**Features**
- selected ids and scores per config and seed, Jaccard overlaps, Neuronpedia URLs if resolvable.

**Indexes and progress**
- `results/index.csv` and `results/summary.json`;
- `progress.json` per job (section 4.5).

**Figures** (regenerated from logs by `dsgx/analysis/plots.py`, PNG and PDF)
- accuracy vs padding;
- rho distributions;
- gate ROC curves;
- forget-vs-utility Pareto plot;
- probe accuracy by layer;
- relearning curves;
- conformal coverage;
- per-language bars with the base-model control;
- N5 rounds;
- T1 predicted vs measured rho;
- feature-overlap heatmaps;
- gibberish rate by gate.

## 8. Reports

**`results/SMOKE_REPORT.md`** (end of P1)
- per-experiment smoke status, time per item, projected runtime, VRAM and RAM;
- sanity-gate result;
- total estimated queue hours by wave;
- blocking problems.

**`results/FINAL_REPORT.md`** (P4), in this order:
1. Executive summary (10 lines).
2. Setup used and deviations.
3. Baseline reproduction (paper vs ours, both views, CIs).
4. Claims table with verdict and evidence.
5. Part I results.
6. Part II results.
7. Part III results.
8. Generality.
9. Surprises and anomalies.
10. Failed, skipped and partial items.
11. Compute used.
12. Next experiments if any claim is inconclusive.
13. Index of figures and run directories.

Every number comes from logs and carries n and a CI. Non-significant results are labelled as such.

## 9. Claude Code must NOT

- Commit to or push main, force-push, or delete branches, worktrees with results, or the old artifact directories.
- Publish anything: no Hugging Face uploads, no PRs, no deployments.
- Install cron jobs or system services without asking me.
- Write hazardous text outside `$DSG_PRIVATE`.
- Tune on test splits, report without a CI, or silently skip an experiment.
- Implement anything in section 5.8.

---

# PART II: PROMPTS

## 10. Prompts

### P1a: Build the foundation and check the baseline (first session)

```
Read docs/MASTER_PLAN.md completely, and REPO_REPORT.md. MASTER_PLAN.md is the source of truth. In this session do ONLY the foundation; do not create experiment branches yet.

1. Write $DSG_RESULTS/PLAN_REVIEW.md: anything impossible or ambiguous with this repo and hardware, your resolution, and anything missing for a publishable result. Log resolutions in DEVIATIONS.md and continue unless something truly blocks you.
2. Environment: clone the env to mechunlearn2, pip install -e dynamic_sae_guardrails, write ENV.md and requirements-lock.txt. Create the DSG_CACHE, DSG_RESULTS and DSG_PRIVATE folders (section 3.2).
3. On a new branch v2-harness cut from main, build:
   - the harness core with model-load sharing;
   - the DSG-faithful method and the dsg-fixed variant;
   - the activation cache refactor;
   - seeded dev/test splits;
   - the leakage checker;
   - the run logger (section 7);
   - the queue system with progress, status, watchdog and safeguards (sections 4.4 to 4.6);
   - the scripts in section 4.1;
   - pytest unit tests.
   Put [DSG DEBUG] behind DSG_DEBUG=1 and remove /tmp paths.
4. Sanity gate: reproduce Bio N=20 seed 0 on the DSG-subset view (WMDP 0.2937, MMLU-u 0.9941, within one question) and tau 0.5458 from the new cache. Debug until it passes, or explain exactly why and stop.
5. Create only the exp/A1-baselines branch and worktree. Run its smoke config through the real queue in tmux, so the scheduler, heartbeats, status command and watchdog are all exercised. Then stop the tmux sessions.
6. Write $DSG_RESULTS/FOUNDATION_REPORT.md:
   - what was built;
   - the sanity-gate numbers;
   - unit test results;
   - smoke result and timing;
   - the status command output;
   - problems and decisions needed from me.
7. Push v2-harness and exp/A1-baselines (never main).

Print a summary under 200 words. Work autonomously; commit at every milestone.
```

### P1b: Build all experiment branches and smoke-test them (second session, after the foundation is approved)

```
Read docs/MASTER_PLAN.md and $DSG_RESULTS/FOUNDATION_REPORT.md. The foundation on v2-harness is approved.

1. Create one branch per remaining experiment ID from v2-harness (exp/<ID>-<name>, section 5) with a worktree in ../dsg_worktrees/. On each branch implement only that experiment: methods, attacks, a full config and a smoke config, porting legacy code per section 5.0. Do not implement section 5.8; write $DSG_RESULTS/DEFERRED.md.
2. Run every smoke config through the real queue in tmux. Fix failures (harness bugs on v2-harness + sync_harness.sh).
3. Write SMOKE_REPORT.md (section 8) with projected hours per wave.
4. Build the full job list (all seeds and grids) but DO NOT start it. Run scripts/preflight.sh and include its output. Stop the tmux sessions. Push v2-harness and the exp/* branches (never main).

Print a summary under 300 words, including total estimated hours and any decision you need from me.
```

### P2: Launch the queue, then exit

```
Read docs/MASTER_PLAN.md sections 0, 3, 4.4 to 4.6, 6 and $DSG_RESULTS/SMOKE_REPORT.md.

1. Run scripts/preflight.sh. If anything fails, fix it or stop and tell me.
2. Queue the full job list in wave and priority order (section 6). Add a PAUSE point after Wave 1: when all Wave 1 jobs finish, the scheduler stops starting new jobs and writes WAVE1_DONE.md with the baseline results (with CIs) for review. It resumes only when I run: python -m dsgx.queue.resume
3. Start everything with scripts/tmux_up.sh (sessions dsg-queue, dsg-workers, dsg-monitor). Wait until the first jobs are running and their progress.json heartbeats appear, and confirm STATUS.md is updating.
4. Write $DSG_RESULTS/RUNBOOK.md for the team:
   - the hourly check commands (section 0.3) and how to read the status table;
   - how to attach to and detach from each tmux session;
   - how to pause, resume, and stop safely;
   - how to restart everything after a reboot (including the optional @reboot cron line; do not install it);
   - what each alert means and what to do.
5. Print: the tmux session names, the number of jobs queued, the ETA by wave, the first 10 jobs, and the exact commands I should run each hour.

Then end the session. The scheduler keeps running without you.
```

### P3: Status check, fix failures (repeat any time)

```
Read docs/MASTER_PLAN.md sections 3, 4.5, 4.6, then run: python -m dsgx.queue.status. Read the logs of FAILED, HUNG and BLOCKED jobs and any alerts in STATUS.md.

1. For each failure: diagnose it. Fix the code on the right branch: harness bugs on v2-harness plus scripts/sync_harness.sh; experiment bugs on that exp branch. Add a unit test for real bugs. Re-queue only the affected jobs. If a fix changes already-finished results, re-queue those too and list them.
2. Look for anything suspicious in finished results: impossible numbers, CIs that are too tight, baseline drift, leakage warnings. Log findings in ANOMALIES.md.
3. If the GPU sits idle while jobs wait, find out why and fix the scheduler (restart it via the supervisor; never kill running jobs unless they are HUNG).
4. Append a dated entry to PROGRESS.md: done, failed, remaining, new ETA.

Print a status summary under 200 words, the output of status --chat, and any decision you need from me. Then end the session; the queue keeps running.
```

### P4: Analyse and write the final report

```
Read docs/MASTER_PLAN.md sections 2.2, 7 and 8. Run python -m dsgx.queue.status to confirm completion (or note that this is an interim report).

1. List every experiment ID as done / partial / failed / skipped / deferred, with reasons.
2. Aggregate all runs. Compute CIs and paired tests (bootstrap + McNemar). Regenerate every figure from section 7 into $DSG_RESULTS/figures/.
3. Write $DSG_RESULTS/summary.json (per experiment and per claim).
4. Write $DSG_RESULTS/FINAL_REPORT.md exactly per section 8, with a verdict for each claim C-H1..C-H7, n and CI on every number, non-significant results labelled, plus surprises and next experiments.
5. Generate the N10 audit cards. Commit the analysis code on v2-harness and push (never main).

Print the 10-line executive summary and the claims table.
```

---

## Appendix: quick reference

| Need | Command or file |
|---|---|
| Hourly check | `python -m dsgx.queue.status` (or `--chat`) |
| Live view | `tmux attach -t dsg-monitor` (detach: Ctrl+b, then d) |
| Scheduler log | `tmux attach -t dsg-queue`, or `results/logs/scheduler.log` |
| Start / stop everything | `scripts/tmux_up.sh` / `scripts/tmux_down.sh` |
| What to send to the planning chat | `STATUS_FOR_CHAT.md` (any time), `SMOKE_REPORT.md` (after P1), `FINAL_REPORT.md` + `summary.json` (after P4) |
