# PREP_PROGRESS — branch prep/later-runs

Goal: everything needed to finish the project without Claude Code after 2026-10-08.
Rules followed: no GPU, no ssh to gpuws, no writes to $DSG_RESULTS / dsg_worktrees / cluster worktree,
no large downloads, no hazardous text.

| # | item | status |
|---|---|---|
| 1 | NO_CLAUDE_RUNBOOK.md + doctor / wave_check / reboot / server.sh | done (commands of items 2-6 are referenced; verified at item 7) |
| 2 | `python -m dsgx.analysis.final_report` | done: smoke run OK (131 runs, 10/12 figures, 3 cards, 3 s); 6 tests |
| 3 | `python -m dsgx.combine` (combination wave) | done: rule + screening + TEST/LOO generation + enqueue (5 tests, incl. temp-worktree enqueue); harness: conformal calib rule, `exp` key for B4/B5 attacks |
| 4 | server job scripts + SERVER_JOBS_MANIFEST.md | done: 7 jobs (script + sbatch + conf), jobcommon, fetch_models.sh, wheels/q2 (29 MB); 14 CPU tests (tiny model) |
| 5 | qualitative track (Q1–Q8) tools | done: Q1 cards (+Neuronpedia fetch, l0_142 only), Q3, Q5, Q6 run on smoke data; Q4/Q8 sheet / kappa / gallery (TOFU-only guard); 5 tests. Fixed open_cache misuse in cluster/jobcommon (would have crashed every server job) |
| 6 | `python -m dsgx.analysis.paper_assets` | done: 23 booktabs tables + 16 paper-size PDFs from smoke; test document compiles with pdflatex; 2 tests |
| 7 | CLAUDE.md, merge notes, final PREP_PROGRESS | done (nothing merged; never main) |
| 8 | rmu-v2 server job (wider RMU grid, probed batch size, TEST vs all comparators) | done: code 64279d0, staged + verified, queued 87→88→89 behind a6-full (afterany) |
| 9 | TMLR paper draft (`~/projects/mechunlearn-project/paper`, own repo) | done: official style files, all non-result sections drafted, results as asset placeholders, compiles with latexmk |

Tests: `CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/mechunlearn2/bin/python -m pytest -q tests` -> 81 passed on CPU (40 original + 41 new). Smoke results were only read; outputs went to a scratch dir. Only download: 29 wheels (29 MB) in `../wheels/q2`.

## Not done / needs a person or the network later
- Downloads for server jobs: `cluster/fetch_models.sh a7-12b | mtbench | q2-graphs | muse` (gemma-3-12b-it needs an accepted HF licence).
- Neuronpedia hosting of DSG's l0_142 SAE is unverified: `python -m dsgx.analysis.qual.q1_feature_cards --probe`.
- MUSE metrics re-implement the muse_bench definitions; swap in the official code after fetching it.
- Roadmap "Section 23" is not on disk; Q1-Q8 were built from the task description.
- Nothing merged into v2-harness / exp branches (CLAUDE.md "Merge notes").

## Server state (session 4, 2026-10-02)
- RMU (79→80→81) finished, fetched, verified; RMU best weights on the lab PC (`dsg_results_cluster/checkpoints/RMU-cluster/best`); server RMU inputs cleaned (corpus kept).
- Fixed `server.sh stage` (only the first STAGE line was copied; `569e462`).
- Overnight chain on gpuws: 83 validate → 84 tofu-full → 85 d1-full → 86 a6-full (afterok). Inputs for all four staged up front (ours 20 GB, 133 GB free at submit).

## Where to continue
NO_CLAUDE_RUNBOOK.md: hourly status + doctor; at the Wave-1 pause `wave_check 1` then `resume`; server: fetch/verify/cleanup tofu-full, d1-full, a6-full after chain 83-86 (CLAUDE.md session 4), then SERVER_JOBS_MANIFEST.md steps 4-7 (a7-12b, mtbench, q2-graphs, muse; downloads first); after Waves 1-3 + N6: `dsgx.combine`; at the end: `final_report`, `paper_assets`, qualitative tools.

## Session 5 (2026-10-02)
- rmu-v2 queued: 87 validate (afterany:86) → 88 rmu-v2-train → 89 rmu-v2-eval. Details and next steps: CLAUDE.md "Session 5".
- Paper: `~/projects/mechunlearn-project/paper` (branch `draft`). Fill results by running
  `python -m dsgx.analysis.paper_assets --out ~/projects/mechunlearn-project/paper/assets`; `\todo{}` markers list what is left;
  11 bib entries carry `% TODO-VERIFY`.

## Session 6 (2026-10-02, server)
- 84 tofu-full failed (load_sae argument order, fix `2ec30f8`); 85/86 cancelled by afterok. Queued 92 validate →
  93 d1-full behind 89 with a hard 09:00 end (deadline + DSG_END_BY). 94 a6-full queued after it (user
  approved) with the same 09:00 stop; it will likely only partly finish. tofu-full rerun waits for the next window. Details: CLAUDE.md "Session 6".

## Session 7 (2026-10-03)
- **A (lab PC queue):** idle 06:03–11:05. Wave deadlock: A1-test-000..003 (wave 1) depend on A1-dev-dsg-subset-ids
  (wave 2, held until the Wave-1 resume, which never comes because wave 1 is not terminal). Fixed in the queue job
  file (wave 1, batch 32→8: A1-dev OOMed at 16 and tasks ignore DSGX_BATCH_FACTOR); 2 DEVIATIONS rows. Started 11:05:08,
  heartbeat fresh. `doctor` now prints `STOP wave-deadlock` (commit 8b1d95d).
- **B (server, last night):** 84 tofu-full FAILED (load_sae, fixed 2ec30f8); 85 d1-full / 86 a6-full CANCELLED (afterok);
  87 validate EXACT; 88 rmu-v2-train + 89 rmu-v2-eval DONE; 92 validate EXACT; 93 d1-full DONE 04:48; 94 a6-full DONE 05:25
  (student, d2 skipped: weights not on the lab PC). Fetched + sha256-verified: rmu-v2 173 files (f3e19827…), d1-full 28
  (09a68f00…), a6-full 49 (e4b3f959…), validate 7. RMU-v2 best → `dsg_results_cluster/checkpoints/RMU-v2/best`.
  RMU v2 (c14: 20×r, α 300, L3, 150 steps; **at grid edge**: steering, steps) TEST WMDP 0.319 vs base 0.644, DSG 0.298
  (diff +0.020 [-0.006, 0.047], n.s.), third-party 0.498, v1 0.556; MMLU 0.548 vs DSG 0.561 (-0.013, p<0.001).
  D1-full: undo a0.3/a0.5 collapse MMLU to chance (0.238/0.232); a0.1 WMDP 0.396, MMLU 0.514.
  Server cleanup: third-party RMU, RMU v1, D1-full a0.3/a0.5 (collapsed, deleted, metrics kept), code-rmu-v2 → ours 35 GB, / 118 GB free.
- **C (server today, another user has priority):** code 3b7ec89 synced. Chain (all `--nice=10000`, afterok):
  95 validate (0:30) → 96 tofu-full rerun (3:00; models cached, per-condition checkpoints, est 1–1.5 h) → 97 a6-full part 2
  (1:00; new targets d1-a0.1, rmu-v2, est 20 min; finished cells skipped). d1-full needs no re-run.
- **D (downloads):** HF licence accepted; `cluster/fetch_models.sh a7-12b` → gemma-3-12b-it (23 GB on disk, 5 shards) +
  gemma-scope-2-12b-it `resid_post/layer_24_width_16k_l0_medium` (1.3 GB). Lab PC free 67 GB afterwards. Judge model skipped.
- **E (paper, repo `paper` ca7e069):** wording matches commit history, contribution 4 = run/scheduled (MUSE conditional),
  eq. 3 uses eq. 1's T, Properties not Propositions, Gemma 2 bib shortened, circuit-breakers TODO-VERIFY; compiles (17 pp,
  0 warnings). paper_assets X1 caption no longer says "pre-registered".

## Session 8 (2026-10-03): heavy experiments to gpuws, D1 v2, DSG figure parity

### Where each experiment runs now (estimates; gpuws ≈ 0.3 × lab-PC time, measured sanity 57 s vs 228 s)
| experiment | jobs | runs now on | est. h lab PC | est. h gpuws | status |
|---|---|---|---|---|---|
| A6 LoRA tampering, DSG hook / no hook | 23 lab jobs | **gpuws** (`a6-lora`, pinned exp/A6 code) | 8.0 | 2.4 | lab: MOVED-TO-SERVER; server: chained |
| A6 LoRA tampering, student / D1-local / D2 | 33 lab jobs | **gpuws** (`a6-baked`) | 11.6 | 3.5 | MOVED-TO-SERVER; staged when lab D1 (Wave 5) and D2 (Wave 3) weights exist |
| C3 layer sweep (8 layers + base + AUROC) | 10 lab jobs | **gpuws** (`c3`, pinned exp/C3 code) | 1.7 | 0.7 | MOVED-TO-SERVER; server: chained |
| A7 Gemma 3 1B / 4B (+ dilution, translate, best fix) | 4 lab jobs → 24 runs | **gpuws** (`a7-small`) | 0.8 (4 runs) / ~4 (24 runs) | 1.2 | MOVED-TO-SERVER; server: chained |
| A7 Gemma 3 12B | 12 runs | **gpuws** (`a7-12b`) | does not fit (16 GB) | 4 | waits for disk (stage after the next cleanups) |
| D1 v2 (α 0.05/0.1/0.2, 4000 steps, DEV sel., TEST, A6) | new | **gpuws** (`d1-v2`) | does not fit (full 2B) | 8.5 | server: chained |
| DSG figure parity (FP-*) | new | **gpuws** (`figs`) | ~17 | 5 | server: chained |
| A6-full, D1-full, RMU v2, TOFU-full | — | gpuws | — | — | done (TOFU-full 96 / a6-full 97 running) |
| everything else (A1–A5, A8, B*, C1/C2/C4–C6, D1-local, D2, D3, N*, T) | 93 | lab PC | lab ETA 3d0h → **2d2h** after the move | — | unchanged |

Lab queue: `python -m dsgx.queue.move --list` (70 moved), `doctor` reports 0 wave / moved-dependency deadlocks.
New tools: `dsgx/queue/move.py` (MOVED-TO-SERVER status, refuses stranded dependents, `--undo`),
`cluster/lab_jobs.py` + `cluster/stage_lab_jobs.sh` (run moved lab jobs on gpuws with their pinned exp-branch
commit via `git archive`; HARDWARE.json marks those runs gpuws), `cluster/a7_server.py` (1B/4B/12B),
`slurm/later.sh` (snapshot `code-later`, budget = Slurm time left − 12 min; every job ≤ 3 h, resumable).

### Session 8 results and state
- **D1 v2** (Part 2): `cluster/d1_v2.py`; reference d1-full α 0.1 on gpuws TEST: WMDP 0.396 [0.358, 0.433], MMLU 0.514
  [0.503, 0.526] (base 0.644 / 0.564, DSG 0.298 / 0.560). v2 queued (100–105).
- **Figure parity** (Part 3): every DSG figure type has a generator for DSG and our gate (figures.py: gate-score
  distributions, forget–utility TEST scatter, relearning by epochs, clamp grid, static vs dynamic, data efficiency,
  multi-topic, latency, TOFU highlights; paper_assets: 8 new tables + "DSG figure parity" status in INDEX.md). Data from
  lab runs where they exist, the rest from the gpuws job `figs` (99, 111–112, 116).
- **Server results fetched:** tofu-full (96) and a6-full (97) sha256-verified; a6-full inputs cleaned. A6-full cells
  (full FT, 200 steps; forget before → after): d1-a0.1 0.403 → 0.47–0.49; rmu-v2 0.373 → 0.52–0.54; dsg-hook 0.407 → 0.30–0.36;
  dsg-nohook 0.53 → 0.51–0.53.
- **Bug found and fixed:** TOFU-full v1 model utility (0.0 everywhere) used `max(0, 1 − mean R)`; TOFU uses the per-item
  mean of `max(0, 1 − Rᵢ)` and option-normalised answer probability on real/world sets. Re-eval queued (113); do not
  use the v1 TOFU numbers. Also fixed before running: the multi-topic union threshold (cache candidates).
- **Latency v1 (job 99) biased** by back-to-back timing on a saturated host (load ~220/224 CPUs); v2 interleaved re-run 116.
- Tests: 97 pass on CPU.

## Session 9 (2026-10-03): trailer rule, paper appendices, MUSE, Q2, dashboard
| # | item | status |
|---|---|---|
| 0 | Chakrish28 trailer rule (CLAUDE.md, memory, paper README, history cleanup incl. past break commits) | done; copy test VERIFY OK |
| 1 | Paper appendices A–D + Section 8 (generated tables, `dsgx.analysis.appendix_tables`) | done; 21 pp, 0 warnings |
| 2 | MUSE (BM1): official metrics, retrain reference, DSG + best gate, <= 3 h resumable | queued 117 → 118–120 (CPU tiny + real-path checks) |
| 3 | Q2 attribution graphs (TOFU: base / DSG / D2 / French attack) | queued 121 → 122 (tiny circuit-tracer run, server `--plan` ok) |
| 4 | Panel dashboard `python -m dsgx.analysis.dashboard` → `results/dashboard.html` | done; snapshot published (private artifact) |

Findings: muse_bench `privleak.eval` crashes on main (eval_data + sweep used); transformers 4.57.3 needs a local
tokenizer path offline; `tofu_full.Gate` scores single tokens during cached generation (see CLAUDE.md session 9;
113 left unchanged). Tests: 107 pass on CPU (+ Q2 overlay test when `env_q2_lab` exists).
