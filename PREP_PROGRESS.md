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
