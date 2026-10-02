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

Tests: `python -m pytest -q tests` -> 76 passed on CPU (40 original + 36 new). Smoke results were only read; outputs went to a scratch dir. Only download: 29 wheels (29 MB) in `../wheels/q2`.

## Not done / needs a person or the network later
- Downloads for server jobs: `cluster/fetch_models.sh a7-12b | mtbench | q2-graphs | muse` (gemma-3-12b-it needs an accepted HF licence).
- Neuronpedia hosting of DSG's l0_142 SAE is unverified: `python -m dsgx.analysis.qual.q1_feature_cards --probe`.
- MUSE metrics re-implement the muse_bench definitions; swap in the official code after fetching it.
- Roadmap "Section 23" is not on disk; Q1-Q8 were built from the task description.
- Nothing merged into v2-harness / exp branches (CLAUDE.md "Merge notes").

## Where to continue
NO_CLAUDE_RUNBOOK.md: hourly status + doctor; at the Wave-1 pause `wave_check 1` then `resume`; server: finish rmu (fetch, cleanup), then SERVER_JOBS_MANIFEST.md steps 1-7; after Waves 1-3 + N6: `dsgx.combine`; at the end: `final_report`, `paper_assets`, qualitative tools.
