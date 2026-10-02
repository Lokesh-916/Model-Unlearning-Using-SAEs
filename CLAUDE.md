# CLAUDE.md — prep/later-runs worktree (tools to finish the project without Claude Code)

Branch: `prep/later-runs` (cut from `cluster/setup`, itself cut from `v2-harness`). Worktree:
`~/projects/mechunlearn-project/prep`. The cluster rules below still apply to every server operation.
Never disturb the lab PC queue (dsg_worktrees, dsg-* tmux sessions, $DSG_RESULTS). Never commit to main.

## What this branch adds (2026-10-02) — start with NO_CLAUDE_RUNBOOK.md
| need | command / file |
|---|---|
| every remaining operation, copy-paste, with "you should see" | `NO_CLAUDE_RUNBOOK.md` |
| failure triage + safe re-queue (incl. BLOCKED dependents) | `python -m dsgx.queue.doctor [--apply \| --requeue J..]` |
| scripted Wave review before resume | `python -m dsgx.queue.wave_check <w>` |
| reboot recovery | `scripts/reboot_recover.sh` (uses baselines_DSG's tmux_up) |
| P4 report in one command | `python -m dsgx.analysis.final_report [--interim] [--hardware gpuws --runs ... --out ...]` |
| claim rules C-H1..C-H7 (fixed; edit only with a DEVIATIONS row) | `dsgx/analysis/claims.py` |
| combination wave (pre-registered DEV rule, LOO, 5-seed TEST) | `python -m dsgx.combine {--dry-run,--enqueue,--select-only}` |
| paper tables (booktabs) + paper-size PDFs | `python -m dsgx.analysis.paper_assets` |
| qualitative track Q1/Q3/Q5/Q6, Q4/Q8 (TOFU-only) | `python -m dsgx.analysis.qual.<q1_feature_cards,q3_never_learned,q5_geometry,q6_trajectory,annotate>` |
| all server work from the lab PC | `cluster/server.sh {status,sync,plan,stage,submit,check,fetch,verify,cleanup,run} <job>` |
| later server jobs, order, disk budget | `SERVER_JOBS_MANIFEST.md`; jobs d1-full, a6-full, tofu-full, a7-12b, muse, mtbench, q2-graphs, rmu-v2 |
| queue a job behind a running chain without touching code/ | `cluster/stage_code_snapshot.sh <name>` (code-<name>/ snapshot) + `server.sh submit <job> --after-any <id>` |
| TMLR paper draft (own git repo, branch `draft`) | `~/projects/mechunlearn-project/paper` (`latexmk -pdf main.tex`; README there) |
| downloads still needed (lab PC, network) | `cluster/fetch_models.sh {a7-12b,mtbench,q2-graphs,muse,list}` |
| progress of this branch | `PREP_PROGRESS.md` |

Harness changes (both backward compatible; existing results stay valid): `dsgx/methods/gates.py`
`calibrate(..., rule="conformal")` (cache key unchanged for the default quantile rule); `dsgx/attacks/transforms.py`
rewrite_cache / suffix accept `exp:` to read another experiment's private artifacts (used by X1).
Tests: `CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/mechunlearn2/bin/python -m pytest -q tests` = 81 pass on CPU (~4.5 min; 41 new `tests/test_prep_*.py`; server jobs run with a
tiny random Gemma-2 via `DSG_TINY=1`).
Gotchas found: Neuronpedia's `3-gemmascope-res-16k` is the canonical **l0_59** SAE, not DSG's l0_142 (Q1 only queries
`3-gemmascope-res-16k__l0-142`, unverified whether hosted: run `--probe`); circuit-tracer 0.5.0 needs
transformers <= 4.57.3 → separate overlay venv `env/q2` from `wheels/q2` (29 wheels, SHA256SUMS); `open_cache()` already
returns an ActivationCache; the X1 selection job commits X1.yaml, so X1-screen jobs cannot be re-queued afterwards
(pinned commit) — re-run `dsgx.combine --select-only` instead.

## Merge notes (do NOT merge yet; for later)
1. Into v2-harness (fast-forward today; `git merge-tree` clean). Best at a queue pause (e.g. the Wave-1 pause):
   `cd ~/projects/mechunlearn-project/baselines_DSG && git merge --ff-only prep/later-runs` (or `--no-ff`), then
   `python -m pytest -q tests`. The scheduler code is unchanged, so the running supervisor needs no restart.
   This also brings in cluster/setup (cluster/, CLUSTER_SETUP_REPORT.md, this CLAUDE.md) — intended.
2. Into the exp branches: `scripts/sync_harness.sh` (idle worktrees only). It will list finished jobs as
   "RE-RUN NEEDED" because their commit differs; **no re-run is needed** for this merge: the gates/transforms changes are
   additive and default behaviour is bit-identical (all 40 original tests pass). Do not re-queue them.
3. The analysis tools never need to be in exp branches (they read $DSG_RESULTS). `exp/X1-combine` is created by
   `dsgx.combine --enqueue` from v2-harness if it contains `dsgx/combine.py`, else from prep/later-runs.
4. After merging, use `~/projects/mechunlearn-project/baselines_DSG` for every runbook command instead of `prep`.

## SERVER FACTS (already checked)
- ssh alias "gpuws" (key auth), user suraj, host iiitdmk-cse-SDI200F0A-48, Ubuntu 26.04.
- Slurm: one partition "gpu", no time limit, 1 node. GPU: 1x RTX 6000 Ada 48 GB, driver 595.91, CUDA 13.2.
- NO internet on the server. Everything is copied from this lab PC.
- 224 CPUs, 123 GB RAM. Shared root disk: only ~152 GB free for ~13 users. /tmp is RAM-backed (no large files there).
- System Python 3.14: do not use it.
- A system restart is pending, so jobs must checkpoint.

## STRICT RULES (shared machine, a friend's account; be careful and accountable)
1. All our files live only in ~/dsg_cluster/ on the server. Never touch other users' files or processes.
2. No sudo, no system packages, no system or network settings, never touch IPMI.
3. All GPU work goes through Slurm (sbatch; srun only for quick checks). Never run GPU code on the login shell. Request only what is needed (1 GPU, e.g. --cpus-per-task=16 --mem=64G) and always set --time.
4. Submitting is fine even if others are queued; Slurm handles the order. Never submit more than one of our jobs at a time unless chained with --dependency=afterok. Every job starts with a check: if less than 40 GB of GPU memory is free (someone using the GPU outside Slurm), it exits cleanly with a clear message.
5. Staged storage: copy only what the current job needs. After the job finishes and results are safely copied back, delete its large inputs (model weights, checkpoints, caches) from the server. Keep only env/, code/ and small files. Our total on the server must stay under 100 GB at all times. Before any large copy, check df; if free space would drop below 50 GB, stop and tell me.
6. Do not disturb the lab PC queue: never modify the dsg_worktrees, the dsg-* tmux sessions, or $DSG_RESULTS. Cluster results go to ~/projects/mechunlearn-project/dsg_results_cluster/ on the lab PC, in the same run-directory format (merged later). Keep lab PC disk free above 60 GB and delete temporary tarballs after transfer.
7. Copy gently: rsync --bwlimit=50000 --partial.
8. Log every server command in ~/dsg_cluster/COMMAND_LOG.md (date, command, purpose).
9. Never open or print hazardous dataset text or generations: ids, hashes and metrics only.
10. Never store passwords. Use the gpuws alias only.

## Per-GPU baselines (user decision, 2026-10-02)
Cross-GPU bf16 gated evals are not bit-reproducible (lab PC RTX 2000 Ada vs gpuws RTX 6000 Ada), so each
machine is its own hardware baseline.
1. **sanity-gpuws** is the exact sanity gate for every server job: WMDP-Bio 161/538, MMLU-u 0.9971
   (108/9/103/84), tau 0.5458, the same 20 legacy features. Set by `DSG_SANITY_TARGET=gpuws` in env.sh
   (`dsgx/checks/sanity.py` TARGETS["gpuws"]), checked exactly by `cluster/validate_check.py`.
   Every server chain starts with `validate.sbatch`; the rest is chained with `--dependency=afterok`.
2. Every compared condition (base, DSG, our RMU, the unverified third-party RMU) runs on gpuws.
3. Never mix lab-PC and gpuws numbers in one table or paired test. Label every result with its hardware
   (`DSG_HARDWARE=gpuws` → `hardware.label` in each run's config.json; summaries state it).

## Hazardous-text rule
Never open, print or quote WMDP questions, hazardous generations, attack prompts or anything in
$DSG_PRIVATE or ~/dsg_cluster private folders; work with ids, hashes and metrics only.

## Staged-storage cycle (every job)
1. Copy inputs (check df first; only what this job needs).
2. Run (sbatch via ~/dsg_cluster/slurm/submit.sh).
3. Fetch results (~/dsg_cluster/slurm/fetch_results.sh → lab PC dsg_results_cluster/).
4. Verify they arrived (file counts / checksums on the lab PC).
5. Clean up large inputs (~/dsg_cluster/slurm/cleanup.sh <job>).

## Where things live
- Server: `~/dsg_cluster/{env,code,hf_cache,data,slurm,logs}`
- Server command log: `~/dsg_cluster/COMMAND_LOG.md`
- Lab PC results: `~/projects/mechunlearn-project/dsg_results_cluster/`

## Session state (update at the end of every session)
### Session 1 — 2026-10-02 (end state)
- **Server disk:** `~/dsg_cluster` = 14 GB; `/` has 138 GB free (152 GB at start; peak use 25 GB).
- **Data on server now:** env/mechunlearn2 (8.2 GB, conda-pack of lab env + hatchling), code/ (cluster/setup,
  CODE_COMMIT.json), hf_cache: gemma-2-2b-it, Gemma Scope 2b-pt-res layer_3/width_16k/average_l0_142 only,
  cais/wmdp, cais/mmlu (57 subjects, no auxiliary_train), wikitext; data/dsg_cache/actcache bio s0 (layer 3);
  data/legacy (bio question ids + sparsity txt). private/ is EMPTY (forget corpus deleted). No checkpoints.
- **Jobs:** 73 dsg-validate ran but **validation FAILED**: WMDP 161/538 (target 158), MMLU-u 0.9971
  (target 0.9941), tau 0.5458 exact, calibration bit-exact. Cross-GPU bf16 numerics (RTX 6000 vs RTX 2000 Ada);
  see CLUSTER_SETUP_REPORT.md §3. 74 dsg-rmu-train / 75 dsg-rmu-eval cancelled by afterok (never ran).
  Nothing of ours is queued or running.
- **Next planned job:** waiting on the user's decision (report §6). Then RMU: re-stage with
  `cluster/stage_rmu.sh` + third-party RMU rsync, `submit.sh rmu_train.sbatch`, then
  `submit.sh rmu_eval.sbatch --dependency=afterok:<id>`.
- **Gotchas:** server scripts need `PYTHONPATH=$DSGC/code` (set in env.sh since session 3); run
  `CUDA_VISIBLE_DEVICES= python cluster/import_check.py` on the login node after code changes; any `--mem` is rejected (node RealMemory=1); always pass `--gres=gpu:1`; no CPU/mem confinement;
  `sacct` unavailable; run conda-unpack via the env's python; never use system python3 on the server.
- Lab PC tools: `cluster/sync_code.sh`, `cluster/stage_rmu.sh`, `cluster/slurm/fetch_results.sh <job>`.

### Session 2 — 2026-10-02 (end state)
- **Decision:** option 1, per-GPU baselines (rules above). Code `0760786` synced (dirty=false).
- **Re-staged** (sha256-verified): stage_rmu.sh inputs incl. private forget corpus (ff48dff6…), third-party RMU
  `models--AMindToThink--gemma-2-2b-it_RMU_s200_a300_layer3` (9.8 GB, listing 65dda2eb…).
  Server: ours 25 GB, `/` 128 GB free. Lab PC 102 GB free.
- **Chain submitted 18:2x:** 76 dsg-validate (sanity-gpuws, exact) → 77 dsg-rmu-train (afterok:76, dev grid)
  → 78 dsg-rmu-eval (afterok:77, TEST). 76 was RUNNING at session end; results not yet seen.
- **Next:** when all three have left the queue: `fetch_results.sh validate`, `fetch_results.sh rmu`, verify, then on the
  server `cleanup.sh rmu` (deletes RMU checkpoints, third-party RMU, forget corpus; keeps base model/SAE/data).
  If 76 MISMATCHES: same-GPU determinism is broken; stop and report (77/78 are auto-cancelled).

### Session 3 — 2026-10-02 (end state)
- Job 76: sanity-gpuws PASSED (`SANITY (gpuws) PASS`), but `python cluster/validate_check.py` failed with
  ModuleNotFoundError dsgx (script dir, not code/, on sys.path) → 77/78 cancelled by afterok.
- Fix `fb2c8c7`: env.sh exports `PYTHONPATH=$DSGC/code`; new `cluster/import_check.py` (CPU, imports only)
  passed on the lab PC and the gpuws login node for validate_check.py, rmu_train.py, rmu_eval.py.
- Inputs still staged from session 2 (ours 25 GB, `/` 128 GB free).
- **Chain resubmitted:** 79 dsg-validate → 80 dsg-rmu-train (afterok:79) → 81 dsg-rmu-eval (afterok:80).
  79 RUNNING at session end. Next: fetch validate + rmu, verify, `cleanup.sh rmu`.

### Session 4 — 2026-10-02 (end state)
- **RMU done:** 79 validate `VALIDATE sanity-gpuws EXACT`; 80 train (dev grid 6 cfgs, selected c4: steering 4.0×r =
  430.84, alpha 300, layer 7, update [5,6,7]); 81 TEST eval. Fetched + sha256-verified: rmu 82 files (listing
  993fab52…), validate 7 files (43fc48f8…; a session-1 local file moved to `dsg_results_cluster/_local_backups/`).
  Summary: `dsg_results_cluster/jobs/rmu/SUMMARY.md`. RMU best weights kept on the lab PC:
  `dsg_results_cluster/checkpoints/RMU-cluster/best` (4.9 GB, tree sha 9dc27d89…).
- **Server cleanup:** deleted RMU-cluster (5.3 G), RMU-cluster-tmp, third-party RMU (9.8 G). Forget corpus kept
  (d1-full / a6-full need it).
- **Bug fixed `569e462`:** `server.sh stage` copied only the first STAGE line (ssh in `push()` read the while-loop
  stdin). Now `ssh -n` / rsync `</dev/null`. All staged inputs re-verified by sha256 against the lab PC.
- **Staged up front:** TOFU (hub + datasets cache; 6 needed configs load offline), forget corpus, RMU best →
  `data/dsg_cache/models/RMU-cluster/best`. Not on the lab PC yet (skipped by a6-full with a note): D1
  `sameref_a0.0`, D2 `nullspace`; X1 `COMBINE_SELECTION.json` (tofu-full uses window 16).
  At submit: ours 20 GB, `/` 133 GB free. Est. peak ≈ 56 GB ours (tofu models 10.4 + D1-full students 15.6 +
  RMU 4.9 + ≤10 transient trainer state), free ≥ ~95 GB.
- **Overnight chain (code `569e462`):** 83 dsg-validate → 84 dsg-tofu-full (afterok:83) → 85 dsg-d1-full (afterok:84)
  → 86 dsg-a6-full (afterok:85, targets dsg-hook, dsg-nohook, d1 = D1-full undo_a0.3, rmu). ~10 h total.
  At session end: 83 `VALIDATE sanity-gpuws EXACT` (passed); 84 tofu-full RUNNING; 85, 86 pending (Dependency).
- **Next:** when all four have left the queue (`cluster/server.sh check a6-full`):
  `cluster/server.sh fetch tofu-full`, `fetch d1-full`, `fetch a6-full` (each also fetches validate), `verify` each,
  then `cleanup tofu-full --yes`, `cleanup d1-full --yes`, `cleanup a6-full --yes` (a6-full's cleanup removes
  D1-full students, RMU best and the corpus). If a step failed, afterok cancels the rest: inputs stay staged;
  resubmit from the failed step with `~/dsg_cluster/slurm/submit.sh <x>.sbatch` (finished cells / DONE runs are skipped).

### Session 5 — 2026-10-02 (end state)
- **RMU v2 (code `64279d0`)**: v1 was under-tuned (TEST WMDP 0.556 vs base 0.644, third-party 0.498; selected
  4x r at the grid edge). New job `rmu-v2` (`cluster/rmu_v2_train.py`, `rmu_v2_eval.py`): pre-registered 16 of
  48 configs (steering {4,8,12,20} x r_L, alpha {100,300,1200}, layer 3 (upd 1,2,3) / 7 (upd 5,6,7), steps
  {150,300}; rationale in GRID_RATIONALE + `jobs/rmu-v2/GRID.md`), train and dev-eval batch size = largest that
  fits min(0.70 x 48 GB, free - 4 GiB) (probed once, stored in grid_state.json), same selection rule, TEST bs=1
  of base / RMU-v2 / DSG paper / third-party RMU (+ paired vs v1 from its gpuws run). Weights:
  `data/dsg_cache/models/RMU-v2` (v1's `RMU-cluster/best`, used by a6-full, untouched).
- **Staged** (sha256-verified): third-party RMU (listing 0fd26642…), corpus (ff48dff6…, already there), code
  snapshot `~/dsg_cluster/code-rmu-v2` = 64279d0 (rmu-v2 runs from it, so `code/` under the chain never changed).
  Budget: ours 39 GB after staging, `/` 113 GB free; worst case ≈ 67 GB ours while the chain runs, ≈ 78 GB at
  rmu-v2 run time. Server `slurm/`: added rmu-v2 sbatch/conf; submit.sh (accepts afterany) and cleanup.sh
  (keeps the shared forget corpus while any dsg job is queued) updated, old copies `*.bak-569e462`.
- **Queue at session end:** 84 tofu-full R → 85 d1-full → 86 a6-full (tonight's chain, unchanged) →
  **87 dsg-validate (afterany:86) → 88 dsg-rmu-v2-train (afterok:87, --time 16 h) → 89 dsg-rmu-v2-eval
  (afterok:88, 4 h)**. Est. rmu-v2 ≈ 3–5 h train + 0.5 h eval.
- **Next:** when 84–89 have all left the queue: fetch/verify/cleanup tofu-full, d1-full, a6-full (as session 4),
  then `cluster/server.sh fetch rmu-v2` (also pulls RMU-v2 best, ~5 GB, to `dsg_results_cluster/checkpoints/RMU-v2/best`),
  `verify rmu-v2`, read `dsg_results_cluster/jobs/rmu-v2/SUMMARY.md` (selected cfg, `at_grid_edge`), then
  `cleanup rmu-v2 --yes` (removes RMU-v2 models, third-party RMU, corpus, code-rmu-v2). If 87 fails, 88/89 are
  cancelled: inputs stay staged; resubmit `submit.sh validate.sbatch` then the two rmu-v2 sbatch with afterok.
- Gotcha: run the test suite with the project env: `CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/mechunlearn2/bin/python
  -m pytest -q tests` (81 pass); the base `python` lacks rouge_score (2 false failures).
- Paper: `~/projects/mechunlearn-project/paper` (see its README). latexmk 4.88 installed in `~/.local/bin`, Latin
  Modern fonts in `~/texmf` (+ `updmap-user --enable Map=lm.map`), because the system TeX Live lacks both.

### Session 6 — 2026-10-02 22:10–22:45 (end state)
- **Diagnosis:** 84 tofu-full FAILED in its eval stage (`RuntimeError: Invalid device string: 'bfloat16'`):
  `jobcommon.load_sae` passed `(device, dtype)` into `get_sae(release, sae_id, dtype, device)`. TINY mode skips
  the real SAE, so the CPU tests missed it. 85 d1-full and 86 a6-full never ran (no logs): cancelled by afterok.
  tofu-full was not fetched (no results; its fine-tuned models `data/dsg_cache/models/A2-tofu-full`, 9.9 GB, stay
  staged so a rerun skips training). 87 validate (afterany:86) passed EXACT; 88 rmu-v2-train running.
- **Fix `2ec30f8`** (synced, import check OK): keyword args; real (non-TINY) jobcommon paths checked on CPU on the
  lab PC (SAE, dsg_features tau 0.5458, corpora, MCQ, dsg_guard forward on gemma-2-2b-it); 19 server-job tests pass.
  d1-full / a6-full sbatch honour `DSG_END_BY` (timeout 5 min before; Trainer checkpoints atomically every 100 steps).
- **Queue (GPU must be free by 2026-10-03 09:00 for another user):** 88 → 89 (untouched; est. done ~01:30) →
  **92 dsg-validate** (afterany:89, --time 0:30, --deadline 09:00) → **93 dsg-d1-full** (afterok:92, --time 7:20,
  --time-min 1:00, --deadline 09:00, DSG_END_BY=2026-10-03T09:00). d1-full estimate ≈ 5–6 h (3 × 2000 full-param
  steps at ~2 s/step + 20 trainer-state saves each + 5 TEST evals ~45 min) → ends ~07:00–07:30.
  **a6-full NOT queued** (~3 h, cannot finish by 09:00). Caveat: 88 keeps its own 16 h limit; if rmu-v2 runs far
  over its estimate, the 09:00 promise depends on it (we did not touch 88/89, as instructed).
- **Next (after 09:00):** if 93 was stopped by the guard/limit, resubmit `submit.sh d1-full.sbatch` (resumes from
  checkpoints; finished students and DONE runs skipped). Then a6-full (afterok d1-full), then tofu-full rerun.
  Fetch/verify/cleanup rmu-v2 as in session 5; fetch d1-full when done.
