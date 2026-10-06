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
| queue idle, nothing running (wave deadlock) | `python -m dsgx.queue.doctor` → `STOP wave-deadlock` line (session 7) |
| reboot recovery | `scripts/reboot_recover.sh` (uses baselines_DSG's tmux_up) |
| P4 report in one command | `python -m dsgx.analysis.final_report [--interim] [--hardware gpuws --runs ... --out ...]` |
| claim rules C-H1..C-H7 (fixed; edit only with a DEVIATIONS row) | `dsgx/analysis/claims.py` |
| combination wave (DEV rule fixed in code, LOO, 5-seed TEST) | `python -m dsgx.combine {--dry-run,--enqueue,--select-only}` |
| paper tables (booktabs) + paper-size PDFs | `python -m dsgx.analysis.paper_assets` |
| qualitative track Q1/Q3/Q5/Q6, Q4/Q8 (TOFU-only) | `python -m dsgx.analysis.qual.<q1_feature_cards,q3_never_learned,q5_geometry,q6_trajectory,annotate>` |
| all server work from the lab PC | `cluster/server.sh {status,sync,plan,stage,submit,check,fetch,verify,cleanup,run} <job>` |
| later server jobs, order, disk budget | `SERVER_JOBS_MANIFEST.md`; jobs d1-full, a6-full, tofu-full, a7-12b, muse, mtbench, q2-graphs, rmu-v2; session 8: figs, d1-v2, c3, a6-lora, a6-baked, a7-small |
| queue a job behind a running chain without touching code/ | `cluster/stage_code_snapshot.sh <name>` (code-<name>/ snapshot) + `server.sh submit <job> --after-any <id>` |
| TMLR paper draft (own git repo, branch `draft`) | `~/projects/mechunlearn-project/paper` (`latexmk -pdf main.tex`; README there) |
| downloads still needed (lab PC, network) | `cluster/fetch_models.sh {a7-12b,mtbench,q2-graphs,muse,list}` |
| progress of this branch | `PREP_PROGRESS.md` |
| mark lab jobs as run on the server (MOVED-TO-SERVER; refuses stranded deps; `--list`, `--undo`) | `python -m dsgx.queue.move` |
| run moved lab jobs on gpuws with their pinned exp-branch code | `cluster/stage_lab_jobs.sh <group> <glob>..` + `cluster/lab_jobs.py` (confs c3, a6-lora, a6-baked) |
| one long server chain (validate first, afterok validate + afterany previous, --nice) | `cluster/submit_chain.sh` |
| D1 v2 (α 0.05/0.1/0.2, 4000 steps, DEV selection, TEST, relearning) | `cluster/d1_v2.py {train,test}` (conf d1-v2) |
| paper appendix tables A-D (splits+hashes, both machines + sanity, B* grids, D1/D2, RMU grids + TEST) | `python -m dsgx.analysis.appendix_tables --out ~/projects/mechunlearn-project/paper/assets` |
| panel dashboard (both machines live, claims per hardware, key figures) | `python -m dsgx.analysis.dashboard [--every 15]` → `results/dashboard.html` (artifact snapshot: https://claude.ai/artifact/188nSVxV3BRGfiqcJgYLKh) |
| MUSE BM1 (official muse_bench metrics, retrain reference, DSG + best gate) | `cluster/muse.py` (conf muse; 3 x 3 h resumable) |
| Q2 attribution graphs, TOFU base / DSG / D2 / French attack | `cluster/q2_graphs.py` (conf q2-graphs; env/q2 on gpuws; transcoders via `cluster/stage_q2_transcoders.sh`) |
| DSG figure parity data (clamp grid, data efficiency, static/dynamic, multi-topic, latency, TOFU highlights) | `cluster/figparity.py` (conf figs); figures + tables in `paper_assets` (INDEX.md "DSG figure parity") |
| every finished result, both machines, CIs, claims C-H1..C-H7 (regenerate at the end of EVERY session) | `python -m dsgx.analysis.results_digest` → `$DSG_RESULTS/RESULTS_DIGEST.md` |
| Cyber forget-utility Pareto curve (Wave-1 decision) | `paper_assets`: figure `cyber_pareto`, table `tab:cyber-pareto` |
| MT-Bench (BM3): base / DSG / window-w16, judge gemma-2-9b-it (same family: label every number) | `cluster/mtbench_open.py` (conf mtbench; 4 × 3 h resumable, snapshot code-later7) |
| skip optimizer state when fetching / prune it on the server in DONE runs | conf `FETCH_EXCLUDE` (a6-lora: `last/trainer.pt`); `ssh gpuws 'PRUNE_ONLY=1 ~/dsg_cluster/slurm/cleanup.sh <job>'` |
| session 13 re-runs (Trainer accumulation fix): TOFU-full, FP-highlight, MUSE, Q2; `train_version 2` guards | `tofu-full-v3`, `figs-hl`, `muse-v2`, `q2-graphs-v2` sbatch (snapshot code-later8); `cluster/q2_rerun_when_ready.sh`; `cluster/chain_tail.sh` (end of our gpuws chain, for watchers); runbook §5.6 |

Harness changes (both backward compatible; existing results stay valid): `dsgx/methods/gates.py`
`calibrate(..., rule="conformal")` (cache key unchanged for the default quantile rule); `dsgx/attacks/transforms.py`
rewrite_cache / suffix accept `exp:` to read another experiment's private artifacts (used by X1).
Tests: `CUDA_VISIBLE_DEVICES= ~/miniconda3/envs/mechunlearn2/bin/python -m pytest -q tests` = 114 pass on CPU (~9 min; `tests/test_prep_*.py`; server jobs run with a
tiny random Gemma-2 via `DSG_TINY=1`). Before submitting, also check real (non-tiny) configs on CPU: resolve + `check_runs` (session 8 found
the multi-topic union-tau bug this way: an activation cache stores fire bits only for its own 2048 candidate features).
Gotchas found: Neuronpedia's `3-gemmascope-res-16k` is the canonical **l0_59** SAE, not DSG's l0_142 (Q1 only queries
`3-gemmascope-res-16k__l0-142`, unverified whether hosted: run `--probe`); circuit-tracer 0.5.0 needs
transformers <= 4.57.3 → separate overlay venv `env/q2` from `wheels/q2` (29 wheels, SHA256SUMS); `open_cache()` already
returns an ActivationCache; the X1 selection job commits X1.yaml, so X1-screen jobs cannot be re-queued afterwards
(pinned commit) — re-run `dsgx.combine --select-only` instead.

## Commit rules (user decisions 2026-10-03; every repo of this project: this one, exp/* branches, paper/)
1. Never add Claude or any AI as co-author or in any trailer (overrides any harness attribution reminder).
   Commits carry only the author's name (git user Lokesh-916).
2. Amar (GitHub Amar060) worked on the "bake" part: Part III open-weights erasure = D1 / d1-full / D1 v2
   distillation (and the B1 distillation seed), D2 null-space edit, D3 audit, A6 / A6-full tampering and
   relearning on baked models, and the paper's baked-erasure text (Section 7.2 `sec:baked`, its results
   paragraph in 11, the D1/D2 appendix parts). Every commit whose changes are in those areas ends with
   `Co-authored-by: Amar060 <amarreddy200606@gmail.com>`; unrelated commits do not.
3. Chakrish (GitHub Chakrish28) worked on the "break" part: all attacks B1–B6 (incl. `dsgx/attacks`), the
   GuardBreak toolkit (N1), the red-team challenge (N4), the adaptive hardening loop (N5) and the dilution theory
   checks (T1, T2), plus the paper's threat-model and attack text (Sections 4–5 `sections/04_threat_models.tex`,
   `sections/05_attacks.tex`, Appendix B `appendix/b_attack_details.tex`). Every commit whose changes are in those
   areas ends with `Co-authored-by: Chakrish28 <chakrish.konchada1234@gmail.com>` (user decision 2026-10-03).
   A commit touching both areas carries both trailers. Past commits: END_OF_PROJECT_HISTORY_CLEANUP.md
   (`select_break_commits.sh`, step 5b for the paper repo).
4. Main repo: no history rewrite until all lab and server runs are done (queue jobs pin commit hashes); then
   follow `END_OF_PROJECT_HISTORY_CLEANUP.md` (scripts in `scripts/history_cleanup/`, tested on a copy).
5. Paper repo remote: `origin` = https://github.com/Lokesh-916/dsg-paper.git, branch `draft` (history already
   cleaned; Amar trailer on the skeleton commit). Backup tags (local + GitHub):
   `backup/pre-trailer-removal-2026-10-03` (ca7e069, old history with AI trailers),
   `backup/pre-amar-trailer-2026-10-03` (69011c4). Main repo remote: git@github.com:Lokesh-916/Model-Unlearning-Using-SAEs.git.

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
  **94 dsg-a6-full** queued too at the user's request (afterok:92,afterany:93, --time 7:00, --time-min 0:30,
  --deadline 09:00, DSG_END_BY): it gets only what is left after d1-full (~1–1.5 h of its ~3 h); cells (DONE per
  target × k, order dsg-hook, dsg-nohook, d1, rmu) resume on resubmit. Caveat: 88 keeps its own 16 h limit; if rmu-v2 runs far
  over its estimate, the 09:00 promise depends on it (we did not touch 88/89, as instructed).
- **Next (after 09:00):** if 93 was stopped by the guard/limit, resubmit `submit.sh d1-full.sbatch` (resumes from
  checkpoints; finished students and DONE runs skipped). Resubmit a6-full the same way for its unfinished cells, then rerun tofu-full.
  Fetch/verify/cleanup rmu-v2 as in session 5; fetch d1-full when done.

### Session 7 — 2026-10-03 11:00–12:xx (end state)
- **Lab PC queue fixed:** idle 06:03–11:05 by a wave deadlock (A1-test-000..003, wave 1, depend on
  A1-dev-dsg-subset-ids, wave 2 in A1-dev.yaml; wave > 1 is held until the Wave-1 resume, which only fires once
  wave 1 is terminal). Queue job file edited (original in `$DSG_RESULTS/queue/_archive/manual-2026-10-03/`): wave 1,
  batch 32 → 8 (all A1-dev jobs OOMed at 16 in `mcq_eval.score_prompts`; task jobs ignore DSGX_BATCH_FACTOR).
  2 DEVIATIONS rows. Started 11:05:08 (est ~5 h at bs 8); then A1-test runs, then the Wave-1 pause (`wave_check 1`).
  No other remaining dev job runs the MCQ scorer at bs ≥ 16; TEST stays bs=1. `doctor` now reports `wave-deadlock`.
- **Server results fetched + verified:** rmu-v2 (173 files), d1-full (28), a6-full (49), validate (7).
  RMU v2 = c14 (20×r, α 300, L3 upd 1-3, 150 steps), **at grid edge** (steering, steps); TEST WMDP 0.319 vs DSG 0.298
  (n.s.), MMLU 0.548 vs 0.561. D1-full undo a0.3/a0.5 collapse MMLU to chance; a0.1 keeps it (0.514, WMDP 0.396).
  A6 `d1` cells used a0.3 (collapsed) and `rmu` cells used v1; new targets `d1-a0.1`, `rmu-v2` (code 3b7ec89).
- **Server cleanup:** deleted third-party RMU, RMU-cluster v1, D1-full a0.3/a0.5 (weights gone everywhere; metrics kept),
  code-rmu-v2. Kept: RMU-v2/best, D1-full/undo_a0.1, forget corpus, A2-tofu-full (for 96/97). Ours 35 GB, / 118 GB free.
- **Queue (another user has priority today; all `--nice=10000`, afterok):** 95 validate EXACT → **96 tofu-full** (R,
  --time 3:00; training cached; each condition saved to `runs/A2-tofu-full/tofu-metrics/partial/`, so a resubmit
  resumes) → **97 a6-full** (--time 1:00; runs only d1-a0.1, rmu-v2; est 20 min).
- **Next:** when 96/97 left the queue: `cluster/server.sh fetch tofu-full`, `verify`, `cleanup tofu-full --yes`;
  `fetch a6-full`, `verify`, `cleanup a6-full --yes` (now also removes RMU-v2 + D1-full + corpus). If 96 hit its time
  limit: `ssh gpuws 'cd ~/dsg_cluster/slurm && ./submit.sh tofu-full.sbatch --nice=10000 --time=03:00:00'`.
  Still not runnable: A6 student/d2 (lab-PC D1/D2 weights not produced yet).
- **Downloads:** gemma-3-12b-it (24.4 GB) + gemma-scope-2-12b-it resid_post/layer_24_width_16k_l0_medium (1.3 GB), HF
  licence accepted; see PREP_PROGRESS. Judge model skipped.
- **Paper** (`paper` repo, ca7e069): "pre-registered" replaced (rules committed after DEV sweeps began, before any
  main TEST result; RMU v2 grid after v1's result), contribution 4 = run/scheduled only (MUSE conditional), eq. 3 in
  eq. 1's T, Properties instead of Propositions, Gemma 2 bib shortened, circuit-breakers TODO-VERIFY. 17 pages, 0 warnings.

### Session 8 — 2026-10-03 13:00–14:xx (end state)
- **Part 1 (move to gpuws):** 70 lab jobs `MOVED-TO-SERVER` (`dsgx.queue.move`; A6 56 = a6-lora 23 + a6-baked 33, C3 10,
  A7 4); doctor 0 deadlocks; lab ETA 3d0h → 2d2h; 3 DEVIATIONS rows. The running scheduler (v2-harness code) skips
  them (it launches only WAITING); its STATUS.md shows them until the merge (prep status shows "moved to server").
  Lab-PC vs gpuws hours: PREP_PROGRESS "Session 8" table.
- **Part 2 (D1 v2):** `cluster/d1_v2.py`, rule fixed before any result (lowest DEV forget, DEV utility drop ≤ 0.02,
  utility excl. high_school_geography). d1-full α 0.1 (TEST, gpuws): WMDP 0.396 [0.358, 0.433], MMLU 0.514
  [0.503, 0.526] vs base 0.644 / 0.564, DSG 0.298 / 0.560.
- **Part 3 (figure parity):** `cluster/figparity.py` + 9 figures + 8 tables + parity index in paper_assets.
- **Server:** fetched + verified tofu-full (11 files, adc4908c…) and a6-full (73, 82c44781…); cleanup a6-full (RMU-v2,
  D1-full deleted on the server; RMU-v2 best stays on the lab PC). **TOFU-full v1 metrics invalid** (model utility
  0.0: clipped the mean truth ratio) → fixed (`METRIC_VERSION 2`), eval-only re-run 113; server marker
  `.fetched/tofu-full` removed so `cleanup tofu-full` refuses until v2 is fetched (models needed by 99 and 113).
  Staged (sizes byte-checked): figs, d1-v2, c3 (+ full Gemma Scope 2B res repo, kept), a6-lora, a7-small.
  Snapshots `code-later` (fa9cc6e), `code-later2` (9976fb0), `code-C3-85805d2`, `code-A6-c2472e5`. Ours 42 GB, / 111 GB free.
- **Queue (all --nice=10000, ≤ 3 h each):** 98 validate → 99 figs-b → 100–103 d1-v2-train → 104 d1-v2-test →
  105 d1-v2-a6 → 106 c3 → 107–109 a6-lora → 110 a7-small → 111–112 figs-a → 113 tofu-full-v2 → 116 figs-lat
  (latency re-run, protocol interleaved-v2, snapshot `code-later3` = ceae159; 99's v1 latency was biased: back-to-back
  timing on a host at load ~220/224 CPUs) (≈ 20 h). 98 validate: `VALIDATE sanity-gpuws EXACT`; 99 TOFU highlights done.
- **Gotcha:** once a job has ended and been purged, `--dependency=afterok:<it>` is rejected ("Job dependency problem");
  when appending to a chain whose validate already passed, depend on the last queued job only (`afterany:<last>`).
- **Next:** runbook §5.3 (fetch/verify/cleanup order; a6-baked when lab D1/D2 weights exist; a7-12b after cleanups;
  A7 translate after lab B3-translate).

### Session 9 — 2026-10-03 16:00–17:4x (end state)
- **Commit rule:** Chakrish28 trailer on break-area commits (rule 3 above); `select_break_commits.sh` + exclusion
  list for past commits, tested on a copy (VERIFY OK: 14 Chakrish, 16 Amar). Paper README updated.
- **Paper** (`paper`, 3 commits): Appendix A (split table, both machines + sanity 158/161), B (attack grids as
  configured; B4 attacker = gemma-2-2b-it, k = 5; B5 = gradient-free coordinate search, 8 tokens, 50/200 steps),
  C (D1 local/full/v2 + D2 tables, X1 rule text), D (RMU grids, c14 at the grid edge, TEST 0.319 vs DSG 0.298 n.s.,
  near-floor caveat), Section 8 (cross-GPU gate flips: 3 items within 0.012 of tau). Tables generated by
  `dsgx.analysis.appendix_tables`. 21 pages, 0 warnings.
- **MUSE** (BM1): data + official code fetched (lab 67 GB free). `privleak.eval` of muse_bench main crashes, so its
  `eval_data` + `sweep` are called; PrivLeak relative to OUR retrain model. Staged (114 files, sha256 ok).
- **Q2:** TOFU mode; French questions `cluster/data/q2_tofu_forget10_fr.json` (NLLB, chrF >= 45); transcoders
  streamed (27 files, 7.4 GB on gpuws); env/q2 built on the login node (CPU, offline); `--plan` passed there.
  transformers 4.57.3 calls the Hub when loading a tokenizer by repo id even offline: Q2 loads it from the snapshot path.
- **Snapshots:** code-later4 (14aaafb, muse), code-later5 (247871a, q2). Ours 59 GB, / 94 GB free.
- **Queue added (all --nice=10000):** 117 validate (afterany:116) → 118–120 muse; 121 validate (afterany:120) → 122 q2-graphs.
- **tofu-full cleanup now keeps `A2-tofu-full/full`** (removed by `cleanup q2-graphs`); server confs updated (`*.bak-0452f70`).
- **Open issue (not changed: 113 is queued):** `tofu_full.Gate` under `generate()` with a KV cache scores each new
  token alone (rho gate: clamps any generated token that fires a selected feature; window gate: never fires after the
  prompt), so TOFU ROUGE for full+dsg / full+best-gate measures that behaviour. MUSE uses prompt-only gating (PromptGate).
- **Next:** runbook §5.3 items 5–6 (fetch/verify/cleanup muse, q2-graphs); regenerate the dashboard.

### Session 10 — 2026-10-04 10:40–12:xx (end state)
- **Jobs 100–122:** done 100–107 (107: 21/23), 111, 113, 116, 117/121 validate EXACT; 112 no-op; failed 108/109 (A6-benign:
  `tatsu-lab/alpaca` not staged), 110 a7-small (sae_lens fetches Gemma Scope 2 shapes over HTTP), 118–120 muse (in-job disk
  guard, ours 85 GB), 122 q2 (`torch.isin` device). Fixes 8e60aa0 (+ `server.sh stage` warns on lab disk, c283b28). Table:
  PREP_PROGRESS "Session 10".
- **Fetched + verified:** figs 585, d1-v2 46, c3 98, tofu-full 11 (v2 metrics valid), a6-lora 166 (without trainer state).
  Cleaned: figs, d1-v2, c3, tofu-full (retain), A6 trainer.pt pruned. Staged a6-lora, a7-small, a7-12b.
  Server: ours 68 GB, free 85 GB at submit.
- **Queue (all `--nice=10000`, no hold, no deadline):** 123 validate → 124 a6-lora (2 benign jobs) → 125 q2-graphs →
  126–127 a7-small → 128–130 muse → 131–132 a7-12b. Snapshot `code-later6` (5866a88) for a7-*/q2; muse keeps code-later4.
- **Lab PC:** 50 GB free (below the 60 GB line all session; nothing deleted; candidates in PREP_PROGRESS). Remaining lab
  waves need ≈ 45–55 GB: free ~25 GB before D2/D1/A2 training (Waves 3–5) or the scheduler stops at 30 GB.
- **Wave 1:** resumed at 10:41:25 before this session (`control.json`); decisions in DEVIATIONS (Bio primary; Cyber Pareto;
  measured 4-subject vs full-MMLU relation; resume time). A4 probe__base done (best layer 19, 0.630 vs control 0.259).
- **Results digest:** `$DSG_RESULTS/RESULTS_DIGEST.md` (regenerate every session). All claims Inconclusive on both machines.
  The C-H7 rule reads exp `A2` only, so the gpuws `A2-tofu-full` result is not used (changing claims.py needs a DEVIATIONS row).
- **Next:** when 124–132 have left the queue: fetch/verify a6-lora (then `cleanup a6-lora --yes`), q2-graphs (then cleanup:
  transcoders + TOFU model), a7-small (cleanup), muse (cleanup), a7-12b (cleanup; then the lab copy of gemma-3-12b-it can go
  if you approve). If a chain step ends `partial`/`INCOMPLETE`, resubmit that sbatch with `--nice=10000`. a6-baked waits for lab
  D1/D2 weights; A7 translate for lab B3. Regenerate the digest.

### Session 11 — 2026-10-04 11:50–12:xx (end state)
- **Lab disk:** deleted (user instruction) `dsg_cache/models/{A2,D1,D2}-smoke`, third-party RMU (HF cache),
  `dsg_results_cluster/checkpoints/RMU-cluster`; no queued job referenced them. Lab free 50 → 104 GB. Legacy artifacts untouched.
- **Server:** a6-lora 124 COMPLETE (A6-benign done) and q2-graphs 125 done: fetched, verified, cleaned (transcoders, TOFU model).
  MT-Bench: judge google/gemma-2-9b-it staged (18 GB); chain **133 validate (afterany:132) → 134–137 mtbench** (`--nice=10000`).
  Server ours 73 GB, free 80 GB; MUSE peak ≈ 89 GB ours (< 100). Queue at session end: 126 a7-small R → 127 → 128–130 muse →
  131–132 a7-12b → 133–137.
- **Rules:** C-H7 also reads the gpuws `A2-tofu-full` (input only; DEVIATIONS). No threshold changed. C-H3 on labpc is now
  **Not supported** by its rule (A4 DSG best layer 9 probe 0.285, CI lo 0.253 < 0.30); it stays so.
- **Bug fixed:** gpuws TOFU / Q2 runs had no hardware label and were read as labpc. Fixed with HARDWARE.json on the server +
  re-fetch, and the scripts now write `hardware_label`. Session 10's parity test failure (Cyber Pareto) is fixed; 107 tests pass.
- **Next:** fetch/verify/cleanup a7-small, muse, a7-12b (then the lab gemma-3-12b-it copy can go if approved), mtbench (read
  `jobs/mtbench/summary.json`; report with `same_family_judge`); regenerate the digest.

### Session 12 — 2026-10-05 10:00–11:xx (end state)
- **Lab:** D1-train-* and A2-tofu-finetune-full OOM fixed (exp/D1 4d086c4, exp/A2 ef5eb17; peak 6.4 / 5.6 GiB, smoke-tested on
  the lab GPU); 19 jobs re-queued, D1-train-sameref first (wave 3). Trainer `zero_grad` moved before `step_fn` (f40dde8): the gpuws
  tofu-full + muse fine-tunes used only 1 of 4 micro-batches per step (DEVIATIONS; not re-run, user decision).
- **X1:** reads the gpuws C3 ranking; enqueued by `scripts/x1_enqueue_when_ready.sh` once the D1 undo checkpoints exist.
- **Server:** a7-small, muse, a7-12b fetched/verified/cleaned (ours 37 → 42 GB after staging, free 78 GB). Held chain:
  134–137 mtbench → 150 validate → 151–152 a6-baked (D2 only). Watchers on the lab PC: `release_when_free.sh`,
  `a6_baked_d1_when_ready.sh` (runbook 5.5). anish's job 149 was running; never touched.
- **Next:** fetch mtbench (report `same_family_judge`) and a6-baked when done; X1 selection result in
  `$DSG_RESULTS/runs/X1-screen/select/COMBINE_SELECTION.md`; regenerate the digest.

### Session 13 — 2026-10-05 11:00–12:xx (end state)
- **Trainer accumulation audit** (PREP_PROGRESS "Session 13" table): affected = gpuws TOFU-full fine-tunes (84/96, eval 113), MUSE
  (128–130) and their dependents FP-highlight (99) + Q2 (125). Not affected: D1-full, D1 v2, RMU v1/v2, A6-full / a6-lora, every lab
  run (lab D1/A2 got accumulation and the fix in one commit). User decision: re-run all affected (DEVIATIONS row supersedes "not re-run").
- **Server:** old outputs moved to `results/_superseded/accbug-2026-10-05/` (+ lab `dsg_results_cluster/_superseded/…`; `.fetched`
  markers moved too, so cleanup refuses until re-fetched). MUSE data re-staged (sha256 ok). Ours 42 GB, free 76 GB at submit.
  One linear held chain: **153 validate → 154–155 tofu-full-v3 → 156 figs-hl → 157–159 muse-v2 → 134–137 mtbench → 150–152 a6-baked**.
  Gotcha: re-pointing a held job ahead needs `scontrol update JobId=<first new> Dependency=` BEFORE pointing the old head at the new
  tail (else "Circular job dependency"); submit.sh refuses an unchained first job while ours are queued, so submit behind the tail first.
- **Watchers (lab):** release_when_free.sh, a6_baked_d1_when_ready.sh (restarted; now appends via chain_tail.sh), x1_enqueue_when_ready.sh,
  new q2_rerun_when_ready.sh (stages q2-graphs + held `validate → q2-graphs-v2` after tofu-full v2 is done and MUSE left the queue).
- **Lab:** A2-tofu-finetune-retain re-run at ef5eb17 (AdamW 8-bit, same as full; DONE 11:31); old run in `runs/A2/_superseded/`;
  A2-tofu-metrics stopped (it had started on the old retain model) and re-queued (running at session end). DEVIATIONS row.
- **C-H2** stays Inconclusive: B2/B3 `attack_success.json` are empty because neither experiment has a no-attack `dsg-faithful` run
  (the attack-success task needs the method's clean run in the same exp). No rule changed.
- **Next:** runbook §5.6 (fetch/verify/cleanup tofu-full after 155, figs after 156, muse after 159, q2-graphs after its chain);
  regenerate the digest after A2-tofu-metrics finishes (C-H7 labpc input).

### Session 14 — 2026-10-05 12:25–13:xx (end state)
- **C-H2 input:** no-attack `dsg-faithful` TEST run (same @forget items, prompt, selected config n20/rp95/m500, bs 1) appended
  last to B2.yaml / B3.yaml (exp/B2 1b892a6, exp/B3 b12cf6c, pushed, Chakrish28 trailer; earlier run indices verified unchanged).
  Lab jobs `B2-clean-dsg`, `B3-clean-dsg` (wave 1, must = head of the queue) → `B2/B3-attack-success-v2` (old empty outputs copied to
  `runs/B*/_superseded/attack-success-noclean-2026-10-05`). C-H2 rule unchanged. DEVIATIONS row.
- **C-H7 input:** `claims.evaluate(..., tofu_extra=)`; the digest passes the lab A2 tofu-metrics run into the gpuws verdict only if
  `lab_a2_tofu_fair` holds (A2 fine-tunes + metrics DONE at exp/A2 ef5eb17, metrics after both fine-tunes). Every TOFU run checked on
  its own, labelled by hardware. DEVIATIONS row. A2-tofu-metrics DONE 12:18:52.
- A5-mia, N2-adapter, N3-demo-check, N4-challenge-check were stale BLOCKED (by the session-13 stop of A2-tofu-metrics): re-queued.
- **Results (13:27–13:33):** B2/B3-clean-dsg and attack-success-v2 DONE (B2 7, B3 13 conditions). Digest regenerated (281 lab, 264 gpuws runs):
  **C-H2 labpc Supported** by its unchanged rule (5 conditions; max B2 decompose split k3 0.694 [0.642, 0.751], B3 spaced 0.449
  [0.389, 0.509], n 265). C-H7 Inconclusive on both machines (no best-fix TOFU condition: X1 not run).
- **Next:** session 13's runbook §5.6 items; digest after every session.

### Session 15 — 2026-10-05 21:29–21:4x (end state)
- **Lab power cut ~14:00, rebooted ~20:35.** `scripts/reboot_recover.sh`: tmux back (dsg-monitor/queue/workers); doctor re-queued
  X1-screen-000 (interrupted) + X1-screen-select (blocked). Canary sanity PASS (labpc); preflight --skip-sanity all PASS except the
  canary's own GPU process. X1-screen-000 running again (fresh heartbeat). Lab ETA ~14h32m. Lab free 60 GiB (at the line).
- **Watchers:** a6_baked_d1 and x1_enqueue had finished before the cut (chain 160–163 submitted; X1 enqueued 13:46). Restarted:
  release watcher as `WAIT_USER=anish EVERY=120 CLEAR_CHECKS=1 nohup cluster/release_when_free.sh >> ~/release_after_anish.log`
  (new `WAIT_USER` option, ssh timeouts, empty-reply retry); `q2_rerun_when_ready.sh`. gpuws: anish 149 R; ours 134–137, 150–163 held.

### Session 16 — 2026-10-06 10:20–10:45 (end state)
- **Re-run chain did not run:** 154–155 tofu-full-v3 and 157–159 muse-v2 refused at their in-job disk guard (free 65 GB − 16 GB
  < 50 GB floor: the MT-Bench judge 18 GB and a6-baked inputs were staged at the same time); 156 figs-hl was a no-op (no TOFU model).
  134–137 mtbench and 150 validate (EXACT) done; 151 a6-baked running (27 of 33 cells done at 10:35), 152/160–163 behind anish's jobs.
- **Fetched + verified:** mtbench (7 files, 525d7a71…); `cleanup mtbench --yes` (judge 18 GB) → ours 44 GB, free 68 GB.
  MT-Bench (judge gemma-2-9b-it, same family): base 7.46 [7.07, 7.83], DSG 7.36, window-w16 7.37 (n 158); paired Δ −0.09/−0.09
  (p 0.29/0.30): only 2–3 of 158 (question, turn) scores differ, i.e. the gate almost never fires on MT-Bench.
- **Resubmitted** (`--nice=10000`, not held) behind 163: **167 validate → 168–169 tofu-full-v3 → 170 figs-hl → 171–173 muse-v2**.
  MUSE needs a6-baked cleaned first (else its guard refuses again): after 163, `fetch a6-baked`, `verify`, `cleanup a6-baked --yes`.
  q2_rerun_when_ready.sh restarted (it had exited at 01:05 when 154–159 left without train_version-2 metrics).
- **X1 DEV selection** (09:19): detector = **cusum**; features, threshold, intervention, baked = default. X1 TEST running (2/33).
- Digest: MT-Bench table + paired Δ, X1 status line (`results_digest.mtbench_section`, `x1_status`; 2 tests).
- **Found:** T3, A8-tables and N10-cards ran 2026-10-04 10:41 right after the Wave-1 resume and read only smoke runs; T5 found no
  traces (n_series 0); T4 is a stub. Re-queue them at the end (`doctor --requeue T-T3 A8-tables N10-cards`). A7: DSG and the window
  gate leave Gemma 3 1B/4B/12B accuracy exactly unchanged although the gate fires on 16–32 % of WMDP items (undiagnosed).
- Report: `$DSG_RESULTS/IMPROVEMENTS_REPORT.md` (copy `docs/IMPROVEMENTS_REPORT.md`): every experiment, both machines, verdicts, presentation list.
- **Next:** after 163: `fetch a6-baked`, `verify`, `cleanup a6-baked --yes` (before MUSE 171 starts); after 169/170/173: runbook 5.6; digest.
