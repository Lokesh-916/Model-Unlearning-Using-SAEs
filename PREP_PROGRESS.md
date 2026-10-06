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

## Session 10 (2026-10-04): server failures fixed, lab disk, Wave-1 decisions, results digest

### Part 1: server jobs 100–122 (all ended by 03:44)
| job | what | outcome | cause / note |
|---|---|---|---|
| 100–103 | d1-v2-train ×4 | done | resumed across copies; all 3 students trained (103 had nothing left) |
| 104 | d1-v2-test | done | DEV rule bound not met (all drops > 0.02): fallback α 0.1 (drop 0.039); TEST WMDP 0.473, MMLU 0.537 |
| 105 | d1-v2-a6 | done | relearn 0.440 → 0.437 / 0.490 / 0.493 (k 10 / 100 / 1000) |
| 106 | c3 | done | 10/10 lab C3 jobs (gate AUROC 0.976–0.993 across layers 1–24) |
| 107 | a6-lora | done 21/23 | A6-benign-dsg-{hook,nohook} failed: `tatsu-lab/alpaca` not staged (offline) |
| 108, 109 | a6-lora | failed fast | same alpaca cause (0.4 min each) |
| 110 | a7-small | failed | sae_lens reads Gemma Scope 2 tensor shapes over HTTP (no offline path) |
| 111 | figs-a | done | clamp 73, data-efficiency 33 |
| 112 | figs-a | skipped (no-op) | nothing left after 111 |
| 113 | tofu-full-v2 | done | metric v2 (model utility: full 0.616, full+dsg 0.495, best gate 0.599, retain 0.608) |
| 116 | figs-lat | done | interleaved latency; DSG overhead +2.2–3.1 % (64–2048 tokens) |
| 117, 121 | validate | done | `VALIDATE sanity-gpuws EXACT` |
| 118–120 | muse ×3 | failed fast (refused) | in-job disk guard: ours 85 GB + 16 GB need ≥ 100 |
| 122 | q2-graphs | failed | `torch.isin` test elements on CPU, graph on cuda:0 |

Fixed (8e60aa0): `jobcommon.offline_sae_shapes()` (reads the safetensors header from the local cache; 1B/4B/12B SAEs
and the Gemma 3 1B bundle load on CPU with sockets disabled; also checked on the gpuws login node), Q2 device,
alpaca staged (a6-lora, a6-baked). Fetched + sha256-verified: figs 585 files, d1-v2 46, c3 98, tofu-full 11,
a6-lora 166 (without `last/trainer.pt`: new `FETCH_EXCLUDE`, pruned on the server only in DONE runs, 16 × 0.95 GB).
Cleanup: figs, d1-v2 (students 15 GB + corpus), c3, tofu-full (retain) → ours 80 → 43 GB, free 73 → 110 GB.
Staged: a6-lora (alpaca, corpus), a7-small (corpus), a7-12b (24.4 GB) → ours 68 GB, free 85 GB.
**Chain (all `--nice=10000`, no hold, no deadline): 123 validate → 124 a6-lora → 125 q2-graphs → 126–127 a7-small →
128–130 muse → 131–132 a7-12b.** Peak ours ≈ 86 GB (MUSE transient 16 GB), free ≥ ~67 GB. Snapshot `code-later6` (5866a88).
Not ready: a6-baked (lab D1/D2 weights), mtbench (judge not downloaded), A7 translate (lab B3 cache).
Lab-side note: the lab PC was below the 60 GB line during the whole session (50–51 GB). `fetch_results.sh` now allows
metric-only fetches (< 1 GB) down to 45 GB; `server.sh stage` warns instead of refusing (it only reads on the lab PC).
D1 v2 student weights were deleted with the d1-v2 cleanup (not fetched: lab disk); the metrics are kept, rerun to rebuild.

### Part 2: lab disk (55 → 50 GB free during this session)
What grew since 2026-10-03 12:00 (files newer than that): `~/.ollama` **+9.3 GB** (personal `qwen3` model blob, created
2026-10-04 00:16; not this project), `dsg_cache` +4.1 GB (lab queue: Cyber activation caches s1–s4 etc.), `~/.cache`
+0.4 GB, `~/.local` +0.3 GB (Claude Code versions), `dsg_results` and `dsg_results_cluster` < 0.1 GB. Earlier (2026-10-03
11:23): gemma-3-12b-it + Gemma Scope 2 12B L24 download, 24.4 GB.
Caches outside the project: `~/.cache/huggingface` 80 GB (gemma-3-12b-it 23, gemma-2-9b-it 18, third-party RMU 9.8,
gemma-3-4b-it 8.1, gemma-2-2b-it 4.9, Gemma Scope 2B res 4.0, NLLB 2.4, gemma-3-1b-it 1.9, the rest < 1.5 each),
`~/.cache/pip` 5.1 GB, `~/.ollama` 8.7 GB (du), `~/miniconda3` 24 GB, `~/.vscode-server` 5.8 GB, `/tmp` 1.8 GB.
No Qwen in the HF cache: the personal Qwen project is the Ollama `qwen3` model.

**Remaining lab waves need ≈ 45–55 GB:** model weights A2 tofu full + retain (2 × 4.9), D1 sameref + undo α 0.1/0.3/0.5
(4 × 4.9), D2 nullspace + ortho (2 × 4.9) = 39 GB; their run dirs (≈ 1 GB trainer state each, from the smokes) ≈ 6 GB;
activation caches (C1, N6, N7 multi-layer; ~0.3–0.4 GB each) 2–4 GB; residual captures A4/D3 and ~150 run dirs 1–5 GB.
At 50 GB free with the scheduler's 30 GB floor (`min_free_disk_gb`), the queue would stop part-way through Waves 3–5
(D2 / D1 / A2 training). About 25 GB must be freed before then.

**Legacy artifacts in baselines_DSG (not deleted; your decision):**
| file (under `artifacts_dynamic_bs1_*/unlearning/gemma-2-2b-it/gemma-scope-2b-pt-res_layer_*/width_16k/average_l0_142/results/sparsities/`) | size | reproduced by the new cache? |
|---|---|---|
| bio `layer_3/act_fgt.pkl` | 17.19 GiB | **yes**, max abs diff 0.0 on all 16,384 features (FOUNDATION_REPORT) |
| bio `layer_3/act_ret.pkl` | 17.19 GiB | **yes**, same check |
| bio `layer_8/act_fgt.pkl` | 17.19 GiB | no l0_142 layer-8 cache exists (C3 used canonical SAEs); not verified |
| bio `layer_8/act_ret.pkl` | 17.19 GiB | same |
| cyber `layer_3/act_fgt.pkl` | 17.19 GiB | same 30 features, 2 order swaps, τ 0.1410 vs 0.1416: **not bit-exact** |
| cyber `layer_3/act_ret_ORIGINAL_wikitext.pkl` | 17.19 GiB | same as above (not bit-exact) |
| cyber `layer_3/act_ret.pkl` (legacy chat-retain corpus) | 2.22 GiB | **no**: build script unknown (REPO_REPORT); keep |
Only the two bio layer-3 files (34.4 GiB) have the diff-0.0 evidence. Keep the 119 small files (1.4 MiB: sparsity
txt, question ids, metrics pkl): `dsgx/checks/sanity.py` reads the legacy ids / sparsities.
Other candidates (not deleted): `dsg_cache/models/{D1,D2,A2}-smoke` 40 GB (runbook T4 lists `*-smoke` models as safe);
`~/.cache/huggingface` gemma-2-9b-it 18 GB (MT-Bench fallback judge; MT-Bench not scheduled), third-party RMU 9.8 GB
(A1-test-003 done; no lab experiment config references it), gemma-3-{1b,4b,12b}-it 33 GB (A7 runs on gpuws; keep
until a7-small / a7-12b are fetched and verified); `dsg_results_cluster/checkpoints/RMU-cluster` 4.9 GB (RMU v1,
superseded by v2); `~/.cache/pip` 5.1 GB; Ollama `qwen3` 9.3 GB (personal).

### Part 3: Wave-1 decisions (DEVIATIONS.md, 4 rows)
Bio primary for all attacks and fixes; Cyber as a forget-utility Pareto curve (`cyber_pareto` figure + `tab:cyber-pareto`,
5866a88). Measured: on TEST the DEV-selected Cyber configs drop full MMLU by 9.1 (chat-retain) and 28.1 (WikiText) points
(5 seeds). DSG's 4-subject metric does **not** hide those drops (it shows −17.6 / −62.3). It understates only mild configs
(DEV: within 1 point while full MMLU drops 2.5–2.8). Logged with the measured numbers.
Resume: the queue was already resumed at 10:41:25 (`control.json`), before this session's review; A3-benign-open then
A4-capture started with fresh heartbeats; `wave_check 1` re-run: VERDICT OK (Cyber utility WARNs expected).

### Part 4: `python -m dsgx.analysis.results_digest` → `$DSG_RESULTS/RESULTS_DIGEST.md`
Every finished result of both machines, separate tables, mean [95% CI] n, claims per machine. Regenerate at the end of
every session. All 7 claims Inconclusive on both machines (inputs missing). Flag: the C-H7 rule reads exp `A2` only,
so the gpuws `A2-tofu-full` result is not used by the gpuws verdict (changing claims.py needs a DEVIATIONS row).
Tests: 107 pass on CPU.

## Session 11 (2026-10-04): disk freed, MT-Bench queued, C-H7 input, digest
- Lab disk 50 → **104 GB** free: deleted smoke models (39.8 GB), third-party RMU (9.8 GB), RMU v1 checkpoint (4.9 GB).
- MT-Bench (BM3): judge **gemma-2-9b-it** (same family as the judged model: labelled), conditions base / DSG / window-w16.
  `dsgx/gen/stream.py` takes an optional `score_fn` (default rho unchanged). On CPU, the streaming window score equals the harness
  `Gate.score` (0.0625 vs 0.0625). Queued 133 → 134–137 behind 132. CPU-calibrated threshold file removed from the lab gate cache
  (the server calibrates on its own GPU).
- a6-lora (A6-benign) and q2-graphs fetched, verified, cleaned. Q2: DSG fires on 1 of 3 TOFU facts (fact 1: P(key) 0.988 → 0.000);
  fact 0 passes in English and in French alike; D2 leaves P(key) unchanged on all 3.
- C-H7 reads A2-tofu-full (gpuws still Inconclusive: no A7 yet). C-H3 (labpc): **Not supported** by its rule.
- Fixed: missing hardware label on gpuws TOFU / Q2 runs; session-10 parity test failure. 107 tests pass.

## Session 12 (2026-10-05): OOM fixes, X1 wiring, server fetch + a6-baked (D2) held
- Digest: new N6 conformal table (tau, calib / held-out / shifted benign FPR, Wilson CI). Regenerated at start and end.
- **Lab OOM fixes** (smoke-tested on the lab GPU after N5 finished, queue paused meanwhile): D1 distill kept the TL bundle
  alive (two 2B models resident); now released, micro-batch 1 + accumulation (same effective batch), gradient checkpointing,
  last-position logits, AdamW 8-bit: peak 6.4 GiB (was 14.8). A2 TOFU full: same + token-weighted accumulation: 5.6 GiB (was 13.9).
  exp/D1 4d086c4, exp/A2 ef5eb17; waiting D1/A2 jobs re-pinned; D1-train-sameref wave 5 → 3 so it runs first; 19 jobs re-queued.
- **Trainer bug** (`zero_grad` after `step_fn`): gpuws tofu-full and muse fine-tunes kept only the last of 4 micro-batches
  (effective batch 4 / 2, ~1/4 of the data per epoch). Fixed (f40dde8, test). Not re-run: user decision. DEVIATIONS row.
- **X1:** C3 ranking from gpuws (layers 16, 24, 8); `--enqueue` waits for the D1 undo checkpoints:
  `scripts/x1_enqueue_when_ready.sh` (lab, nohup; log `$DSG_RESULTS/logs/x1_enqueue.log`). Dry run: 20 candidates, 43 DEV runs, 8.8 GPU-h.
- **Server:** a7-small (86 files), muse (21), a7-12b (43) fetched + verified + cleaned (ours 74 → 37 GB, free 46 → 83 GB).
  a6-baked staged for D2 only (11 jobs, `A6_BAKED_GLOBS`), chain **150 validate → 151–152 a6-baked**, `--hold --nice=10000`,
  behind the held mtbench 134–137. `cluster/release_when_free.sh` (lab, nohup) releases our held jobs when no other user has a job
  queued (two checks 10 min apart). `cluster/a6_baked_d1_when_ready.sh` stages the student/D1 part and submits a held chain once
  D1-train-sameref and undo-a0.3 are DONE.

## Session 13 (2026-10-05): Trainer accumulation audit + re-runs, A2 optimizer fairness, C-H2 diagnosis
- **Audit** (zero_grad after step_fn only hurts step functions that `backward()` earlier micro-batches themselves):
  | run | machine | affected | why |
  |---|---|---|---|
  | TOFU-full fine-tunes full + retain (84/96; eval 113) | gpuws | **yes** | `tofu_full.step_fn` accum 4: only the last micro-batch reached the optimizer |
  | MUSE retrain + target, News + Books (128–130) | gpuws | **yes** | `muse.step_fn` accum 4, same pattern |
  | FP-highlight (99), Q2 graphs (125) | gpuws | **yes (dependents)** | computed on the TOFU-full `full` model |
  | D1-full (93), D1 v2 (100–104) | gpuws | no | forward-only step_fn (one forget + one retain batch per step) |
  | RMU v1 (80) / v2 (88) | gpuws | no | own loop (zero_grad → backward → step), no accumulation, no Trainer |
  | A6-full, A6-full-d1v2 (105), A6 a6-lora (107, 124) | gpuws | no | forward-only step_fn (exp/A6 c2472e5 has no manual backward) |
  | D1-train-sameref / undo-* (lab) | lab | no | accumulation + zero_grad fix landed together (exp/D1 4d086c4) |
  | A2-tofu-finetune-full (lab, 11:08) | lab | no | same (exp/A2 ef5eb17) |
  | A2-tofu-finetune-retain (lab, 5c3229f) | lab | no (bug) | one padded batch per step; **but fp32 AdamW vs 8-bit for full → re-run** |
  | every other Trainer use (exp/* branches, v2-harness) | lab | no | `git grep backward()`: no step_fn backpropagates itself (C1 attribution is not training) |
- **Server re-runs** (snapshot code-later8 = 46d8bfc): `train_version 2` on models / partials / DONE (old ones never reused);
  tofu-full: budget stop at checkpoints, disk guard, retain model deleted after its metrics (MUSE peak stays ≥ 50 GB free).
  Old outputs moved (not deleted) to `results/_superseded/accbug-2026-10-05/` (server) and `dsg_results_cluster/_superseded/…` (lab).
  Chain, all held: **153 validate → 154–155 tofu-full-v3 → 156 figs-hl → 157–159 muse-v2 → 134–137 mtbench → 150–152 a6-baked**
  (134 re-pointed to afterany:159). Q2: `cluster/q2_rerun_when_ready.sh` stages + submits held `validate → q2-graphs-v2` after MUSE.
  `cluster/chain_tail.sh`: watchers append at the end of the chain (a6_baked_d1 watcher restarted with it).
- **Lab A2 fairness:** retain re-pinned to ef5eb17 and re-run (AdamW 8-bit, eff. batch 8, done 11:31); A2-tofu-metrics (had started
  on the old retain model) stopped and re-queued. DEVIATIONS rows (2).
- **C-H2:** Inconclusive because `runs/B2|B3/attack-success/attack_success.json` are empty: `attack_success.compute` needs a clean
  (no-attack) run of the same method in the same experiment; B2/B3 contain only base/none. Missing input: a `dsg-faithful`,
  `attack: none` TEST run (same selected config, bs 1) in B2 and B3. The 0.455 (B2 split k2) is accuracy under attack, not the
  rule's gated-item attack success. Rule unchanged.
- Tests: 111 pass (4 new in `tests/test_prep_accum_rerun.py`). Digest regenerated (276 lab, 264 gpuws runs).

## Session 14 (2026-10-05): C-H2 clean DSG runs (B2/B3), C-H7 reads the lab A2 TOFU result, digest
- B2/B3: one `{attack: {name: none}}` extra (dsg-faithful, TEST @forget, bs 1, same selection) on exp/B2 1b892a6 / exp/B3 b12cf6c;
  queued `B2-clean-dsg` (run 8), `B3-clean-dsg` (run 16) at the head of the queue, then `B2/B3-attack-success-v2`. Rule unchanged.
- C-H7: gpuws verdict also reads the lab A2 TOFU run (fixed loop, 8-bit AdamW for full and retain; provenance check
  `results_digest.lab_a2_tofu_fair`). Each TOFU run checked separately. 1 new test (`test_ch7_reads_tofu_extra_per_machine`).
- DEVIATIONS: 2 rows. Stale BLOCKED A5/N2/N3/N4 re-queued. 112 tests pass.
- Result: clean runs + attack-success DONE 13:33; **C-H2 (labpc) Supported** (B2 decompose split k3 0.694 [0.642, 0.751], B3 spaced
  0.449 [0.389, 0.509], n 265; 5 conditions meet the rule). C-H7 Inconclusive on both machines (no TOFU best-fix condition yet).

## Session 16 (2026-10-06): server fetch, re-runs resubmitted, X1 selection, improvements report
- **Server:** MT-Bench (134–137) fetched, verified, judge cleaned (ours 61 → 44 GB). The re-run chain 154–159 had refused at its
  in-job disk guard (judge + a6-baked inputs staged together); resubmitted as 167 validate → 168–169 tofu-full-v3 → 170 figs-hl →
  171–173 muse-v2 behind a6-baked (151 running, 27/33 cells; rest behind another user's jobs). Clean a6-baked before 171.
- **X1 DEV selection:** detector = CUSUM (dilution Δ −0.401 [−0.447, −0.357], utility cost 0.0002, FPR 0.045); other slots default.
- **Digest:** MT-Bench table + paired Δ (judge scores differ on 2–3 of 158 pairs), X1 status line; 2 tests (114 pass).
- **Report:** `$DSG_RESULTS/IMPROVEMENTS_REPORT.md` (copy in `docs/`): every experiment on both machines with what / why / how /
  result / verdict / lesson, overview table, claims, and the presentation list.
- **Found while writing it:** T3, A8-tables, N10-cards ran on smoke runs only (re-queue at the end); A7 gate fires on Gemma 3 but
  changes no answer (undiagnosed); attack success for leetspeak is near the chance level of the base model under the same encoding.

## Session 17 (2026-10-06): reporting jobs after X1, Gemma 3 diagnosis, self-updating paper
- **Part 1:** `T-T3`, `A8-tables` re-queued (WAITING) with deps on all 33 X1 jobs; T3 re-pinned to exp/T-checks 219ed0e (skips
  smoke / archived / unfinished runs). `N10-cards`: deps likewise, methods taken from summary.json; re-queued by
  `scripts/n10_after_x1.sh` (running, nohup) after `final_report --interim` writes summary.json. Old outputs in
  `runs/{T,A8,N10}/_superseded/smoke-inputs-2026-10-04/`. DEVIATIONS row. T5 left (its glob matches no run).
- **Power cut during Part 1; lab back ~12:37.** Queue recovered (tmux 12:45, doctor clean); unpushed 3d983c1 pushed;
  `n10_after_x1.sh` and `cluster/q2_rerun_when_ready.sh` restarted (both had died with the reboot).
- **Part 2 (A7 Gemma 3 diagnosis, `docs/A7_GEMMA3_DIAGNOSIS.md`, `scripts/diag_gemma3_clamp.py`, CPU only):** not a bug.
  Hook `blocks.<L>.hook_resid_post` valid, error term does not cancel the edit, hooked output feeds layer L+1. Cause: DSG's
  absolute clamp (−500, unit decoder rows) vs Gemma 3's residual norm (1B L13 6,834; 4B L17 30,931; Gemma 2 L3 92):
  edit/resid 0.33 / 0.10 vs 33.6; one clamped token moves the last-token distribution ~9× less. Item level: 1B DSG changes 6
  answers (cancelling), 4B/12B none; non-fired items bit-identical. Reported as a finding; no gpuws re-run queued.
- **Two more power cuts (~13:30, ~13:46):** `scripts/reboot_recover.sh` each time (X1-002 / X1-004 re-queued as interrupted);
  N10 and Q2 watchers restarted each time.
- **Part 3 (self-updating paper):** `scripts/paper_update.sh` = final_report --interim for labpc and gpuws into
  `$P/paper_numbers/{labpc,gpuws}` (never `$DSG_RESULTS/summary.json`) → `dsgx.analysis.paper_numbers` → `paper/numbers.tex`
  (341 macros, `\resX` = value [CI] (n), Val/CI/N(/P); source of each macro in a comment; missing source → red [pending];
  6 pending: B4/B5 have no attack-success summary) → latexmk (23 pages, 0 warnings). final_report summary.json gains a `paper`
  block and run-level CIs for single-seed rows (values now match the digest). Paper: Sections 8, App. A, App. D numbers are
  macros (values unchanged; the Section 8 cross-GPU numbers were recomputed from the two sanity runs and reproduce exactly);
  abstract, Results I–IV and conclusion drafted from the digest, `\interimnote` on X1 TEST (C-H5), a6-baked (C-H6),
  TOFU/MUSE re-runs (C-H7). Digest regenerated.
