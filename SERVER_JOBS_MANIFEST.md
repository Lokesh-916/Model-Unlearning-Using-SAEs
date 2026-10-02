# SERVER_JOBS_MANIFEST — later heavy jobs on gpuws (per CLAUDE.md rules)

Every job: run from the lab PC with `cluster/server.sh` (NO_CLAUDE_RUNBOOK.md §5). Each chain starts with
`validate.sbatch` (sanity-gpuws, exact) and continues with `--dependency=afterok`. Every job script checks
≥ 40 GB free GPU memory (preamble, exit 75 otherwise), uses `--requeue`, resumes from checkpoints / DONE
markers, labels results `hardware_label=gpuws`, and never writes hazardous text outside `private/`.
Server results are reported **only** in gpuws tables (`final_report --hardware gpuws`), never mixed with
lab-PC numbers.

**Disk rule:** ours < 100 GB, server free ≥ 50 GB, lab PC free > 60 GB, at every step. `server.sh stage`
refuses when the job's *peak* (staged inputs + outputs, `EST_DISK_GB`) would break a limit. Only one job's
inputs are on the server at a time, except the d1-full → a6-full pair (a6-full uses d1-full's students).

## Base set kept on the server (≈ 14 GB, never cleaned)
env/mechunlearn2 8.2 GB (+ env/q2 overlay ≈ 0.1 GB after Q2) · code · gemma-2-2b-it 4.9 GB · Gemma Scope
L3 16k SAE 0.3 GB · WMDP / MMLU / wikitext 0.2 GB · bio activation cache s0 0.4 GB · legacy ids.

## Run order and disk budget

| step | job | prerequisites (lab PC) | staged inputs | ours at peak | est. runtime (RTX 6000 Ada) | after fetch: cleanup frees |
|---|---|---|---|---|---|---|
| 0 | rmu (jobs 79→80→81, DONE, fetched, cleaned) | — | (staged) | 25 GB | ~2 h | 11 GB (`cleanup.sh rmu`) |
| 0b | **rmu-v2** (jobs 87→88→89, queued behind a6-full) | rmu fetched | third-party RMU 9.8, corpus 0.7, code snapshot | 39 now; + RMU-v2 best/tmp ≈ 11 at run | ~3–5 h train (16 cfgs, probed bs) + 0.5 h TEST | ≈ 21 GB (`cleanup rmu-v2 --yes`) |
| 1 | **d1-full** | — | forget corpus 0.7 GB | 14 + 0.7 + 15.6 students + 10 trainer (transient) ≈ **41 GB** | ~4 h (3 × 2000 steps + 4 TEST evals) | corpus only; students stay for step 2 |
| 2 | **a6-full** | lab queue D1 (`models/D1/sameref_a0.0`) and D2 (`models/D2/nullspace`) DONE; rmu fetched | D1 sameref 5.2, D2 nullspace 5.2, RMU best 5.2, corpus 0.7 | 14 + 15.6 + 16.3 ≈ **46 GB** | ~3 h (6 targets × 3 k × 200 steps) | 32 GB (students, targets, corpus) |
| 3 | **tofu-full** | (optional) X1 selection done → best gate window size | TOFU 6 MB, COMBINE_SELECTION.json | 14 + 10.4 models + 10 transient ≈ **35 GB** | ~2.5 h | 10.4 GB (models) |
| 4 | **a7-12b** | `cluster/fetch_models.sh a7-12b` (24 GB download, HF licence) ; X1 selection (optional) | gemma-3-12b-it 24 GB, 12B L24 SAE 0.3 GB, translations, corpus | 14 + 24.3 + 1 cache + 0.7 ≈ **40 GB** | ~4 h (cache 0.5 h + 12 runs) | 26 GB |
| 5 | **mtbench** | `cluster/fetch_models.sh mtbench` (19 GB) | Qwen2.5-32B-Instruct bnb-4bit 19 GB | 14 + 19 ≈ **33 GB** | ~6 h (2 × 160 answers + 320 judgments) | 19 GB |
| 6 | **q2-graphs** | `cluster/fetch_models.sh q2-graphs` (≈ 8 GB) ; wheels/q2 (present) | transcoders ≈ 8 GB, wheels 29 MB | 14 + 8 + 1.5 graphs ≈ **24 GB** | ~4 h (80 graphs) | 9.5 GB |
| 7 | **muse** | `cluster/fetch_models.sh muse` (0.2 GB + code) | MUSE News + Books 0.2 GB | 14 + 10.4 + 10 transient ≈ **35 GB** | ~12 h (4 fine-tunes + metrics) | 0.2 GB (models deleted in-job) |

Maximum "ours" at any step: **≈ 46 GB** (step 2) < 100 GB. With 128–138 GB free on `/` today, free space
stays ≥ 80 GB, above the 50 GB floor even if other users add ~30 GB. Steps 3–7 are independent; run them in
any order, one at a time, each followed by fetch → verify → cleanup. Total ≈ 35 GPU-hours.

## Per job: one-command submit and what it produces

All: `cluster/server.sh run <job>` (= plan + stage + submit), then `check`, `fetch`, `verify`, `cleanup --yes`.

| job | script / sbatch | results fetched (lab PC `dsg_results_cluster/`) | --time | what is measured |
|---|---|---|---|---|
| rmu-v2 | `cluster/rmu_v2_train.py`, `rmu_v2_eval.py` / `rmu-v2-{train,eval}.sbatch` | `runs/RMU-v2-{train,dev,test}`, `jobs/rmu-v2/` (summary, SUMMARY.md, GRID.md), `checkpoints/RMU-v2/best` | 16 h + 4 h | 16-config RMU DEV grid at the largest safe batch size; TEST bs=1 vs base, DSG, third-party RMU, RMU v1 |
| d1-full | `cluster/d1_full.py` / `d1-full.sbatch` | `runs/D1-full/*` (harness TEST runs: base, DSG paper config, 3 students), `jobs/d1-full/summary.json` | 12 h | UNDO full-parameter distillation, noise α ∈ {0.1, 0.3, 0.5} |
| a6-full | `cluster/a6_full.py` / `a6-full.sbatch` | `runs/A6-full/relearn-<target>-k<k>/metrics.json` (exp/A6 format) | 10 h | full fine-tune relearning, k ∈ {10, 100, 1000}, 200 steps; C-H6 on gpuws |
| tofu-full | `cluster/tofu_full.py` / `tofu-full.sbatch` | `runs/A2-tofu-full/tofu-metrics/metrics.json` | 8 h | full TOFU FT; DSG + best gate; forget quality, utility, truth ratio, ROUGE |
| a7-12b | `cluster/a7_12b.py` / `a7-12b.sbatch` | `runs/A7-12b/*` (12 harness runs) | 10 h | Gemma 3 12B: base / DSG rule / best fix × clean, dilution 400/1600, translate fr |
| muse | `cluster/muse.py` / `muse.sbatch` | `runs/A5-muse/muse-{news,books}/metrics.json` | 24 h | VerbMem, KnowMem, PrivLeak for target, target+DSG, retrain |
| mtbench | `cluster/mtbench_open.py` / `mtbench.sbatch` | `jobs/mtbench/` (answers, judgments, summary with CIs) | 12 h | MT-Bench base vs DSG, fixed open judge |
| q2-graphs | `cluster/q2_graphs.py` / `q2-graphs.sbatch` | `runs/Q2-graphs/{items,graphs}/` (metrics only; graphs stay private) | 10 h | attribution graphs: share of answer influence through DSG's layer-3 features |

## CPU smoke tests (lab PC, tiny random Gemma-2, no GPU)
`python -m pytest -q tests/test_prep_server_jobs.py` (≈ 3 min): d1-full, a6-full, tofu-full, muse and mtbench
run end to end in `DSG_TINY=1` mode; a7-12b resolves and leakage-checks its 12 configs; Q2's graph metric is
checked on a synthetic graph and `--plan` imports circuit-tracer inside the overlay venv; every conf / sbatch is
checked (gres, requeue, time, no `--mem`, job name `dsg-<job>`, safe cleanup paths). On the server after
`server.sh sync`, `cluster/import_check.py` imports every script on the login node (CPU).

## What is NOT on the lab PC yet (download with `cluster/fetch_models.sh <job>`)

| need | for | size | notes |
|---|---|---|---|
| google/gemma-3-12b-it | a7-12b | ≈ 24 GB | gated licence: accept on huggingface.co and `huggingface-cli login` first |
| google/gemma-scope-2-12b-it (resid_post/layer_24_width_16k_l0_medium only) | a7-12b | ≈ 0.3 GB | listed in sae_lens 6.50 (`gemma-scope-2-12b-it-res`) |
| unsloth/Qwen2.5-32B-Instruct-bnb-4bit | mtbench judge | ≈ 19 GB | pre-quantised nf4, loads with bitsandbytes 0.50.2 offline; fallback judge `google/gemma-2-9b-it` is already present (same family → self-preference risk; label it) |
| mwhanna/gemma-scope-transcoders | q2-graphs | ≈ 8 GB | circuit-tracer's `gemma` transcoder set |
| muse-bench/MUSE-News, muse-bench/MUSE-Books | muse | ≈ 0.2 GB | then `python -m cluster.muse --inspect` to confirm config/split/column names |
| github.com/swj0419/muse_bench (tarball) | muse | < 1 MB | official metric code; our implementation follows its definitions (VerbMem ROUGE-L F1, KnowMem ROUGE-L recall, PrivLeak Min-K% AUC relative to retrain) |
| circuit-tracer 0.5.0 + 28 overlay wheels | q2-graphs | 29 MB | **present** in `~/projects/mechunlearn-project/wheels/q2` (SHA256SUMS); needs transformers 4.57.3, so it runs in a separate venv `env/q2` |

Already present and reused: gemma-2-2b-it, Gemma Scope 2B res L3 (and transcoder L3 only), TOFU (all configs),
WMDP, MMLU, wikitext, alpaca, MT-Bench questions / FastChat judge prompts / GPT-4 reference answers (`mtbench/data`),
gemma-2-9b-it, NLLB translations of WMDP-Bio (private).

## Known limits (stated in the report)
- gemma-2-2b-it targets for MUSE, not the paper's Llama-2-7B targets: MUSE numbers are not comparable with the MUSE paper.
- Gemma Scope SAEs / transcoders were trained on the PT model and are used on the IT model (as DSG itself does).
- The MT-Bench open judge is not GPT-4: absolute scores are not comparable with the paper's 7.78 (decision 9).
- No `--mem` is possible on this cluster (RealMemory=1): a7-12b peaks at ≈ 50 GB host RAM while converting weights.
