# NO_CLAUDE_RUNBOOK: finishing the DSG project without Claude Code

Written 2026-10-02 on branch `prep/later-runs`. Every remaining operation as copy-paste commands.
Each block ends with **You should see:** a line that tells you it worked. If you see something
else, go to section T (troubleshooting). Nothing here needs judgment: where a decision was needed,
a script makes it (`doctor`, `wave_check`, `combine`, `server.sh plan`).

**Safety rule (always):** never open, print or paste WMDP questions, generations, attack prompts or
anything under `dsg_private/` or `~/dsg_cluster/private`. Share only status files, ids, hashes and
metrics. All scripts here print ids and metrics only.

---

## 0. Every new terminal

```bash
export P=~/projects/mechunlearn-project
conda activate mechunlearn2
cd $P/prep                 # the tools in this runbook live here (branch prep/later-runs)
source scripts/env.sh      # sets DSG_RESULTS, DSG_CACHE, DSG_PRIVATE, offline HF
```
**You should see:** no output. `echo $DSG_RESULTS` prints `.../mechunlearn-project/dsg_results`.

> The queue itself (scheduler, tmux) runs from `$P/baselines_DSG` (branch v2-harness). The tools in
> `$P/prep` read and write the same `$DSG_RESULTS/queue`, so `status`, `resume`, `doctor` work from
> either directory. Never start `scripts/tmux_up.sh` from `$P/prep`; use `$P/baselines_DSG/scripts/tmux_up.sh`.
> After `prep/later-runs` is merged into v2-harness (CLAUDE.md, "Merge notes"), use `$P/baselines_DSG` for everything.

---

## 1. Queue status (hourly, ~1 minute)

```bash
python -m dsgx.queue.status --chat     # <= 40 lines; paste into the web chat if you want advice
python -m dsgx.queue.doctor            # classifies any FAILED / BLOCKED job (read-only)
```
**You should see:** `done N | running 1 | waiting M | failed 0 | blocked 0` and
`queue doctor ...: 0 job(s) not healthy ... nothing to do`.

Live view (optional): `tmux attach -t dsg-monitor`, leave with `Ctrl+b` then `d`. Never press
`Ctrl+c` in `dsg-queue`.

---

## 2. Re-queue failed jobs

```bash
python -m dsgx.queue.doctor            # 1) read the classes
python -m dsgx.queue.doctor --apply    # 2) re-queue every SAFE failure + the jobs it blocked
```
**You should see:** `re-queued K job(s): ...`, then in `status` those jobs are WAITING again.

What the classes mean (the doctor prints SAFE / WAIT / STOP in front of each job):

| class | doctor does with `--apply` | you do |
|---|---|---|
| interrupted (reboot, kill) | re-queues | nothing |
| hung (once) | re-queues | nothing |
| oom | re-queues | first check `nvidia-smi` shows no foreign process (T2) |
| blocked | re-queued together with its cause | nothing |
| missing-input | **not** re-queued | wait until the source job is DONE, then `python -m dsgx.queue.doctor --requeue <JOB>` |
| disk | **not** re-queued | free disk (T4), then `--requeue <JOB>` |
| hung-repeat, code-error | **not** re-queued | T3 / T10 |
| leakage, drift | **not** re-queued; exit code 2 | T6 / T7: stop and do not resume |

To force a specific job back (any class): `python -m dsgx.queue.doctor --requeue <JOB_ID> [...]`
(also re-queues the jobs it blocked).

---

## 3. Resume after the Wave-1 pause

When Wave 1 finishes the scheduler writes `$DSG_RESULTS/WAVE1_DONE.md` and pauses
(`status` shows `PAUSED: wave 1 finished ...`).

```bash
python -m dsgx.queue.wave_check 1      # scripted review; exit 0 = fine
python -m dsgx.queue.resume            # only if the VERDICT line says "OK to resume"
```
**You should see:** every line `PASS` (or `WARN`), then `VERDICT: OK to resume`; after resume,
`status` shows the next jobs RUNNING within ~1 minute.

- `WARN utility ...drop +0.0123` or `WARN selection bound not met`: allowed (the dev rule picked the
  closest config; it is logged in the run's config.json). Resume, and the final report flags it.
- `FAIL forget ...` (DSG forget not below base) or `FAIL hardware` or `FAIL canary`: do not resume; T6.
- `VERDICT: wave not finished yet`: some Wave-1 jobs are still running; check again later.

---

## 4. Reboot recovery (lab PC)

```bash
cd $P/prep && source scripts/env.sh
scripts/reboot_recover.sh              # tmux_up (from baselines_DSG) + doctor --apply + status
```
**You should see:** `dsg-monitor`, `dsg-queue`, `dsg-workers` in `tmux ls`, then
`re-queued K job(s): ...` (jobs the reboot killed) and a status table with `running 1`.

If it prints `GPU driver not ready`: wait one minute and run it again. Training jobs resume from their
last checkpoint. Optional auto-start at boot (not installed; add with `crontab -e` only if you want):
`@reboot sleep 90 && bash -lc 'cd ~/projects/mechunlearn-project/prep && scripts/reboot_recover.sh >> ~/projects/mechunlearn-project/dsg_results/logs/reboot.log 2>&1'`

---

## 5. Server jobs (gpuws)

All server work goes through **one lab-PC command**, `cluster/server.sh` (run in `$P/prep`).
It enforces the CLAUDE.md rules: disk limits (server free ≥ 50 GB, ours < 100 GB, lab PC > 60 GB),
the `--bwlimit` copy, the command log, validate-first chains (`sanity-gpuws` exact gate), one of our
jobs at a time, fetch with sha256 verification, and cleanup only after a verified fetch.
The order and the disk budget of every job are in **SERVER_JOBS_MANIFEST.md**.

### 5.1 The cycle for any job `<job>` (rmu, d1-full, a6-full, tofu-full, a7-12b, muse, mtbench, q2-graphs)

Jobs a7-12b, mtbench, q2-graphs and muse need a download on the lab PC first (network, checks disk):
`cluster/fetch_models.sh <job>`; `cluster/fetch_models.sh list` shows what is present.
You should see: `downloaded <repo> -> <path>`.

```bash
cluster/server.sh status               # 0. nothing of ours running; note free disk
cluster/server.sh sync                 # 1. code + slurm scripts to the server, CPU import check
cluster/server.sh plan <job>           # 2. what will be copied, GB, disk after staging
cluster/server.sh stage <job>          # 3. copy inputs (refuses if a limit would break)
cluster/server.sh submit <job>         # 4. validate.sbatch -> the job's sbatch chain (afterok)
cluster/server.sh check <job>          # 5. any time: queue state + last log lines (metrics only)
cluster/server.sh fetch <job>          # 6. after it left the queue: results -> dsg_results_cluster, verified
cluster/server.sh verify <job>         # 7. file counts + summary headline on the lab PC
cluster/server.sh cleanup <job>        # 8a. dry run: what would be deleted
cluster/server.sh cleanup <job> --yes  # 8b. delete the job's large inputs on the server
```
**You should see:**
1. `synced prep/later-runs@xxxxxxx dirty=false` and `IMPORTS OK`.
2. `after staging: free X GB (must stay >= 50), ours Y GB (must stay < 100)` with X ≥ 50, Y < 100.
3. `staged <job>; server free X GB, ours Y GB`.
4. `submitted validate.sbatch as job N`, then one `submitted ... as job N+k` per chain step.
5. While running: `R` state lines; in the validate log `VALIDATE sanity-gpuws EXACT`; at the end of
   the job log a `[<job>] done` / `summary` line.
6. `fetched and verified K files (listing sha ...)` (twice: validate and the job).
7. Non-zero file counts for every path, and a `headline:` line.
8. `deleted <path> (size)` lines, then `du` of `~/dsg_cluster` back near 14–25 GB.

`server.sh run <job>` does 2–4 in one go (asks before copying). Never stage a second job before
the previous job's cleanup (the manifest budget assumes one job's inputs at a time).
Per-job details (inputs, outputs, runtime): SERVER_JOBS_MANIFEST.md.

### 5.3 Session 8 chain (2026-10-03): moved lab experiments, D1 v2, figure parity

Queued on gpuws (all `--nice=10000`, each ≤ 3 h, resumable; `afterok` validate 98, `afterany` the previous):
`98 validate → 99 figs-b → 100–103 d1-v2-train → 104 d1-v2-test → 105 d1-v2-a6 (afterok 104) → 106 c3 →
107–109 a6-lora → 110 a7-small → 111–112 figs-a → 113 tofu-full-v2 → 116 figs-lat` (latency re-run). Est. ≈ 20 h.
To append one more job later: `ssh gpuws 'cd ~/dsg_cluster/slurm && ./submit.sh <x>.sbatch --nice=10000 --dependency=afterany:<last id>'`
(an `afterok:` on a job that already ended is rejected by Slurm).
Jobs run from snapshots `code-later` (prep fa9cc6e) and `code-later2` (9976fb0, tofu-full-v2 only); lab jobs
run with their pinned exp-branch commits (`code-C3-85805d2`, `code-A6-c2472e5`).

```bash
cluster/server.sh status                     # our jobs, GPU, disk
cluster/server.sh check d1-v2                # also: figs, c3, a6-lora, a7-small, tofu-full
ssh gpuws 'cat ~/dsg_cluster/results/jobs/labjobs-c3/status.json | head -30'   # moved lab jobs: done / left
```
A train / lab-jobs / figs sbatch that prints `INCOMPLETE` or `budget used` is normal: the next chained copy
continues. If the last copy of a group ends INCOMPLETE, resubmit that sbatch once more, e.g.
`ssh gpuws 'cd ~/dsg_cluster/slurm && ./submit.sh a6-lora.sbatch --nice=10000'`.
`d1-v2-test` exits 3 (and 105 is cancelled) if a student has no DEV run yet: resubmit `d1-v2-train.sbatch`,
then `d1-v2-test.sbatch`, then `d1-v2-a6.sbatch` with `--dependency=afterok:<previous id>`.

**Fetch / verify / cleanup, in this order, each after its jobs left the queue** (`server.sh fetch` refuses while
a `dsg-<job>*` job is queued):
1. `figs` (after 116): fetch, verify, `cleanup figs --yes` (cyber cache, data-efficiency caches).
2. `d1-v2` (after 105): fetch, verify, read `dsg_results_cluster/jobs/d1-v2/SUMMARY.md`, `cleanup d1-v2 --yes`
   (students; the corpus is kept while any job is queued).
3. `c3`, `a6-lora`, `a7-small`: fetch, verify, cleanup each. **Cleanup a6-lora before a6-baked** (both write runs/A6).
4. `tofu-full` (after 113, the v2 re-eval): fetch, verify, then `cleanup tofu-full --yes`. Since session 9 this
   removes only `A2-tofu-full/retain`: `A2-tofu-full/full` is needed by Q2 (122) and is removed by `cleanup q2-graphs`.
   Do NOT clean tofu-full earlier: 99 (TOFU highlights) and 113 need the models. The v1 marker was removed on purpose.
5. `muse` (session 9 chain: 117 validate → 118–120 muse, each <= 3 h, resumable): fetch, verify, read
   `dsg_results_cluster/runs/A5-muse/muse-{news,books}/metrics.json` (VerbMem, KnowMem, PrivLeak for target,
   target+dsg, target+best-gate, retrain), `cleanup muse --yes`. If 120 ends with "partial", resubmit
   `ssh gpuws 'cd ~/dsg_cluster/slurm && ./submit.sh muse.sbatch --nice=10000'` (finished stages are skipped).
6. `q2-graphs` (121 validate → 122): fetch (~1 GB: 4 full graphs in `runs/Q2-graphs/tofu/pt/`), verify, look at
   `runs/Q2-graphs/tofu/figures/panel_fact0.pdf`, then `cleanup q2-graphs --yes` (transcoders 7.4 GB + TOFU model).
   If it ends "partial", resubmit `q2-graphs.sbatch` the same way.

Panel dashboard (any time, lab PC): `python -m dsgx.analysis.dashboard` → `results/dashboard.html`
(`--every 15` keeps regenerating it; `--no-server` skips the ssh query).

### 5.4 Session 10 chain (2026-10-04)
`123 validate → 124 a6-lora → 125 q2-graphs → 126–127 a7-small → 128–130 muse → 131–132 a7-12b` (all `--nice=10000`, no
deadline; pause by hand with `ssh gpuws 'scontrol hold <ids>'` / `scontrol release <ids>` if the other user needs the GPU).
After each group has left the queue: `cluster/server.sh fetch <job>`, `verify <job>`, `cleanup <job> --yes` for
a7-small, muse, a7-12b, then **mtbench** (session 11: 133 validate → 134–137; judge gemma-2-9b-it, same family as the judged model,
so always report `same_family_judge: true`; resubmit `mtbench.sbatch --nice=10000` if 137 ends `partial`). a6-lora and q2-graphs are
already fetched and cleaned. a6-lora fetches skip `last/trainer.pt` (optimizer state, `FETCH_EXCLUDE`);
cleanup deletes it on the server only inside DONE runs. Fetching metrics (< 1 GB) works down to 45 GB lab free.
**You should see:** `fetched and verified K files`, then `deleted ...` lines.

### 5.5 Session 12 (2026-10-05): held chain + watchers
Queue: `134–137 mtbench → 150 validate → 151–152 a6-baked (D2 only)`, all held (`--hold`) and `--nice=10000`.
Lab-PC background watchers (check with `pgrep -af 'when_ready|release_when'`; restart the same way if the PC rebooted):
`nohup cluster/release_when_free.sh >> $P/dsg_results_cluster/release_watch.log 2>&1 &` (releases our held jobs once no other
user has a job queued), `nohup cluster/a6_baked_d1_when_ready.sh >> $P/dsg_results_cluster/a6_baked_d1_watch.log 2>&1 &`
(student + D1 part of a6-baked, held chain), `nohup scripts/x1_enqueue_when_ready.sh >> $DSG_RESULTS/logs/x1_enqueue.log 2>&1 &`
(X1 `--enqueue` once D1-train-undo-a0.{1,3,5} are DONE). After 137: `fetch mtbench`, verify, cleanup; after 152: `fetch a6-baked`.

**Later (not queued):**
- `a6-baked` (33 moved lab jobs on student / D1-local / D2 weights): when the lab jobs `D1-train-sameref`,
  `D1-train-undo-a0.3` (Wave 5) and `D2-edit-nullspace` (Wave 3) are DONE: `cluster/server.sh plan a6-baked`,
  `stage a6-baked`, then `cluster/submit_chain.sh --nice 10000 a6-baked.sbatch a6-baked.sbatch a6-baked.sbatch a6-baked.sbatch`.
- `a7-12b` (24 GB): after the cleanups above, `cluster/server.sh stage a7-12b`, then
  `cluster/submit_chain.sh --nice 10000 a7-12b.sbatch a7-12b.sbatch`.
- A7 translate runs need the full B3 translation cache (lab B3-translate, Wave 2; the current cache has 6 smoke
  items): afterwards re-stage `dsg_private/translations` and resubmit `a7-small` / `a7-12b` (finished runs are skipped).
- Lab queue view: `python -m dsgx.queue.move --list` (70 MOVED-TO-SERVER); `--undo --jobs '<glob>' --apply` puts
  jobs back to WAITING if a server group must be abandoned.

All these results are gpuws-only: `final_report --hardware gpuws` and `paper_assets --hardware gpuws
--runs $P/dsg_results_cluster/runs` (INDEX.md lists every DSG figure type and whether its data exist).

### 5.6 Session 13 (2026-10-05): re-runs with the fixed Trainer (accumulation bug)
One linear held chain (`--nice=10000`): `153 validate → 154–155 tofu-full-v3 → 156 figs-hl → 157–159 muse-v2 →
134–137 mtbench → 150 validate → 151–152 a6-baked`. Code snapshot `code-later8`. Old outputs: server
`results/_superseded/accbug-2026-10-05/`, lab `dsg_results_cluster/_superseded/accbug-2026-10-05/` (keep; never in tables).
Watcher `nohup cluster/q2_rerun_when_ready.sh >> $P/dsg_results_cluster/q2_rerun_watch.log 2>&1 &` stages q2-graphs and appends a
held `validate → q2-graphs-v2` once tofu-full v2 is done and MUSE has left the queue. Watchers append behind
`cluster/chain_tail.sh` (end of the chain, not the highest id).
1. After 155: `cluster/server.sh fetch tofu-full`, `verify tofu-full`, then `cleanup tofu-full --yes` (the job already
   deleted `retain`; `full` stays for Q2). **You should see:** `metrics.json` with `"train_version": 2`.
2. After 156: `fetch figs`, `verify figs` (FP-highlight). Do not run `cleanup figs` before this fetch (it refuses anyway).
3. After 159: `fetch muse`, `verify muse`, `cleanup muse --yes` (summary.json lists `finished: [news, books]`).
4. After q2-graphs-v2: `fetch q2-graphs`, `verify q2-graphs`, `cleanup q2-graphs --yes` (removes the TOFU model).
5. Regenerate: `python -m dsgx.analysis.results_digest`, then `paper_assets` / `appendix_tables` for the TOFU, MUSE, Q2 and
   FP-highlight assets.

### 5.7 Session 17 (2026-10-06): reporting jobs after X1
`T-T3` and `A8-tables` are re-queued and depend on all 33 X1 jobs (they start by themselves after `X1-attack-success`).
`N10-cards` needs a fresh `summary.json`, so a watcher does it:
`nohup scripts/n10_after_x1.sh >> $DSG_RESULTS/logs/n10_after_x1.log 2>&1 &` (started in session 17; **restart it after a reboot**
if the log has no `N10-cards re-queued` line). **You should see** in the log: `X1 done; writing summary.json`, then
`N10-cards re-queued`. If it says `X1 has BAD:FAILED`: `python -m dsgx.queue.doctor`, fix, rerun the watcher.
T5 is not re-queued: its glob `C2*/*cusum*/traces.npz` matches no run directory (names carry no detector).

### 5.8 Session 20 (2026-10-06): X1 TEST replica + A7-scaled on gpuws
Chain (nice 0): `175–176 tofu-full-v3 → 177 figs-hl → 183 validate → 184–186 x1 → 188 a7-scaled → 189 x1 (catch-up, no-op if
done) → 178–180 muse-v2`. x1 = the lab's 33 X1 TEST jobs at exp/X1-combine 93625f5 (`cluster/lab_jobs.py --group x1`, status in
`results/jobs/labjobs-x1/status.json`); the lab X1 keeps running. a7-scaled = the lab job `A7-scaled-000` (MOVED-TO-SERVER).
1. After 189: `cluster/server.sh fetch x1`, `verify x1` (**you should see** ~192 run dirs + `attack-success` under
   `dsg_results_cluster/runs/X1`, `HARDWARE.json` = gpuws), then `cleanup x1 --yes` (retain corpus, translations, B4/B5, code-X1-*).
   If `labjobs-x1/status.json` says `"complete": false`: `ssh gpuws 'cd ~/dsg_cluster/slurm && ./submit.sh x1.sbatch --dependency=afterany:<last of ours>'`.
2. After 188: `fetch a7-scaled`, `verify a7-scaled`, `cleanup a7-scaled --yes` (Gemma 3 1B + SAE + cache). Do both cleanups before
   MUSE trains if free space is tight: MUSE's in-job guard needs free − 16 GB ≥ 50 GB.
3. Digest: `python -m dsgx.analysis.results_digest` reports X1 per machine; once both are 192/192 + attack-success it prints
   **X1 replicated on two machines** with each machine's own C-H5 verdict (never pooled; `claims.evaluate` refuses mixed hardware).

### 5.2 Server results in the report
Server results land in `$P/dsg_results_cluster/` (same run-directory format). They are reported
in their own **gpuws** tables and never mixed with lab-PC numbers:
`python -m dsgx.analysis.final_report --hardware gpuws --runs $P/dsg_results_cluster/runs --out $P/dsg_results_cluster/report`.

---

### 5.9 Session 22 (2026-10-06): final method (X1 combined gate = CUSUM) — X1-suite (lab) and MT-Bench cusum (gpuws)
1. Lab, nothing to do: `X1-suite-*` (6 jobs) start by themselves after all 33 X1 jobs. Check:
   `python -m dsgx.queue.status | grep X1-suite`. **You should see:** WAITING until X1 is done, then benign-open, leak,
   tofu-metrics, tofu-qa-forget/-retain (GPU, ~6 h in all) and paired (CPU). Results: `$DSG_RESULTS/runs/X1-suite/`
   (`paired/paired.json` = gate − DSG on identical items).
2. gpuws MT-Bench `cusum` mode, **only after MUSE-v2 (178–180) has left the queue** (the 18 GB judge staged earlier makes
   MUSE's disk guard refuse, session 16):
   `cluster/stage_code_snapshot.sh later10` → `cluster/server.sh plan mtbench-x1` (free ≥ 50 GB after staging) →
   `cluster/server.sh stage mtbench-x1` → `cluster/server.sh submit mtbench-x1 --after-any <last job of ours>` →
   when done: `fetch mtbench-x1`, `verify mtbench-x1`, `cleanup mtbench-x1 --yes`.
   **You should see** in `dsg_results_cluster/jobs/mtbench/`: `answers_cusum.jsonl` (160 rows) and
   `judgments_cusum__gemma-2-9b-it.jsonl`; base/dsg/window rows unchanged (skipped as done).
3. Then `scripts/paper_update.sh` and `paper/scripts/fix_framing_check.sh` (lists the macros still [pending]); choose the
   version in `paper/FIX_FRAMING.tex` by `\resLabVerdictCHFive` and apply it by hand.

## 6. Combination wave (after Waves 1–3 and N6 are DONE)

Prerequisites: C1 (feature files), C3 (layer ranking), D1 and D2 (checkpoints) DONE. `--enqueue` refuses
while any of them is missing (`--allow-missing` screens without them; then add a DEVIATIONS.md row).
The dry run prints the GPU-hour estimate (screening ≈ 0.5 h per candidate; TEST ≈ 25–40 h depending on
how many slots are selected).

```bash
python -m dsgx.combine --dry-run       # 1. what would happen: screening jobs, rule, configs
python -m dsgx.combine --enqueue       # 2. creates exp/X1-combine (worktree), enqueues DEV screening,
                                       #    a CPU selection job, and (automatically, after selection)
                                       #    the TEST runs: combined method, leave-one-out ablations,
                                       #    5 seeds, every B attack and the utility benchmarks
python -m dsgx.queue.status --chat     # 3. X1 jobs appear
cat $DSG_RESULTS/runs/X1-screen/select/COMBINE_SELECTION.md    # 4. after the selection job: the decision
```
**You should see:** (1) a table of candidate components per slot and `N screening runs` / `would
enqueue`. (2) `queued X1-...` lines and `worktree: .../dsg_worktrees/exp/X1-combine`. (4) per slot:
`selected: <component>` or `none passed (keeps the DSG default)`, each with its DEV paired p-value,
utility cost and benign FPR.

The pre-registered rule (applied by the script, never by hand; see `dsgx/combine.py`): a component is
included only if, on DEV, it improves its target significantly (paired bootstrap p < 0.05 and McNemar
p < 0.05 where binary), costs ≤ 1 point of full-MMLU utility, and keeps benign FPR ≤ 5%. One per slot:
feature selection (C1), detector (C2/C3/N7), threshold (N6), intervention (C5), plus the best baked
method (D1/D2). TEST is checked once.

---

## 7. Final report, figures and paper assets

```bash
python -m dsgx.analysis.final_report                    # completeness, aggregation, CIs, tests, figures,
                                                        # summary.json, FINAL_REPORT.md, audit cards
python -m dsgx.analysis.paper_assets                    # LaTeX tables (booktabs) + PDF figures at paper size
```
**You should see:** `completeness: done A / partial B / failed C / skipped D / deferred E`, then
`wrote .../FINAL_REPORT.md`, `.../summary.json`, `N figures`, `M audit cards`; paper assets:
`wrote K tables, J figures -> $DSG_RESULTS/paper/`.

Options: `--interim` (allowed before everything is done; the report says so), `--smoke` (smoke runs,
for testing), `--out DIR` (default `$DSG_RESULTS`). Re-running is safe; files are overwritten.
The claims table (C-H1 … C-H7) is computed by fixed rules in `dsgx/analysis/claims.py`; every verdict
cites the runs and numbers it used.

**Self-updating paper (session 17): one command regenerates every number and rebuilds the PDF.**
```bash
scripts/paper_update.sh            # final_report --interim (labpc, gpuws) -> paper_numbers/*/summary.json,
                                   # dsgx.analysis.paper_numbers -> paper/numbers.tex, latexmk -> paper/main.pdf
scripts/paper_update.sh --no-pdf   # numbers.tex only
```
**You should see:** two `claims: ...` lines, `wrote .../paper/numbers.tex: N macros, P pending` (each pending number
listed with its missing source), `LaTeX warnings: 0`, `built .../main.pdf (NN pages)`. Then
`git -C ~/projects/mechunlearn-project/paper diff numbers.tex` and commit it in the paper repo. Result numbers in the
paper are only macros `\resX` (value [CI] (n)), `\resXVal`, `\resXCI`, `\resXN` (diffs also `\resXP`); a missing
source prints a red [pending]; text resting on unfinished runs is marked `\interimnote{...}` (orange): remove those
marks when X1 TEST, the TOFU/MUSE re-runs and a6-baked are in and their numbers are no longer pending.
It never writes `$DSG_RESULTS/summary.json` (the N10 watcher's).

---

Results digest (end of every session; both machines, CIs, claims):
`python -m dsgx.analysis.results_digest` → **You should see:** `wrote .../dsg_results/RESULTS_DIGEST.md (N lab runs, M gpuws runs)`.

## 8. Qualitative track (Q1–Q8)

```bash
python -m dsgx.analysis.qual.q6_trajectory              # layer-by-layer figures from A4/D3 outputs (CPU)
python -m dsgx.analysis.qual.q5_geometry                # representation geometry (CPU)
python -m dsgx.analysis.qual.q3_never_learned           # DSG vs TOFU retain model (CPU)
python -m dsgx.analysis.qual.q1_feature_cards --fetch   # Neuronpedia explanations (network; ~60 small requests)
python -m dsgx.analysis.qual.q1_feature_cards           # build cards from the fetched cache
python -m dsgx.analysis.qual.annotate sheet --n 60      # Q4/Q8: TOFU-only annotation sheet (2 annotators)
python -m dsgx.analysis.qual.annotate kappa A.csv B.csv # Cohen's kappa between the two sheets
python -m dsgx.analysis.qual.annotate gallery           # case-study gallery (TOFU only)
```
**You should see:** `wrote ...` lines under `$DSG_RESULTS/qual/`. Q2 (attribution graphs) runs on the
server: `cluster/server.sh run q2-graphs`.

---

## T. Troubleshooting: the 10 most likely failures

**T1. Jobs FAILED with no exit code after a power cut / reboot.**
`scripts/reboot_recover.sh`. You should see `re-queued K job(s)`.

**T2. `oom` (CUDA out of memory) twice.**
`nvidia-smi` — if a process that is not ours (not `python -m dsgx...`) holds memory, wait for it or ask
its owner. Then `python -m dsgx.queue.doctor --apply`. If it fails again with the GPU otherwise
empty, the job is too big for 16 GB: move it to the server list (SERVER_JOBS_MANIFEST.md) and log a
row in `$DSG_RESULTS/DEVIATIONS.md`.

**T3. `hung-repeat` (heartbeat stale twice).**
Look at the last lines of the log printed by the doctor (`tail -n 20 <log>`; it contains progress
and metrics only). Most common cause: a model or dataset not in the offline cache (`OfflineModeIsEnabled`
/ `LocalEntryNotFoundError`). Fix: `DSGX_OFFLINE=0 python -c "from huggingface_hub import snapshot_download as s; s('<repo id>')"`
then `python -m dsgx.queue.doctor --requeue <JOB>`.

**T4. Disk low (`disk` alert, free < 30 GB, or doctor class `disk`).**
`df -h ~` then `du -sh $DSG_CACHE/* | sort -h | tail`. Safe to delete: `$DSG_CACHE/residuals/<exp>`
of experiments whose probe jobs are DONE, `$DSG_CACHE/models/*-smoke`, `$DSG_RESULTS/runs/_archive`.
Never delete `$DSG_RESULTS/runs/<exp>` (results) or `actcache` (shared). The alert clears by itself.

**T5. GPU idle while jobs are WAITING.**
`python -m dsgx.queue.status` top lines: `PAUSED` → section 3 or the reason text; a dependency
RUNNING on CPU → normal; `python -m dsgx.queue.doctor` prints `STOP wave-deadlock` (a wave-1 job
waits on a later-wave job; happened 2026-10-03) → follow its `->` line; nothing obvious → `tmux ls` must list `dsg-queue`; if not,
`$P/baselines_DSG/scripts/tmux_up.sh`. A stale `queue/scheduler.lock` is replaced automatically.

**T6. Canary drift (`canary` alert, doctor class `drift`, queue PAUSED).**
Do not resume. Check `nvidia-smi` driver version against `580.173.02` (a driver update changes bf16
numerics). Re-run the gate by hand when the GPU is free: `python -m dsgx.run --sanity`.
`SANITY PASS` → it was transient: `python -m dsgx.queue.resume`. Still failing → results after the
last good canary are suspect: list them with `python -m dsgx.analysis.final_report --interim` (runs
are dated), re-queue them with the doctor once the cause is fixed, and log it in DEVIATIONS.md.

**T7. Leakage check failed (doctor class `leakage`, exit 4).**
The calibration or training corpus overlaps an evaluation set. Do not re-queue. Log the job id in
DEVIATIONS.md; the final report lists the experiment as `failed (leakage)`. Fixing needs a code change
(T10).

**T8. Server job cancelled / `[precheck] ABORT: less than 40 GB of GPU memory is free`.**
Someone uses the GPU outside Slurm. Nothing ran; the chain was cancelled by afterok. Wait, then
`cluster/server.sh submit <job>` again (inputs stay staged). `check` shows `DependencyNeverSatisfied`
→ same: `scancel` is not needed; just resubmit.

**T9. Server `VALIDATE sanity-gpuws MISMATCH`.**
Same-GPU determinism is broken (driver or library update on the server). Stop: do not run further
server jobs. `cluster/server.sh fetch validate`, keep the numbers, and write a DEVIATIONS.md row.
Server results from before and after the change must not be compared with each other.

**T10. `code-error` (a Python exception in our code).**
The doctor prints the exception type and message. Without Claude Code:
1. If the message names a missing file of another experiment (`no finished dev runs`, checkpoint path):
   it is `missing-input`, not a bug — wait for the source job and `--requeue`.
2. Otherwise paste the doctor line, the 20 last log lines (`tail -n 20 <log>`; contains no item text)
   and the named source file into the web chat and ask for a patch.
3. Apply the patch on the right branch: harness files (`dsgx/...`) in `$P/baselines_DSG` (v2-harness),
   `git commit`, then `scripts/sync_harness.sh` (merges into idle exp worktrees and lists finished jobs
   that must re-run); experiment files (`experiments/<ID>/...`) in that exp worktree when none of its
   jobs is RUNNING, then `git commit`.
4. `python -m dsgx.queue.doctor --requeue <JOB>`.
5. If it cannot be fixed: leave it FAILED. The final report lists it under "failed, skipped and
   partial items" automatically; add one line to DEVIATIONS.md with the reason.

**Also common:** `fetch_results.sh: VERIFY FAILED` → run `cluster/server.sh fetch <job>` again
(partial copy); `REFUSING to stage` → `cluster/server.sh cleanup <previous job> --yes` first;
`REFUSING: one of our dsg jobs is already queued` → wait for it (`server.sh check`).

---

## W. Using the web chat after Oct 8

Paste only: `status --chat` output, `doctor` output, `wave_check` output, `FINAL_REPORT.md`,
`summary.json`, `COMBINE_SELECTION.md`, server `check` output, and source files. Never paste
anything from `dsg_private/`, `items.parquet` text columns, or generations.

## Command index

| need | command |
|---|---|
| status | `python -m dsgx.queue.status --chat` |
| diagnose / re-queue | `python -m dsgx.queue.doctor [--apply \| --requeue J ...]` |
| pause / resume | `python -m dsgx.queue.pause "why"` / `python -m dsgx.queue.resume` |
| wave review | `python -m dsgx.queue.wave_check <w>` |
| reboot | `scripts/reboot_recover.sh` |
| server | `cluster/server.sh {status,sync,plan,stage,submit,check,fetch,verify,cleanup,run} <job>` |
| combination wave | `python -m dsgx.combine {--dry-run,--enqueue}` |
| final report | `python -m dsgx.analysis.final_report [--interim]` |
| paper assets | `python -m dsgx.analysis.paper_assets` |
| paper numbers + PDF | `scripts/paper_update.sh [--no-pdf]` |
| qualitative | `python -m dsgx.analysis.qual.<q1_feature_cards,q3_never_learned,q5_geometry,q6_trajectory,annotate>` |
