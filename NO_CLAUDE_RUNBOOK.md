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

### 5.2 Server results in the report
Server results land in `$P/dsg_results_cluster/` (same run-directory format). They are reported
in their own **gpuws** tables and never mixed with lab-PC numbers:
`python -m dsgx.analysis.final_report --hardware gpuws --runs $P/dsg_results_cluster/runs --out $P/dsg_results_cluster/report`.

---

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

---

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
RUNNING on CPU → normal; nothing obvious → `tmux ls` must list `dsg-queue`; if not,
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
| qualitative | `python -m dsgx.analysis.qual.<q1_feature_cards,q3_never_learned,q5_geometry,q6_trajectory,annotate>` |
