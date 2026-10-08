# NO_CLAUDE_RUNBOOK: finishing the DSG project without Claude Code

Written 2026-10-02 on branch `prep/later-runs`. Every remaining operation as copy-paste commands.
Each block ends with **You should see:** a line that tells you it worked. If you see something
else, go to section T (troubleshooting). Nothing here needs judgment: where a decision was needed,
a script makes it (`doctor`, `wave_check`, `combine`, `server.sh plan`).

**Safety rule (always):** never open, print or paste WMDP questions, generations, attack prompts or
anything under `dsg_private/` or `~/dsg_cluster/private`. Share only status files, ids, hashes and
metrics. All scripts here print ids and metrics only.

---

## H. State at handover (2026-10-08, last Claude Code session: read this first)

Everything from here on is done by the team with this runbook; no step needs Claude Code.

| area | state on 2026-10-08 (session 32) | what to do |
|---|---|---|
| lab PC queue | 100 % done (176 jobs + 71 moved to gpuws) | nothing (section 1 only if you add jobs) |
| gpuws | 205 ph-union RUNNING, 206–207 pending behind it (post hoc, exploratory); ours 35 GB, `/` 63 GB free | nothing by hand: the watcher `cluster/after_ph_union.sh` (pid alive, log `$P/dsg_results_cluster/after_ph_union.log`) fetches, verifies, cleans, regenerates the digest + both papers, commits numbers.tex locally (section 5.11; restart after a reboot) |
| claims C-H1..C-H7 | final per machine (table in CLAUDE.md, session 31); rules fixed in `dsgx/analysis/claims.py` | never by hand; a change needs a DEVIATIONS row |
| TMLR paper (`$P/paper`, remote `origin` draft) | pushed up to session 31; PH-union numbers fill in when the watcher finishes | section 7.1 steps 2–3 (`scripts/paper_update.sh --final`, commit, `git push origin draft`) |
| SRW paper (`$P/paper-srw`) | BUILD OK, 8 pages, long paper (content ends p.5), anonymous; `SUBMISSION_CHECKLIST.md` | 6 Nov mentorship draft, 15 Dec submission (section 7.1 step 6 + the checklist) |
| backups | paper-srw, presentation, release have **no remote** (gh CLI not installed on the lab PC; release leak scan: SCAN OK on 2026-10-08) | `scripts/push_backups.sh --print` → create the 3 PRIVATE repos on github.com, run the printed commands (or install gh, `gh auth login`, `scripts/push_backups.sh`) |
| D1 v3 (follow-up, prepared, NOT submitted) | `cluster/d1_v3.py`, conf `d1-v3`, dry run `scripts/d1_v3_dryrun.sh` (PLAN OK); needs ≥ 82 GB free on gpuws `/` (63 on 2026-10-08) | README.md "How to run D1 v3 on gpuws without Claude"; section 5.12 |
| DSG-author email, arXiv, history cleanup, final cleanup | not done | section 7.1 steps 4, 5, 7, 8 (in that order; `cleanup_all.sh` last) |

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
   **Automatic since session 24:** `cluster/after_muse_submit.sh` (lab PC, nohup; log `dsg_results_cluster/after_muse_watch.log`)
   does all of the above once MUSE has finished (it resubmits MUSE up to twice if it left the queue incomplete), after
   fetch/verify/cleanup of x1, a7-scaled and muse, and then starts `q2_rerun_when_ready.sh`. Snapshot later10 is already staged.
   **Restart it after a reboot** if its log has no `MT-Bench (cusum) submitted` line:
   `nohup cluster/after_muse_submit.sh >> $P/dsg_results_cluster/after_muse_watch.log 2>&1 &`.
   You still do by hand: `fetch/verify/cleanup mtbench-x1` when it is done.
   **You should see** in `dsg_results_cluster/jobs/mtbench/`: `answers_cusum.jsonl` (160 rows) and
   `judgments_cusum__gemma-2-9b-it.jsonl`; base/dsg/window rows unchanged (skipped as done).
3. Then `scripts/paper_update.sh` and `paper/scripts/fix_framing_check.sh` (lists the macros still [pending]); choose the
   version in `paper/FIX_FRAMING.tex` by `\resLabVerdictCHFive` and apply it by hand.

### 5.10 Session 27 (2026-10-07): X1-suite replica on gpuws (StreamGuard vs DSG, TOFU-full v3 model)
Chain (nice 0): 197 validate (EXACT) → 198 `x1suite-models` (retain-only TOFU reference re-trained on gpuws, ~25 min;
links `ckpt:A2/{tofu_full,tofu_retain}` to the gpuws TOFU-full v3 models) → 199–201 `x1-suite` (the lab's six X1-suite
jobs at b933e20 through `lab_jobs.py --group x1-suite`, order: tofu-metrics, tofu-qa-forget, tofu-qa-retain,
benign-open, leak, paired). When all have left the queue:

    cluster/server.sh check x1-suite          # you should see "group x1-suite COMPLETE" in the last log
    cluster/server.sh fetch x1-suite && cluster/server.sh verify x1-suite
    cluster/server.sh cleanup x1-suite --yes   # retain model, ckpt links, MiniLM, code-X1-suite-*
    cluster/server.sh cleanup x1 --yes         # x1 inputs (wmdp-corpora, translations, B4/B5), no longer needed
    python -m dsgx.analysis.results_digest && scripts/paper_update.sh

If 201 ends `INCOMPLETE`, resubmit: `ssh gpuws 'cd ~/dsg_cluster/slurm && ./submit.sh x1-suite.sbatch'` (DONE jobs are
skipped). If 198 exits 3 (retain fine-tune unfinished), resubmit `x1suite-models.sbatch` then the x1-suite sbatch afterok.
Then the Q2 re-run (watcher refused at 05:44 for disk; there is room once x1-suite is cleaned): `cluster/server.sh run q2-graphs`
with `q2-graphs-v2.sbatch` (runbook 5.6).

### 5.11 Sessions 28–29 (2026-10-07): POST-HOC PH-union / PH-tofucal on gpuws (202–207)
Exploratory, decided after the X1 TEST results, never a claim input (DEVIATIONS 2026-10-07). Chain (nice 0):
202 validate (EXACT) → 203 x1suite-models → **204 ph-union (running, 3 h limit) → 205–207 ph-union, HELD by us**
(`JobHeldUser`: another user needs the GPU). Each ph-union step is resumable: DONE tasks are skipped.

State at session 29 (17:45): PH-tofucal complete (4/4 incl. paired); PH-union TOFU metrics, TOFU QA and benign-open done,
leak running in 204; the MCQ grid (120 runs) and PH-union-paired need 205–207. The finished parts were copied to the lab PC
as an interim copy (sha256-verified, no `.fetched` marker) and are in both papers.
Finding: on TOFU the union never fires (its conformal threshold is the 1e9 sentinel, because DSG's TOFU gate, tau 0.045,
alone fires on > 5 % of MMLU dev); the paper says so. This is not a job failure: do not re-run it.

**Release only when the other user says the GPU is free** (never before; log it):

    ssh gpuws 'scontrol release 205 206 207 && printf -- "- %s | scontrol release 205 206 207 | GPU free again\n" "$(date "+%F %T")" >> ~/dsg_cluster/COMMAND_LOG.md'

You should see: no output; `cluster/server.sh check ph-union` then shows 205 pending/running. ~1–3 h per step.

**Nothing else is manual.** `cluster/after_ph_union.sh` (nohup on the lab PC, log
`$P/dsg_results_cluster/after_ph_union.log`) waits while any ph-union job is queued (held jobs count, so it just waits),
then: fetch + verify + cleanup ph-union, results digest, `scripts/paper_update.sh`, local commits of numbers.tex (paper) and
numbers.tex + main.pdf (paper-srw). Check it is alive: `pgrep -af after_ph_union` (after a reboot:
`nohup cluster/after_ph_union.sh >> $P/dsg_results_cluster/after_ph_union.log 2>&1 &`).
You should see at the end of its log: `ph-union complete: fetch / verify / cleanup` … `done`.

Manual equivalent (if the watcher is not running):

    cluster/server.sh check ph-union                   # "group ph-union COMPLETE"
    cluster/server.sh fetch ph-union && cluster/server.sh verify ph-union && cluster/server.sh cleanup ph-union --yes
    python -m dsgx.analysis.results_digest && scripts/paper_update.sh

**If the held jobs are abandoned** (the GPU stays busy): FIRST stop the watcher, else it resubmits ph-union.sbatch as soon as
the queue is empty: `pkill -f after_ph_union.sh`; then `ssh gpuws 'scancel 205 206 207'` (log it), then the manual
fetch / verify / cleanup above. The paper then keeps "still running" for the union's MCQ part: replace that sentence in
`paper/sections/11_results_fix.tex` (`\ifresdone{\resGpuPhUnionStatus}` … second branch) by "were not run" and do the same
in `paper-srw/main.tex`.

### 5.12 D1 v3 (prepared 2026-10-08, NOT submitted): distillation from the forget corpus, 3 seeds, open-ended, full + LoRA relearning
Full instructions with "you should see" lines: **README.md, section "How to run D1 v3 on gpuws without Claude"**. Short form:

    scripts/d1_v3_dryrun.sh            # submits nothing: tiny tests, real-config CPU plan (PLAN OK), server disk plan, dry chain
    cluster/server.sh run d1-v3        # plan + confirm + stage + validate -> 12 chained 3 h jobs (~21 h); needs >= 82 GB free on /
    cluster/server.sh check d1-v3      # progress
    cluster/server.sh fetch d1-v3 && cluster/server.sh verify d1-v3 && cluster/server.sh cleanup d1-v3 --yes

Results: `$P/dsg_results_cluster/jobs/d1-v3/SUMMARY.md` (gpuws only; follow-up, not a claim input without a DEVIATIONS row).

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

### 7.1 Finishing after Oct 8 (for the team; in this order; every step is copy-paste)
State on 2026-10-08 (session 31): the lab queue is 100 % done (176 jobs incl. B7 and N8; 71 moved to gpuws and run there).
Claims, digest, both final reports, both papers and the deck were regenerated that morning. The only open runs are
gpuws **205–207** (PH-union, post hoc, exploratory; released 2026-10-08, expected to finish early afternoon). Everything
else below is reporting, communication and cleanup. Start every terminal with section 0.

1. **gpuws 205–207 (PH-union) → numbers.** Nothing to do by hand while the watcher runs:

       pgrep -af after_ph_union.sh                         # alive? (after a reboot: section 5.11 restarts it)
       tail -5 $P/dsg_results_cluster/after_ph_union.log   # progress
       cluster/server.sh check ph-union                    # queue state of 205-207

   **You should see** at the end of the log: `ph-union complete: fetch / verify / cleanup` … `paper: numbers.tex committed
   (not pushed)` … `paper-srw: committed` … `done`. The watcher has then fetched, verified and cleaned ph-union on the
   server, regenerated the digest and run `scripts/paper_update.sh` (both papers) and committed numbers.tex locally.
   If a job fails it resubmits up to its retry limit; if the log ends with `stopping`, read the job log
   (`cluster/server.sh check ph-union`) and use the manual path in section 5.11.
   Manual equivalent (watcher not running, group COMPLETE):

       cluster/server.sh fetch ph-union && cluster/server.sh verify ph-union && cluster/server.sh cleanup ph-union --yes

2. **Final numbers, both papers, reports, deck** (also fine to re-run any time; files are overwritten):

       python -m dsgx.analysis.results_digest
       scripts/paper_update.sh --final
       python -m dsgx.analysis.final_report
       python -m dsgx.analysis.final_report --hardware gpuws --runs $P/dsg_results_cluster/runs --out $P/dsg_results_cluster/final_report
       python -m dsgx.analysis.paper_assets
       (cd $P/presentation && ./make_deck.sh)

   **You should see:** `wrote .../RESULTS_DIGEST.md`; `LaTeX warnings: 0`, `built .../main.pdf (46 pages)`,
   `BUILD OK: main.pdf` (SRW); `GpuPhUnionStatus done`, `GpuPhUnionPairedStatus done`; `open: 0 [Interim] marks`;
   the remaining `pending` numbers are only `LabBFour*`, `LabBFive*`, `*FixVsDsgForget`, `*FixVsDsgUtil`, `LabFixAxesWon` (not used in
   either paper; the deck uses the C-H5 rule macros instead); two `claims:` lines that match the table in CLAUDE.md
   (session 31); `final_review.pdf: 28 slides` with `pending on slides: LabBFiveMax LabBFourMax` only.
   The claims never change by hand: `dsgx/analysis/claims.py` is fixed (a change needs a DEVIATIONS row).

3. **Commit and push** (CLAUDE.md commit rules: the author's name only, never an AI trailer; Amar060 / Chakrish28
   trailers only on commits in their areas; numbers-only commits carry none):

       git -C $P/paper status --short            # only numbers.tex / main.pdf-related changes expected
       git -C $P/paper add numbers.tex sections && git -C $P/paper commit -m "Final numbers (PH-union in)" && git -C $P/paper push origin draft
       git -C $P/paper-srw add numbers.tex main.tex main.pdf && git -C $P/paper-srw commit -m "Final numbers"        # no remote
       git -C $P/presentation add -A && git -C $P/presentation commit -m "Deck: final numbers"                        # no remote

   "nothing to commit" is fine (the watcher may have committed already). paper-srw, presentation and release have no
   remote: back them up first as PRIVATE GitHub repos (`scripts/push_backups.sh --print`, section H) or copy them off
   this PC (USB / Drive) before the cleanup in step 8.

4. **Email the DSG authors (before ANY public posting: arXiv, release, talk slides online).** The draft is
   `paper/DSG_AUTHORS_EMAIL.md` (to Aashiq Muhamed, CC Dr. M. Naresh Babu). Lokesh sends it from his own address:
   fill in the CC address, re-check the quoted numbers against `paper/numbers.tex` after step 2, attach the TMLR PDF
   only if the team agrees which version to share. Never attach or paste WMDP items, attack prompts or generations.
   Record the date sent (and any reply) in `paper/REVIEW_NOTES.md` and commit. Give them time to answer before step 5;
   the team decides how long.

5. **arXiv (only after step 4, when the team decides to post):**

       cd $P/paper && scripts/arxiv_build.sh

   **You should see** every check `ok` (citations were verified in session 30, no `TODO-VERIFY` left; authors in
   `paper/arxiv/authors.tex` are `STATUS: CONFIRMED`: K. Lokesh Babu, K. Chakreesh, S. Amarnath Reddy, M. Naresh Babu)
   and `build-arxiv/arxiv-<date>.tar.gz`. The script never uploads: Lokesh uploads the tarball by hand on arxiv.org and
   checks the arXiv-generated PDF page by page. The public code release (`$P/release`, `python scan_release.py
   --private-check ...` must say SCAN OK; LICENSE is still a placeholder: choose one first) also waits for step 4.

6. **EACL 2027 SRW** (`$P/paper-srw`, anonymous ACL template; dates and format in its README):
   - **Fri 6 Nov 2026**: pre-submission mentorship deadline (OpenReview `EACL/2027/SRW_Pre-submission_Mentorship`).
   - 5 Dec 2026: mentorship feedback; apply it in `paper-srw/main.tex` (numbers only through `numbers.tex` macros).
   - **Tue 15 Dec 2026**: direct submission deadline (OpenReview `EACL/2027/SRW`). All deadlines 11:59 pm UTC-12.

       cd $P/paper-srw && ./build.sh

   **You should see** `BUILD OK: main.pdf` with the page, anonymity and font checks passing. Before submitting: first
   author must be a student; Limitations section present (mandatory); any repository link anonymised; the Responsible
   NLP checklist is filled in on OpenReview. Re-run `scripts/paper_update.sh` first if any number changed.

7. **History cleanup of the main repo** (only now: all lab and server runs are done, so no queue job needs its pinned
   hash any more). Follow `END_OF_PROJECT_HISTORY_CLEANUP.md` exactly, steps 0 → 6 (test on a copy first, `VERIFY OK`
   required at every stage, backup tags + bundle, remap recorded hashes, push with `--force-with-lease`).
   Known before step 0: local `main` is 1 commit ahead of `origin/main` (c0535b4 "Add master plan", 2026-10-01, Lokesh):
   push it or drop it first (never commit anything else to main). All other branches were pushed in session 31
   (incl. `exp/PH-X1-conformal`). The paper repo history is already clean (step 5b only if new AI trailers appear:
   `git -C $P/paper log --format=%B | grep -ci claude` must print 0).

8. **Cleanup (last; it removes our access to gpuws).** Dry run first, read every line, then act:

       scripts/cleanup_all.sh            # dry run: checks + what it would do; changes nothing
       scripts/cleanup_all.sh --yes      # acts only if every check is ok

   **You should see** in the dry run `== checks: 0 failure(s)` (warnings for paper-srw / presentation / release having no
   remote are expected). Checks: lab queue idle, no dsg-* job on gpuws, every server result file on the lab PC, every
   branch of the main and paper repos pushed, no uncommitted tracked changes. With `--yes` it stops the watchers,
   copies the server COMMAND_LOG.md to `dsg_results_cluster/server_logs/`, deletes `~/dsg_cluster` on gpuws, removes our
   key line from gpuws `~/.ssh/authorized_keys` (last), stops the dsg-* tmux sessions and removes our crontab lines
   (other lines are kept). Large lab caches (dsg_cache, HF models, conda env, dsg_private) are only **listed** with
   their sizes and delete commands; nothing under `dsg_results*` or the git repos is ever deleted. Tell the friend whose
   gpuws account we used that we are done.

9. Optional, only before step 8: Q2 re-run (needs > 61 GB free on the lab PC): `cluster/server.sh stage q2-graphs &&
   cluster/submit_chain.sh --after-any <last job> --nice 0 q2-graphs-v2.sbatch`, then runbook 5.6.

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
| paper numbers + PDF | `scripts/paper_update.sh [--no-pdf] [--final]` (finish: section 7.1) |
| end-of-project cleanup (gpuws + lab; dry run default) | `scripts/cleanup_all.sh [--yes] [--skip-server] [--skip-lab]` (section 7.1 step 8) |
| D1 v3 (prepared follow-up) | `scripts/d1_v3_dryrun.sh`, `cluster/server.sh run d1-v3` (section 5.12, README.md) |
| private backups of paper-srw / presentation / release | `scripts/push_backups.sh [--print]` (section H) |
| SRW submission list | `$P/paper-srw/SUBMISSION_CHECKLIST.md` |
| qualitative | `python -m dsgx.analysis.qual.<q1_feature_cards,q3_never_learned,q5_geometry,q6_trajectory,annotate>` |
