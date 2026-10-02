# CLAUDE.md — cluster/setup worktree (department GPU server)

Branch: `cluster/setup` (cut from `v2-harness`). Worktree: `~/projects/mechunlearn-project/cluster`.
Never disturb the lab PC queue (dsg_worktrees, dsg-* tmux sessions, $DSG_RESULTS). Never commit to main.

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
- **Gotchas:** any `--mem` is rejected (node RealMemory=1); always pass `--gres=gpu:1`; no CPU/mem confinement;
  `sacct` unavailable; run conda-unpack via the env's python; never use system python3 on the server.
- Lab PC tools: `cluster/sync_code.sh`, `cluster/stage_rmu.sh`, `cluster/slurm/fetch_results.sh <job>`.
