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
_Not yet filled — first session in progress (2026-10-02)._
