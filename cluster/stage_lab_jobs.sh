#!/bin/bash
# Run on the LAB PC. Stage MOVED-TO-SERVER lab-queue jobs for cluster/lab_jobs.py on gpuws:
#   1. read their specs from $DSG_RESULTS/queue/jobs (read-only; never writes the lab queue);
#   2. `git archive` their pinned commit (all jobs of a group share one) into gpuws:dsg_cluster/code-<exp>-<commit7>/
#      with CODE_COMMIT.json (the exp worktree is not touched);
#   3. copy the specs (worktree -> snapshot dir, commit -> null, pinned_commit kept) to
#      gpuws:dsg_cluster/results/queue/jobs/ and write labjobs/<group>/{ORDER,SNAPSHOT}.
#   cluster/stage_lab_jobs.sh <group> '<glob>' ['<glob>' ...]      e.g. stage_lab_jobs.sh c3 'C3-*'
# Order = the lab scheduler's (priority, wave, group key, created, id), then dependencies first.
# LABJOBS_PIN=<commit>: run every job of the group at this commit although the lab pinned some to its ancestors (X1:
#   aa71469 for 4 jobs, 93625f5 = aa71469 + the item-level-resume merge for the rest; per-item outputs identical). Refused
#   unless every lab commit is an ancestor of it. The lab commit is kept in moved_to_server.lab_commit.
# LABJOBS_REPLICA=1: the lab keeps running these jobs too (a replication on gpuws, not a move); recorded in the specs.
set -euo pipefail
GROUP="${1:?usage: stage_lab_jobs.sh <group> <glob>...}"; shift
[ $# -ge 1 ] || { echo "usage: stage_lab_jobs.sh <group> <glob>..."; exit 2; }
P="${DSG_PROJECT:-$HOME/projects/mechunlearn-project}"
Q="${DSG_RESULTS:-$P/dsg_results}/queue/jobs"
REPO="$P/baselines_DSG"
PY="${DSGX_PY:-$HOME/miniconda3/envs/mechunlearn2/bin/python}"
TMP="$(mktemp -d "${TMPDIR:-/tmp}/labjobs-$GROUP.XXXX")"; trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/specs" "$TMP/code"

PIN="${LABJOBS_PIN:-}"
[ -z "$PIN" ] || PIN=$(git -C "$REPO" rev-parse --verify "$PIN^{commit}")
"$PY" - "$Q" "$TMP" "$GROUP" "$PIN" "${LABJOBS_REPLICA:-0}" "$REPO" "$@" <<'PY'
import fnmatch, json, subprocess, sys
from pathlib import Path
q, tmp, group, pin, replica, repo, globs = (Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4],
                                            sys.argv[5] == "1", sys.argv[6], sys.argv[7:])
specs = {p.stem: json.loads(p.read_text()) for p in q.glob("*.json") if any(fnmatch.fnmatchcase(p.stem, g) for g in globs)}
if not specs:
    sys.exit(f"no job matches {globs}")
commits = {s["commit"] for s in specs.values()}
if pin:
    bad = [c for c in commits if c is None or subprocess.run(["git", "-C", repo, "merge-base", "--is-ancestor", c, pin]).returncode]
    if bad:
        sys.exit(f"LABJOBS_PIN {pin[:10]} does not contain the lab commits {bad}")
    commits = {pin}
if len(commits) != 1 or None in commits:
    sys.exit(f"jobs of one group must share one pinned commit, got {commits}")
commit = commits.pop()
rank = {"canary": -1, "must": 0, "should": 1, "stretch": 2, "backlog": 3}
key = lambda j: (rank.get(specs[j].get("priority", "must"), 9), specs[j].get("wave", 0), specs[j].get("group_key", ""), specs[j].get("created", 0), j)
order, seen = [], set()
def visit(j):
    if j in seen:
        return
    seen.add(j)
    for d in sorted(x for x in specs[j].get("deps", []) if x in specs):
        visit(d)
    order.append(j)
for j in sorted(specs, key=key):
    visit(j)
snap = f"{specs[order[0]]['exp_id']}-{commit[:7]}"
for j, s in specs.items():
    s = dict(s, pinned_commit=commit, commit=None, worktree=f"/home/suraj/dsg_cluster/code-{snap}",
             moved_to_server={"group": group, "lab_worktree": s.get("worktree"), "lab_commit": s.get("commit"),
                              "replica": replica})
    (tmp / "specs" / f"{j}.json").write_text(json.dumps(s, indent=1))
(tmp / "ORDER").write_text("\n".join(order) + "\n")
(tmp / "SNAPSHOT").write_text(snap + "\n")
(tmp / "COMMIT").write_text(commit + "\n")
print(f"{len(order)} jobs, commit {commit[:10]}, snapshot code-{snap}")
PY

commit=$(cat "$TMP/COMMIT"); snap=$(cat "$TMP/SNAPSHOT")
git -C "$REPO" archive "$commit" | tar -x -C "$TMP/code" --exclude figures
printf '{"commit": "%s", "branch": "(git archive)", "dirty": false, "synced": "%s", "snapshot": "code-%s"}\n' \
    "$commit" "$(date -Is)" "$snap" > "$TMP/code/CODE_COMMIT.json"
ssh -n -o BatchMode=yes gpuws "printf -- '- %s | %s | %s\n' \"\$(date '+%F %T')\" 'rsync code-$snap/ + $(wc -l < "$TMP/ORDER") job specs + labjobs/$GROUP/' 'stage_lab_jobs.sh $GROUP (moved lab jobs)' >> ~/dsg_cluster/COMMAND_LOG.md"
ssh -n gpuws "mkdir -p ~/dsg_cluster/results/queue/jobs ~/dsg_cluster/labjobs/$GROUP"
rsync --bwlimit=50000 --partial -a --delete "$TMP/code/" "gpuws:dsg_cluster/code-$snap/" < /dev/null
rsync --bwlimit=50000 --partial -a "$TMP/specs/" gpuws:dsg_cluster/results/queue/jobs/ < /dev/null
rsync --bwlimit=50000 --partial -a "$TMP/ORDER" "$TMP/SNAPSHOT" "gpuws:dsg_cluster/labjobs/$GROUP/" < /dev/null
ssh -n gpuws "source ~/dsg_cluster/env.sh && cd ~/dsg_cluster/code-later 2>/dev/null || cd ~/dsg_cluster/code; CUDA_VISIBLE_DEVICES= python cluster/lab_jobs.py --group $GROUP --plan | tail -40"
echo "staged group $GROUP -> code-$snap"
