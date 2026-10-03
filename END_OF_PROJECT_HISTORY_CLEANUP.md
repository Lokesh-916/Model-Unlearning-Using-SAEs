# End-of-project history cleanup (main project repo)

**Do not run before every lab-PC and server run has finished.** Queue job files pin exact commit hashes
(`$DSG_RESULTS/queue/jobs/*.json` → `commit`; the worker refuses to run a job whose worktree HEAD differs),
and run records cite them. This rewrite changes every hash on every branch.

What it does, on all local branches (= the 54 branches on `origin`):
1. removes every AI trailer (`Co-Authored-By: Claude …`, `noreply@anthropic.com`, "Generated with Claude Code");
2. adds `Co-authored-by: Amar060 <amarreddy200606@gmail.com>` to commits in the bake areas (Part III open-weights
   erasure: D1 / d1-full / B1 distillation seed, D2 null-space, D3 audit, A6 / a6-full tampering). Selection =
   non-merge commits whose own diff touches the paths in `scripts/history_cleanup/select_bake_commits.sh`, plus
   the hashes in `scripts/history_cleanup/extra_bake_commits.txt` (review before applying);
3. keeps every tree, author, committer and date identical (checked by `verify.sh`);
4. writes `commit-map.tsv` (old → new) and updates the hashes recorded where they are used operationally.

Tested on a copy on 2026-10-03 (clone of all 55 branches, 500 commits, one linked worktree): dry run → 124 commits
with AI trailers, 14 bake commits; apply in 19 s; `VERIFY OK` (tips' trees and counts equal, 0 AI trailers,
14 Amar trailers, authors/dates identical); linked worktree clean afterwards; `remap_hashes.py` mapped 175/175
queue jobs and the short hashes in CLAUDE.md / PREP_PROGRESS.md.

Tools: `scripts/history_cleanup/{select_bake_commits.sh, msg_filter.py, rewrite.sh, verify.sh, remap_hashes.py}`.
`git filter-repo` is not installed; `rewrite.sh` uses `git filter-branch` with a message filter and a commit
filter that records the map.

## 0. Preconditions (all must hold)
```bash
P=~/projects/mechunlearn-project; R=$P/baselines_DSG          # repo with all branches (worktrees share it)
python -m dsgx.queue.status                                    # nothing queued / running / paused
cluster/server.sh status                                       # no dsg job queued on gpuws
tmux ls | grep dsg- && echo "stop the dsg-* sessions first (scripts/tmux_down.sh)"
git -C $R worktree list | awk '{print $1}' | while read w; do
  [ -z "$(git -C $w status --porcelain --untracked-files=no)" ] || echo "DIRTY $w"; done   # must print nothing
git -C $R fetch origin && for b in $(git -C $R for-each-ref --format='%(refname:short)' refs/heads); do
  [ "$(git -C $R rev-parse $b)" = "$(git -C $R rev-parse origin/$b 2>/dev/null)" ] || echo "NOT PUSHED $b"; done
df -h $P                                                       # > 60 GB free
```
Push or resolve every `NOT PUSHED` branch first, so local and `origin` agree before the rewrite.

## 1. Test on a copy (repeat right before the real run; the history will have grown)
```bash
H=$R/scripts/history_cleanup; T=$(mktemp -d ~/hc-test.XXXX)
git clone -q --no-local $R $T/repo && cd $T/repo
for b in $(git for-each-ref --format='%(refname:short)' refs/remotes/origin | grep -v HEAD); do git branch -q -f "${b#origin/}" "$b"; done
git remote remove origin
$H/rewrite.sh --dry-run $T/repo $T/out        # review the bake list; edit extra_bake_commits.txt; rerun
$H/rewrite.sh --apply   $T/repo $T/out
$H/verify.sh $T/repo $T/out                   # must end with: VERIFY OK
mkdir $T/q && cp $P/dsg_results/queue/jobs/*.json $T/q/
python3 $H/remap_hashes.py $T/out/commit-map.tsv --queue $T/q          # dry run, then add --apply and check
```
Stop if anything other than `VERIFY OK` appears. (The copy has an extra branch `origin` from `origin/HEAD`; harmless.)

## 2. Dry run on the real repo
```bash
cd $R && O=$P/history_cleanup_$(date +%F) && $H/rewrite.sh --dry-run $R $O
```
Must show the same counts as the copy.

## 3. Apply (creates backups first)
```bash
cd $R && $H/rewrite.sh --apply $R $O
$H/verify.sh $R $O                            # VERIFY OK
```
Backups made by `--apply`: tag `backup/pre-history-cleanup-<date>/<branch>` for every branch tip, and
`$O/pre-history-cleanup-<date>.bundle` (all refs; check with `git bundle verify`). Copy the bundle off the
machine too. Linked worktrees stay checked out on their branches and stay clean (same trees).

## 4. Update recorded hashes
```bash
python3 $H/remap_hashes.py $O/commit-map.tsv --queue $P/dsg_results/queue/jobs            # review
python3 $H/remap_hashes.py $O/commit-map.tsv --queue $P/dsg_results/queue/jobs --apply    # old value kept as commit_pre_rewrite
cp $O/commit-map.tsv $P/dsg_results/COMMIT_MAP_history_cleanup.tsv
cp $O/commit-map.tsv $P/dsg_results_cluster/COMMIT_MAP_history_cleanup.tsv
```
Run records (`runs/*/config.json` → `git.commit`, cluster sha256 listings) are **not** edited: they stay
verifiable, and the map translates them. Then the tracked docs, on each branch that carries them
(at least v2-harness / prep/later-runs; `git grep -lE '\b[0-9a-f]{7,40}\b' -- '*.md' '*.yaml'` lists candidates):
```bash
cd $R && python3 $H/remap_hashes.py $O/commit-map.tsv --text $(git grep -lE '\b[0-9a-f]{7,40}\b' -- '*.md' '*.yaml' '*.json')
# review, rerun with --apply, then a normal commit (no AI trailer):
git commit -am "Update recorded commit hashes after the history cleanup (map: COMMIT_MAP_history_cleanup.tsv)"
```
Also the paper (`paper/`), if it cites code commits: same `--text` call on `paper/sections/*.tex appendix/*.tex`.

## 5. Push
```bash
cd $R
git push origin 'refs/tags/backup/pre-history-cleanup-*'      # old history stays reachable on GitHub
while read ref old _; do b=${ref#refs/heads/}
  git push --force-with-lease=$b:$old origin $b; done < $O/tips_before.txt
git push origin $(git for-each-ref --format='%(refname:short)' refs/heads | grep -v -f <(awk '{sub("refs/heads/","",$1);print $1}' $O/tips_before.txt))  # new branches, if any
git ls-remote --heads origin | wc -l                           # = number of local branches
```
`--force-with-lease` with the recorded old tip refuses to overwrite a branch that changed on GitHub since step 0.

## 6. Afterwards
- Other clones (team members): `git fetch && git reset --hard origin/<branch>` (or a fresh clone); never merge
  old and new histories.
- Server copies have no `.git` (`CODE_COMMIT.json`); nothing to do if the server is cleaned up.
- Record the run in `$DSG_RESULTS/DEVIATIONS.md` (date, map file, counts from `verify.sh`).

## Rollback
```bash
for t in $(git -C $R tag -l "backup/pre-history-cleanup-<date>/*"); do
  git -C $R update-ref refs/heads/${t#backup/pre-history-cleanup-<date>/} $t; done
python3 - <<'PY'   # restore queue job commits
import json,glob,os
for f in glob.glob(os.path.expanduser('~/projects/mechunlearn-project/dsg_results/queue/jobs/*.json')):
    j=json.load(open(f))
    if 'commit_pre_rewrite' in j: j['commit']=j.pop('commit_pre_rewrite'); open(f,'w').write(json.dumps(j,indent=2)+'\n')
PY
```
then force-push the restored tips (`--force-with-lease`), as in step 5.
