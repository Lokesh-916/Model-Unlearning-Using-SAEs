#!/usr/bin/env bash
# Rewrite commit messages on every local branch: remove AI trailers, add Amar's trailer to bake commits
# and Chakrish's trailer to break commits.
# Trees, authors and dates are unchanged. Writes <out>/commit-map.tsv (old new), <out>/bake_commits.txt and <out>/break_commits.txt.
# Usage: rewrite.sh --dry-run <repo> <out>   # lists what would change, rewrites nothing
#        rewrite.sh --apply   <repo> <out>   # backup tags + bundle, then rewrite
set -euo pipefail
mode=${1:?--dry-run|--apply}; repo=$(cd "${2:?repo}" && pwd); out=$(mkdir -p "${3:?out}" && cd "$3" && pwd)
here=$(cd "$(dirname "$0")" && pwd); stamp=$(date +%F)
"$here/select_bake_commits.sh" "$repo" > "$out/bake_commits.txt"
"$here/select_break_commits.sh" "$repo" > "$out/break_commits.txt"
ai_re='^\s*(co-authored-by:.*(claude|anthropic|openai|copilot|chatgpt|gpt-)|.*generated with \[?claude code|.*noreply@anthropic\.com)'
git -C "$repo" log --branches --format='%H%x09%B%x00' | python3 -c "
import re,sys
rx=re.compile(r'''$ai_re''',re.I|re.M); n=0
for rec in sys.stdin.read().split('\0'):
    if rec.strip() and rx.search(rec.split('\t',1)[1]): n+=1
print('commits with AI trailers:', n)"
echo "bake commits (Amar trailer): $(wc -l < "$out/bake_commits.txt")  -> $out/bake_commits.txt"
git -C "$repo" log --no-walk=unsorted --format='  %h %ad %s' --date=short $(cat "$out/bake_commits.txt") | cut -c1-140
echo "break commits (Chakrish trailer): $(wc -l < "$out/break_commits.txt")  -> $out/break_commits.txt"
git -C "$repo" log --no-walk=unsorted --format='  %h %ad %s' --date=short $(cat "$out/break_commits.txt") | cut -c1-140
echo "branches: $(git -C "$repo" for-each-ref refs/heads | wc -l); commits: $(git -C "$repo" rev-list --branches | wc -l)"
[[ $mode == --dry-run ]] && { echo "DRY RUN: nothing rewritten"; exit 0; }
[[ $mode == --apply ]] || { echo "unknown mode $mode"; exit 2; }

# backups: one tag per branch tip + a bundle of everything
git -C "$repo" for-each-ref --format='%(refname:short) %(objectname)' refs/heads | while read -r b h; do
  git -C "$repo" tag -f "backup/pre-history-cleanup-$stamp/$b" "$h" >/dev/null
done
git -C "$repo" bundle create "$out/pre-history-cleanup-$stamp.bundle" --all
git -C "$repo" for-each-ref --format='%(refname) %(objectname) %(objectname)^{tree}' refs/heads > "$out/tips_before.txt"

: > "$out/commit-map.tsv"
export BAKE_LIST="$out/bake_commits.txt" BREAK_LIST="$out/break_commits.txt" MAP="$out/commit-map.tsv"
cd "$repo"
FILTER_BRANCH_SQUELCH_WARNING=1 git filter-branch -f \
  --msg-filter "python3 '$here/msg_filter.py'" \
  --commit-filter 'n=$(git commit-tree "$@"); echo "$GIT_COMMIT $n" >> "$MAP"; echo "$n"' \
  -- --branches
git for-each-ref --format='%(refname)' refs/original | xargs -r -n1 git update-ref -d
echo "rewritten: $(wc -l < "$MAP") commits; map: $MAP"
