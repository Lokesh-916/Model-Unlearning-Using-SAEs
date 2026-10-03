#!/usr/bin/env bash
# Check a rewrite: same branch set, identical trees at every tip, same commit counts, no AI trailers left,
# Amar's / Chakrish's trailers on exactly the selected commits. Usage: verify.sh <repo> <out>
set -euo pipefail
repo=$(cd "${1:?repo}" && pwd); out=${2:?out}; fail=0
while read -r ref old _; do
  new=$(git -C "$repo" rev-parse "$ref")
  [[ $(git -C "$repo" rev-parse "$old^{tree}") == $(git -C "$repo" rev-parse "$new^{tree}") ]] || { echo "TREE DIFFERS $ref"; fail=1; }
  [[ $(git -C "$repo" rev-list --count "$old") == $(git -C "$repo" rev-list --count "$new") ]] || { echo "COUNT DIFFERS $ref"; fail=1; }
done < "$out/tips_before.txt"
ai=$(git -C "$repo" log --branches --format=%B | grep -ciE 'co-authored-by:.*(claude|anthropic)|noreply@anthropic|generated with \[?claude code' || true)
amar=$(git -C "$repo" log --branches --format=%H --grep='^Co-authored-by: Amar060 <amarreddy200606@gmail.com>$' | sort -u | wc -l)
# expected = selected commits plus commits that already carried the trailer before the rewrite
had() { git -C "$repo" log $(awk '{print $2}' "$out/tips_before.txt") --format=%H --grep="^$1\$"; }
want=$(cat "$out/bake_commits.txt" <(had 'Co-authored-by: Amar060 <amarreddy200606@gmail.com>') | awk 'NR==FNR{s[$1];next} ($1 in s){print $2}' - "$out/commit-map.tsv" | sort -u | wc -l)
chak=$(git -C "$repo" log --branches --format=%H --grep='^Co-authored-by: Chakrish28 <chakrish.konchada1234@gmail.com>$' | sort -u | wc -l)
cwant=$(cat "$out/break_commits.txt" 2>/dev/null <(had 'Co-authored-by: Chakrish28 <chakrish.konchada1234@gmail.com>') | awk 'NR==FNR{s[$1];next} ($1 in s){print $2}' - "$out/commit-map.tsv" | sort -u | wc -l)
auth=$(git -C "$repo" log --branches --format='%an%x09%ae%x09%ad' | sort | md5sum)
oldauth=$(git -C "$repo" log $(awk '{print $2}' "$out/tips_before.txt") --format='%an%x09%ae%x09%ad' | sort | md5sum)
echo "AI trailers left: $ai; Amar trailers: $amar (expected $want); Chakrish trailers: $chak (expected $cwant); authors/dates identical: $([[ $auth == "$oldauth" ]] && echo yes || echo NO)"
[[ $ai == 0 && $amar == "$want" && $chak == "$cwant" && $auth == "$oldauth" && $fail == 0 ]] && echo "VERIFY OK" || { echo "VERIFY FAILED"; exit 1; }
