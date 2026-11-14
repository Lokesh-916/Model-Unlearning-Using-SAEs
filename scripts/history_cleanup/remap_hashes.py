#!/usr/bin/env python3
"""Replace pre-rewrite commit hashes by their rewritten ones, using commit-map.tsv (old new).

Two kinds of targets (dry run by default; --apply writes):
  --queue DIR  queue job files ($DSG_RESULTS/queue/jobs/*.json): the "commit" field (full hash) is replaced,
               the old value kept as "commit_pre_rewrite". Needed so the worker's HEAD == job.commit check
               still holds if a job is ever re-run.
  --text FILE.. tracked text files (md/yaml/json/tex/txt/sh/py): full hashes and unique short prefixes
               (>= 7 hex, word-bounded) are replaced by the new hash at the same length.
Run records (runs/*/config.json, sha256-verified cluster listings) are NOT edited: put commit-map.tsv next
to them instead (see END_OF_PROJECT_HISTORY_CLEANUP.md).
"""
import argparse, json, re, sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("map")
ap.add_argument("--queue", type=Path)
ap.add_argument("--text", nargs="*", type=Path, default=[])
ap.add_argument("--apply", action="store_true")
a = ap.parse_args()

m = dict(l.split() for l in open(a.map) if l.strip())
m = {o: n for o, n in m.items() if o != n}
byprefix = {}
for o in m:
    for k in range(7, 41):
        byprefix.setdefault(o[:k], set()).add(o)

def new_for(tok):
    hits = byprefix.get(tok.lower())
    if not hits or len(hits) != 1:
        return None
    return m[next(iter(hits))][: len(tok)]

changed = 0
if a.queue:
    for f in sorted(a.queue.glob("*.json")):
        job = json.loads(f.read_text())
        c = job.get("commit")
        if c in m:
            changed += 1
            print(f"queue {f.name}: {c[:10]} -> {m[c][:10]}")
            if a.apply:
                job["commit_pre_rewrite"], job["commit"] = c, m[c]
                f.write_text(json.dumps(job, indent=2) + "\n")

HEX = re.compile(r"(?<![0-9A-Za-z])[0-9a-f]{7,40}(?![0-9A-Za-z])")
for f in a.text:
    s = f.read_text()
    reps = []
    def sub(mt):
        n = new_for(mt.group(0))
        if n:
            reps.append((mt.group(0), n))
            return n
        return mt.group(0)
    t = HEX.sub(sub, s)
    if reps:
        changed += 1
        print(f"text {f}: {len(reps)} replacements " + ", ".join(f"{o}->{n}" for o, n in reps[:6]) + (" ..." if len(reps) > 6 else ""))
        if a.apply:
            f.write_text(t)
print(f"{'APPLIED' if a.apply else 'DRY RUN'}: {changed} files")
