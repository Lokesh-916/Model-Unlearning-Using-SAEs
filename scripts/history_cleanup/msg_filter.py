#!/usr/bin/env python3
"""git filter-branch --msg-filter: drop AI trailers, add the team-member trailers to selected commits.
Env: GIT_COMMIT (set by filter-branch), BAKE_LIST (Amar: bake areas), BREAK_LIST (Chakrish: break areas);
each a file of full hashes (BREAK_LIST optional)."""
import os, re, sys

AMAR = "Co-authored-by: Amar060 <amarreddy200606@gmail.com>"
CHAKRISH = "Co-authored-by: Chakrish28 <chakrish.konchada1234@gmail.com>"
AI = re.compile(r"^\s*(co-authored-by:.*(claude|anthropic|openai|copilot|chatgpt|gpt-)"
                r"|.*generated with \[?claude code|.*noreply@anthropic\.com)", re.I)


def listed(var):
    path = os.environ.get(var)
    return os.environ.get("GIT_COMMIT") in set(open(path).read().split()) if path and os.path.exists(path) else False


msg = sys.stdin.read()
lines = [l for l in msg.split("\n") if not AI.match(l)]
while lines and not lines[-1].strip():
    lines.pop()
for trailer, var in ((AMAR, "BAKE_LIST"), (CHAKRISH, "BREAK_LIST")):
    if listed(var) and not any(l.strip().lower() == trailer.lower() for l in lines):
        last = lines[-1] if lines else ""
        is_trailer_block = bool(re.match(r"^[A-Za-z-]+: ", last)) and len(lines) > 1 and lines[-2].strip() == ""
        lines += ([trailer] if is_trailer_block else ["", trailer])
sys.stdout.write("\n".join(lines) + "\n")
