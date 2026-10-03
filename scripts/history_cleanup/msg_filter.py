#!/usr/bin/env python3
"""git filter-branch --msg-filter: drop AI trailers, add Amar's trailer to selected commits.
Env: GIT_COMMIT (set by filter-branch), BAKE_LIST (file of full hashes)."""
import os, re, sys

AMAR = "Co-authored-by: Amar060 <amarreddy200606@gmail.com>"
AI = re.compile(r"^\s*(co-authored-by:.*(claude|anthropic|openai|copilot|chatgpt|gpt-)"
                r"|.*generated with \[?claude code|.*noreply@anthropic\.com)", re.I)

msg = sys.stdin.read()
lines = [l for l in msg.split("\n") if not AI.match(l)]
while lines and not lines[-1].strip():
    lines.pop()
bake = set(open(os.environ["BAKE_LIST"]).read().split())
if os.environ.get("GIT_COMMIT") in bake and not any(l.strip().lower() == AMAR.lower() for l in lines):
    last = lines[-1] if lines else ""
    is_trailer_block = bool(re.match(r"^[A-Za-z-]+: ", last)) and len(lines) > 1 and lines[-2].strip() == ""
    lines += ([AMAR] if is_trailer_block else ["", AMAR])
sys.stdout.write("\n".join(lines) + "\n")
