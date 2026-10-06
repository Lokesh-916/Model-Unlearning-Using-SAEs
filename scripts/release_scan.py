"""Release scan: exit 1 if the release contains anything that must not be published.

    python scan_release.py                  # structural + text checks (anyone can run these)
    python scan_release.py --private-check  # also compare every text file against the private material

Structural: no data/weight/cache file types (jsonl, parquet, npz, pkl, pt, safetensors, ...), no file or folder named
like private material (private, generations, suffix, translations, rewrites, forget-corpus, checkpoints), no file > 5 MB.
Text: no absolute home paths, local account, server host/account/scheduler details, e-mail or IP addresses, or author
names/handles (the release stays anonymous until de-anonymisation is decided).
Private check (needs the private material locally; nothing is printed except file names and counts): every 10-word
shingle of every release text file is compared with WMDP-Bio/Cyber questions (HF cache), the forget corpora
(--forget-corpus, repeatable) and every text-bearing file under --private-root (repeatable; default $DSG_PRIVATE), which
holds generations, transformed prompts, translations, rewrites and optimised suffixes; strings too short for a shingle
(e.g. suffixes) are matched as whole normalised strings.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SELF = Path(__file__).resolve()
K = 10
MAX_BYTES = 5 * 1024 * 1024
BAD_SUFFIX = {".jsonl", ".parquet", ".npz", ".npy", ".pkl", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".arrow",
              ".gguf", ".h5", ".zip", ".tar", ".gz"}
BAD_NAME = re.compile(r"(^|[/_\-.])(private|generations?|suffix(es)?|translations?|rewrites?|forget[-_]?corpus|"
                      r"checkpoints?|weights)([/_\-.]|$)", re.I)
# A path segment like "dsgx/attacks/transforms.py" is code that *makes* transformations; it is allowed. Names that
# denote stored material (e.g. "b5_suffixes.json", "translations/") are not.
ALLOWED_NAMES = {"transforms.py", "translate.py", "translate_cache.py", "rewrite.py", "suffix_search.py"}
TEXT_SUFFIX = {".py", ".md", ".txt", ".yaml", ".yml", ".json", ".tex", ".csv", ".toml", ".cfg", ".sh", ".ini", ""}
FORBIDDEN = [
    (r"/home/", "absolute home path"),
    (r"amaloch", "local account"),
    (r"mechunlearn-project", "local project folder"),
    (r"suraj|iiitdmk|sdi200f0a", "server account / host"),
    (r"dsg_cluster|\bsbatch\b|\bsqueue\b|\bscontrol\b|ssh\s+\w", "server / scheduler detail"),
    (r"lokesh|chakr(ee|i)sh|amarnath|amar060|amarreddy|naresh|kurnool|iiit", "author / affiliation (anonymous)"),
    (r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "e-mail address"),
    (r"(?<![=\w.])(?:\d{1,3}\.){3}\d{1,3}(?![\w.])", "IP address"),   # not a pinned version (pkg==1.2.3.4)
]
EMAIL_OK = re.compile(r"noreply@|example\.(com|org)|@users\.noreply")
IP_OK = re.compile(r"^(0\.0\.0\.0|127\.0\.0\.1)$")


def release_files():
    for p in sorted(ROOT.rglob("*")):
        if p.is_file() and ".git" not in p.relative_to(ROOT).parts:
            yield p


def is_text(p):
    return p.suffix.lower() in TEXT_SUFFIX


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def shingles(ws):
    return {hash(" ".join(ws[i:i + K])) for i in range(len(ws) - K + 1)}


def strings_in(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from strings_in(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from strings_in(v)


def private_texts(path):
    """Yield the strings of one private file (json/jsonl/text/parquet). Never printed."""
    suf = path.suffix.lower()
    try:
        if suf == ".jsonl":
            with path.open(errors="ignore") as f:
                for line in f:
                    try:
                        yield from strings_in(json.loads(line))
                    except ValueError:
                        yield line
        elif suf == ".json":
            yield from strings_in(json.loads(path.read_text(errors="ignore")))
        elif suf == ".parquet":
            import pyarrow.parquet as pq
            t = pq.read_table(path)
            for c in t.column_names:
                if str(t.schema.field(c).type) in ("string", "large_string"):
                    yield from (v for v in t.column(c).to_pylist() if v)
        elif suf in (".txt", ".csv", ".md", ".tsv", ""):
            yield path.read_text(errors="ignore")
    except Exception as e:  # unreadable file: report, do not guess
        print(f"FAIL  private file unreadable for the scan: {path.name} ({type(e).__name__})")
        raise SystemExit(1)


def wmdp_texts():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset
    for cfg in ("wmdp-bio", "wmdp-cyber"):
        for row in load_dataset("cais/wmdp", cfg, split="test"):
            yield row["question"]
            yield " ".join([row["question"], *row["choices"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--private-check", action="store_true")
    ap.add_argument("--private-root", action="append", default=None)
    ap.add_argument("--forget-corpus", action="append", default=[])
    ap.add_argument("--no-wmdp", action="store_true")
    args = ap.parse_args()
    fails = 0

    def fail(msg):
        nonlocal fails
        fails += 1
        print("FAIL  " + msg)

    files = [p for p in release_files() if p != SELF]
    for p in files:
        rel = str(p.relative_to(ROOT))
        if p.suffix.lower() in BAD_SUFFIX:
            fail(f"{rel}: forbidden file type {p.suffix}")
        if p.name not in ALLOWED_NAMES and BAD_NAME.search(rel):
            fail(f"{rel}: name denotes private material")
        if p.stat().st_size > MAX_BYTES:
            fail(f"{rel}: larger than 5 MB")
    texts = {}
    for p in files:
        if not is_text(p):
            continue
        s = p.read_text(errors="ignore")
        texts[p] = s
        rel = str(p.relative_to(ROOT))
        for pat, what in FORBIDDEN:
            hits = [m.group(0) for m in re.finditer(pat, s, re.I)]
            if what == "e-mail address":
                hits = [h for h in hits if not EMAIL_OK.search(h)]
            if what == "IP address":
                hits = [h for h in hits if not IP_OK.match(h) and all(int(x) < 256 for x in h.split("."))]
            if hits:
                fail(f"{rel}: {what} ({len(hits)} occurrence(s))")
    print(f"info  structural + text checks over {len(files)} files ({len(texts)} text)")

    if args.private_check:
        roots = args.private_root or ([os.environ["DSG_PRIVATE"]] if os.environ.get("DSG_PRIVATE") else [])
        if not roots:
            fail("--private-check needs --private-root or $DSG_PRIVATE")
        index = {}
        for p, s in texts.items():
            for h in shingles(words(s)):
                index.setdefault(h, set()).add(p)
        norm_release = {p: " ".join(words(s)) for p, s in texts.items()}
        hits, n_src, n_short = {}, 0, 0

        def check(text):
            nonlocal n_src, n_short
            n_src += 1
            ws = words(text)
            if len(ws) >= K:
                for h in shingles(ws):
                    for p in index.get(h, ()):
                        hits[p] = hits.get(p, 0) + 1
            elif len(" ".join(ws)) >= 16:      # short strings (e.g. an optimised suffix): whole-string match
                n_short += 1
                needle = " ".join(ws)
                for p, s in norm_release.items():
                    if needle in s:
                        hits[p] = hits.get(p, 0) + 1

        for r in roots:
            for f in sorted(Path(r).rglob("*")):
                if f.is_file():
                    for t in private_texts(f):
                        check(t)
        for c in args.forget_corpus:
            with open(c, errors="ignore") as fh:
                for line in fh:
                    try:
                        for t in strings_in(json.loads(line)):
                            check(t)
                    except ValueError:
                        check(line)
        if not args.no_wmdp:
            for t in wmdp_texts():
                check(t)
        for p, n in sorted(hits.items()):
            fail(f"{p.relative_to(ROOT)}: {n} shingle/string match(es) with private or hazardous material (content not shown)")
        print(f"info  private check: {n_src} source strings ({n_short} short) from {len(roots)} private root(s), "
              f"{len(args.forget_corpus)} corpus file(s), WMDP {'skipped' if args.no_wmdp else 'bio+cyber'}")

    print("SCAN FAILED" if fails else "SCAN OK")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
