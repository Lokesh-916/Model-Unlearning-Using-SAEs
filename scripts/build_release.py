"""Build the public release package (never pushed by this script).

    python scripts/build_release.py [--dest ~/projects/mechunlearn-project/release]

Contents, all taken from committed git objects (not the working tree) or from aggregate result files:
  dsgx/ (harness, without the lab queue/scheduler and the live dashboard), experiments/<X>/ and configs/experiments/
  from every exp/* branch tip, guardbreak/ (N1), dynamic_sae_guardrails/ (upstream DSG code, MIT), data/splits/
  (question ids + hashes only), a CPU test subset, results/ (sanitised summary.json, reports, digest, numbers.tex,
  generated tables) and figures/.
Never copied: hazardous prompts or their transformations, generations, optimised suffixes, forget corpora, model weights,
private folders, server scripts or details (cluster/, scripts/), queue state, personal paths. Absolute local paths in
code defaults and result files are rewritten; the release's own scan_release.py then fails on anything left over.
"""
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROJECT = REPO.parent
PAPER = PROJECT / "paper"

HARNESS_EXCLUDE = (
    "dsgx/queue/",                 # lab scheduler, tmux, worktrees, queue state
    "dsgx/checks/preflight.py",    # lab GPU/queue preflight (imports the queue)
    "dsgx/analysis/dashboard.py",  # live dashboard: ssh status of the server
)
TESTS = ["tests/test_stats.py", "tests/test_splits.py", "tests/test_labels.py", "tests/test_dsg.py",
         "tests/test_config_leakage.py", "tests/test_activation_cache.py", "tests/test_resume_items.py",
         "tests/test_logging.py", "tests/fakeruns.py"]
NEVER_SUFFIX = (".jsonl", ".parquet", ".npz", ".npy", ".pkl", ".pt", ".pth", ".bin", ".safetensors", ".ckpt", ".arrow")

# literal rewrites (order matters; checked again by scan_release.py)
CODE_REWRITES = [
    ('"/home/amaloch/projects/mechunlearn-project"', 'os.path.expanduser("~/dsg-project")'),
    ('Path.home() / "projects/mechunlearn-project/dsg_results_cluster"', 'Path.home() / "dsg-project/dsg_results_cluster"'),
    ('Path.home() / "projects/mechunlearn-project"', 'Path.home() / "dsg-project"'),
    ("~/projects/mechunlearn-project/", "~/dsg-project/"),
]
PATH_RE = re.compile(r"/home/[^/\s\"'`]+/projects/mechunlearn-project/")


def git(*args, text=True):
    return subprocess.run(["git", "-C", str(REPO), *args], check=True, capture_output=True, text=text).stdout


def export_tree(ref, prefix, dest, exclude=()):
    """Extract `prefix` of a commit into dest (git archive; committed content only)."""
    data = git("archive", "--format=tar", ref, prefix, text=False)
    n = 0
    with tarfile.open(fileobj=io.BytesIO(data)) as tf:
        for m in tf.getmembers():
            if not m.isfile() or any(m.name.startswith(e) for e in exclude) or m.name.endswith(NEVER_SUFFIX):
                continue
            out = dest / m.name
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(tf.extractfile(m).read())
            n += 1
    return n


def sanitize_text(s):
    for a, b in CODE_REWRITES:
        s = s.replace(a, b)
    s = s.replace("/dsg_results_cluster/", "/server-results/") if PATH_RE.search(s) else s
    return PATH_RE.sub("<project>/", s)


def sanitize_json(x):
    if isinstance(x, dict):
        return {k: sanitize_json(v) for k, v in x.items() if k != "weights"}   # local paths of model weights
    if isinstance(x, list):
        return [sanitize_json(v) for v in x]
    return sanitize_text(x) if isinstance(x, str) else x


def strip_tex_comments(s):
    out = [re.sub(r"(?<!\\)%.*$", "%", l) for l in s.splitlines() if not l.lstrip().startswith("%")]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=str(PROJECT / "release"))
    args = ap.parse_args()
    dest = Path(args.dest).expanduser()
    dest.mkdir(parents=True, exist_ok=True)
    for p in dest.iterdir():            # rebuild everything except git metadata and the hand-written top-level files
        if p.name in {".git", "README.md", "LICENSE", ".gitignore"}:
            continue
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    sources = {"harness": git("rev-parse", "HEAD").strip(), "branches": {}}

    n = export_tree("HEAD", "dsgx", dest, HARNESS_EXCLUDE)
    n += export_tree("HEAD", "dynamic_sae_guardrails", dest)
    n += export_tree("HEAD", "data/splits", dest)
    for t in TESTS:
        n += export_tree("HEAD", t, dest)
    shutil.copy2(REPO / "scripts" / "release_scan.py", dest / "scan_release.py")
    (dest / "LICENSE-DSG-UPSTREAM").write_text(git("show", "HEAD:LICENSE"))
    (dest / "requirements-lock.txt").write_text(git("show", "HEAD:requirements-lock.txt"))
    for b in git("branch", "--format=%(refname:short)").split():
        if not b.startswith("exp/"):
            continue
        sha = git("rev-parse", b).strip()
        sources["branches"][b] = sha
        # only what the branch adds or changes relative to the harness (its own experiment code and config)
        names = [x for x in git("diff", "--name-only", f"v2-harness...{b}").split("\n") if x]
        names += [x for x in git("ls-tree", "-r", "--name-only", b, "configs/experiments").split("\n")
                  if x and Path(x).stem.split("-")[0] == b.split("/")[1].split("-")[0]]
        for x in sorted(set(names)):
            if x.startswith(("experiments/", "configs/experiments/", "guardbreak/")):
                n += export_tree(b, x, dest)
    for p in dest.rglob("*.py"):
        p.write_text(sanitize_text(p.read_text()))
    for p in list(dest.rglob("*.yaml")) + list(dest.rglob("*.md")):
        if p.parent != dest:
            p.write_text(sanitize_text(p.read_text()))

    # aggregate results (both machines, never merged) + generated tables and figures
    res, figs = dest / "results", dest / "figures"
    for m in ("labpc", "gpuws"):
        src = PROJECT / "paper_numbers" / m
        (res / m).mkdir(parents=True, exist_ok=True)
        (res / m / "summary.json").write_text(json.dumps(sanitize_json(json.loads((src / "summary.json").read_text())), indent=1))
        (res / m / "FINAL_REPORT.md").write_text(sanitize_text((src / "FINAL_REPORT.md").read_text()))
    digest = PROJECT / "dsg_results" / "RESULTS_DIGEST.md"
    (res / "RESULTS_DIGEST.md").write_text(sanitize_text(digest.read_text()))
    (res / "numbers.tex").write_text(strip_tex_comments((PAPER / "numbers.tex").read_text()))
    for m, sub in (("labpc", "assets"), ("gpuws", "assets-gpuws")):
        for kind, d in (("tables", res / "tables" / m), ("figures", figs / m)):
            d.mkdir(parents=True, exist_ok=True)
            for f in sorted((PAPER / sub / kind).glob("*")):
                if f.suffix == ".tex":
                    (d / f.name).write_text(strip_tex_comments(sanitize_text(f.read_text())))
                elif f.suffix in (".pdf", ".png"):
                    shutil.copy2(f, d / f.name)
    mt = PROJECT / "dsg_results_cluster" / "jobs" / "mtbench" / "summary.json"
    if mt.exists():
        (res / "gpuws" / "mtbench_summary.json").write_text(json.dumps(sanitize_json(json.loads(mt.read_text())), indent=1))

    files = sorted(p for p in dest.rglob("*") if p.is_file() and ".git" not in p.parts)
    manifest = {"sources": sources, "files": {str(p.relative_to(dest)): hashlib.sha256(p.read_bytes()).hexdigest()
                                              for p in files if p.name != "MANIFEST.json"}}
    (dest / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    print(f"release built in {dest}: {len(manifest['files'])} files ({n} code/data files from git, "
          f"{len(sources['branches'])} exp branches); now run: python scan_release.py --private-check")


if __name__ == "__main__":
    main()
