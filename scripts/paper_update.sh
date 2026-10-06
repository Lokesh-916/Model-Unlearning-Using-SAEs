#!/usr/bin/env bash
# Self-updating paper: regenerate every result number and rebuild the PDF in one command (CPU only).
#   1. final_report --interim for each machine -> $DSG_PROJECT/paper_numbers/{labpc,gpuws}/summary.json
#      (a side folder: $DSG_RESULTS/summary.json stays the N10 watcher's; lab PC and gpuws never mixed)
#   2. dsgx.analysis.paper_numbers -> paper/numbers.tex (one \newcommand per number, with CI and n)
#   3. latexmk -pdf in the paper repo -> paper/main.pdf
#   scripts/paper_update.sh [--no-pdf]
# Then check `git -C $PAPER diff numbers.tex` and commit numbers.tex in the paper repo.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"
P="$DSG_PROJECT"
PAPER="${PAPER:-$P/paper}"
OUT="$P/paper_numbers"
export CUDA_VISIBLE_DEVICES=
"$DSGX_PY" -m dsgx.analysis.final_report --interim --hardware labpc --no-figures --out "$OUT/labpc" | tail -n 2
"$DSGX_PY" -m dsgx.analysis.final_report --interim --hardware gpuws --runs "$P/dsg_results_cluster/runs" \
  --no-figures --out "$OUT/gpuws" | tail -n 2
"$DSGX_PY" -m dsgx.analysis.paper_numbers --out "$PAPER/numbers.tex"
if [ "${1:-}" != "--no-pdf" ]; then
  export PATH="$HOME/.local/bin:$PATH"   # latexmk 4.88 lives there (session 5)
  (cd "$PAPER" && latexmk -pdf -silent main.tex >/dev/null 2>&1) || { echo "latexmk failed: see $PAPER/main.log"; exit 1; }
  echo "LaTeX warnings: $(grep -c "Warning" "$PAPER/main.log" || true)"
  echo "built $PAPER/main.pdf ($(grep -o 'Output written on main.pdf ([0-9]* pages' "$PAPER/main.log" | grep -o '[0-9]* pages' || echo '? pages'))"
fi
