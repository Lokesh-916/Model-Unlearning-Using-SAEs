#!/usr/bin/env bash
# Self-updating paper: regenerate every result number and rebuild the PDF in one command (CPU only).
#   1. final_report --interim for each machine -> $DSG_PROJECT/paper_numbers/{labpc,gpuws}/summary.json
#      (a side folder: $DSG_RESULTS/summary.json stays the N10 watcher's; lab PC and gpuws never mixed)
#   2. dsgx.analysis.paper_numbers -> paper/numbers.tex (one \newcommand per number, with CI and n)
#   3. latexmk -pdf in the paper repo -> paper/main.pdf
#   4. if ../paper-srw exists (EACL SRW version): its build.sh syncs numbers.tex/figures and rebuilds + checks it
#   5. prints what is still open: pending numbers, \\interimnote marks in the PDF, post-hoc / lab-X1 status macros
#   scripts/paper_update.sh [--no-pdf] [--final]
#   --final: final_report without --interim (it still writes an INTERIM summary, and says why, if anything is unfinished);
#            use it once the lab X1 replication and the gpuws ph-union chain are fetched (runbook section 7.1).
# Then check `git -C $PAPER diff numbers.tex` and commit numbers.tex in the paper repo.
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$REPO/scripts/env.sh"
cd "$REPO"
P="$DSG_PROJECT"
PAPER="${PAPER:-$P/paper}"
OUT="$P/paper_numbers"
NOPDF=0; INTERIM="--interim"
for a in "$@"; do case "$a" in --no-pdf) NOPDF=1;; --final) INTERIM="";; *) echo "unknown option $a"; exit 2;; esac; done
export CUDA_VISIBLE_DEVICES=
"$DSGX_PY" -m dsgx.analysis.final_report $INTERIM --hardware labpc --no-figures --out "$OUT/labpc" | tail -n 2
"$DSGX_PY" -m dsgx.analysis.final_report $INTERIM --hardware gpuws --runs "$P/dsg_results_cluster/runs" \
  --no-figures --out "$OUT/gpuws" | tail -n 2
"$DSGX_PY" -m dsgx.analysis.paper_numbers --out "$PAPER/numbers.tex"
if [ "$NOPDF" = 0 ]; then
  export PATH="$HOME/.local/bin:$PATH"   # latexmk 4.88 lives there (session 5)
  (cd "$PAPER" && latexmk -pdf -silent main.tex >/dev/null 2>&1) || { echo "latexmk failed: see $PAPER/main.log"; exit 1; }
  echo "LaTeX warnings: $(grep -c "Warning" "$PAPER/main.log" || true)"
  echo "built $PAPER/main.pdf ($(grep -o 'Output written on main.pdf ([0-9]* pages' "$PAPER/main.log" | grep -o '[0-9]* pages' || echo '? pages'))"
fi
SRW="${SRW:-$P/paper-srw}"
if [ "$NOPDF" = 0 ] && [ -x "$SRW/build.sh" ]; then
  PAPER="$PAPER" "$SRW/build.sh" | tail -n 3 || { echo "SRW build failed: see $SRW/main.log"; exit 1; }
fi
# what is still open (the paper is final when all three lines read 0 / done)
echo "open: $(sed -n 's/^% [0-9]* macros, \([0-9]*\) pending.*/\1/p' "$PAPER/numbers.tex") pending numbers in numbers.tex"
for m in LabXOneStatus GpuPhUnionStatus GpuPhUnionTofuStatus GpuPhUnionOpenStatus GpuPhUnionPairedStatus GpuPhTofucalStatus; do
  printf '  %-24s %s\n' "$m" "$(sed -n "s/^\\\\newcommand{\\\\res$m}{\\(.*\\)}$/\\1/p" "$PAPER/numbers.tex")"
done
if [ "$NOPDF" = 0 ] && command -v pdftotext >/dev/null; then
  echo "open: $(pdftotext "$PAPER/main.pdf" - | grep -c 'Interim:' || true) [Interim] marks, $(pdftotext "$PAPER/main.pdf" - | grep -c 'pending' || true) lines with 'pending' in paper/main.pdf"
fi
