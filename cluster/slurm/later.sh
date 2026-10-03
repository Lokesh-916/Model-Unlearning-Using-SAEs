# Sourced by the "later" sbatch jobs (session 8) after preamble.sh:
#  * run from the code snapshot code-later/ (stage_code_snapshot.sh later), so code/ is never changed under
#    a running or queued chain;
#  * DSG_BUDGET_MIN = Slurm time left minus 12 min: scripts stop starting work after it (all resumable),
#    so a 3 h --time is never hit mid-step and the next chained sbatch continues.
CODE="$DSGC/code-later"; [ -d "$CODE" ] || { echo "=== missing $CODE (run cluster/stage_code_snapshot.sh later)"; exit 2; }
cd "$CODE"; export PYTHONPATH="$CODE" DSG_WORKTREES="$CODE"
echo "=== code dir: $CODE $(cat CODE_COMMIT.json 2>/dev/null)"
_left=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %L 2>/dev/null | tail -1)
_mins() { local t="$1" d=0 h=0 m=0 s=0; [[ "$t" == *-* ]] && { d=${t%%-*}; t=${t#*-}; }
          IFS=: read -r a b c <<< "$t"; if [ -n "$c" ]; then h=$a; m=$b; s=$c; elif [ -n "$b" ]; then m=$a; s=$b; else m=$a; fi
          echo $(( (10#$d * 24 + 10#$h) * 60 + 10#$m )); }
if [[ "$_left" =~ ^[0-9] ]]; then export DSG_BUDGET_MIN=$(( $(_mins "$_left") - 12 )); else export DSG_BUDGET_MIN=${DSG_BUDGET_MIN:-165}; fi
echo "=== time left ${_left:-?}; work budget ${DSG_BUDGET_MIN} min"
