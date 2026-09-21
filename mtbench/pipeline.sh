#!/bin/bash
# Waits for each stage, then judges, plots, and commits + pushes after every stage. Log: mtbench/logs/pipeline.log
cd "$(dirname "$0")/.."
LOG=mtbench/logs/pipeline.log
say() { echo "[$(date +%T)] $*" | tee -a $LOG; }
wait_for() { until grep -q "DONE mode=$1" "mtbench/logs/generate_$1.log" 2>/dev/null; do sleep 30; done; }
wait_key() { until [ -n "$OPENROUTER_API_KEY" ] || { grep -Eq "^OPENROUTER_API_KEY=.+" .env && ! grep -q "PASTE_YOUR_KEY_HERE" .env; }; do sleep 30; done; }
ship() { git add mtbench && git commit -q -m "$1" && git push -q origin main && say "pushed: $1" || say "nothing to push / push failed: $1"; }

say "pipeline started"
wait_for base;  say "base answers done";  ship "Add MT-Bench answers from base gemma-2-2b-it with generation log"
wait_key;       say "key found, judging base"
python mtbench/judge.py --mode base >> mtbench/logs/judge_base.log 2>&1
ship "Add GPT-4o judge scores for base-model MT-Bench answers"
wait_for dsg;   say "dsg answers done";   ship "Add MT-Bench answers from DSG-guarded gemma-2-2b-it with generation log"
python mtbench/judge.py --mode dsg >> mtbench/logs/judge_dsg.log 2>&1
ship "Add GPT-4o judge scores for DSG-guarded MT-Bench answers"
python mtbench/plots.py >> mtbench/logs/plots.log 2>&1 && ship "Add MT-Bench summary numbers and five result figures"
say "pipeline finished"
