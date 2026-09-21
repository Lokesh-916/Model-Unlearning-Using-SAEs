#!/bin/bash
# Generation-only pipeline: waits for each stage's answers, then commits + pushes them.
# Judging is done separately (by Claude as judge, no API key). Log: mtbench/logs/pipeline.log
cd "$(dirname "$0")/.."
LOG=mtbench/logs/pipeline.log
say() { echo "[$(date +%T)] $*" | tee -a $LOG; }
wait_for() { until grep -q "DONE mode=$1" "mtbench/logs/generate_$1.log" 2>/dev/null; do sleep 30; done; }
ship() { git add mtbench && git commit -q -m "$1" && git push -q origin main && say "pushed: $1" || say "nothing to push / push failed: $1"; }

say "generation-only pipeline started"
wait_for base;  say "base answers done";  ship "Add MT-Bench answers from base gemma-2-2b-it with generation log"
wait_for dsg;   say "dsg answers done";   ship "Add MT-Bench answers from DSG-guarded gemma-2-2b-it with generation log"
say "pipeline finished (judging pending)"
