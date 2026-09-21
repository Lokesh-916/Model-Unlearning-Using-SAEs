#!/bin/bash
# GPU (16 GB) cannot hold two copies at once, so DSG generation waits for the base run to finish.
cd "$(dirname "$0")/.."
while pgrep -f "generate.py --mode base" > /dev/null; do sleep 20; done
python mtbench/generate.py --mode dsg > mtbench/logs/generate_dsg.log 2>&1
