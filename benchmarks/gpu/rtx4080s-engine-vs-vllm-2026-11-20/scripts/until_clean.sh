#!/usr/bin/env bash
# run.sh again until a run finishes with no CI job on the GPU at any moment (run.sh exits 3 while a CI container is up, and leaves DISCARD where one appeared mid-run).
#   until_clean.sh engine|vllm OUTDIR
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
for try in $(seq 1 60); do
    rm -rf "$2"
    bash "$HERE/run.sh" "$1" "$2"; code=$?
    if [ $code = 3 ]; then sleep 20; continue; fi
    if [ -e "$2/DISCARD" ]; then echo "discarded: a CI job ran during it; again"; mv "$2" "$2.discarded.$try"; continue; fi
    exit $code
done
