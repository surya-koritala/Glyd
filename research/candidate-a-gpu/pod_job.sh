#!/usr/bin/env bash
# One approved RTX 4090 matrix validation batch. No model download, dependency install or retry.
# Outer launcher: MAX_MIN=45 with deletion on completion or failure.
set -euo pipefail
R=$HOME/results
W=$HOME/candidate-a-validation
mkdir -p "$R" "$W"
T0=$(date +%s)
SMI=
finish() {
    local rc=$?
    trap - EXIT
    [ -z "$SMI" ] || kill "$SMI" 2>/dev/null || true
    echo "$rc" > "$R/EXIT_CODE"
    touch "$R/DONE"
    exit "$rc"
}
trap finish EXIT
step() {
    local label=$1 cap=$2 remain
    shift 2
    remain=$((T0+40*60-$(date +%s)))
    [ "$remain" -ge 60 ] || return 124
    [ "$cap" -le "$remain" ] || cap=$remain
    echo "$(date -u +%FT%TZ) start $label cap=$cap" >> "$R/steps.txt"
    timeout "$cap" "$@" > "$R/$label.log" 2>&1
    echo "$(date -u +%FT%TZ) pass $label" >> "$R/steps.txt"
}
tar -xf "$HOME/candidate-a-validation.tar" -C "$W"
(cd "$W" && sha256sum -c SHA256SUMS) > "$R/input-checks.txt"
cp "$W/cases.json" "$W/COMMIT" "$W/SHA256SUMS" "$W/environment.json" "$W/pod-image.txt" "$R/"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
nvidia-smi > "$R/machine.txt"
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -l 2 > "$R/smi.csv" &
SMI=$!
step environment 120 python3.12 "$W/environment.py" --expected "$W/environment.json" --out "$R/device-environment.json" --library "$W/libcandidate_a.so" --gpu
step matrices 2100 python3.12 "$W/validate.py" --bundle "$W" --out "$R/matrices"
step fp8 1200 python3.12 "$W/fp8_variant.py" --bundle "$W" --out "$R/fp8"
