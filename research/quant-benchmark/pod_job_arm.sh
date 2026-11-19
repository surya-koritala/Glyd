#!/usr/bin/env bash
# One immutable single-arm quality batch (ARM, default candidate-a7) on the original prepared questions, reported against
# the retained BF16 and FP8 arms in the bundle's references/. The outer launcher owns the paid-pod cap and deletion.
set -euo pipefail
ARM=${ARM:-candidate-a7}
W=$HOME/quant-benchmark
R=$HOME/results
mkdir -p "$W" "$R"
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
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/steps.txt"; }
step() {
    local label=$1 cap=$2 remain
    shift 2
    remain=$((T0+50*60-$(date +%s)))
    [ "$remain" -ge 60 ] || { log 'workload time cap reached'; return 124; }
    [ "$cap" -le "$remain" ] || cap=$remain
    log "start $label (cap ${cap}s)"
    if timeout "$cap" "$@" > "$R/$label.log" 2>&1; then log "pass $label";
    else local rc=$?; log "fail $label (exit $rc)"; return "$rc"; fi
}
tar -xf "$HOME/quant-arm.tar" -C "$W"
(cd "$W" && sha256sum -c SHA256SUMS) > "$R/input-checks.txt"
cp "$W/COMMIT" "$W/CODE_COMMIT" "$W/SHA256SUMS" "$W/prepared/manifest.json" "$W/prepared/model-audit.json" "$R/"
nvidia-smi > "$R/machine.txt"
python3 - <<'PY'
import subprocess
n,cc,m,d=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).strip().split(',')
assert 'A100' in n and cc.strip()=='8.0' and int(m)>75000 and int(d.strip().split('.')[0])>=580
PY
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -l 2 > "$R/smi.csv" &
SMI=$!
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN PYTHONPATH
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false HF_XET_HIGH_PERFORMANCE=1
HERE=$W/research/quant-benchmark
step environment 600 bash "$HERE/pod_environment.sh" "$W" "$W/venv"
PY=$W/venv/bin/python
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 GLYD_CODE_COMMIT=$(cat "$W/CODE_COMMIT")
step model-download 1200 "$PY" "$HERE/download.py" --prepared "$W/prepared" --arms bf16 --out "$R/model-paths.json"
BF16=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["bf16"])' "$R/model-paths.json")
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
step "$ARM" 2700 "$PY" "$HERE/score.py" --prepared "$W/prepared" --model-dir "$BF16" --arm "$ARM" --out "$R/$ARM"
step report 120 "$PY" "$HERE/report.py" --prepared "$W/prepared" --results "$R" --references "$W/references" --candidate "$ARM" --out "$R/report.json"
cp "$W/environment/resolved-versions.json" "$R/"
log "completed fixed-choice quality arm $ARM against the retained references"
