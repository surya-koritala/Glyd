#!/usr/bin/env bash
# One immutable three-arm quality batch. The outer launcher owns the paid-pod cap and deletion.
set -euo pipefail
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
    remain=$((T0+140*60-$(date +%s)))
    [ "$remain" -ge 60 ] || { log 'workload time cap reached'; return 124; }
    [ "$cap" -le "$remain" ] || cap=$remain
    log "start $label (cap ${cap}s)"
    if timeout "$cap" "$@" > "$R/$label.log" 2>&1; then log "pass $label";
    else local rc=$?; log "fail $label (exit $rc)"; return "$rc"; fi
}
tar -xf "$HOME/quant-benchmark.tar" -C "$W"
(cd "$W" && sha256sum -c SHA256SUMS) > "$R/input-checks.txt"
cp "$W/COMMIT" "$W/SHA256SUMS" "$W/prepared/manifest.json" "$W/prepared/model-audit.json" "$R/"
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
export PYTHONNOUSERSITE=1
step model-download 1800 "$PY" "$HERE/download.py" --prepared "$W/prepared" --out "$R/model-paths.json"
BF16=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["bf16"])' "$R/model-paths.json")
FP8=$($PY -c 'import json,sys;print(json.load(open(sys.argv[1]))["fp8"])' "$R/model-paths.json")
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
step bf16 7200 "$PY" "$HERE/score.py" --prepared "$W/prepared" --model-dir "$BF16" --arm bf16 --out "$R/bf16"
step candidate-a 7200 "$PY" "$HERE/score.py" --prepared "$W/prepared" --model-dir "$BF16" --arm candidate-a --out "$R/candidate-a"
step fp8 7200 "$PY" "$HERE/score.py" --prepared "$W/prepared" --model-dir "$FP8" --arm fp8 --out "$R/fp8"
step report 120 "$PY" "$HERE/report.py" --prepared "$W/prepared" --results "$R" --out "$R/report.json"
cp "$W/environment/resolved-versions.json" "$R/"
log 'completed fixed-choice quality baseline; compact GPU memory and serving speed remain separate'
