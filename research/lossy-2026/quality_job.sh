#!/usr/bin/env bash
# One prepared language-quality screen, not a native-format memory or speed benchmark.
# Inputs: ~/lossy27-quality.tar, with verified CI binaries, scripts, immutable evaluation and checksums.
# Outer launcher: one A100 80GB, maximum 90 minutes, automatic deletion. No retries or dependency changes.
set -euo pipefail
R=$HOME/results
W=$HOME/lossy27-quality
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
log() { echo "$(date -u +%FT%TZ) $*" | tee -a "$R/steps.txt"; }
step() {
    local label=$1 cap=$2 remain
    shift 2
    remain=$((T0+80*60-$(date +%s)))
    [ "$remain" -ge 60 ] || { log 'job cap reached'; return 124; }
    [ "$cap" -le "$remain" ] || cap=$remain
    log "start $label (cap ${cap}s)"
    if timeout "$cap" "$@" > "$R/$label.log" 2>&1; then
        log "pass $label"
    else
        local rc=$?
        log "fail $label (exit $rc)"
        return "$rc"
    fi
}
tar -xf "$HOME/lossy27-quality.tar" -C "$W"
(cd "$W" && sha256sum -c SHA256SUMS && sha256sum -c evaluation.sha256) > "$R/input-checks.txt"
cp "$W/COMMIT" "$W/evaluation.json" "$W/evaluation.sha256" "$W/SHA256SUMS" "$R/"
nvidia-smi > "$R/machine.txt"
nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version --format=csv >> "$R/machine.txt"
python3 - <<'PY'
import subprocess
name,cc,mem,driver=subprocess.check_output(['nvidia-smi','--query-gpu=name,compute_cap,memory.total,driver_version','--format=csv,noheader,nounits'],text=True).strip().split(',')
assert 'A100' in name and cc.strip()=='8.0' and int(mem)>75000
assert int(driver.strip().split('.')[0])>=580
PY
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu,power.draw --format=csv,noheader,nounits -l 2 > "$R/smi.csv" &
SMI=$!
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN PYTHONPATH GLYD_CUBLAS
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_HOME=$W/hf TOKENIZERS_PARALLELISM=false HF_XET_HIGH_PERFORMANCE=1
export GLYD_GPU_LIB=$W/libglyd_gpu_cuda13.so GLYD_ENGINE_GEMM=own
chmod +x "$W/glyd-engine" "$W/glyd-lossy-quality"
# Run the same isolated reference bootstrap already checked in the pinned image on CI.
step python-env 900 bash "$W/quality_env.sh" "$W/venv" "$R/environment.json" --require-cuda
PY=$W/venv/bin/python
unset LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1
step model-download 1200 "$PY" - "$W" <<'PY'
import json,sys
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(sys.argv[1]);meta=json.loads((root/'models.json').read_text())['Qwen/Qwen3.8-27B']
assert meta['createdAt'].startswith('2026-')
d=Path(snapshot_download('Qwen/Qwen3.8-27B',revision=meta['sha'],allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model']))
(d/'glyd-study-source.json').write_text(json.dumps({'model':'Qwen/Qwen3.8-27B','revision':meta['sha']}))
(root/'model-path.txt').write_text(str(d))
PY
D=$(cat "$W/model-path.txt")
step kernel-check 300 "$W/glyd-engine" kcheck35
step independent-bf16 900 "$PY" "$W/quality_reference.py" "$D" "$W/evaluation.json" "$R/reference-bf16.json"
step native-control 1500 "$W/glyd-lossy-quality" "$D" "$W/evaluation.json" "$R/native-control.json" --fraction-bits 7
step control-comparison 60 "$PY" "$W/quality_compare.py" "$R/reference-bf16.json" "$R/native-control.json" "$R/control-comparison.json" --control
# A failed independent control stops the job before either lossy candidate is interpreted.
for bits in 3 2; do
    step "candidate-$bits" 1500 "$W/glyd-lossy-quality" "$D" "$W/evaluation.json" "$R/candidate-$bits.json" --fraction-bits "$bits"
    step "comparison-$bits" 60 "$PY" "$W/quality_compare.py" "$R/native-control.json" "$R/candidate-$bits.json" "$R/comparison-$bits.json"
done
log 'completed quality screen; no native-format VRAM or speed result'
