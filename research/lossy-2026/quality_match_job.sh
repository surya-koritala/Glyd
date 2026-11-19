#!/usr/bin/env bash
# One prepared matched weight-quality screen, not a native-format memory or speed benchmark.
# Inputs: ~/lossy27-quality.tar, with verified CI binaries, scripts, immutable evaluation and checksums.
# Outer launcher: one A100 80GB, maximum 75 minutes, automatic deletion. No retries or dependency changes.
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
    remain=$((T0+70*60-$(date +%s)))
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
cp "$W/reference-audit.json" "$W/COMMIT" "$W/evaluation.json" "$W/evaluation.sha256" "$W/SHA256SUMS" "$R/"
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
step prepared-checks 120 "$PY" "$W/check_matched.py" "$W/glyd-lossy-inventory"
step model-download 1800 "$PY" - "$W" <<'PYDOWNLOAD'
import json,sys,time
from pathlib import Path
from huggingface_hub import snapshot_download
root=Path(sys.argv[1]);models=json.loads((root/'models.json').read_text())
audit=json.loads((root/'reference-audit.json').read_text())
assert audit['config_and_tokenizer_match']
for label,name in [('bf16','Qwen/Qwen3.8-27B'),('fp8','Qwen/Qwen3.8-27B-FP8')]:
    meta=models[name]
    assert meta['createdAt'].startswith('2026-') and audit['sources'][name]['revision']==meta['sha']
    time.sleep(2.5)
    d=Path(snapshot_download(name,revision=meta['sha'],max_workers=1,allow_patterns=['*.json','*.safetensors','*.txt','*.jinja','*.model']))
    (d/'glyd-study-source.json').write_text(json.dumps({'model':name,'revision':meta['sha']}))
    (root/f'model-path-{label}.txt').write_text(str(d))
PYDOWNLOAD
D=$(cat "$W/model-path-bf16.txt")
F=$(cat "$W/model-path-fp8.txt")
step kernel-check 300 "$W/glyd-engine" kcheck35
step common-bf16 900 "$PY" "$W/matched_quality.py" "$D" "$W/evaluation.json" "$R/common-bf16.json" --kind bf16
step native-control 1800 "$W/glyd-lossy-quality" "$D" "$W/evaluation.json" "$R/native-control.json" --fraction-bits 7 --ctx 2048
step control-comparison 60 "$PY" "$W/quality_compare.py" "$R/common-bf16.json" "$R/native-control.json" "$R/control-comparison.json" --control
step common-candidate-a 900 "$PY" "$W/matched_quality.py" "$D" "$W/evaluation.json" "$R/common-candidate-a.json" --kind candidate-a
step common-fp8 900 "$PY" "$W/matched_quality.py" "$F" "$W/evaluation.json" "$R/common-fp8.json" --kind fp8
step native-candidate-a 1800 "$W/glyd-lossy-quality" "$D" "$W/evaluation.json" "$R/native-candidate-a.json" --fraction-bits 3 --ctx 2048
step candidate-runtime-control 60 "$PY" "$W/quality_compare.py" "$R/common-candidate-a.json" "$R/native-candidate-a.json" "$R/candidate-runtime-control.json" --control
step a-versus-bf16 60 "$PY" "$W/quality_compare.py" "$R/common-bf16.json" "$R/common-candidate-a.json" "$R/a-versus-bf16.json"
step fp8-versus-bf16 60 "$PY" "$W/quality_compare.py" "$R/common-bf16.json" "$R/common-fp8.json" "$R/fp8-versus-bf16.json"
step a-versus-fp8 60 "$PY" "$W/quality_compare.py" "$R/common-fp8.json" "$R/common-candidate-a.json" "$R/a-versus-fp8.json"
log 'completed quality screen; no native-format VRAM or speed result'
