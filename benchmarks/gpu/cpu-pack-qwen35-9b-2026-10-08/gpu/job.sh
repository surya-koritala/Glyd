#!/usr/bin/env bash
# One approved SM89 pod: this repository's glyd_gpu (its CUDA library built by CI) packs and saves Qwen/Qwen3.5-9B on the
# GPU, loads the save back and generates (gpu_check.py). The outer launcher owns the cap and the pod's deletion; no retry.
set -euo pipefail
R=$HOME/results
W=$HOME/gpu-check
mkdir -p "$R" "$W"
T0=$(date +%s)
finish() {
    local rc=$?
    trap - EXIT
    echo "$rc" > "$R/EXIT_CODE"
    touch "$R/DONE"
    exit "$rc"
}
trap finish EXIT
step() {
    local label=$1 cap=$2 remain
    shift 2
    remain=$((T0+20*60-$(date +%s)))
    [ "$remain" -ge 60 ] || return 124
    [ "$cap" -le "$remain" ] || cap=$remain
    echo "$(date -u +%FT%TZ) start $label cap=$cap" >> "$R/steps.txt"
    timeout "$cap" "$@" > "$R/$label.log" 2>&1
    echo "$(date -u +%FT%TZ) pass $label" >> "$R/steps.txt"
}
tar -xf "$HOME/gpu-check.tar" -C "$W"
(cd "$W" && sha256sum -c SHA256SUMS) > "$R/input-checks.txt"
cp "$W/COMMIT" "$W/SHA256SUMS" "$R/"
nvidia-smi > "$R/machine.txt"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV PIP_BREAK_SYSTEM_PACKAGES PIP_TARGET PIP_PREFIX PIP_USER LD_LIBRARY_PATH HF_TOKEN HUGGING_FACE_HUB_TOKEN
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null PIP_REQUIRE_VIRTUALENV=true HF_HUB_DISABLE_IMPLICIT_TOKEN=1 HF_XET_HIGH_PERFORMANCE=1
# the image's Python refuses installs (PEP 668): a virtual environment over it, its PyTorch kept (as the quality jobs do)
python3.12 -m venv --system-site-packages "$W/venv"
PY=$W/venv/bin/python
step environment 300 "$PY" -m pip install --quiet "transformers==5.18.0" "accelerate>=1.1" "safetensors>=0.8" "huggingface_hub>=1.5,<2"
"$PY" -m pip freeze > "$R/pip-freeze.txt"
export PYTHONPATH="$W/python" GLYD_GPU_LIB="$W/libglyd_gpu_cuda13.so"
step check 1080 "$PY" "$W/gpu_check.py" "Qwen/Qwen3.5-9B" "c202236235762e1c871ad0ccb60c8ee5ba337b9a" "$R" "$W/gpu-save"
