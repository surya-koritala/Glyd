#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
BUNDLE=${1:?bundle root}
VENV=${2:?isolated environment}
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV PIP_BREAK_SYSTEM_PACKAGES PIP_TARGET PIP_PREFIX PIP_USER LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null PIP_REQUIRE_VIRTUALENV=true
python3.12 "$HERE/environment.py" --check "$BUNDLE/base-versions.json"
python3.12 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --no-index --find-links "$BUNDLE/environment/wheels" -r "$BUNDLE/environment/requirements-local.lock"
CUDA_ARGS=(--require-cuda)
if [ "${3:-}" = --cpu-only ]; then CUDA_ARGS=(); fi
"$VENV/bin/python" "$HERE/environment.py" --check "$BUNDLE/environment/resolved-versions.json" "${CUDA_ARGS[@]}"
