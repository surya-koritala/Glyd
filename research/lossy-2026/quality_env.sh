#!/usr/bin/env bash
# Install the reference stack without mutating the pod's system Python.
set -euo pipefail
VENV=${1:?virtual environment directory}
REPORT=${2:?environment report}
HERE=$(cd "$(dirname "$0")" && pwd)
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV PIP_BREAK_SYSTEM_PACKAGES PIP_TARGET PIP_PREFIX PIP_USER LD_LIBRARY_PATH
export PYTHONNOUSERSITE=1 PIP_CONFIG_FILE=/dev/null PIP_REQUIRE_VIRTUALENV=true
# Reuse the image's large CUDA wheels, but verify their identity before installing.
python3.12 - <<'PY'
import importlib.metadata as m
import sys
assert sys.version_info[:2] == (3, 12)
for name, version in [('torch', '2.8.0'), ('torchvision', '0.23.0'), ('torchaudio', '2.8.0')]:
    actual = m.version(name)
    if actual.split('+')[0] != version:
        raise RuntimeError(f'base image has {name} {actual}; expected {version}')
PY
python3.12 -m venv --system-site-packages "$VENV"
"$VENV/bin/python" -m pip install --no-cache-dir -r "$HERE/quality-requirements.txt"
"$VENV/bin/python" "$HERE/quality_env.py" "$REPORT" "${@:3}"
"$VENV/bin/python" -m pip freeze --all > "${REPORT%.json}-freeze.txt"
