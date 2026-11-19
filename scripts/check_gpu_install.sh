#!/usr/bin/env bash
# glyd[gpu] as a user installs it, on a machine with an NVIDIA GPU: a fresh venv, pip install
# "glyd[gpu]==V" from PyPI, a model loaded and generating, exact=True bit for bit with bf16,
# saved and loaded back (gpu/check_api.py, run against the installed package: a copy of it
# outside this checkout, so it does not pick up bindings/python's). One line a step, a final
# verdict.
#
#   scripts/check_gpu_install.sh VERSION [WORKDIR]
#     VERSION   e.g. 0.22.0 (no leading v)
#     WORKDIR   default: a fresh temporary directory. The venv and the check_api.py copy go
#               there; an existing HF_HOME (models already downloaded) is left as is.
set -euo pipefail
V="${1:?usage: check_gpu_install.sh VERSION [WORKDIR]}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AUTO_WORK=0
WORK="${2:-}"
if [ -z "$WORK" ]; then
  WORK="$(mktemp -d)"
  AUTO_WORK=1  # ours to remove when done; a WORKDIR named on the command line is the caller's
fi
mkdir -p "$WORK"
MODEL="Qwen/Qwen3-0.6B"

step() { echo "[check_gpu_install] $*"; }

verdict=FAILED
cleanup() {
  step "verdict: ${verdict}: glyd[gpu]==$V, $MODEL"
  [ "$AUTO_WORK" != 1 ] || rm -rf "$WORK"
}
trap cleanup EXIT

command -v nvidia-smi >/dev/null || { step "no nvidia-smi: not a machine with an NVIDIA GPU"; exit 1; }
step "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"

step "fresh venv: $WORK/venv"
UV="$(command -v uv 2>/dev/null || true)"
if [ -z "$UV" ] && [ -x "$HOME/tools/uv/uv" ]; then
  UV="$HOME/tools/uv/uv"
fi
if [ -n "$UV" ]; then
  "$UV" venv -q --python 3.12 "$WORK/venv"  # a plain venv either way; uv sidesteps a system Python with no ensurepip/venv package
else
  python3 -m venv "$WORK/venv"
fi
PY="$WORK/venv/bin/python"
pip_install() { # a uv-made venv has no pip module in it: install through uv itself, by --python, when it made the venv
  if [ -n "$UV" ]; then "$UV" pip install -q --python "$PY" "$@"; else "$PY" -m pip install --quiet "$@"; fi
}
[ -n "$UV" ] || pip_install --upgrade pip

step "pip install glyd[gpu]==$V from PyPI (retry: PyPI can lag the release)"
ok=0
for i in $(seq 1 80); do
  pip_install "glyd[gpu]==$V" && { ok=1; break; }
  step "glyd[gpu]==$V not resolvable yet on PyPI (attempt $i/80); retrying in 30s"
  sleep 30
done
[ "$ok" = 1 ] || { step "glyd[gpu]==$V never became installable from PyPI"; exit 1; }

got=$("$PY" -c "import glyd; print(glyd.__version__)")
[ "$got" = "$V" ] || { step "installed glyd is $got, expected $V"; exit 1; }
step "glyd.__version__ == $V"

step "gpu/check_api.py on ${MODEL}: from_pretrained generating, exact=True bit for bit with bf16, save_pretrained and load back"
cp "$ROOT/gpu/check_api.py" "$WORK/check_api.py"
"$PY" "$WORK/check_api.py" "$MODEL"

verdict=PASSED
