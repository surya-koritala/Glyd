#!/bin/sh
# Glyd's installer:  curl -LsSf https://getglyd.com/install.sh | sh
#
# Installs Glyd as an isolated tool (uv tool install) on a Python 3.12 that uv manages, so the system's Python is left alone: no virtual
# environment to make, no pip refusing to install (PEP 668), no Python version to choose, no C headers to find. On Linux with an NVIDIA
# GPU it installs the serving stack too (vLLM and PyTorch, several GB), which is what `glyd run MODEL` uses; elsewhere the compression
# tools alone. It installs uv first where there is none (uv's own installer), never uses sudo, and ends with `glyd doctor`.
#
#   GLYD_VERSION   the release to install (default below; a pre-release is named here, e.g. 0.26.0rc3, and only that one is taken)
#   GLYD_SPEC      the package to install instead, as uv takes it: a wheel with its extra ("/path/glyd-...whl[vllm]"), for another build
set -eu

GLYD_VERSION="${GLYD_VERSION:-0.26.0}"
PYTHON=3.12
DRIVER_MIN=580  # the NVIDIA driver vLLM 0.30's PyTorch (2.13, CUDA 13.0) runs on

say() { printf '==> %s\n' "$*"; }
warn() { printf 'glyd install: %s\n' "$*" >&2; }
die() { printf 'glyd install: %s\n' "$*" >&2; exit 1; }

os=$(uname -s)
arch=$(uname -m)
case "$os" in
  Linux | Darwin) ;;
  *) die "this installer is for Linux and macOS (on Windows, use WSL2 with an NVIDIA driver on Windows)." ;;
esac

gpu=no
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then gpu=yes; fi

if [ -n "${GLYD_SPEC:-}" ]; then
  spec=$GLYD_SPEC
elif [ "$os" = Linux ] && [ "$gpu" = yes ] && { [ "$arch" = x86_64 ] || [ "$arch" = aarch64 ]; }; then
  spec="glyd[vllm]==$GLYD_VERSION"
else
  spec="glyd==$GLYD_VERSION"
  if [ "$os" != Linux ]; then
    warn "glyd run needs Linux and an NVIDIA GPU; installing the compression tools only."
  elif [ "$gpu" = no ]; then
    warn "no NVIDIA GPU answered (nvidia-smi); installing the compression tools only. On a machine with an NVIDIA GPU and its driver, run this again to add glyd run."
  else
    warn "no vLLM build for $arch; installing the compression tools only."
  fi
fi

if [ "$gpu" = yes ]; then
  driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1 | tr -d ' ' || true)
  major=${driver%%.*}
  case "$major" in
    '' | *[!0-9]*) ;;
    *) if [ "$major" -lt "$DRIVER_MIN" ]; then
         warn "your NVIDIA driver ($driver) is older than $DRIVER_MIN, which vLLM's PyTorch needs. Installing anyway; before you run a model, update it (Ubuntu: sudo ubuntu-drivers install, then reboot)."
       fi ;;
  esac
fi

trap 'rc=$?; [ "$rc" = 0 ] || warn "the install did not finish (see above). Run the same command again: uv keeps what it downloaded."' EXIT
uv=
for c in "$(command -v uv 2>/dev/null || true)" "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv"; do
  if [ -n "$c" ] && [ -x "$c" ]; then uv=$c; break; fi
done
if [ -z "$uv" ]; then
  say "Installing uv, which manages Python for Glyd (its own installer, astral.sh)"
  command -v curl >/dev/null 2>&1 || die "curl is needed to install uv (Ubuntu: sudo apt install curl)."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  for c in "$HOME/.local/bin/uv" "$HOME/.cargo/bin/uv" "$(command -v uv 2>/dev/null || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then uv=$c; break; fi
  done
  [ -n "$uv" ] || die "uv installed, but it is not where its installer says; open a new terminal and run this again."
fi

case "$spec" in
  *vllm*) say "Installing $spec and Python $PYTHON (PyTorch and vLLM: several GB, a few minutes)" ;;
  *) say "Installing $spec and Python $PYTHON" ;;
esac
"$uv" tool install --force --managed-python --python "$PYTHON" "$spec"

bin=$("$uv" tool dir --bin)
"$uv" tool update-shell >/dev/null 2>&1 || true
first=$(command -v glyd 2>/dev/null || true)
if [ -n "$first" ] && [ "$first" != "$bin/glyd" ]; then
  warn "another glyd ($first) comes first on your PATH: it is the compression program, which has no 'run'. Use $bin/glyd, or put $bin first on your PATH."
fi

say "Checking this machine (glyd doctor)"
"$bin/glyd" doctor || say "Fix the lines marked NO above, then run: glyd doctor"
case "$PATH" in
  *"$bin"*) ;;
  *) say "Open a new terminal (or run: export PATH=\"$bin:\$PATH\") so that glyd is found." ;;
esac
