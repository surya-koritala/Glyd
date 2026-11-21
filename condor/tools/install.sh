#!/usr/bin/env bash
# Install Glyd's engine for Condor under $GLYD_QWEN (default ~/.glyd-qwen-3.8-flash): the glyd-engine program and its GPU library.
# Linux x86-64 or aarch64, an NVIDIA GPU (Ampere or newer) with driver 580+. No sudo.
#
#   tools/install.sh [VERSION]     prebuilt: the engine bundle of a Glyd release (default: the latest), built by the release workflow
#                                  from this repository's source (.github/workflows/release-gpu.yml)
#   tools/install.sh --source      build it here from this checkout: Rust (rustup) and the CUDA toolkit (nvcc 12.8+), about 10 minutes
#
# GLYD_RELEASE_URL: a mirror, or file:///dir, holding glyd-engine-vX.Y.Z-linux-ARCH.tar.gz and its .sha256 (a test before a release).
# GLYD_ENGINE_TARBALL: a local glyd-engine-vX.Y.Z-linux-ARCH.tar.gz to install as it is.
set -euo pipefail
HERE=$(cd "$(dirname "$0")/.." && pwd)          # condor/
ROOT=$(cd "$HERE/.." && pwd)                    # the Glyd checkout it lives in
H=${GLYD_QWEN:-$HOME/.glyd-qwen-3.8-flash}; mkdir -p "$H/bin" "$H/lib"
REPO=surya-koritala/Glyd
LEGACY=https://github.com/surya-koritala/condor/releases/download/v0.2.4   # Condor's last separately built engine (same source, before the merge)
VER=${1:-latest}
command -v nvidia-smi > /dev/null || { echo "no nvidia-smi: an NVIDIA driver is needed" >&2; exit 1; }
case $(uname -m) in x86_64|amd64) ARCH=x86_64;; aarch64|arm64) ARCH=aarch64;; *) echo "unsupported CPU: $(uname -m)" >&2; exit 1;; esac
LIB=libglyd_gpu_cuda13.so

write_env() {
  cat > "$H/env.sh" <<ENV
export GLYD_QWEN=$H GLYD_REPO=$HERE GLYD_ENGINE=$H/bin/glyd-engine GLYD_GPU_LIB=$H/lib/$LIB
ENV
  "$H/bin/glyd-engine" --version
  echo "installed. Next: tools/download.sh, then tools/serve.sh"
}

from_source() {
  command -v cargo > /dev/null || { echo "--source needs Rust: https://rustup.rs" >&2; exit 1; }
  local nvcc=${NVCC:-$(command -v nvcc || true)}
  [ -n "$nvcc" ] || [ -x "${CUDA_HOME:-/usr/local/cuda}/bin/nvcc" ] || { echo "--source needs the CUDA toolkit (nvcc 12.8 or newer; set CUDA_HOME)" >&2; exit 1; }
  # (the machine code for this computer's GPUs, plus Ampere's PTX for any other)
  local caps; caps=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d '.' | sort -u | tr '\n' ' ')
  [ "$caps" = "${caps/90 /}" ] || caps="${caps/90 /90a }"
  echo "building glyd-engine for sm: $caps(from $ROOT)"
  GLYD_ENGINE_ARCHS="$caps" cargo build --release --locked --manifest-path "$ROOT/Cargo.toml" -p glyd-engine
  local out; out=$(mktemp -d)
  bash "$ROOT/gpu/build_lib.sh" "$out"
  LIB=$(cd "$out" && ls libglyd_gpu_cuda*.so | head -1)
  install -m 755 "$ROOT/target/release/glyd-engine" "$H/bin/glyd-engine"
  install -m 644 "$out/$LIB" "$H/lib/$LIB"
  rm -rf "$out"
  echo source > "$H/ENGINE_RELEASE"
  write_env
}

from_tarball() {   # glyd-engine-vX.Y.Z-linux-ARCH.tar.gz: glyd-engine, libglyd_gpu_cuda13.so, LICENSE
  local tgz=$1 tmp; tmp=$(mktemp -d)
  tar -xzf "$tgz" -C "$tmp"
  local d; d=$(find "$tmp" -mindepth 1 -maxdepth 1 -type d | head -1)
  install -m 755 "$d/glyd-engine" "$H/bin/glyd-engine"
  install -m 644 "$d/$LIB" "$H/lib/$LIB"
  # a bundle from 0.33.2 on also holds the CUDA 12 library (not compressed), which a driver older than R580 (CUDA 13) takes
  if [ -f "$d/libglyd_gpu_cuda12.so" ]; then
    install -m 644 "$d/libglyd_gpu_cuda12.so" "$H/lib/libglyd_gpu_cuda12.so"
    drv=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2> /dev/null | head -n 1 | cut -d. -f1 || true)
    case $drv in '' | *[!0-9]*) ;; *) [ "$drv" -ge 580 ] || LIB=libglyd_gpu_cuda12.so ;; esac
  fi
  rm -rf "$tmp"
  write_env
}

from_release() {
  local base tag name
  if [ -n "${GLYD_RELEASE_URL:-}" ]; then
    base=$GLYD_RELEASE_URL; tag=${VER#v}; [ "$VER" != latest ] || { echo "GLYD_RELEASE_URL needs a VERSION" >&2; exit 1; }; tag=v$tag
  else
    if [ "$VER" = latest ]; then
      tag=$(curl -fsSLI -o /dev/null -w '%{url_effective}' "https://github.com/$REPO/releases/latest" | sed 's#.*/tag/##')
    else tag=v${VER#v}; fi
    base=https://github.com/$REPO/releases/download/$tag
  fi
  name=glyd-engine-$tag-linux-$ARCH.tar.gz
  cd "$H"
  if curl -fsSL --retry 3 -o "$name" "$base/$name" && curl -fsSL --retry 3 -o "$name.sha256" "$base/$name.sha256"; then
    sha256sum -c --quiet "$name.sha256" || { echo "$name does not match its sha256" >&2; exit 1; }
    echo "$tag" > "$H/ENGINE_RELEASE"
    from_tarball "$H/$name"; rm -f "$H/$name" "$H/$name.sha256"
    return
  fi
  rm -f "$name" "$name.sha256"
  [ "$ARCH" = x86_64 ] || { echo "Glyd $tag has no engine for linux-$ARCH: run tools/install.sh --source" >&2; exit 1; }
  # (Glyd releases before the open-source merge carry no engine bundle: Condor's own last build of the same source)
  echo "Glyd $tag has no engine bundle yet; installing Condor's last engine build (v0.2.4). tools/install.sh --source builds it here instead."
  for f in glyd-engine $LIB SHA256SUMS; do curl -fsSL --retry 3 -o "$f.new" "$LEGACY/$f" && mv "$f.new" "$f"; done
  grep -q ' glyd-engine$' SHA256SUMS && grep -q " ${LIB//./\\.}\$" SHA256SUMS && sha256sum -c --quiet --ignore-missing SHA256SUMS ||
    { echo "glyd-engine or $LIB does not match SHA256SUMS" >&2; exit 1; }
  mv glyd-engine bin/ && chmod +x bin/glyd-engine && mv "$LIB" lib/
  echo "$tag" > "$H/ENGINE_RELEASE"   # (the Glyd release looked at: a later one with a bundle replaces this build)
  write_env
}

if [ "$VER" = --source ]; then from_source
elif [ -n "${GLYD_ENGINE_TARBALL:-}" ]; then from_tarball "$GLYD_ENGINE_TARBALL"
else from_release; fi
