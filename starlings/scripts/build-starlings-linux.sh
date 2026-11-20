#!/bin/bash
# A release starlings for Linux x86_64 linked against glibc 2.28, as the engine release is, so the
# beta runs on Ubuntu 20.04 and newer. Builds in a Rocky Linux 8 container as the calling user.
# Usage: scripts/build-starlings-linux.sh OUT_DIR [IMAGE]
set -euo pipefail
out=$(mkdir -p "$1" && cd "$1" && pwd)
image=${2:-nvidia/cuda:13.0.3-devel-rockylinux8}
src=$(cd "$(dirname "$0")/../.." && pwd)  # (the repository root: the workspace and its rust-toolchain.toml)
cache=${STARLINGS_BUILD_CACHE:-$HOME/.cache/starlings-build}
mkdir -p "$cache"
toolchain=$(sed -n 's/^channel *= *"\(.*\)"/\1/p' "$src/rust-toolchain.toml")
docker run --rm --user "$(id -u):$(id -g)" \
    -e HOME=/cache -e RUSTUP_HOME=/cache/rustup -e CARGO_HOME=/cache/cargo \
    -e CARGO_TARGET_DIR=/cache/target -v "$src:/src:ro" -v "$cache:/cache" -v "$out:/out" \
    -w /src "$image" bash -euo pipefail -c "
        [ -x /cache/cargo/bin/rustup ] || curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none --no-modify-path
        /cache/cargo/bin/rustup toolchain install $toolchain --profile minimal
        /cache/cargo/bin/cargo +$toolchain build --release --locked -p starlings-cli --bin starlings
        install -m 755 /cache/target/release/starlings /out/starlings"
top=$(objdump -T "$out/starlings" | grep -oE 'GLIBC_[0-9]+(\.[0-9]+)*' | sed 's/GLIBC_//' | sort -uV | tail -1)
echo "$out/starlings: glibc up to $top"
[ "$(printf '%s\n' "$top" 2.28 | sort -V | tail -1)" = 2.28 ] || { echo "needs glibc $top, past 2.28" >&2; exit 1; }
