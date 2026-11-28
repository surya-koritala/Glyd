#!/bin/bash
# GLY-363: the beta download for Linux x86_64 + NVIDIA (CUDA 13 driver): starlings beside the
# engine (glyd-engine and LICENSE, from an unpacked engine bundle of release-gpu.yml: the engine
# loads no GPU library, so none is packaged), with an installer and checksums.
# Usage: scripts/package-beta.sh STARLINGS ENGINE_DIR OUT_DIR
set -euo pipefail
[ $# -eq 3 ] || { echo "usage: $0 STARLINGS ENGINE_DIR OUT_DIR" >&2; exit 2; }
[ -x "$1" ] || { echo "$1 is not an executable starlings" >&2; exit 2; }
for f in glyd-engine LICENSE; do
    [ -f "$2/$f" ] || { echo "$2/$f is missing (ENGINE_DIR is an unpacked glyd-engine-vX.Y.Z-linux-x86_64 bundle)" >&2; exit 2; }
done
version=$("$1" --version | awk '{print $2}')
name=starlings-$version-linux-x86_64-cuda13
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
mkdir "$work/$name"
install -m 755 "$1" "$work/$name/starlings"
install -m 755 "$2/glyd-engine" "$work/$name/glyd-engine"
install -m 644 "$2/LICENSE" "$work/$name/LICENSE-glyd-engine"
install -m 755 "$(dirname "$0")/install.sh" "$work/$name/install.sh"
(cd "$work/$name" && sha256sum starlings glyd-engine LICENSE-glyd-engine install.sh > SHA256SUMS)
mkdir -p "$3"
tar --owner=0 --group=0 --numeric-owner -C "$work" -czf "$3/$name.tar.gz" "$name"
(cd "$3" && sha256sum "$name.tar.gz" > "$name.tar.gz.sha256")
echo "$3/$name.tar.gz"
