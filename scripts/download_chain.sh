#!/usr/bin/env bash
# The Linux 6.10 point releases after the pair in download_versions.sh
# (corpus/versions/linux-6.10.N.tar, N = 2..14): a chain of versions for
# measuring base mode over many steps. About 20 GB unpacked.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/corpus/versions"
mkdir -p "$DIR"
for n in $(seq 2 14); do
    dest="$DIR/linux-6.10.$n.tar"
    if [ -s "$dest" ]; then echo "have $(basename "$dest")"; continue; fi
    echo "downloading $(basename "$dest")"
    curl -fsSL --retry 3 "https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.10.$n.tar.xz" | xz -dc > "$dest.part"
    if [ -s "$dest.part" ]; then mv "$dest.part" "$dest"; else echo "WARNING: $dest failed" >&2; rm -f "$dest.part"; fi
done
ls -la "$DIR"
