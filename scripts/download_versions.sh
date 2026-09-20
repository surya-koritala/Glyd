#!/usr/bin/env bash
# Pairs of consecutive versions of real objects for base mode
# (corpus/versions/): two Wikipedia dumps of one table a month apart,
# two Linux point releases, two builds of the Ubuntu 24.04 cloud root
# filesystem. About 5 GB unpacked. Best-effort, like the other scripts.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/corpus/versions"
mkdir -p "$DIR"
get() { # get <dest> <url> <xz|gz>
    local dest="$1" url="$2" kind="$3"
    if [ -s "$dest" ]; then echo "have $(basename "$dest")"; return; fi
    echo "downloading $(basename "$dest") from $url"
    case "$kind" in
        xz) curl -fsSL --retry 3 "$url" | xz -dc > "$dest.part" ;;
        gz) curl -fsSL --retry 3 "$url" | gzip -dc > "$dest.part" ;;
    esac
    if [ -s "$dest.part" ]; then mv "$dest.part" "$dest"; else echo "WARNING: $dest failed" >&2; rm -f "$dest.part"; fi
}
get "$DIR/simplewiki-20260801-page.sql" https://dumps.wikimedia.org/simplewiki/20260801/simplewiki-20260801-page.sql.gz gz
get "$DIR/simplewiki-20260901-page.sql" https://dumps.wikimedia.org/simplewiki/20260901/simplewiki-20260901-page.sql.gz gz
get "$DIR/linux-6.10.tar" https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.10.tar.xz xz
get "$DIR/linux-6.10.1.tar" https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.10.1.tar.xz xz
get "$DIR/noble-20260826-root.tar" https://cloud-images.ubuntu.com/noble/20260826/noble-server-cloudimg-amd64-root.tar.xz xz
get "$DIR/noble-20260911-root.tar" https://cloud-images.ubuntu.com/noble/20260911/noble-server-cloudimg-amd64-root.tar.xz xz
ls -la "$DIR"
