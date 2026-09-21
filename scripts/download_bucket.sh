#!/usr/bin/env bash
# A realistic bucket (corpus/bucket/): objects that are near-copies of
# each other over time, as object storage holds them. Six builds of the
# Ubuntu 24.04 cloud root filesystem (1.1 GB each), the Linux 6.10 point
# releases (from download_versions.sh and download_chain.sh, linked in),
# two months of three Wikipedia tables, and twelve hours of GitHub
# events. About 35 GB unpacked. Best-effort.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DIR="$ROOT/corpus/bucket"
mkdir -p "$DIR"
get() { # get <dest> <url> <xz|gz|raw>
    local dest="$1" url="$2" kind="$3"
    if [ -s "$dest" ]; then echo "have $(basename "$dest")"; return; fi
    echo "downloading $(basename "$dest")"
    case "$kind" in
        xz) curl -fsSL --retry 3 "$url" | xz -dc > "$dest.part" ;;
        gz) curl -fsSL --retry 3 "$url" | gzip -dc > "$dest.part" ;;
        raw) curl -fsSL --retry 3 "$url" > "$dest.part" ;;
    esac
    if [ -s "$dest.part" ]; then mv "$dest.part" "$dest"; else echo "WARNING: $dest failed" >&2; rm -f "$dest.part"; fi
}
for d in 20260705 20260725 20260801 20260814 20260826 20260911; do
    get "$DIR/noble-$d-root.tar" "https://cloud-images.ubuntu.com/noble/$d/noble-server-cloudimg-amd64-root.tar.xz" xz
done
for f in "$ROOT"/corpus/versions/linux-6.10*.tar; do
    [ -e "$DIR/$(basename "$f")" ] || ln "$f" "$DIR/$(basename "$f")" 2>/dev/null || cp "$f" "$DIR/"
done
for m in 20260801 20260901; do
    for t in page categorylinks page_props; do
        get "$DIR/simplewiki-$m-$t.sql" "https://dumps.wikimedia.org/simplewiki/$m/simplewiki-$m-$t.sql.gz" gz
    done
done
for h in 0 2 4 6 8 10 12 14 16 18 20 22; do
    get "$DIR/gharchive-2024-01-15-$h.json" "https://data.gharchive.org/2024-01-15-$h.json.gz" gz
done
ls -la "$DIR"
