#!/usr/bin/env bash
# The terabyte bucket for the store's gate run: about 1 TB of public
# objects that are versions of each other, as object storage holds
# them. Linux point releases (6.6.1-150, 6.1.1-150, 5.15.1-100, ~1.4 GB
# each), every hour of GitHub events in January 2024 (744 files), five
# English Wikipedia dumps' page, page_props and categorylinks tables,
# nine Simple English dumps' three tables, and the six Ubuntu 24.04
# cloud images on the mirror. Best-effort: a failed download is logged
# and skipped. Parallel where the mirror allows (Wikimedia asks for two
# connections).
#   scripts/gate_corpus.sh <dir>
set -uo pipefail
DIR="${1:?a directory}"
mkdir -p "$DIR"
GUNZIP="gzip -dc"; command -v pigz >/dev/null && GUNZIP="pigz -dc"
get() { # get <dest> <url> <xz|gz|raw>
    local dest="$1" url="$2" kind="$3"
    if [ -s "$dest" ]; then return; fi
    case "$kind" in
        xz) curl -fsSL --retry 3 "$url" | xz -dc > "$dest.part" ;;
        gz) curl -fsSL --retry 3 "$url" | $GUNZIP > "$dest.part" ;;
        raw) curl -fsSL --retry 3 "$url" > "$dest.part" ;;
    esac
    if [ -s "$dest.part" ]; then mv "$dest.part" "$dest"; echo "got $(basename "$dest")"; else echo "FAILED $dest" >&2; rm -f "$dest.part"; fi
}
export -f get; export GUNZIP DIR

# Kernels, eight at a time (xz on one core each).
{
    for n in $(seq 1 150); do echo "$DIR/linux-6.6.$n.tar https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.6.$n.tar.xz xz"; done
    for n in $(seq 1 150); do echo "$DIR/linux-6.1.$n.tar https://cdn.kernel.org/pub/linux/kernel/v6.x/linux-6.1.$n.tar.xz xz"; done
    for n in $(seq 1 100); do echo "$DIR/linux-5.15.$n.tar https://cdn.kernel.org/pub/linux/kernel/v5.x/linux-5.15.$n.tar.xz xz"; done
} | xargs -P 8 -L 1 bash -c 'get "$0" "$1" "$2"' &

# GitHub events, eight at a time.
{
    for d in $(seq -w 1 31); do for h in $(seq 0 23); do
        echo "$DIR/gharchive-2024-01-$d-$h.json https://data.gharchive.org/2024-01-$d-$h.json.gz gz"
    done; done
} | xargs -P 8 -L 1 bash -c 'get "$0" "$1" "$2"' &

# Wikimedia: two connections.
{
    for m in 20260501 20260601 20260701 20260801 20260901; do
        for t in page page_props categorylinks; do
            echo "$DIR/enwiki-$m-$t.sql https://dumps.wikimedia.org/enwiki/$m/enwiki-$m-$t.sql.gz gz"
        done
    done
    for m in 20260101 20260201 20260301 20260401 20260501 20260601 20260701 20260801 20260901; do
        for t in page page_props categorylinks; do
            echo "$DIR/simplewiki-$m-$t.sql https://dumps.wikimedia.org/simplewiki/$m/simplewiki-$m-$t.sql.gz gz"
        done
    done
} | xargs -P 2 -L 1 bash -c 'get "$0" "$1" "$2"' &

# Ubuntu images: whatever dates the mirror holds.
{
    for d in $(curl -fsSL https://cloud-images.ubuntu.com/noble/ | grep -o 'href="20[0-9]*/' | tr -d 'href="/' | sort -u); do
        echo "$DIR/noble-$d-root.tar https://cloud-images.ubuntu.com/noble/$d/noble-server-cloudimg-amd64-root.tar.xz xz"
    done
} | xargs -P 2 -L 1 bash -c 'get "$0" "$1" "$2"' &

wait
echo "corpus: $(ls "$DIR" | wc -l) objects, $(du -sb "$DIR" | cut -f1) bytes"
