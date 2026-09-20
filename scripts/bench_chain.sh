#!/usr/bin/env bash
# A chain of versions (corpus/versions/linux-6.10.N.tar, N = 0..14, from
# download_versions.sh and download_chain.sh): each version against the
# one before it, and against the first, with Glyd --max base mode and
# zstd -3 --patch-from. Sizes in bytes, every rebuild byte-exact.
#   scripts/bench_chain.sh [last N, default 14]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
V="$ROOT/corpus/versions"
LAST="${1:-14}"
T="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
ver() { if [ "$1" = 0 ]; then echo "$V/linux-6.10.tar"; else echo "$V/linux-6.10.$1.tar"; fi; }
glyd() { "$ROOT/target/release/examples/base_delta" "$1" "$2" | sed -E 's/.*base max: ([0-9]+) B.*/\1/'; }
zpatch() { zstd -q -3 -T"$T" --long=31 --patch-from="$1" -c "$2" > "$TMP/p.zst"; zstd -q -d --long=31 --patch-from="$1" -c "$TMP/p.zst" | cmp -s - "$2" || echo "zstd DIFFERS" >&2; size "$TMP/p.zst"; }
plain=$("$ROOT/target/release/glyd" --max -c "$(ver 0)" | wc -c | tr -d ' ')
echo "linux-6.10 alone, Glyd --max: $plain B; zstd -3: $(zstd -q -3 -T"$T" -c "$(ver 0)" | wc -c | tr -d ' ') B ($T threads)"
printf "%-8s %14s %14s %14s %14s\n" version "glyd|prev" "glyd|6.10" "zstd3|prev" "zstd3|6.10"
for n in $(seq 1 "$LAST"); do
    new="$(ver "$n")"; prev="$(ver $((n - 1)))"; first="$(ver 0)"
    printf "%-8s %14s %14s %14s %14s\n" "6.10.$n" "$(glyd "$prev" "$new")" "$(glyd "$first" "$new")" "$(zpatch "$prev" "$new")" "$(zpatch "$first" "$new")"
done
