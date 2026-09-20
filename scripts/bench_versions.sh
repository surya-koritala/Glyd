#!/usr/bin/env bash
# Versions of an object: Glyd base mode against zstd --patch-from on
# pairs of consecutive versions (corpus/versions, see
# experiments/structure/README.md). Sizes and wall times, every rebuild
# byte-exact (Glyd's checked by examples/base_delta; zstd's by cmp).
#   scripts/bench_versions.sh old new [old new ...]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
T="$(nproc 2>/dev/null || sysctl -n hw.ncpu)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
now() { python3 -c 'import time; print(time.time())'; }
while [ $# -ge 2 ]; do
    old="$1"; new="$2"; shift 2
    n=$(size "$new"); wl=27; [ "$n" -gt $((1 << 27)) ] && wl=31
    echo "== $(basename "$old") -> $(basename "$new") ($n bytes, $T threads)"
    for lvl in 3 19; do
        t0=$(now); zstd -q -$lvl -T"$T" --long=$wl --patch-from="$old" -c "$new" > "$TMP/p.zst"; t1=$(now)
        zstd -q -d --long=$wl --patch-from="$old" -c "$TMP/p.zst" | cmp -s - "$new" && ok=exact || ok=DIFFERS
        printf "   zstd -%-2s --patch-from  %12s B  %8.2f s  %6.0f MB/s  %s\n" "$lvl" "$(size "$TMP/p.zst")" "$(echo "$t1 - $t0" | bc -l)" "$(echo "$n / ($t1 - $t0) / 1000000" | bc -l)" "$ok"
    done
    for level in max ultra; do
        "$ROOT/target/release/examples/base_delta" "$old" "$new" $level | sed -E 's/^.*: base /   Glyd --/; s/ \(plain[^)]*\)//'
    done
done
