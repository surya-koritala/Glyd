#!/usr/bin/env bash
# Lever C: the cold tier. Stronger-than-zstd-19 codecs on 64 MB slices
# (xz -9, brotli -q 11 with a 24-bit window, zpaq -m5 context mixing)
# against Glyd --ultra (-r where it applies) and zstd -19: size and
# compress time. What paying 10-100x the CPU buys.
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
now() { python3 -c 'import time; print(time.time())'; }
for f in "$R/corpus/bench/gharchive-2024-01-15-12.json" "$R/corpus/bench/nasa-access-jul95.log" "$R/corpus/bench/enwiki-page_props.sql" "$R/corpus/webster"; do
    head -c 67108864 "$f" > "$TMP/s.bin"; n=$(size "$TMP/s.bin")
    echo "== $(basename "$f") ($n bytes)"
    for c in "zstd -19 -T1" "xz -9 -T1" "brotli -q 11 --large_window=24" "zpaq -m5"; do
        t0=$(now)
        case "$c" in
            zpaq*) rm -f "$TMP/o.zpaq"; zpaq add "$TMP/o.zpaq" "$TMP/s.bin" -m5 -threads 1 >/dev/null 2>&1; out="$TMP/o.zpaq" ;;
            brotli*) brotli -q 11 --large_window=24 -c "$TMP/s.bin" > "$TMP/o.br"; out="$TMP/o.br" ;;
            xz*) xz -9 -T1 -c "$TMP/s.bin" > "$TMP/o.xz"; out="$TMP/o.xz" ;;
            zstd*) zstd -q -19 -T1 -c "$TMP/s.bin" > "$TMP/o.zst"; out="$TMP/o.zst" ;;
        esac
        t1=$(now)
        printf "   %-32s %12s B  %7.1f s  %5.2fx\n" "$c" "$(size "$out")" "$(echo "$t1 - $t0" | bc -l)" "$(echo "$n / $(size "$out")" | bc -l)"
    done
    for lvl in "--ultra" "--ultra -r"; do
        t0=$(now); "$R/target/release/glyd" $lvl -s "$TMP/s.bin" -o "$TMP/o.glyd" >/dev/null 2>&1; t1=$(now)
        printf "   %-32s %12s B  %7.1f s  %5.2fx\n" "Glyd $lvl" "$(size "$TMP/o.glyd")" "$(echo "$t1 - $t0" | bc -l)" "$(echo "$n / $(size "$TMP/o.glyd")" | bc -l)"
    done
done
