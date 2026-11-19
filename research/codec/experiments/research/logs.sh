#!/usr/bin/env bash
# Lever B: system/application logs of other shapes (loghub 2.0: HDFS,
# BGL, Spark, Android). zstd -3/-19 against Glyd --max, --max -r,
# --ultra -r on up to 128 MB of each; whether record mode detects them,
# and the template prototype's estimate of what a smarter shape would get.
set -uo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
W="$R/corpus/research"
size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
for f in "$W/HDFS.log" "$W/BGL.log" "$W/Spark.log" "$W/Android.log" "$W/Linux.log"; do
    [ -f "$f" ] || { echo "missing $f"; continue; }
    head -c 134217728 "$f" > "$W/slice.bin"; n=$(size "$W/slice.bin")
    echo "== $(basename "$f") ($n bytes): $(head -c 300 "$f" | head -2 | cut -c1-110 | tr '\n' '|')"
    for c in "zstd -3 -T1" "zstd -19 -T1"; do
        printf "   %-16s %12s B  %5.2fx\n" "$c" "$($c -q -c "$W/slice.bin" | wc -c | tr -d ' ')" "$(echo "$n / $($c -q -c "$W/slice.bin" | wc -c | tr -d ' ')" | bc -l)"
    done
    "$R/target/release/examples/rec_trial" "$W/slice.bin" 2>&1 | sed -E 's|^.*: shape |   shape |'
    for lvl in "--max" "--max -r" "--ultra -r"; do
        "$R/target/release/glyd" $lvl -m "$W/slice.bin" -o "$W/o.glyd" >/dev/null 2>&1
        printf "   %-16s %12s B  %5.2fx\n" "Glyd $lvl" "$(size "$W/o.glyd")" "$(echo "$n / $(size "$W/o.glyd")" | bc -l)"
    done
    head -c 33554432 "$W/slice.bin" > "$W/slice32.bin"
    python3 "$R/experiments/structure/generic_templates.py" "$W/slice32.bin" 2>&1 | tail -2 | sed 's/^/   templates: /'
done
rm -f "$W/slice.bin" "$W/slice32.bin" "$W/o.glyd"
