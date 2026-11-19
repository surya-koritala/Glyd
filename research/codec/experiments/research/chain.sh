#!/usr/bin/env bash
# Lever A: version chains. Linux 6.10 -> 6.10.1 -> ... -> 6.10.5: each
# version against the previous one (a chain) and against 6.10 (a fixed
# base), Glyd --max --base and zstd -3/-19 --patch-from; plus the whole
# set stored alone. Sizes and times; every rebuild byte-exact.
set -euo pipefail
R="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
V="$R/corpus/versions"; W="$R/corpus/research"
T="$(sysctl -n hw.ncpu 2>/dev/null || nproc)"
vers=("$V/linux-6.10.tar" "$V/linux-6.10.1.tar" "$W/linux-6.10.2.tar" "$W/linux-6.10.3.tar" "$W/linux-6.10.4.tar" "$W/linux-6.10.5.tar")
size() { stat -f %z "$1" 2>/dev/null || stat -c %s "$1"; }
now() { python3 -c 'import time; print(time.time())'; }
echo "version           alone(zstd-19)   chain glyd-max   chain zstd-3     chain zstd-19    vs-6.10 glyd-max"
for i in 1 2 3 4 5; do
    new="${vers[$i]}"; prev="${vers[$((i-1))]}"; base0="${vers[0]}"
    alone=$(zstd -q -19 -T"$T" -c "$new" | wc -c | tr -d ' ')
    g=$("$R/target/release/examples/base_delta" "$prev" "$new" | sed -E 's/.*base max: ([0-9]+) B.*compress ([0-9.]+) s.*/\1 (\2 s)/')
    z3=$(zstd -q -3 -T"$T" --long=31 --patch-from="$prev" -c "$new" | wc -c | tr -d ' ')
    t0=$(now); z19=$(zstd -q -19 -T"$T" --long=31 --patch-from="$prev" -c "$new" | wc -c | tr -d ' '); t1=$(now)
    g0=$("$R/target/release/examples/base_delta" "$base0" "$new" | sed -E 's/.*base max: ([0-9]+) B.*/\1/')
    printf "%-16s %14s   %-16s %-16s %-16s %s\n" "$(basename "$new")" "$alone" "$g" "$z3" "$z19 ($(printf '%.0f' $(echo "$t1 - $t0" | bc -l)) s)" "$g0"
done
