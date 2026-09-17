#!/usr/bin/env bash
# Sweep far-match threshold x window size. Appends durably (host crashes).
set -uo pipefail
export CARGO_TARGET_DIR=/tmp/claude-1000/-home-surya/ae7f39bc-ce5c-4579-9557-fb26287a402a/scratchpad/alk-target
OUT=far_sweep.txt
[ -f "$OUT" ] || echo "window_bits,far_min,total_ratio" > "$OUT"
for WB in 20 22; do
  for FM in 12 16 24 32 48; do
    grep -q "^${WB},${FM}," "$OUT" && continue   # resume: skip done pairs
    sed -i -E "s/pub const WINDOW_SIZE: usize = 1 << [0-9]+;/pub const WINDOW_SIZE: usize = 1 << ${WB};/" src/format.rs
    sed -i -E "s/pub const FAR_MIN_MATCH_LEN: usize = [0-9]+;/pub const FAR_MIN_MATCH_LEN: usize = ${FM};/" src/format.rs
    RUSTFLAGS="-C target-cpu=native" cargo build --release --test regression_floors 2>&1 | grep -E '^error' && continue
    R=$(RUSTFLAGS="-C target-cpu=native" cargo test --release --test regression_floors -- --nocapture 2>&1 \
        | grep 'TOTAL Silesia ratio' | grep -oE '[0-9]+\.[0-9]+' | head -1)
    echo "${WB},${FM},${R}" >> "$OUT"; sync
    echo "window=2^${WB} far_min=${FM} -> ratio ${R}"
  done
done
