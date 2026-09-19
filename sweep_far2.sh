#!/usr/bin/env bash
set -uo pipefail
export CARGO_TARGET_DIR=/tmp
OUT=far_sweep_exact.txt
[ -f "$OUT" ] || echo "config,total_ratio" > "$OUT"
run() { # $1=label $2=window_bits $3=far_min
  grep -q "^$1," "$OUT" && return
  sed -i -E "s/pub const WINDOW_SIZE: usize = 1 << [0-9]+;/pub const WINDOW_SIZE: usize = 1 << $2;/" src/format.rs
  sed -i -E "s/pub const FAR_MIN_MATCH_LEN: usize = [0-9]+;/pub const FAR_MIN_MATCH_LEN: usize = $3;/" src/format.rs
  RUSTFLAGS="-C target-cpu=native" cargo build --release --example ratio_only 2>&1 | grep -E '^error' && return
  R=$("$CARGO_TARGET_DIR/release/examples/ratio_only" | awk '{print $2}')
  echo "$1,$R" >> "$OUT"; sync; echo "$1 -> $R"
}
run "baseline_64KB_nofar" 16 12
run "w1MB_far12"  20 12
run "w1MB_far24"  20 24
run "w4MB_far24"  22 24
run "w4MB_far32"  22 32
run "w16MB_far32" 24 32
