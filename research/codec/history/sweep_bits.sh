#!/usr/bin/env bash
set -uo pipefail
export CARGO_TARGET_DIR=/tmp
for B in 14 15 16 17; do
  grep -q "^h${B}," quick3_log.csv 2>/dev/null && continue
  sed -i -E "s/pub const HASH_BITS: u32 = [0-9]+;/pub const HASH_BITS: u32 = ${B};/" src/x86_compress.rs
  RUSTFLAGS="-C target-cpu=native" cargo build --release --example quick3 2>&1 | grep -E '^error' && continue
  "$CARGO_TARGET_DIR/release/examples/quick3" "h${B}" 3 0.3
done
