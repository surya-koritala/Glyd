#!/usr/bin/env bash
# Round 1's Rust check (r1_job.sh's step d): Qwen3-0.6B packed by python -m glyd.gpu pack (the branch's package and
# library, on the GPU) and by glyd pack (the glyd-gpu crate, on the CPU), in the tiered layout and the 12-bit one (split
# byte, glyd-v3 with its "hb"), every file's sha256 compared; each Rust save verified by glyd verify on the CPU and on the
# GPU and by python -m glyd.gpu verify; the crate's examples on the tiered save (unpack.rs: a pack and a merged one
# against the checkpoint, bit for bit; linear.rs: a product by each route against the route's kernel).
#   PY=python BIN=target/release/glyd-gpu CRATE=src/glyd-gpu SNAP=<Qwen3-0.6B's snapshot> O=<out dir> bash r1_rust.sh
set -u
: "${PY:?}" "${BIN:?}" "${CRATE:?}" "${SNAP:?}" "${O:?}"
fail() { echo "FAIL: $*"; exit 1; }
rm -rf "$O" && mkdir -p "$O"
for layout in mma mma12; do
  echo "== $layout"
  "$PY" -m glyd.gpu pack Qwen/Qwen3-0.6B "$O/py-$layout" --layout "$layout" > "$O/py-$layout.log" 2>&1 || { tail -5 "$O/py-$layout.log"; fail "python -m glyd.gpu pack, $layout"; }
  grep -v "Loading weights\|Fetching" "$O/py-$layout.log" | tail -2
  "$BIN" pack Qwen/Qwen3-0.6B "$O/rs-$layout" --layout "$layout" || fail "glyd pack, $layout"
  ( cd "$O/py-$layout" && sha256sum * ) > "$O/py-$layout.sha256"
  ( cd "$O/rs-$layout" && sha256sum * ) > "$O/rs-$layout.sha256"
  if cmp -s "$O/py-$layout.sha256" "$O/rs-$layout.sha256"; then
    echo "$layout: every file byte-identical to Python's ($(wc -l < "$O/rs-$layout.sha256") files)"
  else
    diff "$O/py-$layout.sha256" "$O/rs-$layout.sha256"; fail "$layout: DIFFERS from Python's save"
  fi
  "$BIN" verify "$O/rs-$layout" || fail "glyd verify, $layout, on the CPU"
  "$BIN" verify "$O/rs-$layout" --device cuda:0 || fail "glyd verify, $layout, on the GPU"
  "$PY" -m glyd.gpu verify "$O/rs-$layout" > "$O/pyverify-$layout.log" 2>&1 || { tail -5 "$O/pyverify-$layout.log"; fail "python -m glyd.gpu verify, $layout"; }
  tail -1 "$O/pyverify-$layout.log"
done
cd "$CRATE" || fail "no crate at $CRATE"
cargo run -q --release --example unpack -- "$O/rs-mma" "$SNAP" || fail "examples/unpack.rs, the first pack"
cargo run -q --release --example unpack -- "$O/rs-mma" "$SNAP" model.layers.0.self_attn.q_proj || fail "examples/unpack.rs, a merged pack"
cargo run -q --release --example linear -- "$O/rs-mma" model.layers.0.mlp.gate_proj || fail "examples/linear.rs"
echo "Rust: all passed"
