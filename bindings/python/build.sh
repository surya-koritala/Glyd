#!/usr/bin/env bash
# Build libglyd and place it in the package; then `pip install .` here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# The store crate's library carries the codec's ABI too, so one file
# serves both; build the codec alone (cargo build --release -p glyd) and
# copy libglyd instead for a BSD/GPL-only package without the store.
cargo build --release --manifest-path "$ROOT/Cargo.toml" -p glyd-store
cp "$ROOT/LICENSE" "$ROOT/COPYING" "$ROOT/bindings/python/"
cp "$ROOT/glyd-store/LICENSE" "$ROOT/bindings/python/LICENSE-glyd-store"
case "$(uname -s)" in
    Darwin) cp "$ROOT/target/release/libglyd_store.dylib" "$ROOT/bindings/python/glyd/" ;;
    *) cp "$ROOT/target/release/libglyd_store.so" "$ROOT/bindings/python/glyd/" ;;
esac
echo "libglyd_store placed in bindings/python/glyd/; now: pip install $ROOT/bindings/python"
