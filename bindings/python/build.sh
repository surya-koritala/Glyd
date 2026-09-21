#!/usr/bin/env bash
# Build libglyd and place it in the package; then `pip install .` here.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cargo build --release --manifest-path "$ROOT/Cargo.toml"
case "$(uname -s)" in
    Darwin) cp "$ROOT/target/release/libglyd.dylib" "$ROOT/bindings/python/glyd/" ;;
    *) cp "$ROOT/target/release/libglyd.so" "$ROOT/bindings/python/glyd/" ;;
esac
echo "libglyd placed in bindings/python/glyd/; now: pip install $ROOT/bindings/python"
