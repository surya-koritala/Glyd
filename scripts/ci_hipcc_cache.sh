#!/usr/bin/env bash
# Stands in for hipcc in CI (HIPCC=scripts/ci_hipcc_cache.sh; glyd-engine/build.rs runs `$HIPCC ... -o OUT ... kernels/engine.cu`): when the bundle of code objects a
# compile of these kernels made is in $HIP_BUNDLE_DIR, it is OUT and hipcc does not run (a change to the engine's Rust code then costs no kernel compile);
# otherwise the real compiler ($HIP_REAL) runs and its bundle is kept there for the next run.
set -euo pipefail
out=""
prev=""
for a in "$@"; do
    [ "$prev" = "-o" ] && out="$a"
    prev="$a"
done
[ -n "$out" ] || { echo "ci_hipcc_cache.sh: no -o in the arguments" >&2; exit 1; }
bundle="$HIP_BUNDLE_DIR/engine.bundle"
if [ -f "$bundle" ]; then
    cp "$bundle" "$out"
    exit 0
fi
"$HIP_REAL" "$@"
mkdir -p "$HIP_BUNDLE_DIR"
cp "$out" "$bundle"
