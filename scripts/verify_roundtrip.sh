#!/usr/bin/env bash
# Exact recovery through the CLI: every file given (default: the whole
# benchmark corpus) at every level, single- and multi-core, must
# decompress to the original bytes (cmp), record mode included, and corrupted copies of the
# compressed file (bit flips, byte overwrites, truncations) must be
# rejected or, if their checksum still passes, decode to the original
# bytes. Exit status is non-zero on any failure.
#   scripts/verify_roundtrip.sh [file ...]
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GLYD="$ROOT/target/release/glyd"
[ -x "$GLYD" ] || cargo build --release --manifest-path "$ROOT/Cargo.toml" >/dev/null
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
FILES=("$@")
[ ${#FILES[@]} -gt 0 ] || FILES=("$ROOT"/corpus/bench/*.{json,log,sql,parquet})
fail=0; ok=0
corrupt() { # corrupt <in> <out> <seed>: one mutation
    python3 - "$1" "$2" "$3" <<'PY'
import sys, random
src, dst, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
d = bytearray(open(src, 'rb').read()); r = random.Random(seed)
k = r.randrange(4)
if k == 0: i = r.randrange(len(d)); d[i] ^= 1 << r.randrange(8)
elif k == 1: i = r.randrange(len(d)); d[i] = r.randrange(256)
elif k == 2: d = d[:r.randrange(len(d))]
else:
    i = r.randrange(len(d)); n = r.randrange(1, 64); d[i:i+n] = bytes(r.randrange(256) for _ in range(n))
open(dst, 'wb').write(d)
PY
}
for f in "${FILES[@]}"; do
    [ -f "$f" ] || continue
    name="$(basename "$f")"
    for level in "-t" "-1" "" "--max" "--ultra" "--max -r" "--ultra -r"; do
        for cores in "-s" "-m"; do
            label="$name level=${level:-default} $cores"
            if ! "$GLYD" $level $cores -c "$f" -o "$TMP/c.glyd" 2>"$TMP/err"; then echo "FAIL compress: $label: $(cat "$TMP/err")"; fail=$((fail+1)); continue; fi
            if ! "$GLYD" -d $cores "$TMP/c.glyd" -o "$TMP/d.bin" 2>"$TMP/err"; then echo "FAIL decompress: $label: $(cat "$TMP/err")"; fail=$((fail+1)); continue; fi
            if ! cmp -s "$TMP/d.bin" "$f"; then echo "FAIL bytes differ: $label"; fail=$((fail+1)); continue; fi
            ok=$((ok+1))
            echo "ok   $label  $(stat -f%z "$f" 2>/dev/null || stat -c%s "$f") -> $(stat -f%z "$TMP/c.glyd" 2>/dev/null || stat -c%s "$TMP/c.glyd")"
        done
        # Corruption: 8 mutations of the multi-core file.
        for seed in 1 2 3 4 5 6 7 8; do
            corrupt "$TMP/c.glyd" "$TMP/bad.glyd" "$seed"
            if "$GLYD" -d "$TMP/bad.glyd" -o "$TMP/bad.bin" 2>/dev/null; then
                if ! cmp -s "$TMP/bad.bin" "$f"; then echo "FAIL corrupted file accepted with wrong bytes: $name level=${level:-default} seed=$seed"; fail=$((fail+1)); fi
            fi
        done
    done
done
echo "round trips ok: $ok, failures: $fail"
[ "$fail" -eq 0 ]
