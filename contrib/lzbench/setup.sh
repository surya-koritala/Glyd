#!/usr/bin/env bash
# Add Glyd to an lzbench checkout and build it:
#   contrib/lzbench/setup.sh /path/to/lzbench
# Glyd is built as a static library into lzbench/misc/glyd/ (this tree,
# copied), the wrapper appended to bench/misc_codecs.cpp, the codec
# declared in bench/codecs.h and listed in bench/lzbench.h. Then:
#   cd lzbench && ./lzbench -eglyd,1,4,5/zstd,1,3,19 file
set -euo pipefail
LZ="$1"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[ -f "$LZ/bench/lzbench.h" ] || { echo "not an lzbench checkout: $LZ" >&2; exit 1; }
mkdir -p "$LZ/misc/glyd"
(cd "$ROOT" && git archive HEAD | tar -C "$LZ/misc/glyd" -xf -)
(cd "$LZ/misc/glyd" && RUSTFLAGS="-C target-cpu=native" cargo rustc --lib --crate-type=staticlib --release -q)
grep -q lzbench_glyd_compress "$LZ/bench/codecs.h" || cat >> "$LZ/bench/codecs.h" <<'H'

#ifndef BENCH_REMOVE_GLYD
    int64_t lzbench_glyd_compress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options);
    int64_t lzbench_glyd_decompress(char *inbuf, size_t insize, char *outbuf, size_t outsize, codec_options_t *codec_options);
#else
    #define lzbench_glyd_compress NULL
    #define lzbench_glyd_decompress NULL
#endif
H
grep -q lzbench_glyd_compress "$LZ/bench/misc_codecs.cpp" || cat "$ROOT/contrib/lzbench/glyd_codec.cpp" >> "$LZ/bench/misc_codecs.cpp"
if ! grep -q '"glyd"' "$LZ/bench/lzbench.h"; then
    VERSION="$(grep -m1 '^version' "$ROOT/Cargo.toml" | cut -d'"' -f2)"
    sed -i.bak "s|^    { \"zstd\",|    { \"glyd\",       \"glyd $VERSION\",              1,   5,    0, FULL_THREADING, lzbench_glyd_compress,       lzbench_glyd_decompress,       NULL,                    NULL },\n    { \"zstd\",|" "$LZ/bench/lzbench.h"
    rm -f "$LZ/bench/lzbench.h.bak"
fi
cd "$LZ"
EXTRA_LD="-L misc/glyd/target/release -lglyd"
case "$(uname -s)" in Darwin) EXTRA_LD="$EXTRA_LD -framework CoreFoundation" ;; esac
make -j"$(nproc 2>/dev/null || sysctl -n hw.ncpu)" USER_CFLAGS="-I misc/glyd/include" USER_LDFLAGS="$EXTRA_LD" 2>&1 | tail -3
echo "built: $LZ/lzbench (try: ./lzbench -eglyd,1,4,5/zstd,1,3,19 <file>)"
