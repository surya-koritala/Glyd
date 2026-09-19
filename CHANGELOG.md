# Changelog

All notable user-facing changes. Measurement history, floors and refuted
ideas live in [CHANGELOG-BENCH.md](CHANGELOG-BENCH.md).
Versioning follows [SemVer](https://semver.org); the on-disk format has its
own version in every block header (v6, v7) and every release decodes
every earlier format.

## v0.2.0 — 2026-09-19

### Levels
- **`--ultra` / `-19`** (format v7, same decoder): optimal parse on a
  binary-tree match finder, every position priced in the coder's own
  bits ([design](docs/design/ultra-parse.md)). Silesia ratio 3.80 vs
  `--max`'s 3.22; zstd -16 3.83, zstd -19 4.01 (3.91 inside Glyd's 2 MB
  window). Compresses at 4.8 MB/s; its output decodes at 2,190 MB/s,
  1.3× zstd -19's. `compress_into_ultra`, `compress_parallel_into_ultra`,
  `compress_with_dict_ultra`; C `glyd_compress_ultra`,
  `glyd_compress_ultra_parallel`.

### Platforms
- x86-64 `--max` decoder: an AVX2+BMI2 entry point and loop shapes for
  16 registers (stream-major entropy batches, the NEON copy structure,
  split walk tables). Sapphire Rapids decode 858 → 1,300 MB/s in the
  published run (zstd -3: 1,260 MB/s); default builds gain the same.
- Cross-platform benchmarks re-run; `ultra_bench` added to the script.

### CLI
- `-19` / `--ultra`; `--single-core` is now `-s` (`-1` was `--fast`).

### Fixed
- Nothing user-visible; see CHANGELOG-BENCH.md for the measurement trail.

## v0.1.0 — 2026-09-19

First public release.

### Levels
- **default** (format v6): LZAV-class parse, minimum match 7. Silesia
  ratio 2.19, decode 6,900 MB/s (1.6× liblz4), compress 340 MB/s.
- **`--fast` / `-1`**: LZ4-class finder, minimum match 5. Ratio 2.18,
  decode 4,900 MB/s, compress 550 MB/s.
- **`--turbo` / `-t`**: minimum match 10. Ratio 1.88, decode 9,200 MB/s
  (2.1× liblz4).
- **`--max` / `-9`** (format v7): 8-way interleaved Huffman literals,
  tANS-coded sequences with repeat offsets, 2 MB window, double-fast
  lazy parse; three-pass decoder. Ratio 3.22 vs zstd -3's 3.20, decode
  1,860 MB/s (1.3× zstd -3), compress 300 MB/s.

### Platforms
- aarch64 NEON decoders for v6 and v7; x86-64 AVX2 decoder for v6 and a
  portable scalar path everywhere else.
- Multi-core compression and decompression over independent 256 KB blocks.

### APIs
- Rust: `compress_into{,_fast,_turbo,_max}`, `compress_parallel_into*`,
  `decompress`, `decompress_into`, `decompress_parallel*`,
  `compress_with_dict` / `decompress_with_dict` (v7), `GlydReader` /
  `GlydWriter` (v6 streaming).
- C ABI (`include/glyd.h`, `libglyd`): `glyd_compress*`,
  `glyd_decompress*`, `glyd_max_compressed_len`, `glyd_version`.
- CLI `glyd`: compress/decompress, level flags, pipes, `-b` benchmark.

### Safety
- Every decoder fuzzed with 1,000,000 random mutations per run into
  exact-size buffers with sentinel guards; no per-call allocation in the
  decoder (1.5 MB thread-local scratch for v7).

### Known gaps
- `--max` compresses at ~90% of zstd -3's speed and decodes 1.3× (not 2×).
- `--max` beats zstd -3 on 3 of 5 extended-corpus files (loses 0.6% on
  repetitive JSON).
- No AVX2 v7 decoder yet (scalar on x86); streaming adapters are v6-only.
