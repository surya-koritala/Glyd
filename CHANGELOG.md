# Changelog

All notable user-facing changes. Measurement history, floors and refuted
ideas live in [CHANGELOG-BENCH.md](CHANGELOG-BENCH.md).
Versioning follows [SemVer](https://semver.org); the on-disk format has its
own version in every block header (v6, v7) and every release decodes
every earlier format.

## Unreleased

### Base mode
- `glyd --base old new` / `compress_with_base`: a new version of an
  object compressed against the old one, decodable with it (`glyd -d
  --base old`, `decompress_with_base`). Units of 32 MB are parsed with
  the base around their own position as history (32 MB of slack each
  way), the long-distance matcher reaching all of it; the decoder reads
  the base in place. Against zstd 1.5.7 `--patch-from` on the same
  machine, byte-exact: Wikipedia page-table dumps a month apart
  `--max` 1.79 MB at 863 MB/s (zstd -3 patch 3.84 MB at 409, zstd -19
  patch 1.30 MB at 2), `--ultra` 1.23 MB; Ubuntu cloud root filesystems
  16 days apart 5.31 MB at 2,018 MB/s (zstd -3 8.82 MB at 654, zstd -19
  5.61 MB at 39), `--ultra` 4.59 MB; Linux 6.10 -> 6.10.1 3.04 MB at
  1,980 MB/s (zstd -3 3.26 MB at 560, zstd -19 2.58 MB at 30), `--ultra`
  2.04 MB. Plain `--max` on those
  files: 33, 287 and 200 MB. Design notes in docs/design/format-v7.md;
  the measurement that led here in experiments/structure/README.md.
- A far match's cap of 130 bytes per sequence applied to the
  repeat-offset continuation as well; a repeat carries no offset bits,
  so the rest of the match is now one sequence. Plain `--max` gains 1%
  on JSON events.
- The long-distance matcher's table grows with the input (a slot per
  16 bytes, up to 2^25 entries).
- The CLI decodes a batch of units at a time into one reused buffer
  and writes as it goes (`decompress_stream`): the memory is a batch,
  not the file, and no output page is touched for the first time after
  the first batch; the library's parallel paths run on scoped worker
  threads (`set_threads`) instead of rayon.

## v0.4.0 — 2026-09-20

### Long-distance matching
- The max and ultra levels find repeats of 32 bytes or more up to 128 MB
  back (`src/ldm.rs`): one pass over the input before the parse, with
  content-defined anchors (one position in 16, found 16 at a time with
  NEON or AVX2) hashing the 32 bytes after them into a 16 MB table whose
  entries carry a hash check; matches are verified, extended both ways
  and handed to the parse, which takes one wherever it beats the local
  finder. Format v9 offsets grow to 27 bits (30 offset codes; v8 blocks
  keep 26, and a table with fewer symbols than its version allows still
  decodes). A far match is capped at 130 bytes per sequence so the
  decoder's one-load walk holds its extra bits, the rest following as a
  repeat-offset sequence. Parallel units grow to one per core, up to
  128 MB (the matcher's reach is the unit).
- The pass runs at 1.2-2 GB/s on one core. The max level gates it:
  after 4 MB and 16 MB (or half the input) it stops on data whose
  repeats are too few or too near to pay (media, Parquet, most SQL
  dumps), which keep 92-96% of their speed; the ultra level runs it
  whole. Where it stays on, the max level compresses at 63-85% of its
  former speed for 3-16% fewer bytes. One core, M1 Max, 128 MB of
  GitHub Archive JSON: `--max` 11.63 -> 9.73 MB at 577 MB/s (zstd -3
  12.90 MB at 895; `zstd -3 --long=27` 10.20 MB at 433), `--ultra`
  7.78 MB (zstd -19 8.96, `zstd -19 --long=27` 7.80); NASA access
  log 13.22 -> 11.92 MB at 435 MB/s (`zstd -3 --long=27` 13.63 MB at
  383); Silesia `--max` 3.259 -> 3.302 at 247 MB/s (290 before; zstd
  -3 3.205 at 335). The 8.7 GB corpus on 10 cores: `--max` 3.89 ->
  3.96 at 2,000 MB/s (2,400 before; zstd -3 3.85 at 4,000), JSON
  events 11.49 -> 13.26 (zstd -3 10.46); `--ultra` 4.65 (zstd -19
  4.66), JSON events 16.40 (zstd -19 15.07).

### Record mode
- `glyd -r` / `compress_records_with`: delimited lines, SQL dumps and
  JSON lines become typed column streams (integer, decimal and
  date-time deltas, dictionaries with recency ranks, text) before the
  level, in parallel 32 MB units, rebuilt byte for byte; other data is
  left as it is; input the transform does not pay on (API events with
  hashes and free text, binaries) takes the plain parallel path. JSON
  lines: a column per key path, typed values leave holes in a frame of
  the structure, keys and text. Telemetry as rows, 128 MB slices, 10
  cores: a cluster trace as CSV `--max -r` 12.6 against zstd -3's 4.5
  and zstd -19's 6.9, as JSON lines 54.6 against 15.7 and 28.7; daily
  weather 19.6 / 47.0 against 7.0 / 18.7 and 12.0 / 31.6; taxi trips
  exported to CSV 8.9 against 5.6 and 8.4 (`scripts/download_ext_corpus.sh`). Whole 8.7 GB corpus, 10 cores: `--ultra -r`
  5.20 against zstd -19's 4.66 (SQL dumps 1.43x smaller, access logs
  1.54x, JSON 1.09x through the plain level's matcher); `--max -r`
  4.75 at 1,100 MB/s. Design notes in
  docs/design/format-v7.md; the prototypes and measurements that led
  here in experiments/structure/.

### Small objects and dictionaries
- `Dict`: a prepared dictionary (trained content plus entropy tables)
  for small objects; `Dict::train` (cover selection as zstd's fastcover,
  scoring each distinct string once), `to_bytes`/`from_bytes`,
  `compress_with_dict`, `compress_with_dict_ultra`,
  `decompress_with_dict`. The object is parsed in place against the
  dictionary's own seeded tables; the decoder copies from the content
  and borrows the dictionary's built tables.
- Format v9: compact framing for blocks of at most 32 KB (one marker
  byte, varint lengths, a 5-10 byte sub-header, single-stream sections
  under 1,024 symbols, no padding on disk): 207 -> 21 bytes of framing
  on a 4 KB object. Every earlier format decodes unchanged
  (tests/format_compat.rs holds v0.2.0, v0.3.0 and v0.4.0 output).
- The ultra level with a dictionary prices its parse from the
  dictionary's tables.
- Per-object work cut: entropy tables and codes built once per
  dictionary, table costs from a lookup, buffers kept across calls,
  single-stream decode paths (three code chains at once, one-load
  batches).

GitHub Archive JSON objects, Apple M1 Max, one core, 110 KB
dictionaries trained on other objects (zstd's numbers without a
checksum; Glyd writes 4 bytes per object): 1 KB objects `--max` + Dict
ratio 4.75 (zstd -3 + dict 4.96), compress 245 MB/s (454), decode 920
MB/s (1,117); 4 KB 6.28 (6.42), 321 (588), 1,209 (1,440); 16 KB 7.65
(7.66), 394 (635), 1,674 (1,890). `--ultra` + Dict: 5.25 / 7.13 / 8.84
(zstd -19 + dict 5.47 / 7.41 / 8.95). Before this work the same objects
compressed to 2.15 / 3.87 / - with a window-only dictionary at 6 MB/s
and decoded at 170 MB/s.

### Verification and benchmarks
- `scripts/verify_roundtrip.sh` (every level and core mode through the
  CLI, byte-compared; corrupted copies rejected or decoded exactly),
  `scripts/download_bench_corpus.sh` (~9 GB of logs, JSON, SQL dumps
  and Parquet with a separate training set), `examples/bench_suite.rs`
  (Glyd against zstd -3, zstd -19 and LZ4 at a stated thread count,
  every decode checked, peak memory, small-object latencies),
  `scripts/s3_workflow.sh` (compress, upload, download, decompress,
  verify, monthly cost), `scripts/bench_aws_suite.sh` (all of it on
  Graviton3 and Sapphire Rapids), `scripts/report_suite.py`.
- The format-compatibility fixtures are now committed (they were
  ignored by the `*.glyd` rule; CI failed on every push since they were
  added).
- Results of the program on Graviton3 and Sapphire Rapids:
  docs/benchmarks/suite-2026-09.md (raw rows in benchmarks/suite/).

### Parallel paths
- The parallel compressors cut the input into units of at least 2 MB
  (v6 levels), 8 MB (`--max`) and 16 MB (`--ultra`) instead of 256 KB,
  growing with the input (two units per core, up to 64 MB): JSON events
  lost 4.7% to 8 MB units against the sequential ratio, 1% at 64 MB. Each unit is
  still a chain of its own (parallel decode, random access), and the
  ratio now stays within 0.5-0.7% of the sequential path; at 256 KB the
  CLI's multi-core default was giving up 3% (default level), 6.6%
  (`--max`) and 13% (`--ultra`). Multi-core decode of files with few
  units is correspondingly less parallel (Silesia on 10 cores: 31,900
  MB/s against 42,900).

### Ultra level
- Blocks are split where the parse's statistics change
  (`v7_ultra::split_points`); prices carry the parse's own prior at
  weight 2 and half a bit per literal. Silesia 3.925 → 3.946 (zstd -19:
  4.006), decode 2,150 → 2,090 MB/s.
- x86-64: the walk's and the tANS batch's per-stream state through
  memory: +6% max-level decode on Sapphire Rapids.

## v0.3.0 — 2026-09-19

### Format v8
Every level of the entropy-coded family now writes format v8; v7 (and
v6) files from earlier releases decode unchanged (tests/format_compat.rs
holds v0.2.0 output as fixtures).
- 8 MB window (26 offset codes).
- Section layout: 24-bit sub-stream sizes and one padding per section
  instead of per stream; tANS counts as width + mantissa; literal tables
  as nibbles with unused-symbol runs. Per-block overhead 950 → 421
  bytes.
- Length codes with direct codes to 15 (runs) and 34 (matches) and
  short buckets before the log2 ones.

Silesia, M1 Max, same run: `--ultra` 3.80 → **3.93** (zstd -16 3.83,
zstd -19 4.01), decode 2,150 MB/s (1.3× zstd -19); `--max` 3.22 →
**3.25** (zstd -3 3.20), decode 1,890 MB/s (1.27× zstd -3). Design
notes: docs/design/format-v7.md, "Format v8".

### Library
- The ultra finder sizes its tables to the input per call: 2 MB for a
  256 KB chunk, 64 MB for an input that fills the window.
- `bits::Stream` (a sub-stream with its own length and the bytes to the
  section's end) replaces bare slices in the decoders' signatures.

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
