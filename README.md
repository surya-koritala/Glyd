# Alatirok: High-Throughput SIMD-First Streaming Lossless Codec

[![Rust](https://img.shields.io/badge/rust-1.80%2B-blue.svg)](https://www.rust-lang.org)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![SIMD: AVX2 | NEON](https://img.shields.io/badge/SIMD-AVX2%20%7C%20NEON-orange.svg)]()
[![C ABI](https://img.shields.io/badge/C%20ABI-include%2Falatirok.h-brightgreen.svg)]()
[![CI](https://github.com/Sigbound/alatirok/actions/workflows/ci.yml/badge.svg)](https://github.com/Sigbound/alatirok/actions)

**Alatirok** is a SIMD-first LZ77 codec in Rust, built for the places where
decompression throughput is the bottleneck: object-store chunk fetches, gRPC
payloads, columnar scans, and KV-cache paging for LLM serving.

The compressed block is split into four homogeneous streams (**tokens**,
**offsets**, **extras**, **literals**) instead of one interleaved byte stream.
That is what lets the decoder pre-decode 32 tokens per AVX2 pass, check
bounds once per chunk, and run a copy-only loop, which an inline format such
as LZ4's cannot do.

**Status (2026-09-18, commit `54f0a24`): single-core decode beats liblz4 on
Silesia, measured in the same run, at a higher ratio.** Compression speed is
the open axis. See [Where we are](#where-we-are-and-what-is-next).

---

## Format v6 (current)

| Stream | Layout |
| :--- | :--- |
| **Token** (1 byte per token) | bits 0..2 literal length 0..6 (7 = escape), bits 3..6 match length code (0 = none, 1..14 = length 7..20, 15 = escape), bit 7 = offset bit 16 |
| **Offsets** | fixed 2 bytes per match; 17-bit offsets, 128 KB window |
| **Extras** | one byte per escaped length (`255` + `u16` continuation for longer runs) |
| **Literals** | raw bytes, copied 32 at a time |

Blocks are 256 KB, 36-byte header, AVX2-vectorized Adler32, `FLAG_CHAIN_RESET`
for independent parallel decode. The match finder is a port of LZAV 4.3's
parse (komihash 6-byte hash, 2-way buckets, back-matching, adaptive skip) with
a masked-u32 candidate check, minimum match 7.

---

## Format v7 (`--max` level)

Format v7 (`compress_into_max` / `compress_parallel_into_max`, CLI `-9` /
`--max`) replaces the single interleaved token stream with five
independently entropy-coded streams: **literals** (Huffman, 8 interleaved
bitstreams, max code length 11, table sent as packed 4-bit lengths or a
"reuse previous block" flag), **literal lengths** and **match lengths**
(tANS over a small code alphabet, 0-15 direct plus log2-bucket codes above
that), **offsets** (tANS; codes 0-2 are repeat offsets `rep0`/`rep1`/`rep2`
with zstd-style semantics, 3+ are log2 buckets, up to 21 bits for a 2 MB
window), and **extra bits** (one raw LSB-first bitstream, also 8-way
interleaved). Every stream falls back to raw storage per block when coding
does not pay for itself. The decoder runs three passes over thread-local
scratch (~1.5 MB, allocated once per thread, never per call): an entropy
pass that unpacks streams 2-5 into flat sequence arrays (repeat-offset
resolution happens here), a literal pass that Huffman-decodes the block's
literals, and a copy pass that walks the arrays and literal buffer through
the existing NEON/AVX2 copy loop -- the same three-register decode
discipline as format v6, just fed by entropy-coded streams instead of raw
bytes.

**Measured (Silesia, same run, M1 Max, one core):**

| Codec | Ratio | Comp GB/s | Decode GB/s |
| :--- | ---: | ---: | ---: |
| **Alatirok `--max` (v7)** | **3.218** | 0.305 | **1.86** |
| zstd -3 | 3.205 | 0.335 | 1.45 |
| zstd -1 | 2.894 | 0.555 | 1.55 |

Honest framing: `--max` beats zstd -3's ratio (not by much: +0.4%) and
decodes 1.29x as fast, at 91% of its compression speed. It does not reach
2x zstd -3 decode or match its compression speed outright; see
`CHANGELOG-BENCH.md` (`v7 milestone 5`) for the per-file breakdown and the
milestones that got here.

### Dictionary API

`compress_with_dict(dict, input, output)` / `decompress_with_dict(dict,
compressed)` preload the 2 MB window with `dict`'s bytes before the first
block, so early matches and repeat offsets can reach into it -- the same
mechanism zstd's dictionaries use, aimed at small, similarly-shaped inputs
(e.g. one JSON event) where a cold `--max` block has nothing to match
against yet. `dict_id(dict)` (`compute_checksum(dict)`, with checksum 0
remapped to 1) is stored in the block header; a decompress call is
rejected unless it is given the same dictionary bytes. Dictionaries are a
sequential-only feature in this milestone -- `compress_parallel_into_max`
and the parallel decompress path do not take one.

### Extended corpus (gate G2)

`scripts/download_corpus.sh` also populates `corpus/ext/` with real-world
formats beyond Silesia/enwik8 (GitHub Archive JSON lines, NASA HTTP logs,
NYC taxi Parquet, an uncompressed Linux source tarball, an OpenStreetMap
PBF extract, and TPC-H `lineitem` when DuckDB is on `PATH`) so the ratio
claim above is not a Silesia-only artifact. `examples/v7_bench.rs` runs
gate G2 over every file present: v7's ratio must be >= zstd -3's, file by
file.

| File | v7 ratio | zstd -3 ratio | G2 |
| :--- | ---: | ---: | :---: |
| gharchive.json (912 MB, GH Archive JSON lines) | 10.5913 | 10.6560 | FAIL (-0.6%) |
| liechtenstein.osm.pbf (3.4 MB, OSM PBF) | 1.0001 | 1.0000 | PASS |
| linux.tar (64 MB, kernel source) | 4.9371 | 4.8983 | PASS (+0.8%) |
| nasa_access.log (205 MB, HTTP log) | 9.7894 | 9.7822 | PASS |
| yellow_tripdata.parquet (50 MB, NYC taxi) | 1.0007 | 1.0038 | FAIL (-0.3%) |
| **Total** | **7.1117** | **7.1343** | **FAIL (2/5 files)** |

**Honest result: G2 does not hold on this corpus.** Silesia is where the
format was tuned, and it holds there; two of five extended-corpus files
lose to zstd -3. The Parquet loss is noise at the incompressible floor
(both ratios round to 1.00 -- Parquet already applies its own internal
compression, so this measures framing overhead, not entropy coding).
The GitHub Archive loss is real, if small: zstd -3 is both denser and
faster on that file, most likely because its search is more thorough
than v7's `dfast` (double-fast) parse on this file's very repetitive
JSON structure. Full per-file numbers are in `CHANGELOG-BENCH.md`
(`v7 milestone 5`).

---

## Measured against the field

All numbers: Silesia corpus (202 MB, 12 files), AMD Ryzen 9 7950X3D, one pinned
core, `-C target-cpu=native`, median of repeated runs. **Every Alatirok decode
number is paired with liblz4 from the same run** (`examples/quick3.rs`) so
the comparison cannot be met by run-to-run drift (which is ±3-10% in this
WSL2 VM).

### Single core, Silesia total

| Codec | Decode GB/s | % of memcpy wall | Ratio | Comp GB/s |
| :--- | ---: | ---: | ---: | ---: |
| memcpy (the physical ceiling) | 22.9 | 100 | - | - |
| **Alatirok v6** | **6.05** | **26** | **2.192** | 0.35 |
| liblz4 (same run) | 5.54 | 24 | 2.101 | 0.85 |
| lz4_flex | 3.7 | 16 | 2.097 | 0.67 |
| LZAV | 3.1 | 14 | 2.450 | 0.49 |
| zstd -1 / 1 / 3 | 2.2 / 1.7 / 1.6 | 7-10 | 2.24-3.20 | 0.3-0.6 |
| snappy | 2.1 | 9 | 2.076 | 0.78 |

liblz4 is the open-source decode-speed champion; every other measured codec is
slower. Alatirok is the only one above it, and does so while 4% denser. RAD
Oodle (commercial, closed) is the unmeasured bar above that.

### Per file, same run (2026-09-18)

| File | Ratio | Alatirok decode GB/s | liblz4 decode GB/s | vs liblz4 |
| :--- | ---: | ---: | ---: | ---: |
| dickens | 1.815 | 5.99 | 5.19 | **+15%** |
| mozilla | 1.926 | 5.42 | 4.83 | **+12%** |
| mr | 1.902 | 6.65 | 5.57 | **+19%** |
| nci | 6.846 | 7.13 | 7.24 | -2% |
| ooffice | 1.335 | 6.31 | 4.68 | **+35%** |
| osdb | 2.294 | 6.27 | 5.11 | **+23%** |
| reymont | 2.378 | 5.04 | 4.49 | **+12%** |
| samba | 2.858 | 6.38 | 6.14 | **+4%** |
| sao | 1.038 | 13.71 | 7.32 | **+87%** |
| webster | 2.250 | 4.70 | 4.85 | -3% |
| xml | 4.949 | 6.42 | 5.56 | **+16%** |
| x-ray (stored raw) | 1.000 | 46.6 | 18.1 | **+157%** |
| **Total** | **2.192** | **6.05** | **5.54** | **+9%** |

Beats liblz4 on 10 of 12 files; trails on nci and webster by 2-3%.

### Apple M1 Max, same run (2026-09-18, NEON port, one core)

| Codec | Decode GB/s | % of memcpy wall | Ratio |
| :--- | ---: | ---: | ---: |
| memcpy | 39.9 | 100 | - |
| **Alatirok v6 (NEON)** | **6.94** | **17** | **2.192** |
| liblz4 (same run) | 4.36 | 11 | 2.101 |

Beats liblz4 on 12 of 12 files (+53% total). The measured wall for this
format on this chip is ~12.5 GB/s (one dependent 32-byte copy per token on
the serial output pointer); see `examples/floor.rs` and the changelog.

Levels, same run (`quick3`, M1 Max, one core):

| Level | Comp GB/s | Ratio | Decode GB/s |
| :--- | ---: | ---: | ---: |
| `--fast` (`compress_into_fast`) | 0.55 | 2.176 | 4.96 |
| default | 0.34 | 2.192 | 6.94 |
| `--turbo` (`compress_into_turbo`) | 0.28 | 1.884 | **9.23** |
| `--max` (`compress_into_max`, format v7) | 0.305 | **3.218** | 1.86 |
| liblz4 | 0.66 | 2.101 | 4.39 |

Turbo is the default parse at minimum match 10 (`FLAG_TURBO` blocks):
fewer tokens, so decode is 211% of liblz4 at 10% less ratio. Decode
scales with tokens per byte (min 8: 8.2 GB/s at 2.055; 12: 10.5 at 1.75).

`--max` is not measured against liblz4 above -- an entropy-coded format
at LZ4-class ratios makes no sense; its competitor is zstd -3. See the
"Format v7" section below.

### Field survey: does anything dominate liblz4? (`examples/field_survey.rs`, Silesia, M1 Max, one core)

Every codec measured in the same run, `Alatirok-max` included:

```
codec      |   ratio  vs lz4 | comp GB/s  vs lz4 |  dec GB/s  vs lz4 | dominates lz4?
-------------------------------------------------------------------------------------------------
Alatirok-max |  3.2176  1.532x |     0.277  0.454x |     1.733  0.422x | wins: ratio
zstd-3     |  3.2045  1.525x |     0.319  0.523x |     1.361  0.331x | wins: ratio
zstd-1     |  2.8942  1.378x |     0.535  0.876x |     1.493  0.363x | wins: ratio
LZAV-hi    |  2.8032  1.334x |     0.091  0.148x |     3.185  0.775x | wins: ratio
LZAV       |  2.4500  1.166x |     0.426  0.699x |     3.128  0.761x | wins: ratio
zstd--1    |  2.4380  1.160x |     0.614  1.007x |     2.153  0.524x | wins: ratio+comp
zstd--3    |  2.2399  1.066x |     0.684  1.122x |     2.307  0.562x | wins: ratio+comp
Alatirok   |  2.1924  1.044x |     0.312  0.511x |     6.507  1.584x | wins: ratio+dec
Alatirok-fast |  2.1760  1.036x |     0.501  0.821x |     4.670  1.137x | wins: ratio+dec
liblz4     |  2.1009  1.000x |     0.610  1.000x |     4.108  1.000x | (baseline)
lz4_flex   |  2.0971  0.998x |     0.633  1.037x |     3.004  0.731x | wins: comp
snappy     |  2.0761  0.988x |     0.607  0.996x |     1.495  0.364x | no
zstd--5    |  2.0570  0.979x |     0.746  1.222x |     2.484  0.605x | wins: comp
Alatirok-turbo |  1.8837  0.897x |     0.263  0.431x |     8.647  2.105x | wins: dec
```

(`zstd-N` is the normal level N; `zstd--N` is `--fast=N`, zstd's low-ratio
ultra-fast mode.) `Alatirok-max` has the best ratio of the field, ahead of
zstd -3; nothing here beats liblz4 on ratio, compression and decode at
once, `Alatirok-max` included -- it wins on ratio alone in this survey,
same as zstd -3. Raw per-file numbers are logged to
`field_survey_partial.csv` by every run.

Multi-core decode (10 threads, independent 256 KB blocks): **42.9 GB/s**
over Silesia, above this machine's single-core memcpy (`examples/mc.rs`).

The compression ceiling for any greedy LZ finder is one data-random branch
per probed position (~0.7-1 GB/s per core); see `examples/cfloor.rs`. `src/neon_decompress.rs` is a
lane-for-lane port of the AVX2 decoder; the finder is still scalar on arm64
(comp 0.27 GB/s).

### Multi-core (16 cores / 32 threads, 256 KB independent blocks)

13+ GB/s decode on mozilla, nci, webster, samba (GOAL3 floor S1.4: >= 10 GB/s).
Compression scales 4-7x on 16 cores. Full table in `RESULTS.md`; refresh
with `examples/bench.rs`.

### How the decode number was reached

| Step | Decode, % of liblz4 (same run) |
| :--- | ---: |
| Pivot to speed (GOAL3) baseline | 55% |
| Credit-based bounds checks | 57% |
| Format v6: 3-bit literal, fixed 2-byte offsets | 72% |
| AVX2 32-token pre-pass | 75% |
| Fix chunk-retry waste (`careful` counter) | 90% |
| 255-continuations decoded inside the chunk | 97% |
| Copy loop in its own function | 98% |
| Minimum match 7 (tokens -20%) | **106-109%** |

Every step and every refuted idea is recorded with its numbers in
`CHANGELOG-BENCH.md`.

---

## Where we are, and what is next

**Done (GOAL3 Tier S1):** decode >= liblz4 in the same run, Silesia ratio >=
2.1009 (liblz4's), 16-core decode >= 10 GB/s, decoder allocates nothing beyond
the output, 25 tests green including 1M-mutation fuzz.

**The trade that was made:** to get here the minimum match went 6 -> 7, which
took the ratio from 2.39 to 2.19 and compression from 0.42 to 0.35 GB/s.
Ratio and decode speed trade against each other in this design; that was
measured every way we could think of (entropy-coded tokens, bit-packed
offsets, bigger windows, all refuted in `CHANGELOG-BENCH.md`) before choosing
speed.

**Next, in order:**

1. **Fast compression level (GOAL3 S3).** Built (`-1`/`--fast`): 5-byte
   hash, 1-way 32 KB table, skip, minimum match 5, offsets down to 1. On
   the M1 it is at 83% of liblz4's compression speed with 3.6% better
   ratio and 10% faster decode. The parse alone is faster than liblz4;
   the gap is the four-stream emit (profile in the changelog). Handles
   x-ray (1.004) instead of storing it raw.
2. **Decode toward the wall (GOAL3 S2).** memcpy of the output is 22.9 GB/s;
   we are at 26% of it, liblz4 at 24%. The remaining cost is ~9 cycles per
   token in the copy loop; the plausible next stop is ~9-10 GB/s. nci and
   webster are the two files still behind. Needs hardware counters, which
   WSL2 does not expose: profile on bare-metal Linux or macOS.
3. **Multi-core against DRAM bandwidth.** 16-core decode is 13+ GB/s; the
   question is how close to the memory wall it gets on a bare-metal box.

**Not worth retrying (all measured, see changelog):** Huffman or rANS on the
token stream (~3 ns/symbol table-load wall), bit-packed offsets (ratio 2.47
but 3.3 ns/match), windows above 2 MB, prefetching in the decoder, branchless
escape handling, 16-byte copies, scalar two-pass decode, packed u32 lanes.

**Goal documents:** `GOAL3.md` (current: speed), `GOAL2.md` (ratio tiers,
superseded but its rules still bind), `GOAL.md` (original).

---

## Universal Compatibility

### 1. Standalone CLI Utility (`alatirok`)
```bash
cargo install --path .
```

```bash
# Compress with multi-core parallelism (default)
alatirok -c telemetry.json -o telemetry.json.alk

# Decompress to original file
alatirok -d telemetry.json.alk -o telemetry_restored.json

# Streaming UNIX pipes (zero temporary disk files)
cat raw_stream.log | alatirok -c | curl -X POST https://s3-bucket.internal/upload --data-binary @-

# Fast benchmark mode
alatirok -b bigdata.csv

# Inspect SIMD hardware capabilities (AVX-512 / AVX2 / BMI2)
alatirok -v
```

### 2. Standard C / C++ ABI (`include/alatirok.h`)
Any C, C++, Go (CGO), Python (ctypes/CFFI), or Java (JNI) program can link directly with `libsimd_stream_codec.so` or `libsimd_stream_codec.a`.

```c
#include "alatirok.h"

// 1. Calculate safe destination buffer capacity
size_t max_out = alatirok_max_compressed_len(src_len);
uint8_t* compressed = malloc(max_out);

// 2. Compress using all CPU cores in parallel
int64_t comp_bytes = alatirok_compress_parallel(src, src_len, compressed, max_out);

// 3. Decompress with checksum validation
uint8_t* restored = malloc(src_len);
int64_t decomp_bytes = alatirok_decompress_parallel(compressed, comp_bytes, restored, src_len);
```

### 3. Standard Rust Streaming I/O (`std::io::Read` / `std::io::Write`)
```rust
use std::fs::File;
use std::io::{copy, BufReader, BufWriter};
use simd_stream_codec::streaming::{AlatirokWriter, AlatirokReader};

// Transparent streaming compression to disk or network
let out_file = File::create("archive.alk")?;
let mut writer = AlatirokWriter::new(BufWriter::new(out_file));
writer.write_all(b"high-throughput streaming data")?;
writer.flush()?;

// Transparent streaming decompression on the fly
let in_file = File::open("archive.alk")?;
let mut reader = AlatirokReader::new(BufReader::new(in_file));
let mut decoded = Vec::new();
reader.read_to_end(&mut decoded)?;
```

---

## Cloud & AI Applications

### 1. Cloud Object Store & gRPC Streaming (`examples/cloud_stream.rs`)
- Indexed chunk storage: read arbitrary records from the middle of a 100 MB object without decompressing the rest.
  ```bash
  cargo run --release --example cloud_stream
  ```

### 2. AI / LLM PagedAttention KV-Cache Offload (`examples/ai_kv_cache.rs`)
- Compresses inactive FP16/BF16 KV-cache blocks into host DRAM and restores them at multi-GB/s.
  ```bash
  cargo run --release --example ai_kv_cache
  ```

---

## Quickstart

### Build and run tests
Builds are done in WSL2 on this machine; see `GOAL2.md` section 3 for the
measurement protocol and `scratch/test_all.sh` for the full test run.
```bash
CARGO_BUILD_JOBS=6 cargo test --release
```

### Same-run comparison against liblz4 (the S1 gate)
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example quick3 -- label 3 0.3 -v
```

### Field survey (liblz4, lz4_flex, LZAV, zstd, snappy, ...)
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example field_survey
```

### Physical ceiling on this machine
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example speed_ceiling
```

### C ABI verification
```bash
# macOS
clang -O3 tests/test_c_abi.c -Iinclude -Ltarget/release -lsimd_stream_codec -o target/release/test_c_abi
DYLD_LIBRARY_PATH=target/release ./target/release/test_c_abi

# Linux
gcc -O3 tests/test_c_abi.c -Iinclude -Ltarget/release -lsimd_stream_codec -o target/release/test_c_abi
LD_LIBRARY_PATH=target/release ./target/release/test_c_abi
```

---

## License

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
