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
| `--turbo` (`compress_into_turbo`) | 0.33 | 2.055 | **8.23** |
| liblz4 | 0.66 | 2.101 | 4.39 |

Turbo is the default parse at minimum match 8 (`FLAG_TURBO` blocks):
fewer tokens, so decode is 188% of liblz4 at 2% less ratio than liblz4.

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
gcc -O3 tests/test_c_abi.c -Iinclude -Ltarget/release -lsimd_stream_codec -o target/release/test_c_abi
LD_LIBRARY_PATH=target/release ./target/release/test_c_abi
```

---

## License

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
