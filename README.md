# Alatirok: High-Throughput SIMD-First Streaming Lossless Codec

[![Rust](https://img.shields.io/badge/rust-1.80%2B-blue.svg)](https://www.rust-lang.org)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![AVX2 / AVX-512](https://img.shields.io/badge/SIMD-AVX2%20%2F%20AVX--512-orange.svg)]()
[![C ABI](https://img.shields.io/badge/C%20ABI-include%2Falatirok.h-brightgreen.svg)]()
[![CI](https://github.com/Sigbound/alatirok/actions/workflows/ci.yml/badge.svg)](https://github.com/Sigbound/alatirok/actions)

**Alatirok** is an ultra-high-throughput, SIMD-first streaming lossless compression and decompression engine engineered in Rust. It is purpose-built to eliminate throughput bottlenecks in **Cloud Infrastructure** (S3/GCS chunk storage, gRPC microservices, ClickHouse/Parquet columnar data) and **AI/LLM Serving** (PagedAttention KV-cache host-accelerator memory streaming).

By decoupling the compressed bitstream into three separate, homogeneous columnar streams (**Tokens**, **Match Offsets**, and **Literals**), Alatirok eliminates tag-branch mispredictions and unlocks true multi-core CPU parallel execution.

---

## Key Architectural Advantages

| Feature | Google Snappy | LZ4 | Alatirok (Ours) |
| :--- | :--- | :--- | :--- |
| **Bitstream Architecture** | Interleaved variable tag bytes, offsets, literals. | Interleaved 1-byte token, literals, 2-byte offsets. | **Decoupled Columnar**: 3 homogeneous independent streams (Tokens, Offsets, Literals). |
| **Token Representation** | Variable-length tag bytes (4 element types, 00..11). | 1-byte packed: `(lit << 4) \| match` + varints. | **Uniform 16-bit Token**: 5-bit literal len, 11-bit match len + Format v2 extended literal escapes. |
| **SIMD Vector Engine** | Partial scalar wildcopy. | 8-byte / 16-byte wildcopy with bounds traps. | **AVX-512 (64-byte) & AVX2 (32-byte)** vector wildcopies with split hot-loop boundaries. |
| **Periodic Matches (1..15 bytes)** | Scalar byte copy loops. | Scalar byte copies for offsets < 8. | **AVX2 Shuffle Tables** (`_mm_shuffle_epi8`) with precomputed periodicity masks. |
| **Multi-Core Scaling** | None (Single-threaded bitstream). | None (Serial stream dependencies). | **256 KB Parallel Pipeline**: Independent chunk units with `FLAG_CHAIN_RESET` and sequential fallback. |
| **Streaming I/O** | Custom framing. | LZ4 Frame format. | **Standard `std::io::Read` & `Write`** (`AlatirokReader` / `AlatirokWriter`) with chunk framing. |
| **Universal Plug-in** | C++ library. | C library. | **Standard C ABI (`include/alatirok.h`)**, `.so` / `.a` libraries, and standalone CLI (`alatirok`). |
| **Integrity Verification** | CRC32. | xxHash32. | **AVX2-Vectorized Adler32** (12–50 GB/s line rate). |

---

## Performance Highlights

Tested on an **AMD Ryzen 9 7950X3D (Zen 4, 16-Core / 32-Thread, AVX-512, 128 MB L3 Cache)** with native CPU target flags (`-C target-cpu=native`):

### 1. Like-for-Like Single-Core Comparison (1C vs 1C)

All compressors and decompressors executing on a single dedicated core:

| File Target | Orig Size | Alatirok 1C Comp | LZ4 1C Comp | Alatirok 1C Decomp | LZ4 1C Decomp | Snappy 1C Decomp | Alatirok Ratio | LZ4 Ratio | Deficit vs LZ4 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nci`** | 32.00 MB | 1.12 GB/s | 1.19 GB/s | **5.21 GB/s** | 4.45 GB/s | 3.09 GB/s | 5.41x | 6.06x | -10.8% |
| **`osdb`** | 9.62 MB | 0.49 GB/s | 0.76 GB/s | **4.16 GB/s** | 3.58 GB/s | 2.72 GB/s | **2.06x** | 1.91x | **+8.0% (Beats LZ4)** |
| **`xml`** | 5.10 MB | 0.91 GB/s | 1.08 GB/s | **4.37 GB/s** | 4.33 GB/s | 2.95 GB/s | 3.82x | 4.35x | -12.2% |
| **`samba`** | 20.61 MB | 0.63 GB/s | 0.77 GB/s | 2.99 GB/s | 3.71 GB/s | 2.40 GB/s | 2.44x | 2.80x | -12.9% |
| **`mozilla`** | 48.85 MB | 0.35 GB/s | 0.69 GB/s | 2.16 GB/s | 2.83 GB/s | 1.91 GB/s | 1.77x | 1.93x | -8.5% |
| **`mr`** | 9.51 MB | 0.41 GB/s | 0.82 GB/s | 1.98 GB/s | 3.86 GB/s | 1.86 GB/s | 1.58x | 1.83x | -13.6% |
| **`dickens`** | 9.72 MB | 0.31 GB/s | 0.42 GB/s | 1.54 GB/s | 3.74 GB/s | 1.36 GB/s | 1.34x | 1.59x | -15.3% |
| **`webster`** | 39.54 MB | 0.35 GB/s | 0.50 GB/s | 1.91 GB/s | 2.97 GB/s | 1.55 GB/s | 1.78x | 2.06x | -13.3% |
| **`x-ray`** | 8.08 MB | 0.20 GB/s | 2.66 GB/s | 3.46 GB/s | 16.41 GB/s | 26.61 GB/s | **1.02x** | 1.01x | **+1.4% (Beats LZ4)** |
| **TOTAL SILESIA** | **202.12 MB** | — | — | — | — | — | **1.90x** | **2.10x** | **-9.3%** |

### 2. Multi-Core Scaling on 16 Cores / 32 Threads

Alatirok's parallel pipeline splits inputs into 256 KB chained units scheduled across Rayon worker threads:

| File Target | Orig Size | 1-Core Comp | 16-Core Comp | Comp Scaling | 1-Core Decomp (Ver) | 16-Core Decomp (Ver) | Decomp Scaling |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`nci`** | 32.00 MB | 1.12 GB/s | **8.01 GB/s** | **7.2x** | 4.57 GB/s | **12.85 GB/s** | **2.8x** |
| **`samba`** | 20.61 MB | 0.63 GB/s | **4.75 GB/s** | **7.5x** | 2.56 GB/s | **9.58 GB/s** | **3.7x** |
| **`mozilla`** | 48.85 MB | 0.35 GB/s | **1.48 GB/s** | **4.2x** | 2.08 GB/s | **15.03 GB/s** | **7.2x** |
| **`webster`** | 39.54 MB | 0.35 GB/s | **1.66 GB/s** | **4.7x** | 1.80 GB/s | **12.88 GB/s** | **7.2x** |
| **`osdb`** | 9.62 MB | 0.49 GB/s | **3.37 GB/s** | **6.9x** | 3.42 GB/s | **5.21 GB/s** | **1.5x** |
| **`xml`** | 5.10 MB | 0.91 GB/s | **3.08 GB/s** | **3.4x** | 3.48 GB/s | **4.21 GB/s** | **1.2x** |

### 3. Real-World Application Workloads (25 MB Payloads)

| Workload | Application Domain | Ratio | Alatirok 1C Raw | Alatirok 16C Ver | LZ4 1C | Snappy 1C | Deficit vs LZ4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Binary RPC** | Protobuf / gRPC microservice packed payloads | **9.65x** | **14.62 GB/s** | **11.74 GB/s** | 8.55 GB/s | 9.21 GB/s | **+4.4% (Beats LZ4)** |
| **JSON Logs** | Kubernetes / CloudWatch structured JSON logs | **7.08x** | **9.70 GB/s** | **12.42 GB/s** | 6.31 GB/s | 5.40 GB/s | -9.4% |
| **Columnar DB** | Parquet / ClickHouse timestamp & metric tables | **1.68x** | **3.08 GB/s** | **10.75 GB/s** | 2.35 GB/s | 1.95 GB/s | -22.5% |
| **Source Code** | Codebase repositories, ASTs, and syntax trees | **30.43x** | **20.54 GB/s** | **12.11 GB/s** | 16.34 GB/s | 7.79 GB/s | -87.7% |

---

## Universal Compatibility

Alatirok is designed to be a universal, drop-in replacement across any stack.

### 1. Standalone CLI Utility (`alatirok`)
Install globally:
```bash
cargo install --path .
```

Compress and decompress files or Unix streams:
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
- High-throughput streaming ingestion into S3 / GCS / gRPC (>2.4 GB/s streaming, >7.0 GB/s multi-core).
- **Indexed Chunk Storage**: Query arbitrary records in the middle of a 100 MB object without decompressing the rest of the file:
  ```bash
  cargo run --release --example cloud_stream
  ```

### 2. AI / LLM PagedAttention KV-Cache Offload (`examples/ai_kv_cache.rs`)
- Compresses inactive FP16/BF16 KV-cache blocks into host DRAM.
- Expands effective context capacity while retaining multi-gigabyte/second restoration throughput:
  ```bash
  cargo run --release --example ai_kv_cache
  ```

---

## Quickstart

### Build and Run Tests
```bash
cargo test --release
```

### Run C ABI Verification Test
```bash
gcc -O3 tests/test_c_abi.c -Iinclude -Ltarget/release -lsimd_stream_codec -o target/release/test_c_abi
LD_LIBRARY_PATH=target/release ./target/release/test_c_abi
```

### Run Silesia & Workload Benchmarks
```bash
RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench
```

---

## License

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
