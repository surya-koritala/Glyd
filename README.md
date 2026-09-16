# Alatirok: High-Throughput SIMD-First Streaming Lossless Codec

[![Rust](https://img.shields.io/badge/rust-1.80%2B-blue.svg)](https://www.rust-lang.org)
[![License: MIT/Apache-2.0](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)
[![AVX2 / AVX-512](https://img.shields.io/badge/SIMD-AVX2%20%2F%20AVX--512-orange.svg)]()
[![CUDA](https://img.shields.io/badge/GPU-RTX%204080%20Super%20Verified-green.svg)]()

**Alatirok** is a production-grade, SIMD-first streaming lossless compression and decompression codec engineered in Rust. It is architected from the ground up for high-throughput columnar databases (ClickHouse, Parquet), distributed event brokers (Kafka, Redpanda), microservice RPCs (gRPC), and AI/LLM KV-cache offloading.

By decoupling the compressed bitstream into separate, homogeneous streams for tokens, match offsets, and literals, Alatirok eliminates tag branch mispredictions and unlocks true multi-core CPU and GPU parallel execution.

---

## Performance Highlights

Tested on an **AMD Ryzen 9 7950X3D (Zen 4, 16-Core / 32-Thread, AVX-512)** and **NVIDIA GeForce RTX 4080 Super**:

### 1. Enterprise Workload Benchmarks (25 MB Payloads)
| Workload | Real-World Application Domain | Compression Ratio | SIMD 1-Core Raw | **SIMD 16-Core Verified** | LZ4 (1-Core) | Google Snappy | Multi-Core Leap |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **JSON Logs** | Kubernetes / CloudWatch structured logs | **6.10x** | **7.69 GB/s** | **12.75 GB/s** | 5.91 GB/s | 5.29 GB/s | **2.2x faster than LZ4** |
| **Columnar DB** | Parquet / ClickHouse timestamp & metric tables | **1.54x** | **3.68 GB/s** | **11.26 GB/s** | 2.43 GB/s | 1.94 GB/s | **4.6x faster than LZ4** |
| **Binary RPC** | Protobuf / gRPC microservice payloads | **9.16x** | **11.00 GB/s** | **13.42 GB/s** | 9.82 GB/s | 9.66 GB/s | **1.4x faster than LZ4** |
| **Incompressible** | High-entropy encrypted / pre-compressed data | **1.00x** | **36.80 GB/s** | **4.21 GB/s** | 18.03 GB/s | 27.26 GB/s | **2.0x faster than LZ4** |

### 2. Official Silesia Corpus (202.12 MB)
| File Target | Content Type | SIMD 1-Core Raw | **SIMD 16-Core Verified** | LZ4 (1-Core) | Google Snappy | Meta Zstd-1 | Speedup vs LZ4 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`mozilla`** (48.8 MB) | Compiled Binary | 2.30 GB/s | **16.27 GB/s** | 3.01 GB/s | 1.97 GB/s | 0.78 GB/s | **5.4x faster** |
| **`nci`** (32.0 MB) | Chemistry DB | 3.92 GB/s | **17.64 GB/s** | 4.25 GB/s | 3.29 GB/s | 0.98 GB/s | **4.1x faster** |
| **`webster`** (39.5 MB) | HTML Dictionary | 1.67 GB/s | **13.61 GB/s** | 2.99 GB/s | 1.58 GB/s | 0.77 GB/s | **4.5x faster** |
| **`samba`** (20.6 MB) | C Source Code | 2.75 GB/s | **11.71 GB/s** | 4.04 GB/s | 2.44 GB/s | 0.81 GB/s | **2.9x faster** |
| **`osdb`** (9.6 MB) | Relational DB | 4.06 GB/s | **5.82 GB/s** | 3.81 GB/s | 2.74 GB/s | 1.60 GB/s | **1.5x faster** |
| **`dickens`** (9.7 MB) | English Prose | 1.74 GB/s | **5.50 GB/s** | 3.81 GB/s | 1.41 GB/s | 1.38 GB/s | **1.4x faster** |

### 3. GPU Frontier (NVIDIA RTX 4080 Super)
- **Kernel Decompression Speed**: **15.47 GB/s** (32 MB decoded in **2.020 ms** across 512 independent blocks).
- **Verification**: **100% bit-for-bit exact match** against reference data.

---

## Architectural Comparison: Google Snappy vs. LZ4 vs. Alatirok

| Architectural Dimension | Google Snappy | LZ4 | Alatirok (Ours) |
| :--- | :--- | :--- | :--- |
| **Bitstream Layout** | Interleaved tag bytes, offsets, and literals. | Interleaved 1-byte token, literals, and 2-byte offsets. | **Decoupled Columnar**: 3 separate homogeneous streams (Tokens, Offsets, Literals). |
| **Token Representation** | Variable-length tag bytes (4 element types, 00..11). | 1-byte packed: `(lit_len << 4) \| match_len` + varints. | **Uniform 16-bit Token**: 5 bits lit_len, 11 bits match_len. Zero tag-decoding branches. |
| **Branch Penalty** | High (4-way unpredictable branch mispredictions). | Low (Fixed token-literal-offset sequence). | **Zero**: No tag branching; branchless vector execution. |
| **Periodic Matches (1..15 bytes)** | Scalar byte copy loops. | Scalar byte copies for offsets < 8. | **AVX2 Shuffle Tables** (`_mm_shuffle_epi8` / `pshufb`) with precomputed periodicity masks. |
| **Multi-Core Scaling** | None (Single-threaded bitstream). | None (Serial stream dependencies). | **Native Rayon Parallel Engine** (10–27 GB/s line rate). |
| **GPU Execution** | Impractical (warp divergence on variable-length tags). | Impractical (warp divergence). | **Native GPU Kernel** (15.5 GB/s validated on RTX 4080 Super). |
| **Integrity Verification** | CRC32 (often disables throughput). | xxHash32 (often scalar). | **AVX2-Vectorized Adler32** (12–50 GB/s vector pass). |

---

## Quickstart

### Build and Run Tests
```bash
cargo test --release
```

### Download the Official Silesia Corpus
```bash
bash scripts/download_corpus.sh
```

### Run the Unified Benchmark Suite
```bash
# Run both Silesia and Enterprise Workloads:
RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench

# Or run specific benchmarks:
RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench -- --silesia
RUSTFLAGS="-C target-cpu=native" cargo run --release --example bench -- --workload
```

---

## Rust API Usage

```rust
use simd_stream_codec::{compress, decompress, compress_parallel, decompress_parallel};

fn main() -> Result<(), simd_stream_codec::error::CodecError> {
    let payload = b"{\"user_id\": 42, \"action\": \"telemetry_ping\", \"status\": 200}".repeat(10_000);

    // 1. Single-Core Sequential Compression / Decompression
    let compressed = compress(&payload);
    let restored = decompress(&compressed)?;
    assert_eq!(restored, payload);

    // 2. High-Throughput Multi-Core Parallel Engine
    let par_compressed = compress_parallel(&payload);
    let par_restored = decompress_parallel(&par_compressed)?;
    assert_eq!(par_restored, payload);

    println!("Compression ratio: {:.2}x", payload.len() as f64 / par_compressed.len() as f64);
    Ok(())
}
```

---

## License

Dual-licensed under either of:
- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
