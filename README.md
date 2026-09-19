<h1 align="center">Glyd</h1>
<p align="center"><strong>The world's fastest-decoding open-source compression.</strong><br>
Fewer bytes than zstd's default level. Reads 1.3× faster than zstd, up to 2.1× faster than LZ4.</p>

<p align="center">
<a href="https://github.com/surya-koritala/Glyd/actions"><img alt="CI" src="https://github.com/surya-koritala/Glyd/actions/workflows/ci.yml/badge.svg"></a>
<a href="LICENSE"><img alt="License: BUSL-1.1" src="https://img.shields.io/badge/license-BUSL--1.1-blue.svg"></a>
<img alt="Rust 1.80+" src="https://img.shields.io/badge/rust-1.80%2B-blue.svg">
<img alt="SIMD: AVX2 | NEON" src="https://img.shields.io/badge/SIMD-AVX2%20%7C%20NEON-orange.svg">
<a href="include/glyd.h"><img alt="C ABI" src="https://img.shields.io/badge/C%20ABI-include%2Fglyd.h-brightgreen.svg"></a>
<a href="https://github.com/surya-koritala/Glyd/releases"><img alt="Release" src="https://img.shields.io/github/v/release/surya-koritala/Glyd?include_prereleases&label=release"></a>
</p>

<p align="center">
<a href="#at-a-glance">At a glance</a> ·
<a href="#what-glyd-saves-you">Savings</a> ·
<a href="#quick-start">Quick start</a> ·
<a href="#levels">Levels</a> ·
<a href="#benchmarks">Benchmarks</a> ·
<a href="#how-it-works">How it works</a> ·
<a href="ROADMAP.md">Roadmap</a> ·
<a href="#license">License</a>
</p>

---

## At a glance

**Glyd** is a lossless data compression library and CLI, written in Rust with
a C ABI, for workloads where **decompression speed** and **storage cost**
decide the bill: object storage and data lakes (Parquet, ORC), columnar
scans, RPC and message payloads, game and app assets, and KV-cache paging
for LLM inference. It is a drop-in alternative to **LZ4**, **Snappy** and
**zstd**.

| Level | Ratio | Compress | **Decompress** | vs the reference, same run |
| :--- | ---: | ---: | ---: | :--- |
| ⚡&nbsp;**Glyd&nbsp;‑‑turbo** | 1.88 | 280&nbsp;MB/s | **9,200&nbsp;MB/s** | **2.1×** liblz4 (4,400&nbsp;MB/s) |
| ⚡&nbsp;**Glyd&nbsp;default** | 2.19 | 340&nbsp;MB/s | **6,900&nbsp;MB/s** | **1.6×** liblz4, better ratio |
| ⚡&nbsp;**Glyd&nbsp;‑‑max** | **3.22** | 300&nbsp;MB/s | **1,860&nbsp;MB/s** | **1.3×** zstd&nbsp;-3 (1,440&nbsp;MB/s); denser (3.20) |

<sub>Silesia corpus (202 MB), Apple M1 Max, one core; every Glyd number is paired with the reference library measured in the same process. Multi-core decode reaches <b>43,000 MB/s</b> on 10 cores, the machine's memory wall. The same story holds on AWS Graviton3; on x86 the v6 levels lead too, while <code>--max</code> still runs its scalar decoder (AVX2 port next). Cross-platform results: <a href="benchmarks/">benchmarks/</a>.</sub>

- 🚀 **Fastest decode at every ratio point** measured, against liblz4, lz4_flex, LZAV, zstd (7 levels) and snappy, in the same run.
- 📦 **Fewer bytes than zstd -3** with the `--max` level, at 30% faster reads.
- 🧱 **One container, three levels**, any mix of blocks decodes; independent 256 KB blocks scale across cores.
- 🛡️ **Fuzzed** with a million mutations per run into exact-size buffers; no per-call allocation in the decoder.
- 🔌 **Rust, C ABI, CLI**, streaming `std::io` adapters, dictionaries for small objects.

---

## What Glyd saves you

> **Try it:** [docs/savings.html](docs/savings.html) — enter what you store and what you compress with today.

Stored bytes scale with `1 / ratio`. Most analytics data today is compressed
with Snappy or LZ4 (Parquet's default codec is Snappy). Moving it to Glyd
`--max` cuts the bytes stored and moved by about a third; moving from zstd
saves CPU on every read instead.

| You&nbsp;store&nbsp;today | Compressed&nbsp;with | ⚡&nbsp;**Glyd&nbsp;‑‑max** | Bytes&nbsp;saved | **Saved&nbsp;per&nbsp;year** ($21/TB‑month) |
| ---: | :--- | ---: | ---: | ---: |
| 100&nbsp;TB | Snappy (2.08) | 64.5&nbsp;TB | 35.5% | **$8,900** |
| 1 PB | Snappy | 645 TB | 35.5% | **$89,000** |
| 1 PB | LZ4 (2.10) | 653 TB | 34.7% | **$87,000** |
| 10 PB | Snappy | 6.45 PB | 35.5% | **$895,000** |
| 100 PB | Snappy | 64.5 PB | 35.5% | **$8.9 M** |
| 1&nbsp;PB | zstd&nbsp;-3 (3.20) | 996&nbsp;TB | 0.4% | $1,000, plus **22% fewer decode CPU‑seconds** on every read |

Formula: `saved_per_year = stored_TB × (1 − old_ratio / 3.22) × price_per_TB_month × 12`.
Ratios are Silesia, same run; your data will differ — measure it with
`glyd -b yourfile` before believing any table, including this one.

**At market scale:** object storage holds hundreds of exabytes (AWS said in
March 2026 that S3 alone stores "hundreds of exabytes" across 500 trillion
objects). At list price, **every 1% fewer bytes across 100 EB is about
$250 M a year**; moving the Snappy/LZ4-compressed share of it to Glyd
`--max` is worth billions a year.

Where the numbers come from and what they do not say: the byte savings
apply when you are on Snappy/LZ4 today; against zstd -3 the saving is CPU,
not bytes. Glyd `--max` compresses at 89–91% of zstd -3's speed. See
[Known gaps](#known-gaps).

---

## Quick start

```bash
cargo install --git https://github.com/surya-koritala/Glyd
```

```bash
glyd -9 data.parquet -o data.parquet.glyd     # --max: zstd-class ratio, faster reads
glyd    telemetry.json -o telemetry.glyd      # default: LZ4-class ratio, 6.9 GB/s reads
glyd -t assets.bin -o assets.glyd             # --turbo: 9 GB/s reads
glyd -d data.parquet.glyd -o data.parquet     # decompress (level is in the stream)
cat log | glyd -c | curl -X POST https://store/upload --data-binary @-
glyd -b bigfile                               # benchmark all levels on your data
```

Rust:

```rust
let mut out = Vec::new();
glyd::compress_into_max(&input, &mut out);          // or compress_into / compress_into_fast / compress_into_turbo
let back = glyd::decompress(&out)?;                  // any level, any block mix

// Dictionaries for small objects (JSON documents, records):
let mut out = Vec::new();
glyd::compress_with_dict(&dict, &doc, &mut out);
let back = glyd::decompress_with_dict(&dict, &out)?;

// std::io streaming (v6 levels):
let mut w = glyd::GlydWriter::new(std::io::BufWriter::new(file));
```

C / C++ / Go / Python (ctypes): link `libglyd` and include [`include/glyd.h`](include/glyd.h):

```c
size_t cap = glyd_max_compressed_len(n);
int64_t clen = glyd_compress_max_parallel(src, n, dst, cap);   // or glyd_compress / _parallel
int64_t dlen = glyd_decompress_parallel(dst, clen, out, n);
```

---

## Levels

| Level | Use it for | How it works |
| :--- | :--- | :--- |
| **‑‑turbo**&nbsp;(‑t) | Data read far more often than written, where read CPU is the cost: in-memory caches, game assets, KV-cache paging | v6 format, minimum match 10: fewest tokens, one 32-byte copy per token |
| **default** | The LZ4/Snappy slot with better ratio and 1.6× LZ4's read speed | v6 format, LZAV-class match finder, minimum match 7 |
| **‑‑fast**&nbsp;(‑1) | When you need LZ4-class compression speed | v6 format, LZ4-class finder, minimum match 5 |
| **‑‑max**&nbsp;(‑9) | The zstd slot: fewest bytes, 30% faster reads than zstd -3 | v7 format: 8-way interleaved Huffman literals + tANS-coded sequences, repeat offsets, 2 MB window, double-fast lazy parse |

All levels produce the same container; the decoder reads any mix. Blocks
are 256 KB; `FLAG_CHAIN_RESET` blocks decode independently across cores.

---

## Benchmarks

Every number below is from a single process that also runs the reference
library, so a comparison cannot be met by run-to-run drift. Reproduce with
the commands at the end of this section; cross-platform runs on AWS
Graviton3 and Sapphire Rapids are in [`benchmarks/`](benchmarks/) with the
script that produced them.

### The field, one run (Silesia, Apple M1 Max, one core)

| Codec | Ratio | Compress MB/s | **Decompress MB/s** | |
| :--- | ---: | ---: | ---: | :--- |
| ⚡&nbsp;**Glyd&nbsp;‑‑max** | **3.218** | 277 | **1,733** | ✅ best ratio; 1.27× zstd&nbsp;-3 decode |
| zstd&nbsp;-3 | 3.205 | 319 | 1,361 | |
| zstd&nbsp;-1 | 2.894 | 535 | 1,493 | |
| LZAV-hi | 2.803 | 91 | 3,185 | |
| LZAV | 2.450 | 426 | 3,128 | |
| zstd&nbsp;‑‑fast=1 | 2.438 | 614 | 2,153 | |
| zstd&nbsp;‑‑fast=3 | 2.240 | 684 | 2,307 | |
| ⚡&nbsp;**Glyd&nbsp;default** | **2.192** | 312 | **6,507** | ✅ 1.6× liblz4 decode, better ratio |
| ⚡&nbsp;**Glyd&nbsp;‑‑fast** | **2.176** | 501 | **4,670** | ✅ 1.1× liblz4 decode, better ratio |
| liblz4 | 2.101 | 610 | 4,108 | |
| lz4_flex | 2.097 | 633 | 3,004 | |
| snappy | 2.076 | 607 | 1,495 | |
| zstd&nbsp;‑‑fast=5 | 2.057 | 746 | 2,484 | |
| ⚡&nbsp;**Glyd&nbsp;‑‑turbo** | 1.884 | 263 | **8,647** | ✅ fastest decode, 2.1× liblz4 |

(`examples/field_survey.rs`. This run was taken with other work on the
machine; the headline table above is from a quiet run of the paired
harnesses, which is why its numbers are a few percent higher across the
board — the ordering is the same.)

<details>
<summary><b>Default level vs liblz4, per file</b> (same run)</summary>

| File | Ratio | ⚡ **Glyd MB/s** | liblz4 MB/s | Glyd advantage |
| :--- | ---: | ---: | ---: | ---: |
| dickens | 1.815 | 5,990 | 5,190 | +15% |
| mozilla | 1.926 | 5,420 | 4,830 | +12% |
| mr | 1.902 | 6,650 | 5,570 | +19% |
| nci | 6.846 | 7,130 | 7,240 | −2% |
| ooffice | 1.335 | 6,310 | 4,680 | +35% |
| osdb | 2.294 | 6,270 | 5,110 | +23% |
| reymont | 2.378 | 5,040 | 4,490 | +12% |
| samba | 2.858 | 6,380 | 6,140 | +4% |
| sao | 1.038 | 13,710 | 7,320 | +87% |
| webster | 2.250 | 4,700 | 4,850 | −3% |
| xml | 4.949 | 6,420 | 5,560 | +16% |
| x-ray | 1.000 | 46,600 | 18,100 | +157% |

(AMD Ryzen 9 7950X3D, AVX2 path. On the M1 Max NEON path Glyd wins 12 of 12.)

</details>

<details>
<summary><b><code>--max</code> vs zstd -3, per file</b> (same run, M1 Max)</summary>

| File | ⚡ **Glyd ratio** | zstd -3 ratio | ⚡ **Glyd MB/s** | zstd -3 MB/s |
| :--- | ---: | ---: | ---: | ---: |
| dickens | 2.833 | 2.782 | 1,300 | 1,139 |
| mozilla | 2.776 | 2.810 | 1,697 | 1,275 |
| mr | 2.819 | 2.811 | 1,421 | 1,240 |
| nci | 11.156 | 11.840 | 3,530 | 2,600 |
| ooffice | 1.991 | 1.968 | 1,326 | 978 |
| osdb | 2.863 | 2.880 | 2,093 | 1,608 |
| reymont | 3.483 | 3.420 | 1,665 | 1,339 |
| samba | 4.362 | 4.360 | 2,401 | 1,947 |
| sao | 1.318 | 1.312 | 1,936 | 832 |
| webster | 3.496 | 3.427 | 1,661 | 1,389 |
| xml | 8.245 | 8.414 | 3,115 | 2,400 |
| x-ray | 1.462 | 1.393 | 1,094 | 839 |
| **Total** | **3.218** | **3.204** | **1,841** | **1,417** |

(Same run; `--max` wins ratio on 8 of 12 files and decode on 12 of 12.)

</details>

### Other machines (AWS, same script, same-run references)

| Decompress&nbsp;MB/s | ⚡&nbsp;**Glyd&nbsp;default** | liblz4 | ⚡&nbsp;**Glyd&nbsp;‑‑turbo** | ⚡&nbsp;**Glyd&nbsp;‑‑max** | zstd&nbsp;‑3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Graviton3&nbsp;(c7g.2xlarge, NEON) | **3,690** | 3,160 | **4,980** | **1,205** | 912 |
| Sapphire&nbsp;Rapids (c7i.2xlarge, AVX2) | **4,430** | 3,775 | **5,350** | 858 | 1,291 |

Ratios are identical across machines (the format is deterministic).
`--max` on x86 uses the portable scalar decoder until its AVX2 port lands
(ROADMAP item 1), so it trails zstd -3 there today. Raw outputs and the
launch script: [`benchmarks/`](benchmarks/).

### Multi-core

10 threads, independent 256 KB blocks, default level: **42,900 MB/s** over
Silesia on the M1 Max (`examples/mc.rs`), above the machine's single-core
`memcpy`; 27,100 MB/s on 8 Graviton3 vCPUs, 21,400 MB/s on 8 Sapphire
Rapids vCPUs.

### Beyond Silesia

`EXT_CORPUS=1 scripts/download_corpus.sh` adds real-world formats;
`examples/v7_bench.rs` checks `--max` against zstd -3 file by file:

| File | ⚡ **Glyd --max** | zstd -3 | |
| :--- | ---: | ---: | :--- |
| Linux kernel source tarball (64 MB) | 4.937 | 4.898 | +0.8% |
| NASA HTTP server log (205 MB) | 9.789 | 9.782 | + |
| GitHub Archive JSON events (912 MB) | 10.591 | 10.656 | −0.6% |
| OpenStreetMap PBF | 1.000 | 1.000 | tie (already compressed) |
| NYC taxi Parquet (50 MB) | 1.001 | 1.004 | tie (already compressed) |

### Reproduce

```bash
scripts/download_corpus.sh                                         # Silesia + enwik8
RUSTFLAGS="-C target-cpu=native" cargo run --release --example quick3 -- label 3 0.3 -v   # v6 levels vs liblz4
RUSTFLAGS="-C target-cpu=native" cargo run --release --example v7_bench                  # --max vs zstd -3 / -1
RUSTFLAGS="-C target-cpu=native" cargo run --release --example field_survey 3 0.3        # everything
AWS_PROFILE=... scripts/bench_aws.sh main                          # Graviton3 + Sapphire Rapids, ~$1
```

---

## How it works

The compressed block is split into homogeneous streams — tokens, offsets,
lengths, literals — instead of one interleaved byte stream. That is what
lets the decoder pre-decode 32 tokens per SIMD pass, check bounds once per
chunk and run a copy-only loop, which an inline format such as LZ4's
cannot. The `--max` level keeps the layout and adds 8-way interleaved
entropy coding (Huffman literals, tANS sequences with repeat offsets), so
the entropy decoders run as straight-line SIMD-friendly loops and the
copies stay a separate pass.

Design: [docs/design/format-v7.md](docs/design/format-v7.md). Every
measurement, refuted idea and floor: [docs/engineering-notes.md](docs/engineering-notes.md)
and [CHANGELOG-BENCH.md](CHANGELOG-BENCH.md). What comes next: [ROADMAP.md](ROADMAP.md).

Safety: the decoder is fuzzed with a million random mutations per run
into exact-size buffers with sentinel guards, on every level; it never
allocates per call (a 1.5 MB thread-local scratch for `--max`), and every
unsafe block carries its bound.

---

## Known gaps

- `--max` compresses at 89–91% of zstd -3's speed.
- `--max` decodes 1.3× zstd -3, not the 2× the design aimed at.
- On the extended corpus `--max` beats zstd -3 on 3 of 5 files; it loses
  0.6% on very repetitive JSON.
- `GlydReader`/`GlydWriter` (std::io streaming) carry v6 levels only.
- The `--max` decoder has a NEON path and a scalar fallback: on x86 it
  decodes at 0.66× zstd -3 today (858 vs 1,291 MB/s on Sapphire Rapids).
  The AVX2 port is roadmap item 1; the v6 levels already have AVX2 and lead
  on x86.

---

## Releases and versioning

Current release: **v0.1.0** ([CHANGELOG.md](CHANGELOG.md), [releases](https://github.com/surya-koritala/Glyd/releases)).
Glyd follows SemVer. The on-disk format is versioned separately in every
block header (v6 for default/fast/turbo, v7 for `--max`); every release
decodes every earlier format, and a format change always gets a new
format number, never a silent reinterpretation. Tags are `vMAJOR.MINOR.PATCH`;
each tag ships with release notes and the benchmark tables measured at
that commit.

Contributing: open an issue with the measured number for anything that
touches speed or ratio (`examples/quick3.rs`, `examples/v7_bench.rs` and
`examples/field_survey.rs` all print same-run comparisons); pull requests
run the full suite including the 1M-mutation fuzz in CI.

---

## License

Glyd is released under the [Business Source License 1.1](LICENSE).

- Free to read, modify, redistribute and use for development, testing,
  personal, educational, research and other non-commercial purposes.
- **Commercial or revenue-generating production use requires a commercial
  license.** Contact suryakoritala1324@gmail.com.
- Each version converts to the Apache License 2.0 on its Change Date
  (four years after its first public release; 2030-09-18 for this one).
