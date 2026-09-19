<h1 align="center">Glyd</h1>
<p align="center"><strong>The world's fastest-decoding open-source compression.</strong><br>
Fewer bytes than zstd -3; a <code>--ultra</code> level denser than zstd -16 and within 2% of zstd -19. Reads 1.3× faster than zstd, up to 2.1× faster than LZ4.</p>

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
| ⚡&nbsp;**Glyd&nbsp;‑‑max** | **3.25** | 310&nbsp;MB/s | **1,890&nbsp;MB/s** | **1.3×** zstd&nbsp;-3 (1,490&nbsp;MB/s); denser (3.20) |
| ⚡&nbsp;**Glyd&nbsp;‑‑ultra** | **3.93** | 3.8&nbsp;MB/s | **2,150&nbsp;MB/s** | **1.3×** zstd&nbsp;-19 (1,640&nbsp;MB/s); denser than zstd&nbsp;-16 (3.83), 2% below zstd&nbsp;-19 (4.01) |

<sub>Silesia corpus (202 MB), Apple M1 Max, one core; every Glyd number is paired with the reference library measured in the same process. Multi-core decode reaches <b>43,000 MB/s</b> on 10 cores, the machine's memory wall. The same story holds on AWS Graviton3; on x86 (Sapphire Rapids) the v6 levels lead and <code>--max</code> decodes 1.03× zstd -3. Cross-platform results: <a href="benchmarks/">benchmarks/</a>.</sub>

- 🚀 **Fastest decode at every ratio point** measured, against liblz4, lz4_flex, LZAV, zstd (7 levels) and snappy, in the same run.
- 📦 **Fewer bytes than zstd -3** with the `--max` level, at 30% faster reads.
- 🗜️ **`--ultra`: denser than zstd -16** (Silesia 3.93, zstd -16 3.83, zstd -19 4.01) on an optimal parse, and its output reads 1.3× faster than zstd -19's. Same decoder, same container.
- 🧱 **One container, three levels**, any mix of blocks decodes; independent 256 KB blocks scale across cores.
- 🛡️ **Fuzzed** with a million mutations per run into exact-size buffers; no per-call allocation in the decoder.
- 🔌 **Rust, C ABI, CLI**, streaming `std::io` adapters, dictionaries for small objects.

---

## What Glyd saves you

> **Try it:** [surya-koritala.github.io/Glyd/savings.html](https://surya-koritala.github.io/Glyd/savings.html) — enter what you store and what you compress with today.

Stored bytes scale with `1 / ratio`. Most analytics data today is compressed
with Snappy or LZ4 (Parquet's default codec is Snappy). Moving it to Glyd
`--max` cuts the bytes stored and moved by about a third; moving from zstd
saves CPU on every read instead.

| You&nbsp;store&nbsp;today | Compressed&nbsp;with | ⚡&nbsp;**Glyd&nbsp;‑‑max** | Bytes&nbsp;saved | **Saved&nbsp;per&nbsp;year** ($21/TB‑month) |
| ---: | :--- | ---: | ---: | ---: |
| 100&nbsp;TB | Snappy (2.08) | 63.8&nbsp;TB | 36.2% | **$9,100** |
| 1 PB | Snappy | 638 TB | 36.2% | **$91,000** |
| 1 PB | LZ4 (2.10) | 646 TB | 35.4% | **$89,000** |
| 10 PB | Snappy | 6.38 PB | 36.2% | **$912,000** |
| 100 PB | Snappy | 63.8 PB | 36.2% | **$9.1 million** |
| 1&nbsp;PB | zstd&nbsp;-3 (3.20) | 985&nbsp;TB | 1.5% | $3,800, plus **22% fewer decode CPU‑seconds** on every read |

Formula: `saved_per_year = stored_TB × (1 − old_ratio / 3.25) × price_per_TB_month × 12`.
Ratios are Silesia, same run; your data will differ — measure it with
`glyd -b yourfile` before believing any table, including this one.

**At market scale:** object storage holds hundreds of exabytes (AWS said in
March 2026 that S3 alone stores "hundreds of exabytes" across 500 trillion
objects). At list price, **every 1% fewer bytes across 100 EB is about
$250 million a year**; moving 100 EB from Snappy or LZ4 to Glyd `--max`
(35% fewer bytes) is worth about **$8.8 billion a year**.

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
| **‑‑max**&nbsp;(‑9) | The zstd slot: fewer bytes than zstd -3, 30% faster reads | v7 format: 8-way interleaved Huffman literals + tANS-coded sequences, repeat offsets, 2 MB window, double-fast lazy parse |
| **‑‑ultra**&nbsp;(‑19) | Write once, read many: cold storage, release assets, datasets. Fewest bytes; compresses at single-digit MB/s | v7 format on an optimal parse: binary-tree match finder, every position priced in the coder's own bits, cheapest path through the block ([design](docs/design/ultra-parse.md)) |

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
| ⚡&nbsp;**Glyd&nbsp;‑‑max** | **3.254** | 277 | **1,733** | ✅ best ratio; 1.27× zstd&nbsp;-3 decode |
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
| dickens | 2.848 | 2.782 | 1,367 | 1,220 |
| mozilla | 2.801 | 2.810 | 1,734 | 1,316 |
| mr | 2.827 | 2.811 | 1,519 | 1,320 |
| nci | 11.735 | 11.840 | 3,696 | 2,746 |
| ooffice | 1.999 | 1.968 | 1,403 | 1,035 |
| osdb | 2.903 | 2.880 | 2,243 | 1,731 |
| reymont | 3.506 | 3.420 | 1,764 | 1,445 |
| samba | 4.471 | 4.360 | 2,480 | 2,039 |
| sao | 1.326 | 1.312 | 1,901 | 889 |
| webster | 3.538 | 3.427 | 1,640 | 1,456 |
| xml | 8.386 | 8.414 | 3,194 | 2,526 |
| x-ray | 1.465 | 1.393 | 1,157 | 866 |
| **Total** | **3.254** | **3.204** | **1,891** | **1,487** |

(Same run; `--max` wins ratio on 8 of 12 files and decode on 12 of 12.)

</details>

### Other machines (AWS, same script, same-run references)

| Decompress&nbsp;MB/s | ⚡&nbsp;**Glyd&nbsp;default** | liblz4 | ⚡&nbsp;**Glyd&nbsp;‑‑turbo** | ⚡&nbsp;**Glyd&nbsp;‑‑max** | zstd&nbsp;‑3 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Graviton3&nbsp;(c7g.2xlarge, NEON) | **3,800** | 3,170 | **5,170** | **1,210** | 916 |
| Sapphire&nbsp;Rapids (c7i.2xlarge, AVX2) | **3,790** | 3,250 | **4,590** | **1,180** | 1,110 |

| Decompress&nbsp;MB/s | ⚡&nbsp;**Glyd&nbsp;‑‑ultra** | zstd&nbsp;‑16 | zstd&nbsp;‑19 |
| :--- | ---: | ---: | ---: |
| Graviton3&nbsp;(c7g.2xlarge, NEON) | **1,390** | 1,024 | 924 |
| Sapphire&nbsp;Rapids (c7i.2xlarge, AVX2) | **1,325** | 1,218 | 1,077 |

Ratios are identical across machines (the format is deterministic).
Absolute speeds on shared cloud instances move by up to 10% between runs;
the pairings within one run are the comparison. Raw outputs and the
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
RUSTFLAGS="-C target-cpu=native" cargo run --release --example ultra_bench               # --ultra vs zstd -16 / -19
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
- On x86 (Sapphire Rapids) `--max` decodes at 1.03× zstd -3, not the 1.3×
  it reaches on ARM: x86-64's 16 general registers spill the 8-stream
  entropy loops that ARM's 31 keep in registers.
- `--ultra` is 2% less dense than zstd -19 (3.93 vs 4.01, both with an
  8 MB window); the gap sits on structured data (mozilla, xml, samba
  3-4%), text and binaries are within 1-2%.

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
