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

<sub>Silesia corpus (202 MB), Apple M1 Max, one core; every Glyd number is paired with the reference library measured in the same process. Multi-core decode reaches <b>43,000 MB/s</b> on 10 cores, the machine's memory wall. The same story holds on AWS Graviton3; on x86 (Sapphire Rapids) the v6 levels lead, <code>--max</code> decodes about as fast as zstd -3 and <code>--ultra</code> 1.1-1.2× zstd -19. Cross-platform results: <a href="benchmarks/">benchmarks/</a>.</sub>

- 🚀 **Fastest decode at every ratio point** measured, against liblz4, lz4_flex, LZAV, zstd (7 levels) and snappy, in the same run.
- 📦 **Fewer bytes than zstd -3** with the `--max` level, at 30% faster reads.
- 🗜️ **`--ultra`: denser than zstd -16** (Silesia 3.93, zstd -16 3.83, zstd -19 4.01) on an optimal parse, and its output reads 1.3× faster than zstd -19's. Same decoder, same container.
- 🧱 **One container, four levels**, any mix of blocks decodes; independent units (2-16 MB by level) scale across cores.
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

// Dictionaries for small objects (JSON documents, records): train once
// on samples of the data, keep the bytes, prepare on every process.
let dict = glyd::Dict::train(&samples, 110 * 1024);   // samples: &[&[u8]], content budget
std::fs::write("events.glyddict", dict.to_bytes())?;
let dict = glyd::Dict::from_bytes(&std::fs::read("events.glyddict")?).unwrap();
let mut out = Vec::new();
glyd::compress_with_dict(&dict, &doc, &mut out);       // or compress_with_dict_ultra
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
are 256 KB; the parallel paths cut the input into independent units
(2 MB for the v6 levels, 8 MB for `--max`, 16 MB for `--ultra`: the first
block of each carries `FLAG_CHAIN_RESET`) that compress and decode one per
core and cost 0.5-0.7% of ratio against the sequential path; `-s` on the
CLI takes the sequential path.

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
| Graviton3&nbsp;(c7g.2xlarge, NEON) | **3,880** | 3,180 | **5,230** | **1,200** | 928 |
| Sapphire&nbsp;Rapids (c7i.2xlarge, AVX2) | **4,410** | 3,760 | **5,100** | 1,170 | 1,270 |

| Decompress&nbsp;MB/s | ⚡&nbsp;**Glyd&nbsp;‑‑ultra** | zstd&nbsp;‑16 | zstd&nbsp;‑19 |
| :--- | ---: | ---: | ---: |
| Graviton3&nbsp;(c7g.2xlarge, NEON) | **1,380** | 1,040 | 950 |
| Sapphire&nbsp;Rapids (c7i.2xlarge, AVX2) | **1,335** | 1,362 | 1,180 |

Ratios are identical across machines (the format is deterministic).
Absolute speeds on shared cloud instances move by up to 20% between runs
(three c7i runs put zstd -3 at 1,060, 1,110 and 1,270 MB/s); the pairings
within one run are the comparison, and on Sapphire Rapids they say
`--max` decodes at 0.9-1.06× zstd -3 and `--ultra` at 1.1-1.2× zstd -19,
against 1.3× and 1.45× on Graviton3. Raw outputs and the launch script:
[`benchmarks/`](benchmarks/).

### Multi-core

10 threads, independent 2 MB units, default level: **31,900 MB/s** over
Silesia on the M1 Max (`examples/mc.rs`; Silesia's files are 6-50 MB, so
most have fewer units than the machine has cores; 42,900 MB/s with 256 KB
units, which cost 3% of ratio and were the default before v0.3.1). The
AWS numbers in `benchmarks/` are from the 256 KB units.

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

### Small objects (JSON events, one core)

Objects cut from GitHub Archive events, 2,000 per size, each codec with
its own 110 KB dictionary trained on 2,000 other objects (zstd's
trainer for zstd, `Dict::train` for Glyd; zstd's output carries no
checksum, Glyd's 4 bytes per object). Apple M1 Max, one core,
`examples/small_objects.rs` and `examples/small_speed.rs`:

| Object | zstd&nbsp;-3&nbsp;+&nbsp;dict | ⚡&nbsp;**Glyd&nbsp;‑‑max&nbsp;+&nbsp;Dict** | zstd&nbsp;-19&nbsp;+&nbsp;dict | ⚡&nbsp;**Glyd&nbsp;‑‑ultra&nbsp;+&nbsp;Dict** |
| ---: | ---: | ---: | ---: | ---: |
| 1 KB | 4.96 | **4.75** | 5.47 | **5.25** |
| 4 KB | 6.42 | **6.28** | 7.41 | **7.13** |
| 16 KB | 7.66 | **7.65** | 8.95 | **8.84** |

| Object | Codec | Compress | Decompress |
| ---: | :--- | ---: | ---: |
| 1 KB | ⚡&nbsp;**Glyd&nbsp;‑‑max&nbsp;+&nbsp;Dict** | 245&nbsp;MB/s | **920&nbsp;MB/s** |
| 1 KB | zstd -3 + dict | 454&nbsp;MB/s | 1,117&nbsp;MB/s |
| 4 KB | ⚡&nbsp;**Glyd&nbsp;‑‑max&nbsp;+&nbsp;Dict** | 321&nbsp;MB/s | **1,209&nbsp;MB/s** |
| 4 KB | zstd -3 + dict | 588&nbsp;MB/s | 1,440&nbsp;MB/s |
| 16 KB | ⚡&nbsp;**Glyd&nbsp;‑‑max&nbsp;+&nbsp;Dict** | 394&nbsp;MB/s | **1,674&nbsp;MB/s** |
| 16 KB | zstd -3 + dict | 635&nbsp;MB/s | 1,890&nbsp;MB/s |

On small objects zstd is ahead: 1-4% denser at `-3` and 1-4% at `-19`,
1.6-1.9× faster to compress and 1.1-1.2× faster to decode. Without a
dictionary Glyd `--max` is 6-12% less dense than zstd -3 on objects
under 16 KB.

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

## Real data, real machines

The verification and benchmark program (8.7 GB of logs, JSON, SQL
dumps and Parquet; zstd -3, zstd -19 and LZ4 on the same AWS machines
and thread counts; small objects with dictionaries; a real S3 round
trip costed at list prices) and its results: [docs/benchmarks/](docs/benchmarks/README.md)
and [docs/benchmarks/suite-2026-09-20.md](docs/benchmarks/suite-2026-09-20.md).
The short version, Graviton3 and Sapphire Rapids: `--max` stores 2.8%
less than zstd -3 over the corpus (22% less on JSON, 5% on logs,
parity on SQL and Parquet), decodes 3-6x faster with 8 cores and 1.03x
(Graviton3) / 0.80x (Sapphire Rapids) on one, and compresses at
0.58-0.66x zstd -3's speed; `--ultra` equals zstd -19 (10% smaller on
JSON); `--max -r` stores 19% less than zstd -3 and 2% less than
zstd -19 at 540-675 MB/s on 8 cores, `--ultra -r` 10% less than
zstd -19. A terabyte-year in S3 at one read a month costs within 2%
either way for the plain levels; the record-mode rows are in the
report.

## Record mode: logs and table dumps as columns

Byte-level matching is a plateau: on real data zstd -19, xz and Glyd
`--ultra` land within a few percent of each other. The redundancy of a
log or a table dump is not in nearby bytes but in the same field of
every record. `glyd -r` (record mode, [design](docs/design/format-v7.md#record-mode-v040-typed-columns-before-the-level))
detects delimited lines, SQL dumps and JSON lines, turns them into one
typed stream per field or key path (integer, decimal and date-time
deltas, dictionaries with recency ranks, text), compresses those with
the chosen level in parallel 32 MB units, and rebuilds the bytes
exactly. Anything else is left as it is.

The 8.7 GB benchmark corpus, 10 cores, every decode byte-checked
(`examples/bench_suite.rs`, rows in [`benchmarks/suite/m1-max-v0.4.0/`](benchmarks/suite/m1-max-v0.4.0/)):

| Data | Glyd&nbsp;‑‑max | Glyd&nbsp;‑‑ultra | ⚡&nbsp;**Glyd&nbsp;‑‑max&nbsp;‑r** | ⚡&nbsp;**Glyd&nbsp;‑‑ultra&nbsp;‑r** | zstd&nbsp;-3 | zstd&nbsp;-19 | **‑‑ultra&nbsp;‑r vs zstd&nbsp;-19** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| SQL dumps (3.9 GB) | 4.96 | 6.98 | **8.38** | **10.14** | 4.97 | 7.10 | **1.43× smaller** |
| Access logs (0.55 GB) | 10.56 | 14.40 | **21.4** | **23.0** | 9.53 | 14.9 | **1.54× smaller** |
| Pageview logs (0.71 GB) | 3.71 | 4.68 | **4.10** | **4.92** | 3.56 | 4.82 | 1.02× |
| JSON events (2.6 GB) | **13.26** | **16.40** | 13.26 | 16.40 | 10.46 | 15.07 | **1.09× smaller** |
| Parquet (1.0 GB) | 1.01 | 1.02 | 1.01 | 1.02 | 1.01 | 1.02 | 1.00× |
| Whole corpus | 3.96 | 4.65 | **4.75** | **5.20** | 3.85 | 4.66 | **1.11× smaller** |

`--max -r` is 2% smaller than zstd -19 over the corpus while
compressing at 1,100 MB/s against 19 (10 cores); `--ultra -r` is 11%
smaller than zstd -19, and plain `--ultra` now equals it. JSON events
are not record-shaped (their redundancy is inside each record and
across the whole file), so `-r` hands them to the plain level, where
the long-distance matcher (repeats up to 128 MB back, on at `--max`
and `--ultra`) does the work: 13.26 and 16.40 against 11.49 and 14.59
with the 8 MB window, 27% and 9% smaller than zstd -3 and zstd -19.
Costs: `--max` compresses the corpus at 2,000 MB/s (2,400 without the
matcher; zstd -3 4,000), record-mode reads run at 4,400 MB/s instead
of 6,500-8,700 for plain `--max`.

Telemetry is where the multiples are. Measurements as rows (a cluster
trace, daily weather, taxi trips) in CSV or as JSON lines
(`scripts/download_ext_corpus.sh`, 128 MB slices, 10 cores, rows in
[`benchmarks/suite/m1-max-v0.4.0/bench_suite_ext2.txt`](benchmarks/suite/m1-max-v0.4.0/bench_suite_ext2.txt)):

| Data | ⚡&nbsp;**Glyd&nbsp;‑‑max&nbsp;‑r** | ⚡&nbsp;**Glyd&nbsp;‑‑ultra&nbsp;‑r** | zstd&nbsp;-3 | zstd&nbsp;-19 | **‑‑max&nbsp;‑r vs zstd&nbsp;-3** | **‑‑ultra&nbsp;‑r vs zstd&nbsp;-19** |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| Alibaba cluster machine usage, CSV | **12.6** | **13.7** | 4.46 | 6.90 | **2.8× smaller** | **2.0× smaller** |
| the same as JSON lines | **54.6** | **58.9** | 15.7 | 28.7 | **3.5× smaller** | **2.1× smaller** |
| NOAA daily weather, CSV | **19.6** | **23.1** | 6.99 | 12.0 | **2.8× smaller** | **1.9× smaller** |
| the same as JSON lines | **47.0** | **58.0** | 18.7 | 31.6 | **2.5× smaller** | **1.8× smaller** |
| NYC taxi trips, CSV export | **8.93** | **9.37** | 5.64 | 8.38 | **1.6× smaller** | 1.1× smaller |
| Common Crawl index (a hash per line) | 7.73 | 9.36 | 7.55 | 9.66 | 1.02× | 0.97× |

`--max -r` compresses these at 340-680 MB/s (10 cores) and reads back
at 770-2,500 MB/s. The last row is the honest limit: a line that is
mostly a hash has nothing a column can model, and API events with
hashes and free text (GitHub Archive) gain 1.6% from columns and stay
plain.

## Known gaps

- `--max` compresses at 0.58-0.66x zstd -3's speed on server cores
  (Graviton3, Sapphire Rapids; 0.65-0.78x before the matcher): its
  2 MB of finder tables miss a 1 MB L2, and the long-distance pass
  takes another 15-37% where it stays on (text, logs, JSON: 3-16%
  fewer bytes for it; `zstd -3 --long=27` pays 17-60% for the same
  window). Record mode's transform halves the write speed again
  (200-400 MB/s per core).
- `--max` decodes 1.3× zstd -3, not the 2× the design aimed at.
- On the extended corpus `--max` beats zstd -3 on 3 of 5 files; it loses
  0.6% on very repetitive JSON.
- Small objects with a dictionary: zstd is 1-4% denser and 1.6-1.9×
  faster to compress (table above).
- `GlydReader`/`GlydWriter` (std::io streaming) carry v6 levels only.
- On x86 (Sapphire Rapids) `--max` decodes at 0.80x zstd -3 on the real-data
  corpus (0.9-1.06x on Silesia), not the 1.03-1.3x it reaches on ARM:
  x86-64's 16 general registers spill the 8-stream entropy loops that
  ARM's 31 keep in registers, and the 8 MB window's far copies miss its
  smaller caches.
- The CLI spends more CPU per decoded byte than zstd's (a checksum per
  block, a whole-file buffer, the parallel decode's threads): 1.3x on
  Graviton3, 2x on Sapphire Rapids over the corpus.
- `--ultra` is 1.2% less dense than zstd -19 on Silesia (3.96 vs
  4.01); the gap sits on structured data (mozilla, xml, samba 3-4%),
  text and binaries are within 1-2%. On the real-data corpus the two
  are equal.

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
