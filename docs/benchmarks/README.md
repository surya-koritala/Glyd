# The verification and benchmark program

What has to be true before Glyd's numbers are used for a storage
decision, and how each point is checked. Results of each run are in
[`../../benchmarks/suite/`](../../benchmarks/suite/) as JSON lines, one
directory per machine, with the log of the run; the tables in
[`suite-2026-09.md`](suite-2026-09.md) are generated from them by
`scripts/report_suite.py`.

## 1. Correctness

- `cargo test --release`: 80 tests, among them round trips of every
  level through the container, exact-size destination buffers,
  mutation fuzzing of every decoder (`tests/fuzz_safety.rs`: a million
  mutations per run into sentinel-guarded buffers; `tests/v7_fuzz.rs`
  with `V7_FUZZ=1000000` for the entropy-coded family, dictionaries and
  compact blocks), and the compatibility fixtures.
- Compatibility: `tests/format_compat.rs` decodes files written by
  v0.2.0 (formats v6 and v7), v0.3.0 (v8) and v0.4.0 (v9 compact
  blocks, a serialized dictionary and objects compressed with it). A
  fixture is never regenerated.
- Exact recovery at scale: `scripts/verify_roundtrip.sh` compresses
  every file of the benchmark corpus at every level, single- and
  multi-core, through the CLI, and compares the decompressed bytes with
  `cmp`; eight corrupted copies per file and level (bit flips, byte
  overwrites, truncations, runs of random bytes) must be rejected, or,
  when their checksum still passes, decode to the original bytes.
- Every decode in the benchmark harness is compared with its input
  before a time is recorded.

## 2. Data

`scripts/download_bench_corpus.sh`, 8.7 GB measured:

| Kind | Files | Bytes |
| :--- | :--- | ---: |
| JSON events | GitHub Archive, three hours (2024-01-15 12:00 and 18:00, 2024-01-16 12:00) | 2.56 GB |
| Logs | NASA HTTP July 1995, ClarkNet August and September 1995, Wikipedia pageviews (three hours of 2024-01-15) | 1.26 GB |
| Database exports | Wikipedia SQL dumps: simplewiki page, pagelinks, categorylinks, templatelinks; enwiki redirect, page_props | 3.92 GB |
| Parquet (already compressed) | NYC TLC for-hire trips January and February 2024, yellow taxi February 2024 | 0.99 GB |

Small-object dictionaries are trained on `corpus/bench/train/`: another
hour of GitHub Archive (2024-01-14), the NASA August 1995 log, another
day of pageviews. No measured byte is in the training data.

## 3. Comparison

`examples/bench_suite.rs`, the same binary for every codec:

- Codecs: Glyd default, `--max`, `--ultra`; zstd -3 and -19 (libzstd
  1.5 through the `zstd` crate; `-T N` via zstdmt at the same thread
  count as Glyd); LZ4 (liblz4 frame, level 1). Small objects: every
  codec with a 110 KB dictionary — zstd's trainer for zstd, `Dict::train`
  for Glyd, zstd's trained content for LZ4 (lz4_flex's block API with an
  external dictionary; liblz4's Rust binding exposes none).
- Threads: `--threads 1` and `--threads <vCPUs>` on the same machine.
  zstd and LZ4 decompress on one thread whatever the setting (their
  formats offer no more; the CLIs do the same); Glyd's parallel decode
  uses the given count.
- Repeats: three per compress and decompress, the median reported (one
  for `--ultra` and zstd -19 single-thread, whose passes over a
  gigabyte take minutes; two at all threads).
- Bytes: every codec's whole output, framing and checksums included;
  small objects with and without the dictionary's own size.
- Memory: peak resident set of one compress and decompress of the
  file's first 64 MB in a fresh process (input, compressed and
  decompressed buffers included, so 64 MB + compressed + 64 MB is the
  floor).
- Small-object latency: per object, one thread, the best of the repeats
  per object, p50 / p90 / p99.

## 4. One real workflow

`scripts/s3_workflow.sh`, on the instance, for each codec's CLI at all
threads (`glyd --max -m`, `glyd --ultra -m`, `zstd -3 -T0`, `zstd -19
-T0`, `lz4 -1`, and the uncompressed baseline): compress the corpus,
`aws s3 cp` it to a bucket in the same region, download it to a fresh
directory, decompress, sha256 every file against the original. Wall
and CPU seconds per step. Cost per month of holding the corpus in S3
Standard and reading it back once, ten and a hundred times, at the
region's public on-demand prices (dated in the script): storage +
PUT and GET requests + instance seconds for the compress and the
decompress; transfer between EC2 and S3 in one region is free, and
internet egress per read is shown separately.

## Acceptance

Exact recovery everywhere (the verification above), improvements that
hold on both machines and across repeats, and a lower monthly cost for
the workload than the same workflow with zstd — or an honest statement
of where it is not.
