# GOAL: Beat LZ4 on every single-core metric with ONE default configuration

Work is DONE only when every gate G1 through G8 below PASSES on a single commit,
measured by the procedure in Section 1, and reproduced on a second run from a
fresh clone. Until then, keep working. Do not change any threshold in this file.

---

## 0. Definition of done

1. All gates G1 to G8 PASS on the same commit.
2. `RESULTS.md` at the repo root contains the gate table (Section 4) for that commit.
3. A fresh clone into a clean Linux directory (NOT under OneDrive or /mnt/c) is
   rebuilt and the full benchmark is run twice. Both runs pass every gate.
4. Branch is pushed and a pull request `compression-v2 -> main` is opened with
   `RESULTS.md` and the two CSVs attached.

---

## 1. Measurement procedure (mandatory, identical for every codec)

Machine: AMD Ryzen 9 7950X3D (16C/32T, Zen 4), WSL2, 30 GB RAM, RTX 4080 Super.

Before any timed run:
- Stop `llama-server` and any compile jobs. `nvidia-smi` and `top` must show the box idle.
- Build once: `RUSTFLAGS="-C target-cpu=native" cargo build --release --example bench`
- Single-core measurements are pinned: `taskset -c 4 target/release/examples/bench ...`
  Use the same core for every codec in the same session.

Baselines, built into the same benchmark binary with the same flags:
- **LZ4 reference = C liblz4** via the `lz4` crate (links `lz4-sys`), block API,
  default acceleration (1), `compress_default` / `decompress_safe`.
  This is the gate baseline. `lz4_flex` stays as an informational column only.
- Snappy via `snap` (informational). Zstd level 1 via `zstd` (informational).

Timing policy for every (codec, file, direction):
- 1 warm-up iteration, then repeat until at least 1.0 s has elapsed and at least
  5 iterations have run. That is one measurement. Take 5 measurements. Report the median.
- Throughput = original bytes / elapsed seconds, in GB/s (1024^3).
- Decompression is measured raw-to-raw: no checksum on either side.
  The verified (checksum) path is reported separately and is not gated.
- Ratio = original bytes / compressed bytes, INCLUDING all headers and framing.
- Corpus-total throughput = (sum of bytes) / (sum of time), not the mean of per-file rates.

Corpora:
- **Silesia** (12 files, primary): `bash scripts/download_corpus.sh`
- **enwik8** (holdout, 100 MB): http://mattmahoney.net/dc/enwik8.zip
  Add it to `scripts/download_corpus.sh`. Never tune against it; run it only at gate time.
- **Workloads** (4 synthetic sets in `examples/bench.rs`): JSON Logs, Columnar DB,
  Binary RPC, Source Code. All four must be reported. None may be dropped or renamed.

`bench --csv` must emit one row per (corpus, file) with every column needed to
evaluate the gates. `bench --gates` must print the Section 4 table with PASS/FAIL.

---

## 2. Gates

All gates apply to ONE default configuration (the codec's default level with no
flags). Extra levels may exist but do not count toward any gate.

| Gate | Metric | Threshold |
|---|---|---|
| **G1** | Correctness | `cargo test --release --all-targets` green. Silesia and enwik8 round-trip through sequential, parallel, and cross paths (seq->par, par->seq). Fuzz: 1,000,000 random mutations of compressed streams, zero panics, zero out-of-bounds. |
| **G2** | Ratio, Silesia | Corpus total ratio >= 1.02 x LZ4 total. Every one of the 12 files >= 0.95 x its LZ4 ratio. |
| **G3** | Decompression, 1 core, Silesia | Corpus-total throughput >= 1.05 x LZ4. Every file >= 0.90 x LZ4. At least 9 of 12 files >= LZ4. |
| **G4** | Compression, 1 core, Silesia | Corpus-total throughput >= 1.00 x LZ4. Every file >= 0.80 x LZ4, x-ray and sao included. |
| **G5** | Holdout, enwik8 | Ratio >= LZ4. Decompression 1C >= LZ4. Compression 1C >= 0.90 x LZ4. |
| **G6** | Workloads | Ratio >= LZ4 on all 4. Decompression 1C >= LZ4 on all 4. |
| **G7** | Multi-core guard | 16C decompression >= 10 GB/s on mozilla, nci, webster, samba. 16C compression >= 4 x the 1C figure on the same files. |
| **G8** | Memory | Compressor working set <= 1 MB per stream beyond I/O buffers. Decompressor allocates nothing beyond the output buffer plus padding. |

---

## 3. Rules (anti-gaming)

- One default configuration must satisfy G2, G3, and G4 at the same time.
  Satisfying ratio with a "strong" level and speed with a "fast" level is a FAIL.
- No dictionaries or tables derived from any benchmark corpus. No per-file or
  per-corpus tuning. No detection of benchmark inputs.
- Baseline codecs are unmodified upstream crates at default settings. Never edit them.
- Never change a threshold in this file. If a gate looks impossible, write the
  evidence in `RESULTS.md` under "Blocked gates" and keep working on the others.
- Same `RUSTFLAGS`, same core pinning, same timing policy for every codec.
- CI (`.github/workflows/ci.yml`) builds `x86-64-v3` and validates correctness
  only. It is not a gate measurement and must not be quoted as one: that build
  has no AVX-512, which shifts single-core compression about +6% and
  decompression about -7% against the `target-cpu=native` build this section
  mandates.
  A measurement that breaks this is invalid and must be rerun.
- Every commit does one thing and ships its own `bench --csv` output plus a
  one-line delta in `CHANGELOG-BENCH.md`. Commit messages state what changed and why.
- G1 may never regress. Any format change bumps `CURRENT_VERSION`, adds round-trip
  tests for the new paths, and keeps the fuzz test green.
- Do not edit `README.md` performance tables until all gates pass. Then the README
  tables are copied from `RESULTS.md` verbatim.

---

## 4. Reporting: RESULTS.md

Regenerate `RESULTS.md` on every commit that changes a number. It contains:

1. Commit hash, date, machine, `rustc --version`, `RUSTFLAGS`, taskset core.
2. Gate table: one row per gate with Measured, Threshold, LZ4 value, PASS/FAIL.
3. Per-file tables for Silesia, enwik8, and workloads: ratio, comp 1C, decomp 1C,
   for Glyd, liblz4, lz4_flex, Snappy, Zstd-1.
4. "Blocked gates" section, if any, with evidence.
5. Path to the CSV for this run.

---

## 5. Baseline (verified 2026-09-16, branch compression-v2, commit 21bdaf4)

Independently measured on this machine. Baseline here is `lz4_flex`; the agent's
first task is to add liblz4 and re-baseline. Expect liblz4 to be equal or faster.

Silesia total ratio: Glyd 1.90x, LZ4 2.10x (deficit 9.3%).

| File | Ratio Alat / LZ4 | Decomp 1C GB/s Alat / LZ4 | Comp 1C GB/s Alat / LZ4 |
|---|---|---|---|
| dickens | 1.34 / 1.59 | 1.58 / 3.72 | 0.31 / 0.43 |
| mozilla | 1.77 / 1.93 | 2.10 / 2.98 | 0.36 / 0.68 |
| mr | 1.58 / 1.83 | 2.0 / 3.8 | 0.41 / 0.82 |
| nci | 5.41 / 6.06 | 5.00 / 4.48 | 1.06 / 1.18 |
| ooffice | 1.34 / 1.41 | 2.0 / 3.7 | 0.24 / 0.75 |
| osdb | 2.06 / 1.91 | 4.24 / 3.61 | 0.47 / 0.76 |
| reymont | 1.59 / 2.08 | 1.64 / 3.42 | 0.42 / 0.51 |
| samba | 2.44 / 2.80 | 3.0 / 3.8 | 0.64 / 0.77 |
| sao | 1.05 / 1.06 | 2.6 / 4.9 | 0.21 / 0.88 |
| webster | 1.78 / 2.06 | 1.83 / 3.07 | 0.34 / 0.52 |
| xml | 3.82 / 4.35 | 4.2 / 4.2 | 0.87 / 1.07 |
| x-ray | 1.02 / 1.01 | 3.4 / 17.0 | 0.19 / 2.65 |

Gate status at baseline: G1 PASS. G2 FAIL. G3 FAIL. G4 FAIL. G5 not run.
G6 FAIL (Source Code ratio, Columnar ratio). G7 PASS. G8 not measured.

---

## 6. Suggested order of attack (guidance, not a rule)

1. Harness first: add liblz4 baseline, taskset pinning, median-of-5 policy, enwik8
   download, `--csv` columns, `--gates` output, `RESULTS.md` generator. Re-baseline.
2. Early-abort incompressibility detection in the compressor: sample the match rate
   in the first few KB of a block and bail to a raw block. Targets G4 on x-ray and sao.
3. Faster default parser: single hash probe, smaller table, SIMD multi-position
   hashing. Keep lazy matching only if G4 still passes with it.
4. Batch token decode: prefix-sum literal and match lengths for 8 or 16 tokens with
   SIMD to compute output positions up front, then execute copies. Flatten the nested
   offset-size branches. Targets G3.
5. Repeat-offset codes and match lengths beyond the 2047 cap. Targets G2 and G6
   (Source Code) at zero decode cost.
6. Window to the full 256 KB chunk (format v3). Targets G2 on mozilla and webster.
7. Optimal parsing in the default level only if G4 still holds after it.

Decode speed is governed by token count, not vector width. Every change that
reduces tokens helps G2 and G3 together. Measure after every step.
