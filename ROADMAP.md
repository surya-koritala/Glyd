# Roadmap

Every item has a measured gate, taken in the same run as the reference
codec, on the corpus and machine named. Nothing ships on a number that
was not reproduced.

## Where the levels stand (Silesia, one core, Apple M1 Max)

| Level | Ratio | Compress MB/s | Decompress MB/s | Reference (same run) |
| :--- | ---: | ---: | ---: | :--- |
| turbo | 1.88 | 280 | 9,200 | liblz4 2.10 / 660 / 4,400 |
| default | 2.19 | 340 | 6,900 | liblz4 |
| max | 3.25 | 310 | 1,890 | zstd -3 3.20 / 340 / 1,490 |
| ultra | 3.93 | 3.8 | 2,150 | zstd -16 3.83 / 8.0 / 1,790; zstd -19 4.01 / 4.0 / 1,640 |

Measured floors that bound further tuning of these levels are recorded in
`CHANGELOG-BENCH.md` and `docs/design/ultra-parse.md`: the v6 copy loop
and the v7 pass-1 walk are at their instruction floors, the entropy coders
within 20% of their symbol-rate floors, and the ultra parse's finder depth
and cutoffs are past their knees. Format v8 (v0.3.0) took the window to
8 MB and the per-block overhead to 421 bytes; smaller blocks lose more
overhead than they gain in adaptivity (measured: 128 KB blocks -0.1%,
64 KB -0.9%).

## Next: move the needle, not the decimals

1. **The last 2% to zstd -19.** It sits on structured data (mozilla,
   xml, samba, nci: 3-4% behind; text and binaries within 1-2%). What
   remains per block is ~420 bytes of framing and tables (3% of an nci
   block), and modeling the parse cannot see: rep-code semantics with a
   literal-length-zero context (zstd's), literal tables by context class.
   Gate: `--ultra` ratio >= zstd -19 on Silesia, decode unchanged.

2. **x86-64 parity for the max level's decoder.** 1.03× zstd -3 on
   Sapphire Rapids against 1.3× on ARM; the remaining cost is instruction
   count in the 8-stream loops (~150 per sequence; ARM does it in two
   thirds). Hand-scheduled BMI2 loops for the tANS batch and the walk.
   Gate: >= 1.2× zstd -3 in the published run.

3. **Compression speed of `--max`.** 0.9× zstd -3 on ARM, 0.7× on x86.
   The finder probe loop is at liblz4's efficiency; what is left is the
   entropy stage (histograms, table builds, two-pass literal decision).
   Gate: >= zstd -3's compression speed in the same run.

4. **Modeling the decoder can afford.** Literal tables by context
   class and larger tANS tables, each fractions of a percent; the finer
   length buckets landed in v8 (worth 1% on nci, a wash on text).

5. **Dictionaries, prepared.** A prepared-dictionary object (pre-seeded
   tables, pre-built entropy tables through the existing reuse flags) for
   small objects, where most stored objects live.

6. **Streaming for v7.** `GlydReader`/`GlydWriter` carry v6 blocks only.

## Known gaps, stated

- `--max` compresses at 89-91% of zstd -3's speed on ARM, 70% on x86.
- `--max` decodes 1.3x zstd -3 on ARM and 1.0x on x86, not the 2x the
  design aimed at; the remaining cost is per-sequence and inherent to
  the sequence format.
- `--ultra` is 2% less dense than zstd -19 (same 8 MB window); it
  decodes 1.3x faster than zstd -19's output.
- Extended corpus: `--max` beats zstd -3 on 3 of 5 files; loses 0.6% on
  JSON event logs and ties at the incompressible floor on Parquet.
- x86 vs ARM bit-identical output for `--max` and `--ultra` is not yet
  verified.
