# Roadmap

Every item has a measured gate, taken in the same run as the reference
codec, on the corpus and machine named. Nothing ships on a number that
was not reproduced.

## Where the levels stand (Silesia, one core, Apple M1 Max)

| Level | Ratio | Compress MB/s | Decompress MB/s | Reference (same run) |
| :--- | ---: | ---: | ---: | :--- |
| turbo | 1.88 | 280 | 9,200 | liblz4 2.10 / 660 / 4,400 |
| default | 2.19 | 340 | 6,900 | liblz4 |
| max | 3.22 | 300 | 1,860 | zstd -3 3.20 / 330 / 1,440 |
| ultra | 3.80 | 4.8 | 2,190 | zstd -16 3.83 / 8.0 / 1,780; zstd -19 4.01 / 4.0 / 1,640 |

Measured floors that bound further tuning of these levels are recorded in
`CHANGELOG-BENCH.md` and `docs/design/ultra-parse.md`: the v6 copy loop
and the v7 pass-1 walk are at their instruction floors, the entropy coders
within 20% of their symbol-rate floors, and the ultra parse's finder depth
and cutoffs are past their knees.

## Next: move the needle, not the decimals

1. **A larger window (format v8).** zstd -19 gains 2.5% from its 8 MB
   window over 2 MB; ours is a format limit (21 offset bits, and the
   decoder's one-load sequence walk holds 57 bits: 18 + 18 + 21 fits a
   4 MB window, 8 MB needs a second load or shorter length fields). With
   it, the per-block overhead measured at 1.45% (sub-stream size tables
   and padding, entropy tables) comes down too: shared padding, compact
   tables. Gate: `--ultra` ratio >= zstd -19 at its default window
   (4.01) on Silesia, decode unchanged, every v7 file still decoded.

2. **x86-64 parity for the max level's decoder.** 1.03× zstd -3 on
   Sapphire Rapids against 1.3× on ARM; the remaining cost is instruction
   count in the 8-stream loops (~150 per sequence; ARM does it in two
   thirds). Hand-scheduled BMI2 loops for the tANS batch and the walk.
   Gate: >= 1.2× zstd -3 in the published run.

3. **Compression speed of `--max`.** 0.9× zstd -3 on ARM, 0.7× on x86.
   The finder probe loop is at liblz4's efficiency; what is left is the
   entropy stage (histograms, table builds, two-pass literal decision).
   Gate: >= zstd -3's compression speed in the same run.

4. **Modeling the decoder can afford.** Finer length buckets are worth
   0.6% of the sequence section; literal tables by context class and
   larger tANS tables each fractions of a percent. After 1, one at a
   time.

5. **Dictionaries, prepared.** A prepared-dictionary object (pre-seeded
   tables, pre-built entropy tables through the existing reuse flags) for
   small objects, where most stored objects live.

6. **Streaming for v7.** `GlydReader`/`GlydWriter` carry v6 blocks only.

## Known gaps, stated

- `--max` compresses at 89-91% of zstd -3's speed on ARM, 70% on x86.
- `--max` decodes 1.3x zstd -3 on ARM and 1.03x on x86, not the 2x the
  design aimed at; the remaining cost is per-sequence and inherent to
  the sequence format.
- `--ultra` is 2.7% less dense than zstd -19 in the same window and 5%
  less at zstd's default window; it decodes 1.3x faster than zstd -19's
  output.
- Extended corpus: `--max` beats zstd -3 on 3 of 5 files; loses 0.6% on
  JSON event logs and ties at the incompressible floor on Parquet.
- x86 vs ARM bit-identical output for `--max` and `--ultra` is not yet
  verified.
