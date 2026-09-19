# Roadmap

Every item has a measured gate, taken in the same run as the reference
codec, on the corpus and machine named. Nothing ships on a number that
was not reproduced.

## Where the levels stand (Silesia, one core, Apple M1 Max)

| Level | Ratio | Compress GB/s | Decompress GB/s | Reference (same run) |
| :--- | ---: | ---: | ---: | :--- |
| turbo | 1.88 | 0.28 | 9.2 | liblz4 2.10 / 0.66 / 4.4 |
| default | 2.19 | 0.34 | 6.9 | liblz4 |
| max | 3.22 | 0.30 | 1.86 | zstd -3 3.20 / 0.33 / 1.44 |

Measured floors that bound further tuning of these levels are recorded in
`CHANGELOG-BENCH.md` ("where the floor is", "the ceiling", "decoder on the
real parse"): the v6 copy loop and the v7 pass-1 walk are at their
instruction floors, the parse probe loop is at liblz4's efficiency, and
the entropy coders are within 20% of their symbol-rate floors.

## Next: move the needle, not the decimals

1. **x86 parity for the max level.** The v7 decoder is NEON with a scalar
   fallback; the AVX2 port is mechanical and required before the x86
   numbers in `benchmarks/` mean anything.
   Gate: c7i decode of `--max` >= 1.25x zstd -3 in the same run.

2. **Optimal parsing (`--ultra`).** The decoder does not change. A
   binary-tree match finder and a backward cost-model parse over the
   coder's actual costs, the way zstd's high levels win their ratio.
   Gate: Silesia ratio >= 3.55 (+11% over zstd -3) at >= 0.08 GB/s
   compression, decode unchanged. This is the "write once, read many"
   proposition: fewer bytes stored and moved than zstd's default, faster
   reads, slower writes.

3. **Modeling the decoder can afford.** Finer length buckets, literal
   tables by context class, larger tANS tables. Each is worth fractions
   of a percent; done after 2, measured one at a time.

4. **Dictionaries, prepared.** A prepared-dictionary object (pre-seeded
   tables, pre-built entropy tables through the existing reuse flags) for
   small objects, where most stored objects live.

5. **Streaming for v7.** `GlydReader`/`GlydWriter` carry v6 blocks only.

## Known gaps, stated

- `--max` compresses at 89-91% of zstd -3's speed.
- `--max` decodes 1.3x zstd -3, not the 2x the design aimed at; the
  remaining cost is per-sequence and inherent to the sequence format.
- Extended corpus: `--max` beats zstd -3 on 3 of 5 files; loses 0.6% on
  JSON event logs and ties at the incompressible floor on Parquet.
- x86 vs ARM bit-identical output for `--max` is not yet verified.
