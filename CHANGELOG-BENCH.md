## Harness + G1: liblz4 baseline, real 1M fuzz, OOB write fix

- Baseline switched to reference C liblz4. Against it the real gap is far
  wider than against lz4_flex: Silesia decomp 2.66 vs 5.66 GB/s (0.47x),
  comp 0.42 vs 0.89 GB/s (0.47x), ratio 1.90x vs 2.10x (0.91x). 0/12 files
  beat liblz4 on decompression.
- Fixed: Snappy compression was timed against a zero-length output slice, so
  it errored instantly and reported ~720,000 GB/s.
- Fixed: G1 gate was hardcoded to PASS. It now reads .g1-status.json written
  by the fuzz test and FAILs when absent or under 1,000,000 mutations.
- SECURITY: heap-buffer-overflow in the tail phase of both vector decoders.

## decode: branchless literal wildcopy
Silesia 1C decomp 2.694 -> 4.053 GB/s (0.488x -> 0.733x liblz4). Ratio and
compression unchanged.
Hash-table sweep (16KB..256KB) recorded: ratio and compression speed trade
off almost exactly, no setting reaches G2 and G4 together.

## compress: widen raw-block bypass from 2% to 4% savings
Silesia 1C decomp 3.882 -> 4.209 GB/s (0.708x -> 0.759x liblz4).
Total ratio 1.9022 -> 1.9002 (-0.1%). x-ray decode 6.15 -> 12.68 GB/s with
ratio 1.011 (1.001x liblz4, still clear of the 0.95 floor).

## decode: remove the AVX-512 path (measured slower on Zen 4)
Silesia 1C decomp 4.12 -> 4.38 GB/s (0.75x -> 0.79x liblz4). AVX-512 was
slower than AVX2 on all 12 files by 5-15% (dickens +12.7%, mr +15.0%).
Function, dispatch and badge removed.

## analysis: stream composition and token histograms
Silesia: token stream is 31.9% of compressed output, offsets 31.9%,
literals 28.4%. 17.74M tokens. A 1-byte token models to ratio 2.1833 vs
liblz4 2.1015 = 1.039x, which clears G2's 1.02x threshold.

## format v3: 1-byte token. G2 PASSES.
Silesia ratio 1.9002 -> 2.19 (0.904x -> 1.04x liblz4, min file 0.97x).
G2 now PASS. Decode 4.38 -> 3.73 GB/s (0.79x -> 0.68x). Compression
unchanged.

## format: decouple match window (64 KB) from output block size (256 KB)
Silesia ratio 2.19 -> 2.20 (min file 0.97x -> 0.98x), decomp 3.60 -> 3.75.
Source Code workload ratio 36.75 -> 74.18 (liblz4 246.81).

## analysis: v3 G2/G4 frontier sweep (6 configs)
No configuration passes both G2 and G4, and none passes G4 at all. The
fastest point on the frontier is 0.698x liblz4 compression (16KB table,
lazy off), 30% short of the 1.00x threshold, and it fails G2 at 0.946x.

## compress: prefetch hash buckets
Silesia comp 0.420 -> 0.428 GB/s (0.485x -> 0.498x liblz4), ratio identical
at 2.1950. Small gain.

## verification: strict-mode run from a fresh clone
Section 1 procedure (median-of-5, >=1s each, fresh clone, re-downloaded
corpus incl. enwik8). Verdicts unchanged: G1, G2, G7, G8 PASS; G3, G4, G5,
G6 FAIL. Silesia decomp 0.66x, comp 0.48x, ratio 1.04x.

## verification: second fresh-clone strict reproduction
Two independent clones agree within 0.3% on ratio, compression and
decompression, with identical gate verdicts. Both required reproductions
are now complete.

## analysis: disassembly shows compiler-emitted AVX-512 in the decoder
`target-cpu=native` makes LLVM auto-vectorise the literal wildcopy to
vmovdqu64/zmm. Disabling AVX-512 codegen: comp 0.482x -> 0.509x, decomp
0.678x -> 0.630x. Neither passes G3/G4. CI builds x86-64-v3 (no AVX-512)
so it measures a 6-7% different configuration than the documented local build.

## analysis: interleaved token+offset layout tested and rejected
Merging the two hottest streams into fixed 3-byte records made decode 3.8%
slower (0.678x -> 0.647x liblz4) with identical ratio. Reverted.

## diagnostic: emission is 13.3% of compression time
Search-only build (all output writes removed) reaches 0.556x liblz4 vs
0.482x for the full compressor. Even with free emission, G4's 1.00x
threshold is unreachable. The match search is 86.7% of the time.

## T1.2 attempt: match window beyond 64 KB - REFUTED
Format v4 (far offsets) implemented and measured across window sizes and
far-match thresholds. Every configuration is WORSE than the 64 KB baseline:
  baseline 64KB no-far 2.19502 | 1MB/12 2.17908 | 4MB/24 2.18653 | 16MB/32 2.18781
Window size is not the lever; the match finder is. Reverted.

## T1.3: hash entries store the match word (adapted from LZAV, MIT)
comp 0.387 -> 0.4744 GB/s (+22.6%, now 97.0% of the 0.489 target)
decomp 3.529 -> 3.8795 GB/s (+9.9%, T1.4 floor passing)
ratio 2.19502 -> 2.17972 (HASH_BITS 16 -> 15 to hold memory flat)

## T1 frontier mapped; T1.3 and T1.4 solved, T1.2 is the gap
Best comp 0.5318 (target 0.489, PASSES). Best decomp 4.75 (floor 3.130).
Best ratio 2.2637 (target 2.4500, +8.2% still needed).
Measured and rejected: larger window (negative), repeat offsets (3.3% of
matches, +1.23%).

## T1.2 path PROVEN: entropy coding reaches ratio 2.502 > 2.4500 target
Shannon entropy of the real streams: total ideal saving 12.89% of output
-> ratio 2.17972 becomes 2.50225.

## T1.2 entropy coding measured per block with the real huffman.rs
examples/huff_probe.rs, 802 blocks, table cost included, raw kept when smaller.
  tokens only -> 2.35170 | tokens+offset-hi -> 2.46615 | all four -> 2.69078
Decode cost is the wall. Current decode_into: 3.3 ns/sym, tokens alone 60 ms,
against a T1.4 budget of 63 ms for the WHOLE decode (LZ loop already 44-54 ms).
Huffman on the current format reaches T1.2 but cannot hold T1.4 on the same
commit.

## T1.2: LZAV's advantage decomposed. Format is worth 0.3%, the parse is everything
examples/lzav_decomp.rs re-costs our exact token stream in LZAV's stream
format 2 and parses LZAV's real output with a port of lzav_decompress_2
(verified: match + literal bytes sum to the input exactly).
  ours in our format 2.17972 | ours in LZAV format 2.18666 (+0.3%)
  LZAV in LZAV format 2.45004 | LZAV parse in our format 2.49475

## T1.2: stream formats costed exactly on LZAV's parse (lzav_decomp format study)
Candidate stream formats costed exactly on LZAV's parse:
  E (chosen, format v5): 2.37690 raw, 2.51861 with per-block Huffman on the
     13.77M token bytes (T1.2 OK, +2.8% margin)
  S: 2.42876 raw, 2.57700 with Huffman on 18.32M header bytes
  v3 layout on the same parse: 2.14686 (u16 offsets cannot even hold it)

## Format v5 + LZAV finder port: ratio 2.388 on one configuration
Finder ported from LZAV 4.3. x-ray 1.080 (floor 1.01).
Measured 5x1s median, three repeats:
  ratio 2.38817 | comp 0.434-0.452 | decomp 2.74-2.88 (T1.4 FAIL)
Without the dense retry the same finder measured decomp 3.2246 (3x0.3s).
LZAV per-file: we trail on ratio on 10 of 12 files (dickens -7.5%, webster
-7.3%); x-ray costs us 24 ms of compression against LZAV's 0.7 ms.

## Window frontier mapped; finder cleanup; parse is not the gap
Window sweep (5x1s median, idle), ratio | comp | decode:
  64K 2.205|0.378|3.16  256K 2.277|0.406|3.19  1M 2.344|0.440|3.21
  2M 2.366|0.451|3.24  4M 2.382|0.427|3.00  8M 2.388|0.452|3.19
No window passes all three.
Removed a per-block std::env::var syscall from the finder hot path (window is
now read once, cached); comp 0.43 -> 0.452 at 8M.
Diagnosis via lzav_decomp: our parse is as dense as LZAV's (14.38M refs vs
13.77M, 35.1M lit bytes vs 40.3M, avg match 12.15 vs 12.47). The gap is
encoding:
  our parse, our v5 format   88.75 MB  2.38817
  our parse, LZAV bit format 86.95 MB  2.43757  (format costs ~2%)
  LZAV parse+format          86.50 MB  2.45004

## T1.2 entropy coding REFUTED for the decode budget (huff_speed)
Interleaved Huffman decode of the 14.38M-token stream, pinned, best of 7:
  N=1 4.28 ns/sym | N=2 3.81 | N=4 3.06 | N=8 3.10  (~0.3 GB/s on tokens)
Huffman on the tokens saves ~4.5% of output (ratio ~2.50), but decoding them
costs ~43 ms on top of the ~63 ms LZ decode: decode would fall 3.19 -> ~1.9
GB/s, well under the 3.130 floor. Table Huffman cannot pay for itself here
even interleaved.

## Moonshot REFUTED: rANS decode hits the same wall as Huffman (rans_speed)
Interleaved 32-bit rANS, 12-bit freqs, decode of the 14.38M-token stream:
  N=1 4.59 ns/sym | N=2 4.09 | N=4 3.56 | N=8 3.62 | N=16 3.84 | N=32 3.72
Best ~0.26 GB/s on tokens. Entropy-coding tokens by ANY table method adds
~36-43 ms to the ~63 ms decode: decode 3.19 -> ~2.1 GB/s, under the 3.130
floor.

## Bit-packed offsets: ratio prize 2.47 (beats LZAV) but decode cost looks prohibitive
bitoff_probe on the 14.38M real offsets. Bit-packing saves 8.86% of offset
bytes (33.17M -> 30.23M = 2.94M = 3.31% of output), taking ratio 2.388 ->
~2.470, above LZAV's 2.450. A bitstream read per match costs ~3.3 ns/match
isolated: 14.38M matches ~ 47.8 ms of offset decode vs ~10-12 ms today, so
integrated decode very likely falls from 3.19 to ~2.1-2.5 GB/s, under the
3.130 floor.

## GOAL3: pivot to speed. Physics measured, LZ4 is the gate, dense retry retired
speed_ceiling (single core, pinned): memcpy of Silesia 22.9 GB/s is the hard
decode ceiling (DRAM ~19, L3 ~48, L2 ~67). liblz4 decodes at 25% of it, we at
14%. GOAL3.md: Tier S1 = dominate liblz4 on decode+ratio, measured in the same
run (quick3 now reports liblz4 alongside). Dense retry made opt-in
(GLYD_DENSE=1); x-ray floor set to liblz4's own 1.00-1.01; total ratio
floor RAISED 1.85 -> 2.1009.
  goal3_baseline: ratio 2.37232 | comp 0.4651 | decomp 3.1239 vs liblz4 5.6526 (55.3%)

## GOAL3 decode: ablation maps the loop; credit guards +4%; window 256 KB
examples/dec_ablate.rs re-runs the fast loop over every block's real streams
with pieces switched off (cold cache, whole corpus, 13.8M tokens, ~66 ms).
Same-run decode 55.3% -> 57.1% of liblz4.
Finder window default 8 MB -> 256 KB (FINDER_WINDOW): decode 57.0/57.1% on
repeats vs 51.5/55.5% at 8 MB. Costs: ratio 2.372 -> 2.272 (floor 2.10), comp
0.457 -> 0.418 (a level, not a gate, under GOAL3).
  credit_win256k_default: ratio 2.27212 | comp 0.4184 | decomp 3.2126 vs
  liblz4 5.6378 = 57.0%

## Format v6: 3-bit literal, 4-bit match, fixed 2-byte offsets. Decode +26%
token_stats on the real stream (256 KB window, 12.9M tokens): escaped tokens
31.7% -> 21.5%.
  v6: ratio 2.27262 | comp 0.4204 | decomp 4.05/4.05/3.97 vs liblz4 5.66/5.66/5.63
      = 71.6 / 71.5 / 70.5% of liblz4 (was 57.0%)
Per file we are 61-85% of LZ4 on decode; x-ray (raw) decodes at 45 GB/s.
x-ray regression floor set to a raw store (0.99) until the fast level exists.

## GOAL3 S2 landed: AVX2 32-token pre-pass. Decode 90% of liblz4
  cold harness (lib): 64 -> 47 -> 39.9 ms
  quick3: ratio 2.27262 | comp 0.412 | decomp 5.06 / 5.02 / 5.04 vs liblz4
          5.60 / 5.59 / 5.61 same run = 90.2 / 89.8 / 89.8% (was 71.6%)
Per file we now beat liblz4 on decode on ooffice, osdb and (raw) x-ray, tie
reymont, and trail by 4-33% elsewhere (mr, sao, nci worst).

## Continuation escapes decoded inside the chunk: decode 97% of liblz4
  quick3: ratio 2.27262 | comp 0.412-0.422 | decomp 5.50 / 5.47 / 5.45 vs
          liblz4 5.65 / 5.63 / 5.60 same run = 97.4 / 97.1 / 97.2% (was 90%)
Per file vs liblz4 decode: beat it on mozilla, ooffice, osdb, reymont, sao,
xml, x-ray (7 of 12); trail on dickens 94%, mr 98%, nci 93%, samba 95%,
webster 86%.

## GOAL3 S1 PASSED: decode beats liblz4 on Silesia, same run, ratio floor held
Minimum match 6 -> 7: tokens 12.5M -> 10.0M (-20%). Ratio 2.2726 -> 2.1923
(floor 2.1009). Compression 0.41 -> 0.345, which GOAL3 assigns to the fast
level. The candidate check was generalized so any minimum 4..8 is sound (7 and
8 were corrupting output before, trusting unverified bytes).
Also fixed on the way, a latent decoder bug: the chunked path keeps lengths
in u16 arrays, and a literal run at MAX_LIT_LEN (65797) wrapped to 261. Only
the parallel Silesia stream contained one. Such lengths now take the careful
path; tests/roundtrip.rs pins it.
  quick3: ratio 2.19234 | comp 0.345 | decomp 6.15 / 6.03 / 6.05 vs liblz4
          5.67 / 5.69 / 5.73 same run = 108.5 / 106.0 / 105.6%  S1 OK
Per file we beat liblz4 on 10 of 12 (dickens 114%, ooffice 143%, sao 189%,
xml 115%...), trail on nci 94% and webster 96%. x-ray (raw) 49 GB/s.

## Docs refresh: README, GOAL3 status, per-file S1 table
Fresh run for the README table:
  quick3 readme_s1: ratio 2.19234 | comp 0.346 | decomp 6.05 vs liblz4 5.54
                    same run = 109.3%  S1 OK
Per file: beat liblz4 on 10 of 12 (sao +87%, ooffice +35%, osdb +23%, mr
+19%); nci -2%, webster -3%.

## aarch64: NEON port of the v6 decoder (Apple M1 Max)
The crate did not build on arm64 (x86 modules were unguarded) and the
scalar fallback decoded Silesia at 1.28 GB/s against liblz4's 4.43 in the
same run (29%). `src/neon_decompress.rs` is a NEON port of the AVX2 decoder.
Same-run quick3, M1 Max, one core:

Silesia 1C decomp 1.28 -> 5.32 GB/s vs liblz4 4.36 = 122% (S1.2 PASS on
this host). Beats liblz4 on 12/12 files (nci +8%, webster +15%, sao +97%).
Ratio 2.192, comp 0.27 GB/s (scalar finder). memcpy ceiling here is 39.9
GB/s; we are at 13% of it, liblz4 at 11%.
25 tests green including the 1M-mutation fuzz. Checksum stays scalar on
arm64 (not on the raw decode path that the gate measures).

## aarch64: where the floor is, and loop-free escapes (M1 Max)
New harness `examples/floor.rs`, Silesia, 10.0M tokens, 21.2 bytes/token, one
core: the real decoder was at 3.70 ns/token (5.3 GB/s) before; memcpy is 0.45
ns/token (44.1 GB/s). The wall for this format on this chip is ~1.6 ns/token
(~12.5 GB/s). Pass 1 in isolation: 1.14 -> 0.45 ns/token, bit-exact on all
794 blocks.

Silesia 1C decomp 5.32 -> 6.68 GB/s vs liblz4 4.36 = 153% (was 122%).
12/12 files up; osdb 5.3 -> 8.5, ooffice 5.7 -> 7.4, samba 5.3 -> 7.0.
3.01 ns/token against the 2.2 ns copy loop.
25 tests green including the 1M-mutation fuzz.

## aarch64: pipelined pre-pass (M1 Max)
Silesia 1C decomp 6.68 -> 6.79 GB/s (155% of liblz4): +1.5%. The decoder is at
2.94 ns/token against the harness floor of ~2.6.

## Compression: the ceiling, and the fast level (M1 Max)
New harness `examples/cfloor.rs`, Silesia, one core: memcpy 44 GB/s; liblz4
0.66 GB/s; default finder alone (before) 0.33; compress_into (before) 0.28;
block checksum, scalar (before) 2.6.

Changes.
1. NEON checksum: 75 -> 8.4 ms on Silesia (2.6 -> 23.5 GB/s).
2. Emit path: 8.5 -> ~4 ns per token.
3. Fast level (GOAL3 S3): `compress_into_fast`, `compress_parallel_into_fast`,
   CLI `-1/--fast`. Table size sweep (parse only, 5-byte hash, min 5):
   12 bits 0.72 GB/s at 2.00, 13 bits 0.68 at 2.10, 14 bits 0.57 at 2.18,
   16 bits 0.38 at 2.25. 13 bits chosen.

Same run (quick3), Silesia, one core:

| level | comp GB/s | ratio | decode GB/s | vs liblz4 decode |
|---|---:|---:|---:|---:|
| liblz4 | 0.66 | 2.101 | 4.38 | 100% |
| fast | 0.54 | 2.098 | 5.02 | 114% |
| default | 0.34 (was 0.28) | 2.192 | 6.80 | 155% |

Fast is at 81% of liblz4's compression speed at its ratio, and x-ray now
compresses (1.004) instead of being stored raw.
27 tests green (fast-level round trips added).

## Fast level: offset floor 1, the profile, and where the gap is (M1 Max)
- The fast finder now accepts offsets down to 1. Parse-only sweep at 13 bits:
  ratio 2.104 -> 2.183 and 0.68 -> 0.70 GB/s. End to end: ratio 2.098 -> 2.176.
- Decoder: offsets under 8 are no longer a byte loop. Fast-level decode 4.48 ->
  4.84 GB/s (mozilla 3.3 -> 3.9, mr 3.05 -> 4.1).

Same run (quick3), Silesia, one core:

| level | comp GB/s | ratio | decode GB/s | vs liblz4 decode |
|---|---:|---:|---:|---:|
| liblz4 | 0.66 | 2.101 | 4.39 | 100% |
| fast | 0.55 | 2.176 | 4.84 | 110% |
| default | 0.34 | 2.192 | 6.82 | 156% |

Fast beats liblz4 on ratio (+3.6%) and decode (+10%) and is at 83% of its
compression speed. Table size remains the speed dial (12 bits: ~0.60 GB/s
at 2.08).

## Decode: super-chunks, and the remaining levers (M1 Max)
- Fast phase restructured into super-chunks. Silesia 1C decode 6.82 -> 6.94
  GB/s (159% of liblz4).
- Measured, not adopted: minimum match 8 gives 8.05 GB/s (+18%) at ratio
  2.056, under the liblz4 floor.
- Multi-core (`examples/mc.rs`, parallel-compressed independent 256 KB
  blocks, 10 threads): 42.9 GB/s aggregate over Silesia, above this
  machine's single-core memcpy (40 GB/s); per file 26-71 GB/s.

## Turbo level (M1 Max)
`compress_into_turbo` / `compress_parallel_into_turbo` / CLI `-t --turbo`:
the default finder at minimum match 8. Same run:
decode 8.23 GB/s = 188% of liblz4 (default 6.94 = 159%), ratio 2.055
(default 2.192, liblz4 2.101), comp 0.33 GB/s. Per file 7.0-11.2 GB/s on
compressible data; sao is stored raw at this level (its 1.037 is under the
4% threshold) and decodes at memcpy. Decode scales with tokens per byte:
min 7 -> 8 cut tokens 18% and bought 18%.

## Turbo at minimum match 10; the compression dial (M1 Max)
- Turbo sweep, same run: min 8 8.2 GB/s at 2.055, 9 8.8 at 1.975, 10 9.3
  at 1.884, 12 10.5 at 1.751. Turbo set to 10: decode 211% of liblz4,
  +34% over the default. Hash table 17/18 bits: +0.005 ratio, not worth
  the L2 traffic (comp -15%).
- Fast level: 12-bit table gives 0.585 GB/s (+7%) at ratio 2.078, under
  liblz4's 2.101. Kept at 13 bits (2.176).

## v7 milestone 1: entropy coders
- Huffman: `huff8.rs`, 8-stream interleaved canonical Huffman. Silesia literals
  (dickens + mozilla, 235 blocks, 20 MB; the full 12-file corpus gives the same
  number at 794 blocks/55 MB, so this is not a sampling artifact), best of 5,
  `target-cpu=native`: **1.06 -> 0.85 -> 0.61 ns/symbol** across two rounds,
  against the 0.6 ns/symbol gate (`examples/huff_spike.rs`: 0.46 on the same
  data). Roundtrip, invalid-code (Kraft-sum) and overrun tests pass.
- tANS: `tans::encode8`/`decode8`, 8-stream interleaved tANS. 14M symbols
  (36-symbol alphabet, skewed), best of 5, `target-cpu=native`: first cut
  **0.873 ns/symbol**, then **0.738**; final, confirmed: **0.74 ns/symbol**,
  still over the 0.6 gate.

## v7 milestone 2: container + decoder on the default parse
- Container: `parse_header` accepts VERSION_V7; `compress_into_max` /
  `compress_parallel_into_max` use the default (Lzav) finder with the v7
  encoder per 256 KB block; a block the coder cannot shrink below the chunk is
  stored as a v6 raw block. Round trip through every `decompress*` entry point
  on empty, one byte, constant, random (raw), periodic (periods 3..70000), text
  and word-salad inputs.
- Pass 3 (copies): 6.0 -> 2.9 ns/sequence; Silesia decode 0.99 -> 1.20 GB/s
  (mr, short offsets, 0.62 -> 1.14).
- `examples/v7_bench.rs`, Silesia, M1 Max, `target-cpu=native`, median
  of 3 runs of >= 0.3 s, zstd 1.5.7 (bulk API, contexts reused) in the
  same run:
  **v7: ratio 2.7406, comp 0.133 GB/s, decomp 1.203 GB/s** |
  zstd-3: ratio 3.2045, comp 0.326, decomp 1.434 |
  zstd-1: ratio 2.8942, comp 0.551, decomp 1.543.
  Ratio is in the brief's 2.6-2.8 band (+25% over v6's 2.19 on the same
  parse); decode is under the 2.5 GB/s stop rule, and 3-4 GB/s was never
  in reach of this pipeline.

## v7 milestone 2b: pass 1 on fast readers
- Pass 1's extra-bits walk, per sequence, Silesia, same machine state (zstd-3
  1.41-1.43 in those runs): 6.73 (baseline) -> 3.40 -> 2.88 -> 2.38 -> 2.40
  -> 2.25. Output bounds checked once per batch: the three code streams 2.42
  -> 2.14 ns/sequence, pass 1 5.27 -> 4.99; pass 2 0.203 -> 0.169 ns/byte.
- `examples/v7_bench.rs`, same protocol as milestone 2, before and after
  in the same machine state:
  before **v7 decomp 1.218 GB/s** | zstd-3 1.479 | zstd-1 1.576;
  after **v7: ratio 2.7406, comp 0.142 GB/s, decomp 1.883 GB/s** |
  zstd-3: ratio 3.2045, comp 0.341, decomp 1.474 |
  zstd-1: ratio 2.8942, comp 0.561, decomp 1.558.
  Pass 1 9.79 -> 5.0 ns/sequence (brief: <= 5), decode +55% (brief:
  >= ~1.7 GB/s); v7 decode is now 1.28x zstd -3 and 1.21x zstd -1 on
  this parse.

## v7 milestone 2c: encoder throughput
- Stage costs per pass over Silesia (212 MB, 10.0 M sequences, 63.3 M
  literals; `compress_into_max` replayed stage by stage with the Lzav
  finder at 2.9-3.0 ns/byte in the same run, M1 Max, `target-cpu=native`):
  the v7 encode side was 3.8 ns/byte. After the changes, each byte-identical
  on the whole corpus (FNV of every `compress_into_max` output against the
  milestone-2 binary): 0.95 ns/byte.
- `examples/v7_bench.rs`, same protocol as milestone 2, milestone-2
  binary and this one back to back: **comp 0.134 -> 0.237 GB/s**, ratio
  2.7406 and decomp 1.19 GB/s unchanged (zstd-3 comp 0.326 / 0.321 in the
  two runs). The brief's 0.5 ns/byte (~0.28 GB/s) is not reached.

## v7 milestone 3/4: modeling + double-fast parse
- `v7_encode::find_sequences_dfast` replaces the milestone-2 bridge in
  `compress_into_max`: a double-fast parse with lazy matching and zstd's
  post-match insertions. Minimum match 4, window 2 MB across the blocks of
  one call.
- Silesia ratio (this coder), the steps: greedy, 17/16-bit tables (zstd -3's
  sizes): 2.7406 -> 3.0475, under the spec's 3.10 stop rule, so lazy matching
  went in: 3.1337; zstd's insertions: 3.1621; tables 18/18 (2 MB): **3.2219**
  -- G2 (>= 3.20) met, zstd -3 is 3.2045.
  Table size is the dial: 17/16 3.162, 18/17 or 17/18 3.204, 18/18
  3.222; the parse is ~7% slower at 18/18 than at 17/16, the extra on
  the binaries (x-ray +30%).
- The parse's share, measured by running zstd -3's own sequences
  (libzstd's `ZSTD_generateSequences` through the zstd-sys static
  library, its 128 KB blocks merged pairwise) through `encode_block`:
  3.1398. So zstd's dfast parse through this coder is 2.0% behind
  zstd -3, and this parse beats zstd's dfast on every Silesia file through
  the same coder (+2.6% total).
- `examples/v7_bench.rs`, Silesia, M1 Max, `target-cpu=native`, median
  of 3 runs of >= 0.3 s, zstd 1.5.7 in the same run (two other agents'
  builds running; zstd-3 within 3% of the milestone-2 run):
  **v7: ratio 3.2219, comp 0.125 GB/s, decomp 0.983 GB/s** |
  zstd-3: ratio 3.2045, comp 0.319, decomp 1.411 |
  zstd-1: ratio 2.8942, comp 0.540, decomp 1.517.
  G2 (ratio >= 3.20) is met. G4 (comp >= 0.34 GB/s = 2.94 ns/byte for
  parse and coder together) is not: the parse alone is 3.37 ns/byte
  (byte-weighted over Silesia, `find_sequences_dfast` timed by itself
  in 256 KB blocks; 2.96 at 17/16 tables), the coder the other 3.8 of
  the 7.2 ns/byte total, so even a free coder leaves G4 short.
  Per file, parse ns/byte: nci 1.1, xml 1.6,
  samba 2.6, mozilla 3.4, osdb 3.4, reymont 3.8, mr 4.1, webster 4.3,
  ooffice 4.4, sao 5.0, dickens 5.1, x-ray 5.5. G3 (decode >= 3.0)
  is the decoder task's; per byte the current decoder is 17% slower than at
  milestone 2 (1.18 -> 0.98).

  | file    | v7 ratio | comp GB/s | decomp GB/s | zstd-3 ratio | zstd-3 comp | zstd-3 decomp |
  |---------|---------:|----------:|------------:|-------------:|------------:|--------------:|
  | dickens |   2.8376 |     0.091 |       0.722 |       2.7822 |       0.201 |         1.152 |
  | mozilla |   2.7821 |     0.122 |       0.914 |       2.8101 |       0.345 |         1.264 |
  | mr      |   2.8191 |     0.100 |       0.816 |       2.8106 |       0.251 |         1.248 |
  | nci     |  11.2417 |     0.333 |       1.858 |      11.8403 |       0.844 |         2.607 |
  | ooffice |   1.9930 |     0.089 |       0.728 |       1.9680 |       0.254 |         0.982 |
  | osdb    |   2.8612 |     0.119 |       1.090 |       2.8804 |       0.344 |         1.629 |
  | reymont |   3.4815 |     0.115 |       0.868 |       3.4197 |       0.253 |         1.350 |
  | samba   |   4.3658 |     0.168 |       1.291 |       4.3604 |       0.418 |         1.922 |
  | sao     |   1.3171 |     0.092 |       0.989 |       1.3120 |       0.201 |         0.842 |
  | webster |   3.4994 |     0.105 |       0.849 |       3.4272 |       0.250 |         1.368 |
  | xml     |   8.3375 |     0.248 |       1.679 |       8.4138 |       0.641 |         2.401 |
  | x-ray   |   1.4617 |     0.071 |       0.621 |       1.3926 |       0.190 |         0.829 |
  | total   |   3.2219 |     0.125 |       0.983 |       3.2045 |       0.319 |         1.411 |

## v7 milestone 2d: decoder on the real parse
- Steps, each measured against its predecessor with the two binaries
  run alternately (other sessions' builds moved single runs by up to
  10%); the final column is the whole corpus back to back in one state:

  | step | ns/sequence | ns/byte |
  |---|---|---|
  | baseline f25f4f9 | p1 4.80, p2 0.82, p3 2.67 | 0.553 |
  | copy pass | p3 2.67 -> 2.15 | 0.518 |
  | tANS decode | codes 2.53 -> 2.15 | 0.499 |
  | groups of eight | walk 2.10 -> 1.90, tail 0.17 -> 0.14 | 0.489 |
  | huff8 decode | p2 0.82 -> 0.74 | 0.484 |
  | table build | tables 0.46 -> 0.38 | 0.479 |

- Where it stands: 7.0 ns/sequence; 2.5 GB/s needs 5.3 ns/sequence.
- `examples/v7_bench.rs`, Silesia, M1 Max, `target-cpu=native`, median
  of 3 runs of >= 0.3 s, zstd 1.5.7 in the same run, the f25f4f9 and
  HEAD binaries back to back (zstd-3 1.439 / 1.435 in the two):
  before **v7 decomp 1.603 GB/s** | zstd-3 1.439 | zstd-1 1.538;
  after **v7: ratio 3.2219, comp 0.214 GB/s, decomp 1.822 GB/s** |
  zstd-3: ratio 3.2045, comp 0.329, decomp 1.435 |
  zstd-1: ratio 2.8942, comp 0.545, decomp 1.534.
  +13.7%, 1.27x zstd -3 and 1.19x zstd -1; the brief's 2.5 GB/s and
  gate G3's 3.0 are not met. Per file, decode GB/s before -> after:
  dickens 1.14 -> 1.32, mozilla 1.49 -> 1.67, mr 1.29 -> 1.47, nci 3.14
  -> 3.49, ooffice 1.20 -> 1.34, osdb 1.87 -> 2.11, reymont 1.43 ->
  1.65, samba 2.14 -> 2.39, sao 1.63 -> 1.92, webster 1.39 -> 1.60, xml
  2.73 -> 3.09, x-ray 0.94 -> 1.11.

## v7 milestone 4b: parse speed, and the coder's modeling gap
Goal: comp >= 0.30 GB/s at ratio >= 3.20 on Silesia. Start (f25f4f9,
`v7_bench`): ratio 3.2219, comp 0.211 GB/s, zstd-3 0.327 in the same run.

Steps, each measured alone (`examples/v7_parse`: the parse by itself in
256 KB blocks, min of 5 runs, next to `compress_into_max`'s total ns/byte
and ratio; the machine was shared with other agents, so the later steps
were A/B'd as alternating binaries, min of 3):

| step | parse ns/B | total ns/B | ratio |
|---|---:|---:|---:|
| f25f4f9 | 3.24 | 4.10 | 3.2219 |
| final, quiet machine, min of 5 | 2.39* | 2.93 | 3.2176 |

(* parse column includes the code emission from that step on.)

`examples/v7_bench.rs`, Silesia, M1 Max, `target-cpu=native`, median of
3 runs of >= 0.3 s, zstd 1.5.7 in the same run, load average ~3.5:
**v7: ratio 3.2176, comp 0.304 GB/s, decomp 1.623 GB/s** | zstd-3:
ratio 3.2045, comp 0.333, decomp 1.446 | zstd-1: 2.8942, 0.553, 1.545.
Gate met: comp 0.304 >= 0.30 at ratio 3.2176 >= 3.20; 91% of zstd-3's
compression speed at +0.4% ratio (start: 65%, +0.5%).

  | file    | v7 ratio | comp GB/s | decomp GB/s | zstd-3 ratio | zstd-3 comp | zstd-3 decomp |
  |---------|---------:|----------:|------------:|-------------:|------------:|--------------:|
  | dickens |   2.8331 |     0.206 |       1.135 |       2.7822 |       0.212 |         1.176 |
  | mozilla |   2.7760 |     0.307 |       1.518 |       2.8101 |       0.359 |         1.286 |
  | mr      |   2.8188 |     0.250 |       1.265 |       2.8106 |       0.261 |         1.234 |
  | nci     |  11.1565 |     0.749 |       3.180 |      11.8403 |       0.884 |         2.679 |
  | ooffice |   1.9914 |     0.221 |       1.198 |       1.9680 |       0.263 |         1.020 |
  | osdb    |   2.8629 |     0.317 |       1.873 |       2.8804 |       0.356 |         1.700 |
  | reymont |   3.4834 |     0.272 |       1.459 |       3.4197 |       0.269 |         1.412 |
  | samba   |   4.3619 |     0.423 |       2.112 |       4.3604 |       0.437 |         1.998 |
  | sao     |   1.3176 |     0.200 |       1.717 |       1.3120 |       0.209 |         0.862 |
  | webster |   3.4962 |     0.255 |       1.417 |       3.4272 |       0.263 |         1.406 |
  | xml     |   8.2448 |     0.545 |       2.803 |       8.4138 |       0.665 |         2.471 |
  | x-ray   |   1.4621 |     0.165 |       0.967 |       1.3926 |       0.197 |         0.848 |
  | total   |   3.2176 |     0.304 |       1.623 |       3.2045 |       0.333 |         1.446 |

## v7 milestone 5: levels, corpus, field survey
Starting point, not previously recorded in this file: Task 9b (parse/coder
work, merged as 5241056) took the combined Silesia numbers from milestone
4b's 3.2176 / 0.304 / 1.623 to **ratio 3.2176, comp 0.305 GB/s, decode
1.862 GB/s** (zstd-3 in the same run: 3.2045 / 0.335 / 1.448). This task
adds the `--max` level to the CLI and C ABI, an extended real-world
corpus, and the field survey, then checks all of it against zstd honestly
rather than only on Silesia.

**CLI**: `-9`/`--max` added to `src/bin/glyd.rs`
(`compress_into_max` / `compress_parallel_into_max`). Verified with `cmp`:
`glyd -9 corpus/dickens -o d.glyd && glyd -d d.glyd -o d && cmp d
corpus/dickens` -- byte-identical. Multi-core default: 10,192,446 ->
3,912,217 bytes (ratio 2.6055, independent `PARALLEL_CHUNK_SIZE` chunks
cost some ratio vs a single chained stream); `--single-core`: 10,192,446
-> 3,597,654 (ratio **2.8331**, matching milestone 4b's per-file dickens
number exactly).

**C ABI**: `glyd_compress_max` / `glyd_compress_max_parallel`
added to `src/c_api.rs` and `include/glyd.h`, mirroring
`glyd_compress[_parallel]` exactly (same signature, same error
codes). No new decompress entry point was needed --
`glyd_decompress[_parallel]` already dispatch on the block header's
version, so v7 payloads round-trip through the existing calls.
`tests/test_c_abi.c` extended with a max-level sequential and parallel
round trip on its existing 1 MB structured buffer; built with clang and
run with `DYLD_LIBRARY_PATH` on macOS (`README.md`'s C ABI section now
shows both clang/`DYLD_LIBRARY_PATH` for macOS and gcc/`LD_LIBRARY_PATH`
for Linux):

```
====================================================
  Testing Glyd C ABI Interface (Shared Library)
====================================================
Glyd Version: 0.1.0
Uncompressed size: 1048576 bytes, Max compressed buffer: 1052736 bytes
1. Single-core compress: written 14960 bytes (ratio: 70.09x)
2. Single-core decompress: restored 1048576 bytes
   Single-core verification: PASS
3. Multi-core compress: written 17840 bytes
4. Multi-core decompress: restored 1048576 bytes
   Multi-core verification: PASS
5. Max-level compress: written 720 bytes (ratio: 1456.36x)
   Max-level decompress: restored 1048576 bytes
   Max-level verification: PASS
5b. Max-level parallel compress: written 852 bytes
    Max-level parallel decompress: restored 1048576 bytes
    Max-level parallel verification: PASS
7. Undersized buffer test: error code -1
8. Null pointer safety test: error code -2
====================================================
  All C ABI Tests Passed 100% Cleanly!
====================================================
```

**Extended corpus** (`scripts/download_corpus.sh` -> `corpus/ext/`, each
entry best-effort so one flaky host does not block the rest): GitHub
Archive one-hour JSON-lines sample (912 MB), NASA HTTP logs July 1995
(205 MB), NYC yellow taxi Parquet for 2024-01 (50 MB), the first 64 MB of
the linux-6.6 source tarball, and a small OpenStreetMap PBF extract
(Liechtenstein, 3.4 MB). TPC-H `lineitem` skipped (no `duckdb` on `PATH`)
and `vmlinux` skipped (no local kernel build available).

`examples/v7_bench.rs` now iterates `corpus/ext/*` after Silesia and
checks gate G2 (v7 ratio >= zstd -3 ratio) per file. Fresh Silesia total
in the same run (`RUSTFLAGS="-C target-cpu=native" cargo run --release
--example v7_bench`, confirms the 9b numbers above within normal
run-to-run noise on a machine shared with other agents' worktrees during
this task):

  | file    | v7 ratio | comp GB/s | decomp GB/s | zstd-3 ratio | zstd-3 comp | zstd-3 decomp |
  |---------|---------:|----------:|------------:|-------------:|------------:|--------------:|
  | dickens |   2.8331 |     0.219 |       1.382 |       2.7822 |       0.217 |         1.211 |
  | mozilla |   2.7760 |     0.317 |       1.737 |       2.8101 |       0.369 |         1.303 |
  | mr      |   2.8188 |     0.256 |       1.517 |       2.8106 |       0.268 |         1.313 |
  | nci     |  11.1565 |     0.771 |       3.673 |      11.8403 |       0.914 |         2.722 |
  | ooffice |   1.9914 |     0.227 |       1.401 |       1.9680 |       0.269 |         1.021 |
  | osdb    |   2.8629 |     0.329 |       2.224 |       2.8804 |       0.365 |         1.725 |
  | reymont |   3.4834 |     0.277 |       1.770 |       3.4197 |       0.261 |         1.441 |
  | samba   |   4.3619 |     0.428 |       2.489 |       4.3604 |       0.448 |         2.031 |
  | sao     |   1.3176 |     0.207 |       2.050 |       1.3120 |       0.210 |         0.882 |
  | webster |   3.4962 |     0.263 |       1.701 |       3.4272 |       0.265 |         1.444 |
  | xml     |   8.2448 |     0.564 |       3.227 |       8.4138 |       0.680 |         2.520 |
  | x-ray   |   1.4621 |     0.172 |       1.144 |       1.3926 |       0.201 |         0.864 |
  | total   |   3.2176 |     0.314 |       1.911 |       3.2045 |       0.339 |         1.476 |

Extended corpus, gate G2 per file:

  | file | size | v7 ratio | v7 comp | v7 decomp | zstd-3 ratio | zstd-3 comp | zstd-3 decomp | G2 |
  |---|---:|---:|---:|---:|---:|---:|---:|:---:|
  | gharchive.json | 912 MB | 10.5913 | 0.745 | 4.064 | 10.6560 | 0.955 | 3.329 | **FAIL** |
  | liechtenstein.osm.pbf | 3.4 MB | 1.0001 | 1.002 | 30.645 | 1.0000 | 5.038 | 47.806 | PASS |
  | linux.tar | 64 MB | 4.9371 | 0.373 | 2.183 | 4.8983 | 0.400 | 1.793 | PASS |
  | nasa_access.log | 205 MB | 9.7894 | 0.682 | 3.072 | 9.7822 | 0.825 | 2.369 | PASS |
  | yellow_tripdata.parquet | 50 MB | 1.0007 | 1.797 | 25.705 | 1.0038 | 0.770 | 6.798 | **FAIL** |
  | total | | 7.1117 | 0.713 | 3.820 | 7.1343 | 0.861 | 3.053 | **FAIL (2/5)** |

**G2 does not hold on the extended corpus** -- honest result, not the
Silesia-only 3.2176 >= 3.2045 story. The Parquet "loss" is noise at the
incompressible floor (both ratios round to 1.00; Parquet already applies
its own internal compression, so this is framing overhead, not entropy
coding, and zstd is 2.3x faster to compress it too). The GitHub Archive loss
is real, if small (-0.6% ratio, and zstd -3 is faster there too: 0.955 vs
0.745 GB/s). liechtenstein.osm.pbf, linux.tar and nasa_access.log all pass,
two of them by a comfortable margin. Net: `--max` is a solid zstd -3
substitute on Silesia-like text and source code, roughly a wash on
structured/repetitive JSON, and already-compressed containers (Parquet,
PBF) are a wash for any general-purpose byte-level coder by construction.

**Field survey** (`examples/field_survey.rs` gained an `Glyd-max` row,
index 13, `compress_into_max` / `decompress_into_raw`, same pattern as
the fast/turbo rows). `RUSTFLAGS="-C target-cpu=native" cargo run
--release --example field_survey 3 0.3`, Silesia, M1 Max, one core, every
codec in the same run:

  ```
  codec      |   ratio  vs lz4 | comp GB/s  vs lz4 |  dec GB/s  vs lz4 | dominates lz4?
  -------------------------------------------------------------------------------------------------
  Glyd-max   |  3.2176  1.532x |     0.277  0.454x |     1.733  0.422x | wins: ratio
  zstd-3     |  3.2045  1.525x |     0.319  0.523x |     1.361  0.331x | wins: ratio
  zstd-1     |  2.8942  1.378x |     0.535  0.876x |     1.493  0.363x | wins: ratio
  LZAV-hi    |  2.8032  1.334x |     0.091  0.148x |     3.185  0.775x | wins: ratio
  LZAV       |  2.4500  1.166x |     0.426  0.699x |     3.128  0.761x | wins: ratio
  zstd--1    |  2.4380  1.160x |     0.614  1.007x |     2.153  0.524x | wins: ratio+comp
  zstd--3    |  2.2399  1.066x |     0.684  1.122x |     2.307  0.562x | wins: ratio+comp
  Glyd       |  2.1924  1.044x |     0.312  0.511x |     6.507  1.584x | wins: ratio+dec
  Glyd-fast  |  2.1760  1.036x |     0.501  0.821x |     4.670  1.137x | wins: ratio+dec
  liblz4     |  2.1009  1.000x |     0.610  1.000x |     4.108  1.000x | (baseline)
  lz4_flex   |  2.0971  0.998x |     0.633  1.037x |     3.004  0.731x | wins: comp
  snappy     |  2.0761  0.988x |     0.607  0.996x |     1.495  0.364x | no
  zstd--5    |  2.0570  0.979x |     0.746  1.222x |     2.484  0.605x | wins: comp
  Glyd-turbo |  1.8837  0.897x |     0.263  0.431x |     8.647  2.105x | wins: dec
  ```
`Glyd-max` is the best ratio in the field (ahead of zstd -3, same
survey conclusion as always: nothing dominates liblz4 on all three axes).
Its own comp/decode numbers here (0.277 / 1.733) read lower than the
v7_bench total above (0.314 / 1.911) -- both are honest measurements of
the same binary, just different harnesses (field_survey's `timed` warms
up and loops per codec across 14 codecs x 12 files back to back) and,
per the standing note in this file, a machine shared with other agents'
worktrees during this task; run-to-run spread here is in the same
±3-10% band already on record.

## v8 decoder experiments (2026-09-19, Sapphire Rapids c7i.2xlarge dev box, M1 Max)

- Repeat offsets resolved in the copy pass instead of the sequence walk: M1
  neutral (2,187 vs 2,190 MB/s on the corpus loop), x86 -5% (1,292 vs
  1,360). Reverted.
- Format v8 (8 MB window) against v7 on the same x86 instance, same
  binaries side by side: max-level decode 1,390/1,370 -> 1,355/1,337
  MB/s (corpus loop), 1,197 -> 1,183 (v7_bench); zstd -3 1,157. The
  published c7i run of 56cc09d measured every codec 5-25% below its
  previous run on that instance (noisy neighbour); re-run on a fresh
  instance for the release.
- Match sources as positions with a prefetch: M1 -3.6%, x86 -10% without
  the prefetch and -4%/-8% with it. Reverted.
- Stream positions kept in memory on x86-64: +1.5% (1,350 -> 1,370 MB/s).
  Kept.

## Ultra parse pricing (2026-09-19, M1 Max, Silesia, ultra_bench)

Kept: own prior at weight 2, half a bit per literal: 3.925 -> 3.946, decode
2,150 -> 2,087 MB/s (more short matches). Block splitting (`split_points`):
margin 2x/3x/4x overhead gives 3.938/3.935/3.931 at 2,097/2,119/2,142 MB/s;
4x kept.
