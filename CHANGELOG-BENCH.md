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
compression unchanged. Removed the per-token `if lit_len > 0` guard and the
<=32 / >32 dual path in favour of an unconditional vector store loop.
Hash-table sweep (16KB..256KB) recorded: ratio and compression speed trade
off almost exactly, no setting reaches G2 and G4 together.

## compress: widen raw-block bypass from 2% to 4% savings
Silesia 1C decomp 3.882 -> 4.209 GB/s (0.708x -> 0.759x liblz4).
Total ratio 1.9022 -> 1.9002 (-0.1%). x-ray decode 6.15 -> 12.68 GB/s with
ratio 1.011 (1.001x liblz4, still clear of the 0.95 floor).
Rejected first: a sampling pre-scan (stride-16 4-byte hash hit rate). It
never fired on x-ray yet did fire on mozilla, costing 3.7% ratio there.
Measured savings beat predicted savings.

## decode: remove the AVX-512 path (measured slower on Zen 4)
Silesia 1C decomp 4.12 -> 4.38 GB/s (0.75x -> 0.79x liblz4). AVX-512 was
slower than AVX2 on all 12 files by 5-15% (dickens +12.7%, mr +15.0%).
Zen 4 executes 512-bit ops at half rate. Function, dispatch and badge removed.

## analysis: stream composition and token histograms
Silesia: token stream is 31.9% of compressed output, offsets 31.9%,
literals 28.4%. 17.74M tokens. lit_len <= 6 covers 94.20% of tokens;
match_len in 4..33 covers 96.56%. A 1-byte token (3-bit literal code,
5-bit match code, 2-byte escapes) models to ratio 2.1833 vs liblz4 2.1015
= 1.039x, which clears G2's 1.02x threshold. 4/4 split is worse (1.212
bytes/token vs 1.185).

## format v3: 1-byte token. G2 PASSES.
Silesia ratio 1.9002 -> 2.19 (0.904x -> 1.04x liblz4, min file 0.97x).
G2 now PASS. Decode 4.38 -> 3.73 GB/s (0.79x -> 0.68x): the escape branches
cost more than the smaller stream saved, partly recovered by a 256-entry
token decode table (3.20 -> 3.73). Compression unchanged.

## format: decouple match window (64 KB) from output block size (256 KB)
Silesia ratio 2.19 -> 2.20 (min file 0.97x -> 0.98x), decomp 3.60 -> 3.75.
Source Code workload ratio 36.75 -> 74.18 (liblz4 246.81). Matches were
being cut at the block boundary, not by the window.

## analysis: v3 G2/G4 frontier sweep (6 configs)
No configuration passes both G2 and G4, and none passes G4 at all. The
fastest point on the frontier is 0.698x liblz4 compression (16KB table,
lazy off), 30% short of the 1.00x threshold, and it fails G2 at 0.946x.

## compress: prefetch hash buckets
Silesia comp 0.420 -> 0.428 GB/s (0.485x -> 0.498x liblz4), ratio identical
at 2.1950. Small gain, and the size of it is the finding: compression is
throughput-bound in the inner loop, not stalled on the 256 KB table's L2
latency.

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
slower (0.678x -> 0.647x liblz4) with identical ratio. The columnar layout
is not the decoder's bottleneck. Reverted.

## diagnostic: emission is 13.3% of compression time
Search-only build (all output writes removed) reaches 0.556x liblz4 vs
0.482x for the full compressor. Even with free emission, G4's 1.00x
threshold is unreachable. The match search is 86.7% of the time.

## T1.2 attempt: match window beyond 64 KB - REFUTED
Format v4 (far offsets: zero marker in the u16 offset stream, true offset in
extras) implemented and measured across window sizes and far-match thresholds.
Every configuration is WORSE than the 64 KB baseline:
  baseline 64KB no-far 2.19502 | 1MB/12 2.17908 | 4MB/24 2.18653 | 16MB/32 2.18781
Cause: the hash table keeps only the most recent position per hash, so a far
candidate only appears when a pattern has NOT recurred recently. Taking it
costs 7 bytes vs 3 and consumes positions that would have yielded better near
matches. Window size is not the lever; the match finder is. Reverted.

## T1.3: hash entries store the match word (adapted from LZAV, MIT)
comp 0.387 -> 0.4744 GB/s (+22.6%, now 97.0% of the 0.489 target)
decomp 3.529 -> 3.8795 GB/s (+9.9%, T1.4 floor passing)
ratio 2.19502 -> 2.17972 (HASH_BITS 16 -> 15 to hold memory flat; sweep next)
A probe now rejects a miss with a register compare instead of a random,
cache-missing read of the candidate position in the source.

## T1 frontier mapped; T1.3 and T1.4 solved, T1.2 is the gap
Best comp 0.5318 (target 0.489, PASSES). Best decomp 4.75 (floor 3.130).
Best ratio 2.2637 (target 2.4500, +8.2% still needed).
Measured and rejected: larger window (negative), repeat offsets (3.3% of
matches, +1.23%). Offsets are 37.8% of output, the largest single component.

## T1.2 path PROVEN: entropy coding reaches ratio 2.502 > 2.4500 target
Shannon entropy of the real streams: tokens 5.344 b/sym (saves 5.96 MB),
offset-hi 6.247 (3.93 MB), literals 7.524 (2.16 MB), offset-lo 7.785 (0.48 MB).
Total ideal saving 12.89% of output -> ratio 2.17972 becomes 2.50225.
Tokens + offset-hi alone give 10.17% -> 2.42641. First proven route to T1.2.
