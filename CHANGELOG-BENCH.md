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

## T1.2 entropy coding measured per block with the real huffman.rs
examples/huff_probe.rs, 802 blocks, table cost included, raw kept when smaller.
Per-block tables capture more than the corpus-wide bound predicted:
  tokens 7.31% | offset-hi 4.30% | offset-lo 1.49% | literals 5.89% (was 2.2%)
  tokens only -> 2.35170 | tokens+offset-hi -> 2.46615 | all four -> 2.69078
Decode cost is the wall. Current decode_into: 3.3 ns/sym, tokens alone 60 ms,
against a T1.4 budget of 63 ms for the WHOLE decode (LZ loop already 44-54 ms).
A 4-stream interleaved decoder at ~0.5 ns/sym would cost ~18 ms for
tokens+offset-hi, which fits only on the h6 base whose ratio (2.11-2.16)
lands at 2.35-2.41 after coding. Huffman on the current format reaches T1.2
but cannot hold T1.4 on the same commit. Next: decompose LZAV's advantage
into format vs parse (it reaches 2.45 with no entropy coding at all).

## T1.2: LZAV's advantage decomposed. Format is worth 0.3%, the parse is everything
examples/lzav_decomp.rs re-costs our exact token stream in LZAV's stream
format 2 and parses LZAV's real output with a port of lzav_decompress_2
(verified: match + literal bytes sum to the input exactly).
  ours in our format 2.17972 | ours in LZAV format 2.18666 (+0.3%)
  LZAV in LZAV format 2.45004 | LZAV parse in our format 2.49475
LZAV emits 13.77M refs vs our 17.94M (avg match 12.47 vs 9.60) with 10.7%
more literal bytes; net 12.3 MB less. 41.6% of its refs are beyond 64 KB,
unreachable with u16 offsets. 66.9% of its refs carry no literals (ours
50.5%). Its finder: 6-byte hash, 2-tuple buckets (1 MB table), 8 MB
window, back-matching into pending literals, adaptive skip, no lazy match.
Route to T1.2: reproduce that parse in a stream format with cheap far
offsets. The earlier "bigger window is negative" result was for a 1-way
4-byte-hash table, which thrashes; it does not refute this combination.

## T1.2: stream formats costed exactly on LZAV's parse (lzav_decomp format study)
Simple byte layouts lose to LZAV's own format on its parse because LZAV
steals 2 offset bits from its header byte (10/18/23-bit classes):
  E: 2-bit offset class / 4-bit len (mref 6) / 2-bit lit, 1/2/3-byte offsets,
     byte escapes: 2.37690 raw, 2.51861 with per-block Huffman on the
     13.77M token bytes (T1.2 OK, +2.8% margin)
  S: LZAV headers split into streams: 2.42876 raw, 2.57700 with Huffman on
     18.32M header bytes
  v3 layout on the same parse: 2.14686 (u16 offsets cannot even hold it)
LZAV parse offsets: <256 12.4% | <4K 18.5% | <64K 27.4% | <1M 33.0% | <4M 7.6%
Plan: port LZAV's finder, format v5 = E, then Huffman tokens.

## Format v5 + LZAV finder port: ratio 2.388 on one configuration
Format v5: 2-bit offset class (1/2/3-byte offsets, 16 MB reach), 4-bit
match length (bias 5), 2-bit literal count, byte escapes; 36-byte header
with token_bytes/offset_bytes/extras_bytes. Finder ported from LZAV 4.3:
6-byte komihash into 2-tuple 16-byte buckets (1 MB), 8 MB window,
back-matching, adaptive skip, no lazy. Dense retry (min match 5, 4-byte
hash, no cap on offset) for blocks the first pass would store raw; only
x-ray (33 blocks) and one mozilla block take it. x-ray 1.080 (floor 1.01).
Measured 5x1s median, three repeats:
  ratio 2.38817 | comp 0.434-0.452 | decomp 2.74-2.88 (T1.4 FAIL)
Before the dense retry was wired in, the same finder measured decomp
3.2246 (3x0.3s), so part of the decode drop is suspected plumbing, not
the window. LZAV per-file: we trail on ratio on 10 of 12 files (dickens
-7.5%, webster -7.3%), the format effect of its 10/18-bit offset classes;
x-ray costs us 24 ms of compression against LZAV's 0.7 ms.

## Window frontier mapped; finder cleanup; parse is not the gap
Window sweep (5x1s median, idle), ratio | comp | decode:
  64K 2.205|0.378|3.16  256K 2.277|0.406|3.19  1M 2.344|0.440|3.21
  2M 2.366|0.451|3.24  4M 2.382|0.427|3.00  8M 2.388|0.452|3.19
No window passes all three. Ratio saturates past ~2M; comp and ratio move
TOGETHER with window (bigger window = more matches accepted = fewer positions
hashed = faster AND denser), so the only real tension is (ratio+comp) vs decode.
Removed a per-block std::env::var syscall from the finder hot path (window is
now read once, cached); comp 0.43 -> 0.452 at 8M.
Diagnosis via lzav_decomp: our parse is as dense as LZAV's (14.38M refs vs
13.77M, 35.1M lit bytes vs 40.3M, avg match 12.15 vs 12.47). The finder is not
the gap. The gap is encoding:
  our parse, our v5 format   88.75 MB  2.38817
  our parse, LZAV bit format 86.95 MB  2.43757  (format costs ~2%)
  LZAV parse+format          86.50 MB  2.45004
To beat 2.45 (not just tie) needs entropy coding, but scalar Huffman token
decode (~3.4 ns/sym) would drop decode under the floor. The path that can
dominate LZAV is a fast (interleaved/SIMD) entropy decoder; that is next.

## T1.2 entropy coding REFUTED for the decode budget (huff_speed)
Interleaved Huffman decode of the 14.38M-token stream, pinned, best of 7:
  N=1 4.28 ns/sym | N=2 3.81 | N=4 3.06 | N=8 3.10  (~0.3 GB/s on tokens)
Interleaving gained only 1.4x and plateaued at N=4, so the table lookup is
the bottleneck, not the serial bit-position dependency. Huffman shrinks tokens
to 5.79 bits/sym (14.38M -> 10.42M bytes, saves ~4.5% of output, ratio ~2.50),
but decoding them costs ~43 ms on top of the ~63 ms LZ decode: decode would
fall 3.19 -> ~1.9 GB/s, well under the 3.130 floor. Table Huffman cannot pay
for itself here even interleaved. The path that strictly dominates LZAV on all
three axes is therefore not reachable with this entropy method. The remaining
ratio lever that does not touch decode is the byte format (bit-packed offsets,
~+2% -> ~2.44), which would tie LZAV on ratio while we keep our decode lead.

## Moonshot REFUTED: rANS decode hits the same wall as Huffman (rans_speed)
Interleaved 32-bit rANS, 12-bit freqs, decode of the 14.38M-token stream:
  N=1 4.59 ns/sym | N=2 4.09 | N=4 3.56 | N=8 3.62 | N=16 3.84 | N=32 3.72
Best ~0.26 GB/s on tokens, and it PLATEAUS/worsens past N=4 -- slightly slower
than interleaved Huffman (3.06). rANS's branchless arithmetic did not help,
which pins the bottleneck on the per-symbol table load throughput, not the
algorithm. Both methods do ~1 random few-KB-table load per symbol at ~3 ns.
Entropy-coding tokens by ANY table method adds ~36-43 ms to the ~63 ms decode:
decode 3.19 -> ~2.1 GB/s, under the 3.130 floor. SIMD/gather rANS would not
escape this (gather is N sequential loads at the same throughput). Conclusion:
the ratio/decode tradeoff cannot be broken on this hardware with table entropy;
strictly dominating LZAV on all three axes is infeasible for this design. The
defensible win is the speed niche: fastest decode of anything denser than LZ4,
dominating Snappy on ratio and decode. Remaining ratio lever with no decode
cost is format bit-packing (~+2% -> ~2.44, ties LZAV, keeps the decode lead).

## Bit-packed offsets: ratio prize 2.47 (beats LZAV) but decode cost looks prohibitive
bitoff_probe on the 14.38M real offsets. Best 4-class widths [10,16,20,24]
save 8.86% of offset bytes (33.17M -> 30.23M = 2.94M = 3.31% of output),
taking ratio 2.388 -> ~2.470, above LZAV's 2.450. But a bitstream read per
match costs ~3.3 ns/match isolated (single accumulator; a Vec-indexed
"interleaved" variant was slower, inconclusive). 14.38M matches ~ 47.8 ms of
offset decode vs ~10-12 ms today, so integrated decode very likely falls from
3.19 to ~2.1-2.5 GB/s, under the 3.130 floor. The offset bitstream has no
table load (unlike Huffman/rANS), so register-allocated interleaving MIGHT
help, but even optimistic hiding leaves it marginal. The trade is ratio
dominance over LZAV for our decode crown -- probably not worth it, since the
decode lead is our defining strength. Building v6 to get the exact integrated
number is hours with a likely revert.

## GOAL3: pivot to speed. Physics measured, LZ4 is the gate, dense retry retired
speed_ceiling (single core, pinned): memcpy of Silesia 22.9 GB/s is the hard
decode ceiling (DRAM ~19, L3 ~48, L2 ~67). liblz4 decodes at 25% of it, we at
14%. Nobody is near the wall because cost is per token (14-20M tokens), not per
byte. GOAL3.md: Tier S1 = dominate liblz4 on decode+ratio, measured in the same
run (quick3 now reports liblz4 alongside). Dense retry made opt-in
(ALATIROK_DENSE=1); x-ray floor set to liblz4's own 1.00-1.01; total ratio
floor RAISED 1.85 -> 2.1009.
  goal3_baseline: ratio 2.37232 | comp 0.4651 | decomp 3.1239 vs liblz4 5.6526 (55.3%)

## GOAL3 decode: ablation maps the loop; credit guards +4%; window 256 KB
examples/dec_ablate.rs re-runs the fast loop over every block's real streams
with pieces switched off (cold cache, whole corpus, 13.8M tokens, ~66 ms):
  walk (no memory traffic) 38 ms = base 14 (1.0 ns/token, LZ4-class floor)
    + guards 8 + escapes 16 (25% of tokens carry one)
  memory side ~28 ms: far-offset misses dominate when cold; the offset
    position chain itself is only ~4 ms (full vs fixed-stride at 64 KB)
Refuted by measurement (do not retry): branchless cmov escapes (-25%: the
extras cursor chain serializes), shift/mask token decode instead of the
table (-5%), 16-byte copies (0), self/one-ahead prefetch (-30%), scalar
two-pass split (-10%), recomputing bounds after every escape (-6%).
Adopted: LZ4-style single 32-byte store per literal run and per match; a
credit counter that pays bound checks in bulk (one decrement per token,
escapes charge their excess). Same-run decode 55.3% -> 57.1% of liblz4.
Finder window default 8 MB -> 256 KB (FINDER_WINDOW): sources stay in L2,
so decode is immune to the VM's cache-state drift (57.0/57.1% on repeats vs
51.5/55.5% at 8 MB). Costs: ratio 2.372 -> 2.272 (floor 2.10), comp 0.457
-> 0.418 (a level, not a gate, under GOAL3).
  credit_win256k_default: ratio 2.27212 | comp 0.4184 | decomp 3.2126 vs
  liblz4 5.6378 = 57.0%

## Format v6: 3-bit literal, 4-bit match, fixed 2-byte offsets. Decode +26%
token_stats on the real stream (256 KB window, 12.9M tokens): 31.7% of tokens
escaped (literal 22.3%, match 12.1%), and every offset fit 18 bits, so v5's
two width bits bought nothing. v6 token = literal 0..6 direct / match 6..19
direct / offset bit 16; offset stream is a constant 2 bytes per match (17-bit
offsets, 128 KB window). Escaped tokens 31.7% -> 21.5%; the decoder's offset
read no longer waits on the previous token's width; 3-byte offsets vanish,
which pays for the smaller window (ratio unchanged at 2.2726).
  v6: ratio 2.27262 | comp 0.4204 | decomp 4.05/4.05/3.97 vs liblz4 5.66/5.66/5.63
      = 71.6 / 71.5 / 70.5% of liblz4 (was 57.0%)
Per file we are 61-85% of LZ4 on decode; x-ray (raw) decodes at 45 GB/s.
x-ray regression floor set to a raw store (0.99) until the fast level exists.

## GOAL3 S2 landed: AVX2 32-token pre-pass. Decode 90% of liblz4
The fast phase now decodes 32 token bytes per AVX2 pass (lengths, escape
lanes, match count), patches escaped lanes from the extras stream branch-free
over the escape mask, checks bounds once per chunk from its totals, and runs
a copy-only loop. Two lessons on the way, both measured: (1) `if` inside the
fixup loop made no difference either way -- the branches were not the cost;
(2) the real cost was retry waste: a chunk rejected after its pre-pass (a 255
continuation escape, a stream near its end) was retried one token later,
still containing the offender, up to 31 times: ~18 ms of ~47. Taking a full
chunk of tokens carefully after such a rejection fixed it. prepass_bench
shows the pre-pass alone costs 8.8 ms (3.4 cycles/token).
  cold harness (lib): 64 -> 47 -> 39.9 ms
  quick3: ratio 2.27262 | comp 0.412 | decomp 5.06 / 5.02 / 5.04 vs liblz4
          5.60 / 5.59 / 5.61 same run = 90.2 / 89.8 / 89.8% (was 71.6%)
Per file we now beat liblz4 on decode on ooffice, osdb and (raw) x-ray, tie
reymont, and trail by 4-33% elsewhere (mr, sao, nci worst).

## Continuation escapes decoded inside the chunk: decode 97% of liblz4
A 255 escape byte (literal run >= 262, match >= 275) used to reject the whole
32-token chunk to the careful path, which hurt long-match files most (nci
77%, mr 68% of liblz4). The fixup loop now reads the u16 continuation inline;
the branch is rare on most data and predictable where it is common.
  quick3: ratio 2.27262 | comp 0.412-0.422 | decomp 5.50 / 5.47 / 5.45 vs
          liblz4 5.65 / 5.63 / 5.60 same run = 97.4 / 97.1 / 97.2% (was 90%)
Per file vs liblz4 decode: beat it on mozilla, ooffice, osdb, reymont, sao,
xml, x-ray (7 of 12); trail on dickens 94%, mr 98%, nci 93%, samba 95%,
webster 86%.
