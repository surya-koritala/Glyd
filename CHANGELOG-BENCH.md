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

## GOAL3 S1 PASSED: decode beats liblz4 on Silesia, same run, ratio floor held
Minimum match 6 -> 7: tokens 12.5M -> 10.0M (-20%), and decode time tracks
tokens. Ratio 2.2726 -> 2.1923 (floor 2.1009). Compression 0.41 -> 0.345,
which GOAL3 assigns to the fast level. The candidate check was generalized
to a masked u32 over bytes 4..7 so any minimum 4..8 is sound (7 and 8 were
corrupting output before, trusting unverified bytes).
Also fixed on the way, a latent decoder bug: the chunked path keeps lengths
in u16 arrays, and a literal run at MAX_LIT_LEN (65797) wrapped to 261. Only
the parallel Silesia stream contained one. Such lengths now take the careful
path; tests/roundtrip.rs pins it. Refuted this round: packing lit/hi/ml into
one u32 lane (0), near-offset store-forwarding stalls (0), hoisting the copy
loop into its own function (+3%, kept).
  quick3: ratio 2.19234 | comp 0.345 | decomp 6.15 / 6.03 / 6.05 vs liblz4
          5.67 / 5.69 / 5.73 same run = 108.5 / 106.0 / 105.6%  S1 OK
Per file we beat liblz4 on 10 of 12 (dickens 114%, ooffice 143%, sao 189%,
xml 115%...), trail on nci 94% and webster 96%. x-ray (raw) 49 GB/s.

## Docs refresh: README, GOAL3 status, per-file S1 table
README rewritten from the format-v2 era to the current state: v6 layout, the
field table, a fresh same-run per-file table, the decode trajectory, and a
"where we are / what is next" section. GOAL3 S1 marked passed with start/now
columns; S3 table now lists fast (not built), default (min match 7), dense
(ALATIROK_DENSE=1). Fresh run for the README table:
  quick3 readme_s1: ratio 2.19234 | comp 0.346 | decomp 6.05 vs liblz4 5.54
                    same run = 109.3%  S1 OK
Per file: beat liblz4 on 10 of 12 (sao +87%, ooffice +35%, osdb +23%, mr
+19%); nci -2%, webster -3%.

## aarch64: NEON port of the v6 decoder (Apple M1 Max)
The crate did not build on arm64 (x86 modules were unguarded) and the
scalar fallback decoded Silesia at 1.28 GB/s against liblz4's 4.43 in the
same run (29%). `src/neon_decompress.rs` is a lane-for-lane port of the
AVX2 decoder: the 32-token pre-pass runs as two 16-lane vectors, movemask
is bit-select + three pairwise adds, sums are `vaddlvq_u8`, copies are
16-byte pairs (ldp/stp q). Same-run quick3, M1 Max, one core:

Silesia 1C decomp 1.28 -> 5.32 GB/s vs liblz4 4.36 = 122% (S1.2 PASS on
this host). Beats liblz4 on 12/12 files (nci +8%, webster +15%, sao +97%).
Ratio 2.192, comp 0.27 GB/s (scalar finder, no NEON prefix compare yet).
memcpy ceiling here is 39.9 GB/s; we are at 13% of it, liblz4 at 11%.
25 tests green including the 1M-mutation fuzz. Checksum stays scalar on
arm64 (not on the raw decode path that the gate measures).

## aarch64: where the floor is, and loop-free escapes (M1 Max)
New harness `examples/floor.rs`: each Silesia block pre-decoded once into
flat length arrays, then a ladder of loops each removing one physical cost.
Silesia, 10.0M tokens, 21.2 bytes/token, one core:

| rung | ns/token | GB/s |
|---|---:|---:|
| real decoder (before) | 3.70 | 5.3 |
| copy loop only, real branches | 2.26 | 8.7 |
| one unconditional 32-byte copy per run, no branches | 1.57 | 12.5 |
| same with every offset >= 32 | 1.56 | 12.6 |
| one 32-byte store per token, no match load | 0.84 | 23.4 |
| pointer chain only (dst += lit + ml) | 0.69 | 28.6 |
| memcpy | 0.45 | 44.1 |

Findings. (1) The wall for this format on this chip is ~1.6 ns/token
(~12.5 GB/s): the serial dst chain plus one dependent load+store per token.
memcpy (39.9 GB/s) is unreachable at 21 bytes/token; throughput = bytes per
token / ns per token, and 0.45 ns/token is below the pointer chain.
(2) Short offsets cost nothing on Silesia (97.8% are >= 32); the
store->load hazard GOAL3 worried about is 0.1 ns. (3) Each rare copy path
(lit > 32: 1.9%, offset < 32: 2.2%, ml > 32: 6.6%) costs one mispredict,
additive: 0.15 + 0.12 + 0.31 ns. Branch-free versions (unconditional extra
copies) cost more than they save. (4) The escape patch loop was 1.07 of
the 1.3 ns pre-pass: not its branches (a branchless select variant was
0.96) but the loop itself. The mask walk alone with an empty body costs
0.64 ns/token: variable trip count -> exit mispredict every chunk -> the
flush exposes the movemask latency chain each time.

Change: escapes are expanded without a loop. fields per lane (0..2) ->
exclusive prefix sum (4 ext+add steps per half) -> vqtbl4q gather from
the next 64 extras bytes -> u16 blend with the direct lengths. A 255
continuation in any consumed field sends the chunk to the old scalar loop.
Pass 1 in isolation: 1.14 -> 0.45 ns/token, bit-exact on all 794 blocks.
Copy loop: match and literal tails up to 128 bytes are fixed 3x32 stores;
the loop only runs past that (5% in the harness). Offset < 32 is a cold
function.

Silesia 1C decomp 5.32 -> 6.68 GB/s vs liblz4 4.36 = 153% (was 122%).
12/12 files up; osdb 5.3 -> 8.5, ooffice 5.7 -> 7.4, samba 5.3 -> 7.0.
3.01 ns/token against the 2.2 ns copy loop: ~0.8 ns of pass 1 left.
25 tests green including the 1M-mutation fuzz.

## aarch64: pipelined pre-pass (M1 Max)
Chunk k+1's pre-pass now runs before chunk k's copy loop (double-buffered
length arrays), to overlap the vector chain and the array store->load with
the copies. Silesia 1C decomp 6.68 -> 6.79 GB/s (155% of liblz4): +1.5%,
so the out-of-order core was already hiding most of it. The decoder is at
2.94 ns/token against the harness floor of ~2.6 (copy loop with fixed
tails 2.16 + loop-free pass 1 0.45). What is left is per-token bookkeeping
in the copy loop (offset validation, the token's offset bit) and the 10% of
chunks that hit a 255 continuation; neither is worth more than ~5%. The
next lever is the format: a token wide enough to make escapes rare (they
are 31% of tokens at 3+4 bits) trades ~10% ratio, below the liblz4 floor.

## Compression: the ceiling, and the fast level (M1 Max)
New harness `examples/cfloor.rs`, Silesia, one core:

| rung | GB/s | ns/byte |
|---|---:|---:|
| memcpy | 44 | 0.02 |
| hash every position into a 64 KB table, probe, no branch | 4.1 | 0.23 |
| greedy LZ4-shape parse (hit/miss branch, extend), no output | 0.52 | 1.80 |
| liblz4 | 0.66 | 1.41 |
| default finder alone (before) | 0.33 | 2.83 |
| compress_into (before) | 0.28 | 3.33 |
| block checksum, scalar (before) | 2.6 | 0.36 |

The ceiling for an LZ finder is not memory: it is one data-random branch
(match or not) per probed position, ~7 cycles average, so a greedy parse
that probes every literal position tops out around 0.7-1 GB/s per core.
liblz4 is there. Going faster means probing fewer positions, which is the
ratio trade.

Changes.
1. NEON checksum: 75 -> 8.4 ms on Silesia (2.6 -> 23.5 GB/s). Vector-only
   formulation (block sums, k-weighted block sums, position-weighted sums),
   four independent accumulators; chained into one it was 12 cycles per
   32 bytes.
2. Emit path: cursors by value, unchecked writes with per-block reserve,
   32-byte wild literal copies bounded by the input end, branchless escape
   and offset writes. 8.5 -> ~4 ns per token.
3. Fast level (GOAL3 S3): `compress_into_fast`, `compress_parallel_into_fast`,
   CLI `-1/--fast`. LZ4-class finder: one position per bucket, 5-byte hash
   (as liblz4 on 64-bit), a u64 compare that is both the hit test and the
   length, LZ4 skip acceleration, back-match, minimum match 5 emitted as
   FLAG_DENSE blocks. Table size sweep (parse only, 5-byte hash, min 5):
   12 bits 0.72 GB/s at 2.00, 13 bits 0.68 at 2.10, 14 bits 0.57 at 2.18,
   16 bits 0.38 at 2.25. 13 bits chosen. Refuted for speed: 64 KB window
   (slower: fewer matches, more probes), liblz4's forward-hash loop shape
   (slower here, the core overlaps it already), 4-byte hash.

Same run (quick3), Silesia, one core:

| level | comp GB/s | ratio | decode GB/s | vs liblz4 decode |
|---|---:|---:|---:|---:|
| liblz4 | 0.66 | 2.101 | 4.38 | 100% |
| fast | 0.54 | 2.098 | 5.02 | 114% |
| default | 0.34 (was 0.28) | 2.192 | 6.80 | 155% |

Fast is at 81% of liblz4's compression speed at its ratio, and x-ray now
compresses (1.004) instead of being stored raw. What remains in the fast
path is per-token work in the parse loop (extend, back-match, two
variable-trip loops) that liblz4 also pays; per byte the parse alone is at
liblz4 parity (0.67 vs 0.66), the rest is emit and framing.
27 tests green (fast-level round trips added).

## Fast level: offset floor 1, the profile, and where the gap is (M1 Max)
- The fast finder now accepts offsets down to 1 (the default parse refuses
  < 8 to keep its copies wide). Parse-only sweep at 13 bits: ratio 2.104 ->
  2.183 and 0.68 -> 0.70 GB/s. End to end: ratio 2.098 -> 2.176.
- Decoder: offsets under 8 were a byte loop; now eight byte stores lay the
  pattern down and 8-byte copies continue at a stride rounded up to a
  multiple of the period (liblz4's trick). Fast-level decode 4.48 -> 4.84
  GB/s (mozilla 3.3 -> 3.9, mr 3.05 -> 4.1).
- Instruments profile of the fast level (`examples/fastloop.rs`, Time
  Profiler, 4000 samples): 31% of samples on the instruction after the
  hit branch, 19% on the miss-loop top, 7.5% waiting on the candidate
  compare. Half the time is resolving the data-random hit/miss branch;
  the loop body is 12 instructions. Per byte the parse alone (0.70 GB/s)
  is faster than liblz4 end to end (0.66); our four-stream emit is 2.8 ns
  per token (~9 cycles, 4 stores) and is the whole remaining gap.
- Refuted for speed, all measured in `cfloor`: NEON 32-byte extend and
  u64 back-match (355 vs 292 ms: the vector->scalar latency lands on the
  serial pos chain), liblz4's forward-hash loop shape (338), acceleration
  2-4 (ratio falls below 2.10 before speed passes liblz4), 64 KB window
  (slower: fewer matches, more probes), record-then-emit two-pass (equal),
  u16 escape store and 16-byte literal copy (noise).

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
- Fast phase restructured: pass 1 fills L1-resident length arrays for up
  to 1024 tokens (32 at a time, running totals checked against the block
  bounds), pass 2 is one copy loop. Bookkeeping per 1024 tokens instead of
  per 32. Silesia 1C decode 6.82 -> 6.94 GB/s (159% of liblz4).
- Refuted: two independent blocks decoded interleaved on one core (2.62
  vs 2.17 ns/token in the harness; the loop is mispredict-bound, and a
  flush kills both chains); folding the offset bit into the length array
  (pass 1 pays what pass 2 saves).
- Measured, not adopted: minimum match 8 gives 8.05 GB/s (+18%) at ratio
  2.056, under the liblz4 floor. Decode is a straight dial on token count.
- Multi-core (`examples/mc.rs`, parallel-compressed independent 256 KB
  blocks, 10 threads): 42.9 GB/s aggregate over Silesia, above this
  machine's single-core memcpy (40 GB/s); per file 26-71 GB/s.
