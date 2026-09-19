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
(GLYD_DENSE=1); x-ray floor set to liblz4's own 1.00-1.01; total ratio
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
(GLYD_DENSE=1). Fresh run for the README table:
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

## Turbo level (M1 Max)
`compress_into_turbo` / `compress_parallel_into_turbo` / CLI `-t --turbo`:
the default finder at minimum match 8, emitted as FLAG_TURBO blocks (new
token table and escape base; the decoders are bias-generic). Same run:
decode 8.23 GB/s = 188% of liblz4 (default 6.94 = 159%), ratio 2.055
(default 2.192, liblz4 2.101), comp 0.33 GB/s. Per file 7.0-11.2 GB/s on
compressible data; sao is stored raw at this level (its 1.037 is under the
4% threshold) and decodes at memcpy. Decode scales with tokens per byte:
min 7 -> 8 cut tokens 18% and bought 18%.

## Turbo at minimum match 10; the compression dial (M1 Max)
- Finder: minimums above 8 are now sound (the word plus verify mask prove
  8 bytes, the extension must reach the minimum or the hit is dropped).
- Turbo sweep, same run: min 8 8.2 GB/s at 2.055, 9 8.8 at 1.975, 10 9.3
  at 1.884, 12 10.5 at 1.751. Turbo set to 10: decode 211% of liblz4,
  +34% over the default. Hash table 17/18 bits: +0.005 ratio, not worth
  the L2 traffic (comp -15%).
- Fast level: 12-bit table gives 0.585 GB/s (+7%) at ratio 2.078, under
  liblz4's 2.101; unbounded back-match and dropping the post-match
  re-insert are noise. Kept at 13 bits (2.176). The dial is documented in
  finder.rs; the parse loop has no ratio-neutral speed left.

## v7 milestone 1: entropy coders
- Huffman: `huff8.rs`, 8-stream interleaved canonical Huffman over
  `bits::BitReader`/`BitWriter` (LSB-first, bit-reversed codes, length
  capped at 11 so decode is one table lookup). Symbol i goes to sub-stream
  i % 8; decode refills all 8 streams every 4 symbols per stream (44 bits
  <= the 56-bit refill guarantee). Roundtrip, invalid-code (Kraft-sum) and
  overrun tests pass. Silesia literals (dickens + mozilla, 235 blocks,
  20 MB; the full 12-file corpus gives the same number at 794 blocks/55 MB,
  so this is not a sampling artifact), best of 5, `target-cpu=native`:
  first cut **1.06 ns/symbol**, still over the 0.6 ns/symbol gate against
  `examples/huff_spike.rs`'s 0.46 on the same data/table/loop shape.
  Closed part of the gap in `bits::BitReader`: `consume()` no longer
  touches any counter (just `bits >>= n; cnt -= n`); overrun accounting
  moved into `refill()` instead, folding in the whole 4-symbols-per-stream
  window's consumption in one step (4x less often than per-symbol). A
  pure position-derived count (`p - start`) was tried first and reverted:
  once `refill`'s OOB-safety clamp kicks in, `p` gets rebased to
  `last + small_delta` on every subsequent call instead of continuing to
  grow, so it plateaus right at the real/pad boundary and can't
  distinguish "ended exactly at the last symbol" from "read arbitrarily
  far past it" -- confirmed by two opposite test failures (a false-positive
  overrun on a valid decode, a false-negative on a genuine 10x over-read).
  `huff8::decode`'s table lookup is also unchecked now (`get_unchecked`,
  justified by `peek(TB) < table.entries.len()` by construction). Result:
  **0.85 ns/symbol**, ~20% faster, still over the 0.6 gate. objdump of the
  main loop (`target/release/deps/v7_codecs-*`, `huff8::decode` vs.
  `huff_spike::decode::<8>`) shows both loop bodies at ~290 instructions,
  but huff8's spends proportionally more of them on stack traffic (74 str
  + 39 ldr of 292, vs. the spike's 35 str + 22 ldr of 316) -- `BitReader`
  carries 6 fields per stream (`p, last, bits, cnt, filled, budget`)
  against the spike's unchecked `St { p, bits, cnt }` (3), and across 8
  unrolled streams that state doesn't fit in registers.
- Round 2: restructured `huff8::decode` per the register-pressure finding
  above. Added `bits::FastReader { p, bits, cnt }` -- the spike's exact
  3-field, unclamped, unaccounted reader, plus `BitReader::to_fast`/
  `resume`/`last()`. `resume` reconciles the round-1 `budget`/`filled`
  scheme after an unaccounted excursion: since `FastReader` never clamps
  (the caller proved it wouldn't need to), the bits it loaded are exactly
  `(f.p - old_p) * 8` (the same per-call identity `refill` relies on:
  bytes advanced == bits newly loaded), so `consumed = old_cnt + loaded -
  f.cnt` folds back exactly, no approximation. `huff8::decode`'s hot loop
  now runs on 8 `FastReader`s; an outer loop computes, each pass,
  `iters = min(remaining/32, min_k safe_refills(fast[k], last_k))`
  (`safe_refills(last) = (last-p)/7 + 1` for `p <= last`, since a refill
  advances `p` by at most 7 bytes) and runs that many 32-symbols-at-a-time
  batches -- the spike's exact loop body, unsafe and 3-field. Once some
  stream runs low on margin (or fewer than 32 symbols remain), `resume`
  hands the state back to the clamped `BitReader`s for the existing
  per-symbol tail path, which is also what keeps `overrun` exact.
  Result: 235 blocks / 20 MB, best of 5, `target-cpu=native`:
  **0.61 ns/symbol** (1.06 -> 0.85 -> 0.61 across the two rounds, ~43%
  faster than round 1's start), a hair over the 0.6 gate. objdump of the
  new inner loop confirms the fix landed: 186 instructions (was 292) with
  memory ops back to ~18% (was ~39%), matching the spike's proportions.
  Added `huff8_short_codes_roundtrip` (200,000 symbols, 3 distinct byte
  values / ~2-bit codes) to exercise the outer loop's re-evaluation, since
  short codes make the pointer creep forward slowly and `safe_refills`'
  worst-case-7-bytes assumption undershoots badly there -- passes, just
  costs more (small) outer passes instead of one big one, as intended.
- tANS: `tans::encode8`/`decode8`, 8-stream interleaved tANS over the same
  `bits::BitReader`/`FastReader` split as Huffman above. `encode8` splits
  symbol i to sub-stream i % 8 across 8 independent `Encoder`s (unchanged
  from Task 3). `decode8` is `huff8::decode`'s exact structure ported to
  tANS: clamped `BitReader`s read each stream's initial TL-bit state and
  handle the tail, unclamped `FastReader`s run the hot loop 4 symbols/
  stream per refill (4 * TL = 40 <= 56), `safe_refills` bounds how many
  batches run before any stream might need the clamp. First cut, with
  Task 3's 4-byte `DecodeEntry { sym: u8, nbits: u8, base: u16 }` struct:
  14M symbols (36-symbol alphabet, skewed), best of 5, `target-cpu=native`:
  **0.873 ns/symbol**, over the 0.6 gate. `size_of::<DecodeEntry>()` is
  already 4 (no padding -- the three fields pack exactly into a `u16` +
  two `u8`s; the brief's 6-byte guess didn't hold here), so the fix isn't
  removing padding but removing the struct itself: packed the same three
  fields into a `u32` (`sym | nbits << 8 | base << 16`) and switched
  `DecodeTable`'s storage from `Vec<DecodeEntry>` to `Vec<u32>`. Result:
  **0.738 ns/symbol**, ~15% faster, still over the 0.6 gate -- the same
  shortfall shape as huff8's first cut (1.06) before its two
  register-pressure rounds. Unlike Huffman's fixed `peek(TB)` table key,
  tANS's key is `st[k]`, a per-stream state threaded from one decoded
  symbol's `base + bits` to the next lookup: that's a 4th live value per
  stream on top of `FastReader`'s 3, and the likely next place to look if
  this needs closing further (not attempted here; out of this task's
  scope). Roundtrip tests (n = 0, 1, and non-multiples of 8) and
  `tans8_streams_of_different_lengths` (one sub-stream all rare/long-code
  symbols, the other seven all common/short-code, exercising the outer
  loop's per-stream `safe_refills` minimum when streams end up very
  different lengths) pass.
- tANS rounds 1-2 (coordinator-directed register-pressure fix, bounded
  attempts, final for this task): round 1 tried splitting `decode8`'s
  8-stream inner loop into two sequential groups of 4 (to bring the
  per-iteration live set -- `FastReader`'s 3 fields + tANS's own `st[k]`,
  x 8 streams -- under the ~31 GPRs available); it measured slower, not
  faster (0.77 ns/symbol vs 0.74), so round 2 reverted it back to the
  single 8-way interleaved loop. Kept from round 1: `tans8_overrun_is_an_error`.
  Final, confirmed: **0.74 ns/symbol** (packed-u32 table), still over the
  0.6 gate.

## v7 milestone 2: container + decoder on the default parse
- Container: `parse_header` accepts VERSION_V7; `decode_block` dispatches
  v7 payloads to `v7_decode` with a thread-local table carry, reset at
  FLAG_CHAIN_RESET (the parallel path decodes each chained unit whole on
  one thread starting at such a block, so it sees what the sequential
  path sees). `compress_into_max` / `compress_parallel_into_max`: the
  default (Lzav) finder, its v6 streams bridged to a sequence list by
  `sequences_from_streams`, `v7_encode::encode_block` per 256 KB block;
  a block the coder cannot shrink below the chunk is stored as a v6 raw
  block. Round trip through every `decompress*` entry point on empty,
  one byte, constant, random (raw), periodic (periods 3..70000), text
  and word-salad inputs, the last checked to exercise literal- and
  sequence-table reuse across blocks (the periodic/text inputs never do:
  `close()` demands identical support, and one rare code per block --
  a block-boundary literal run, a 46-byte match once in 256 KB -- breaks
  it; ~0.1% of ratio, a later task's call).
- Pass 3 (copies) took `neon_decompress::copy_run`'s measured shape per
  sequence: unconditional 32-byte copy, fixed 3x32 tails, cold
  `short_match` under offset 32 (`copy32`/`short_match` shared with the
  v6 decoder), portable twin under `cfg(not(aarch64))`. The 3x32 tails
  overshoot a sequence by up to 95 bytes (a 33-byte run copies 128), so
  the wild path needs 96 bytes of room past the sequence in both `dst`
  and the literal buffer -- the brief's 64 was not enough; the last 96
  bytes of a block whose `dst` has no slack (the parallel path's
  per-block slices are exact) take an exact copy, so nothing is written
  past `dst` (sentinel test with the worst-case shape fails at 64,
  passes at 96). Pass 3: 6.0 -> 2.9 ns/sequence; Silesia decode 0.99 ->
  1.20 GB/s (mr, short offsets, 0.62 -> 1.14). The per-sequence
  block/literal bound checks, redundant with pass 1's totals, cost 0.3
  ns/sequence (2% of decode); kept.
- `examples/v7_bench.rs`, Silesia, M1 Max, `target-cpu=native`, median
  of 3 runs of >= 0.3 s, zstd 1.5.7 (bulk API, contexts reused) in the
  same run:
  **v7: ratio 2.7406, comp 0.133 GB/s, decomp 1.203 GB/s** |
  zstd-3: ratio 3.2045, comp 0.326, decomp 1.434 |
  zstd-1: ratio 2.8942, comp 0.551, decomp 1.543.
  Ratio is in the brief's 2.6-2.8 band (+25% over v6's 2.19 on the same
  parse); decode is under the 2.5 GB/s stop rule, and 3-4 GB/s was never
  in reach of this pipeline: per byte, pass 1 (sequences) is 0.46 ns =
  50%, pass 2 (literals) 0.18 ns = 19%, pass 3 (copies, before the NEON
  loop) 0.29 ns = 31%. Inside pass 1 the three tANS streams take ~3
  ns/sequence (~1 ns/symbol) and the extra-bits walk 6.7 ns/sequence --
  the largest single cost in the decoder (35% of the total): three
  dependent clamped `BitReader::get`s through `ers[i % 8]` (state in
  memory, not registers) plus the branchy `Reps::resolve`. The stop
  rule's remedy, double-symbol Huffman tables, targets the 19% pass;
  the walk is where the time is, and it wants the huff8 treatment (8
  unclamped `FastReader`s in a loop unrolled by 8, `safe_refills`
  bound, clamped tail) before Task 9 -- not implemented here.

## v7 milestone 2b: pass 1 on fast readers
- The extra-bits walk in `v7_decode::sequences` (three dependent clamped
  `BitReader::get`s per sequence through `ers[i % 8]`, eight readers in
  memory) now has the `huff8::decode` shape: an unclamped batch loop of
  8 sequences (one per stream) under a proven load bound, the clamped
  readers for the tail. Its hot state is one bit position per stream
  rather than a `FastReader`: a valid sequence's extra bits are at most
  18 + 18 + 20 = 56 (lengths up to 2^18, offsets below 2^21) and one
  unaligned 8-byte load shifted by the sub-byte position holds >= 57,
  so a sequence is one load, three field extractions and one add --
  no accumulator, count or refill, 8 live registers instead of 24. The
  load bound is `safe_seqs`: at most 58 bits (codes 31, 31, 23) = 8
  bytes per sequence whatever the code bytes hold; a corrupt 58-bit
  sequence reads a zero for its last bit and advances exactly. The tail
  starts its clamped readers at the walk's positions
  (`BitReader::new_at`, exact accounting, `overrun` unchanged). Walk
  tables `v7_format::*_WALK` (256 x u64, `base << 32 | mask << 8 | nb`)
  replace the branchy code functions: per field an AND with the mask
  and an add of the base as shifted operands, a shift by `nb` -- three
  instructions where building the mask cost two more. `Reps::update`
  selects (`select_unpredictable`) instead of matching. Totals are
  summed over the arrays after the walk. `tans::decode8` and
  `huff8::decode` bounds-check their output batch once instead of per
  symbol.
- Steps, Silesia, per sequence, same machine state (zstd-3 1.41-1.43 in
  those runs): walk 6.73 (baseline) -> 3.40 (8 `FastReader`s, 32-entry
  u32 tables, select reps; readers spilled to the stack, 56 instructions
  per sequence) -> 2.88 (bit positions, 256-entry tables) -> 2.38
  (streams written out so positions live in registers) -> 2.40 (scalar
  positions, totals after the walk: nil then, 0.15 better once the
  table layout landed) -> 2.25 (u64 entries, base-high/mask-mid layout).
  Rejected: branchy `Reps::update` 2.71 (rep codes are 4.0% of offsets
  on this parse and it still lost 0.3); raw-pointer batch slices 2.32
  (3%, not worth the unsafe). Then `decode8` once-per-batch bounds
  check: the three code streams 2.42 -> 2.14 ns/sequence, pass 1 5.27
  -> 4.99; the same in `huff8::decode`: pass 2 0.203 -> 0.169 ns/byte.
  Pass 1 now: walk 2.25 + tANS decode 2.14 + tANS table builds 0.58
  (three 1024-entry builds per block; reuse rarely fires) + 0.04.
- `examples/v7_bench.rs`, same protocol as milestone 2, before and after
  in the same machine state:
  before **v7 decomp 1.218 GB/s** | zstd-3 1.479 | zstd-1 1.576;
  after **v7: ratio 2.7406, comp 0.142 GB/s, decomp 1.883 GB/s** |
  zstd-3: ratio 3.2045, comp 0.341, decomp 1.474 |
  zstd-1: ratio 2.8942, comp 0.561, decomp 1.558.
  Pass 1 9.79 -> 5.0 ns/sequence (brief: <= 5), decode +55% (brief:
  >= ~1.7 GB/s); v7 decode is now 1.28x zstd -3 and 1.21x zstd -1 on
  this parse. Per byte: pass 1 0.21, pass 2 0.17, pass 3 0.13.
## v7 milestone 2c: encoder throughput
- Stage costs per pass over Silesia (212 MB, 10.0 M sequences, 63.3 M
  literals; `compress_into_max` replayed stage by stage with the Lzav
  finder at 2.9-3.0 ns/byte in the same run, M1 Max, `target-cpu=native`):
  the v7 encode side was 3.8 ns/byte -- `sequences_from_streams` 0.27,
  `encode_block` 3.53 (75 ns/sequence): the codes + extra-bits loop
  0.130 s, literal histogram + lengths 0.158 s (of which
  `huffman::build_lengths` 0.125 s = 150 us/block: a stable re-sort of
  the live nodes per merge, two or three builds per block for the length
  cap), `huff8::encode` 0.198 s (3.1 ns/literal), code histograms +
  normalize 0.039 s, the three tANS streams 0.220 s (2.4 ns/symbol),
  assembly 0.004 s. Steps, each byte-identical on the whole corpus
  (FNV of every `compress_into_max` output against the milestone-2
  binary) and measured in the same harness: (1) `huffman_lengths` sorts
  once and inserts merged nodes at `partition_point` (identical ties):
  lengths 0.125 -> 0.027 s, encode 3.53 -> 3.18 ns/B. (2) `bits::BitCursor`
  (64-bit accumulator, one unconditional 8-byte `write_unaligned` per
  put, cursor by value in the caller's frame) and `bits::write_streams`
  (the section layout -- 8 u32 sizes, 8 padded streams -- reserved from
  the caller's bit bound, `B / 8 + 16` bytes per stream); `huff8` and
  `tans` write their sections straight into the payload with stride-8
  walks, the tANS reverse pass runs the 8 stream states side by side
  (independent chains overlap in the pipeline) into a packed `u32` chunk
  scratch; `encode_block` writes every section into `out` behind a
  sub-header placeholder, with codes and the per-sequence extra bits (ll,
  ml, off concatenated: at most 18 + 18 + 20 = 56 bits, one put) in a
  caller-owned `EncScratch`: 3.18 -> 1.32 ns/B. (3) 4-way then 8-way
  interleaved histograms (`[u32; N]`, no `vec!`): code hists 0.041 ->
  0.011 s, literal hist 0.038 -> 0.020 s. (4) tANS reverse pass without
  the four per-symbol bounds checks (`chunks_exact` walks, a 256-entry
  `sym` table so a `u8` indexes it -- the `state_table` check stays as the
  zero-count safety net), `u32` wrapping `delta_find`, the chunk stored as
  `state | nbits << 16` with the masking moved to the forward pass, four
  chunks per put: 2.01 -> 1.17 ns/symbol; `huff8` four symbols per put
  from split code/length tables: 3.1 -> 0.44 ns/literal. (5)
  `sequences_from_streams` in one pass into a reused `Vec<Sequence>`:
  0.27 -> 0.20 ns/B. (6) `huffman_lengths` on fixed arrays with the exact
  two-queue merge (leaf ties by descending symbol, merged ties by newest,
  merged over leaf -- the original stable sort's order, checked identical
  on 20 815 histograms): lengths 0.027 -> 0.013 s. (7) Zipped cursors in
  the codes loop (the `Vec`s behind `&mut EncScratch` reloaded pointer
  and length per store), two sequences' extras per put when they fit:
  0.130 -> 0.035 s and 0.013 -> 0.008 s. Rejected, measured slower or
  equal: a branch-free `Reps::code_for` (the compiler already selects,
  and the rep0/rep1 branches predict well on this parse), branch-light
  escape decoding in the bridge.
- After: encode side 0.95 ns/byte (bridge 0.20, `encode_block` 0.75 = 16
  ns/sequence: codes loop 0.035 s, literal hist + lengths 0.035 s, huff8
  0.028 s, code hists 0.011 s, tANS 3 x 0.014 s, extras 0.008 s).
  `examples/v7_bench.rs`, same protocol as milestone 2, milestone-2
  binary and this one back to back: **comp 0.134 -> 0.237 GB/s**, ratio
  2.7406 and decomp 1.19 GB/s unchanged (zstd-3 comp 0.326 / 0.321 in the
  two runs). The brief's 0.5 ns/byte (~0.28 GB/s) is not reached: the
  parse alone is 2.9-3.0 ns/byte here, and what is left on the encode
  side is per-sequence work at its instruction-throughput floor -- the
  bridge (4 ns/sequence, gone with Task 9's parse), the codes loop (3.5
  ns, of which the rep-offset state chain is 0.8) and the three tANS
  reverse passes (~14 instructions/symbol with 8 chains in flight).
## v7 milestone 3/4: modeling + double-fast parse
- `v7_encode::find_sequences_dfast` replaces the milestone-2 bridge in
  `compress_into_max`: zstd -3's double-fast shape -- a long table keyed
  by an 8-byte hash and a short table keyed by a 5-byte hash, one
  candidate each; at every position the three repeat offsets are tried
  first (a 4-byte compare each), then the long candidate, then the
  short one -- plus lazy matching by one position (a match at pos + 1
  that is 4+ bytes longer wins) and zstd's post-match insertions (match
  start + 2 in both tables, end - 2 long, end - 1 short). Minimum match
  4, window 2 MB across the blocks of one call (positions absolute in
  the input). The tables live in a thread-local and are never cleared:
  a stale entry is at or past the current position (rejected) or is
  compared byte for byte like any candidate, so the parallel path's one
  call per 256 KB chunk pays no 2 MB calloc. The repeat offsets mirror
  `Reps::code_for`'s update and reset per block as `encode_block`'s do.
- Silesia ratio (this coder), the steps: greedy, 17/16-bit tables
  (zstd -3's sizes): 2.7406 -> 3.0475, under the spec's 3.10 stop rule,
  so lazy matching went in: 3.1337; zstd's insertions: 3.1621; tables
  18/18 (2 MB): **3.2219** -- G2 (>= 3.20) met, zstd -3 is 3.2045.
  Table size is the dial: 17/16 3.162, 18/17 or 17/18 3.204, 18/18
  3.222; the parse is ~7% slower at 18/18 than at 17/16, the extra on
  the binaries (x-ray +30%). Skip strength 8 instead of 6 is +0.1%
  (x-ray +1%), not taken; a short rep hit yielding to a 4+ longer long
  candidate is +0.1%, not taken; long-before-reps is -0.7%.
- The parse's share, measured by running zstd -3's own sequences
  (libzstd's `ZSTD_generateSequences` through the zstd-sys static
  library, its 128 KB blocks merged pairwise) through `encode_block`:
  3.1398. So zstd's dfast parse through this coder is 2.0% behind
  zstd -3 -- that 2% is the coder's (coarser length codes above 16,
  1 + 2n-byte tANS tables and 128-byte Huffman tables per block, 8 x
  u32 sub-stream sizes per section) -- and this parse beats zstd's
  dfast on every Silesia file through the same coder (+2.6% total).
- Container test: the word salad no longer reuses *sequence* tables
  under this parse (a rare code -- a literal run of 5, a new offset
  bucket while the window fills -- flickers between blocks and
  `close()` demands identical support); a 64-byte-record input whose
  blocks parse to the same few codes asserts that instead, the salad
  keeps the literal-table assertion.
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
  the 7.2 ns/byte total, so even a free coder leaves G4 short -- the
  parse needs zstd's pipelining (the next position's hashes and table
  loads issued a probe early; each probe is a hash -> table -> candidate
  latency chain now) and fewer branches per match (two probes per match
  with the lazy step, each with five data-dependent branches; ~39 ns
  per match on dickens). Per file, parse ns/byte: nci 1.1, xml 1.6,
  samba 2.6, mozilla 3.4, osdb 3.4, reymont 3.8, mr 4.1, webster 4.3,
  ooffice 4.4, sao 5.0, dickens 5.1, x-ray 5.5 (its 4-byte short
  matches every ~15 bytes keep the probe step at 1: with the long table
  alone it parses at 0.36 ns/byte and loses 7% of its ratio, at 17/16). G3 (decode >= 3.0)
  is the decoder task's; the parse's sequences are shorter than the
  Lzav parse's (dickens 8.3 bytes per sequence, x-ray 9), so per byte
  the current decoder is 17% slower than at milestone 2 (1.18 -> 0.98).

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
- Profile at the double-fast parse (Silesia, 14.2 bytes per sequence,
  temporary per-pass timers, M1 Max): per byte pass 1 0.318 ns (57%:
  the three tANS streams 2.11 ns/sequence, the extra-bits walk 2.08,
  table builds 0.42, tail 0.17), pass 3 (copies) 0.177 (32%, 2.67
  ns/sequence), pass 2 (literals) 0.055 (10%). Sequence statistics:
  ll = 0 in 62% of sequences, ll > 32 in 0.2%, ml > 32 in 4.9%, offsets
  under 32 in 4.3% (mozilla 15%), rep codes 11% of offsets (mozilla 32%,
  nci 25%, the text files ~1%). Pass 2 is under the 25% that would have
  bought the double-symbol Huffman table.
- Steps, each measured against its predecessor with the two binaries
  run alternately (other sessions' builds moved single runs by up to
  10%); the final column is the whole corpus back to back in one state:

  | step | what | ns/sequence | ns/byte |
  |---|---|---|---|
  | baseline f25f4f9 | | p1 4.80, p2 0.82, p3 2.67 | 0.553 |
  | copy pass | bounds implied by the totals dropped from the fast loop; one compare for the offset; the last sequence (the only ml == 0) and the last 96 bytes of a slack-less `dst` exact | p3 2.67 -> 2.15 | 0.518 |
  | tANS decode | bit positions + a window per stream (2 live values, no spills); u64 entries `nbits, sym << 8, mask << 16, base << 32`, the 32 symbols of a batch written out: 5 ALU ops per symbol; table rebuilt in place in `DecTables` | codes 2.53 -> 2.15 | 0.499 |
  | groups of eight | codes and lengths in `[ll x 8, ml x 8, off x 8]` groups (`tans::decode8_rows`): one pointer per array in the walk instead of three, positions as absolute bit addresses; the walk's batch 316 -> 263 instructions, ~80 -> 6 stack references | walk 2.10 -> 1.90, tail 0.17 -> 0.14 | 0.489 |
  | huff8 decode | the same position scheme: 4 ALU ops per symbol, no spills | p2 0.82 -> 0.74 | 0.484 |
  | table build | spread from the index (no running position), fill with a 256-entry counter, u64 math and a mask table: 2.2 -> 1.5 us per table | tables 0.46 -> 0.38 | 0.479 |

  Rejected, measured: a per-batch "no rep code among the eight" branch
  that skips the LRU selects (walk 2.08 -> 2.13: text gains 0.15, the
  binaries lose 0.5, the branch is unpredictable there); the walk with
  `chunks_exact` zips over six arrays (2.34: the compiler sank the
  position adds and spilled the table entries); the walk and the copies
  fused per batch (4.33 vs 4.15 separate: the copy state pushed the walk
  back into spilling); the rep LRU resolved in the copy pass instead
  (walk -0.32, copies +0.35: no idle slots there after all); an inline
  16-byte path for offsets 16..31 (within noise). The 4-chain table fill
  and a chain-free fill were slower than the plain one in a microbench:
  the fill is throughput-bound, not chain-bound.
- Where it stands: tANS 1.7 + walk 1.9 + copies 2.2 + literals 0.7 +
  tables 0.4 + tail 0.14 = 7.0 ns/sequence. The walk is 21 ALU ops per
  sequence (3 position, 8 fields, 3 adds, 7 for the rep LRU) and runs
  at ~3.4 of them per cycle, as does the tANS loop at 5 per symbol --
  the shifted-operand ops the entry layouts rely on look like the
  ceiling; the copy loop is ~20 instructions at 2.7 IPC, latency- and
  mispredict-bound (the floor analysis's shape). 2.5 GB/s needs 5.3
  ns/sequence: a shorter sequence format (fewer fields, or reps out of
  the per-sequence chain) more than a faster loop.
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
Profile (Instruments Time Profiler on `examples/v7_parse --loop`): 96% of
samples in `find_sequences_dfast`; per probe, the long candidate's
compare+branch took 24% and the short one's 19% (both directions hot: on
text 88% of probes hit and which table hits is a coin toss), the lazy
re-probe 15%, the literal memcpy call 4%. Counters: dickens 0.26 probes
per byte, 46% of them the lazy re-probe, which wins 7%; rep checks hit
0.1-1% of probes on text, 16% on mozilla.

Steps, each measured alone (`examples/v7_parse`: the parse by itself in
256 KB blocks, min of 5 runs, next to `compress_into_max`'s total ns/byte
and ratio; the machine was shared with other agents, so the later steps
were A/B'd as alternating binaries, min of 3):

| step | parse ns/B | total ns/B | ratio |
|---|---:|---:|---:|
| f25f4f9 | 3.24 | 4.10 | 3.2219 |
| lazy step probes the long table only (reps + short at pos + 1 were +0.2% ratio for 3x the cost) | 2.86 | 3.68 | 3.2162 |
| literal runs <= 16 copied as two 8-byte stores into reserved slack, no memcpy call | 2.79 | 3.61 | 3.2162 |
| table entries = 24-bit position + 8-bit hash tag: a miss resolves from the entry, verify + extend is one loop | 2.50 | 3.32 | 3.2162 |
| lazy win out of line (`#[cold]`): a branch, not csel/cinc on the next position | -3% | | 3.2162 |
| sequence tables reused by cost (195/815 blocks, was 0), literal table when it covers the block (156, was 61) | | | 3.2176 |
| software pipelined: pos + 1's slot and entries loaded while pos is checked; a hit's lazy step and a step-1 miss's next probe use them | 2.33 | 3.16 | 3.2176 |
| the parse writes the code bytes and extra-bit words (`EncScratch::push_codes`), `encode_block_coded` skips the codes loop; rep update as selects | 2.61* | 3.22 | 3.2176 |
| final, quiet machine, min of 5 | 2.39* | 2.93 | 3.2176 |

(* parse column includes the code emission from that step on.)

Rejected, measured: rep0-only checks (-0.9% ratio, no faster); all five
checks evaluated branchlessly under one branch (3.35 vs 2.86 ns/B: the
early-outs beat the csel chain and five always-issued loads); one branch
for the long + short tag checks with a csel'd candidate (+8% and, on the
pipelined loop, +4%); dropping the match-start + 2 insertions (-0.9%
ratio, no faster); skip strength 5 (-0.9% time, -0.12% ratio); lazy only
when the first match is under 16 (-0.26% ratio, noise); a 4-byte
pre-check at offset rc before the lazy extension (+2.5%: it makes the
lazy chain depend on rc); pipelining pos + 1 for the lazy step only (text
-10%, sao/x-ray +10-15%, a wash); code histograms accumulated in the
parse (a wash); no `o <= pos` guards on the rep checks (no change); the
`Sequence` push is 1.2% (kept: the tests and the harness read it). Table
sizes with tags: 18/17 3.1990 at -6% parse time, 17/17 3.1770, 17/16
3.1572 (-6%); 18/18 stays. 128 KB blocks: 3.1990.

Encoder-only modeling (bitstream unchanged): reuse by cost +0.04%
(29.7 KB); the raw-vs-coded 2% margin at 0 is +0.09% (3.2205), all sao's
literals going Huffman, i.e. slower decode on near-incompressible blocks
-- not taken, the decoder owner's call; the literal length limiter
(halve-and-rebuild at 11 bits) is within 1 KB of package-merge over the
815 blocks, a 12-bit cap another 0.5 KB.

Output composition (bytes, entropy of the code alphabets + raw extras):
literals 24.56 MB (27.2 M literals, 7.2 bits each), ll codes 2.92 MB +
0.11 MB extras, ml 5.73 + 1.07, offsets 7.09 + 23.61 MB of raw mantissa
(36% of the output), tables + size tables + headers ~0.8 MB. Format
changes and their estimated gains on these sequences: two length codes
per octave above 16 (one raw bit less, the split entropy coded): ll -13
KB, ml -65 KB (0.12%); zstd's ll == 0 rep semantics (rep0 is impossible
there; code rep0 +- 1): 7300 sequences, ~18 KB (0.03%); FSE-style
compressed tANS count tables (~30 bytes instead of 1 + 2n): ~60 KB
(0.09%); u16 or varint sub-stream size tables (5 x 8 x u32 = 160
bytes/block): ~90 KB (0.14%); compressed Huffman lengths instead of 128
packed bytes: ~36 KB (0.06%); 12-bit Huffman: 0.5 KB. Together ~0.45%.
The 2.0% measured in milestone 3/4 for zstd's own sequences through this
coder is not explained by these; re-measure that harness before
spending on the format.

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

**CLI**: `-9`/`--max` added to `src/bin/glyd.rs` as the highest-priority
arm of the `(mc, fast, turbo, max)` match (`compress_into_max` /
`compress_parallel_into_max`). Verified with `cmp`:
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
(205 MB, fetched over plain FTP -- it still works), NYC yellow taxi
Parquet for 2024-01 (50 MB), the first 64 MB of the linux-6.6 source
tarball, and a small OpenStreetMap PBF extract (Liechtenstein, 3.4 MB)
all downloaded on the first try. TPC-H `lineitem` skipped (no `duckdb`
on `PATH`) and `vmlinux` skipped (no local kernel build available) --
both noted by the script rather than failing it. The linux.tar entry
needed a fix after the first run: `curl -fsSL URL | xz -dc | head -c
67108864 > dest` reports a broken-pipe error from `curl` (exit 56) once
`head` stops reading at 64 MB, even though `dest` is exactly the right
64 MB -- under `set -euo pipefail` that would abort the whole script, so
this entry now runs the pipeline inside an `if` (exempt from `set -e`
regardless of its exit status) and judges success by `[ -s dest ]`
instead of the pipeline's exit code.

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
coding, and zstd is 2.3x faster to compress it too -- likely its
raw-block short-circuit is cheaper than v7's). The GitHub Archive loss is
real, if small (-0.6% ratio, and zstd -3 is faster there too: 0.955 vs
0.745 GB/s): on this file's very repetitive JSON structure zstd -3's
search finds matches v7's `dfast` (double-fast, greedy-with-one-step-lazy)
parse does not -- consistent with milestone 4b's unexplained ~2% modeling
gap on zstd's own sequences through this coder. liechtenstein.osm.pbf,
linux.tar and nasa_access.log all pass, two of them by a comfortable
margin. Net: `--max` is a solid zstd -3 substitute on Silesia-like text
and source code, roughly a wash on structured/repetitive JSON, and
already-compressed containers (Parquet, PBF) are a wash for any
general-purpose byte-level coder by construction.

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

**Cleanup**: `examples/huff_spike.rs` (the milestone-1 throwaway spike,
superseded by the real decoder in `src/huffman.rs` and `src/v7_decode.rs`)
and its `Cargo.toml` `[[example]]` entry deleted. `cargo test --release`
after deletion still shows exactly the 5 pre-existing warnings and no
others: `src/lib.rs` unused `avx2` (line ~458) and `min_match` (line
~503), `src/bin/glyd.rs` an unreachable `"-1" | "--single-core"` arm
(pre-existing -- `"-1"` already matches `"--fast"` above it, so
single-core can only be selected with the long flag; not introduced or
fixed here, out of this task's scope), and `examples/floor.rs` an
unused `mut` and a dead `file_base` field. None of the 5 came from the
spike, before or after its removal. `tests/v7_fuzz.rs` (Task 10, a
different worktree's concurrent work, not yet merged into this branch)
does not exist here yet, so `V7_FUZZ=1000000 cargo test --release --test
v7_fuzz` cannot run in this worktree; `fuzz_safety.rs`'s existing
1,000,000-mutation container-format test (`test_corruption_mutation_fuzz_1m`)
is green as part of the full suite.

## v8 decoder experiments (2026-09-19, Sapphire Rapids c7i.2xlarge dev box, M1 Max)

- Resolving repeat offsets in the copy pass instead of the sequence
  walk (the walk stores the raw offset value; the copy loop, sequential
  anyway, runs the three-select chain): M1 neutral (2,187 vs 2,190 MB/s
  on the corpus loop), x86 -5% (1,292 vs 1,360). The chain moved from a
  pass that overlaps eight streams onto the address of every match
  source load, where the x86 core cannot run ahead of it. Reverted.
- Format v8 (8 MB window) against v7 on the same x86 instance, same
  binaries side by side: max-level decode 1,390/1,370 -> 1,355/1,337
  MB/s (corpus loop), 1,197 -> 1,183 (v7_bench); zstd -3 1,157. The
  published c7i run of 56cc09d measured every codec 5-25% below its
  previous run on that instance (noisy neighbour); re-run on a fresh
  instance for the release.
