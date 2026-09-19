# Format v7: beat zstd -3 on ratio at 2x its decode speed

Date: 2026-09-18. Status: implemented (`--max` level); measured results in CHANGELOG-BENCH.md.

## Goal

A new compression level (`--max`, format v7) that replaces zstd at its
default level: ratio >= zstd -3 (3.20 on Silesia) with compression speed
>= zstd -3 (0.34 GB/s) and decode >= 2x zstd -3 (>= 3.0 GB/s), measured
in the same run on one core. Existing levels (fast, default, turbo, all
format v6) are untouched.

Why this target: at hyperscale the cost that matters is bytes stored and
moved, which scales with ratio; compression speed is a hard floor because
every object is compressed once on the provider's CPUs; decode only needs
to be "not the bottleneck". zstd -3 is what gets deployed. A codec that
matches its ratio, its compression speed, and decodes 2x faster is a
switch a storage provider can flip.

## Measured basis

- Ratio above ~2.5 is not reachable by parsing alone (LZAV-hi: 2.80 at
  0.10 GB/s). Entropy coding is required.
- Order-0 Huffman on the v6 streams gives only 2.18 -> 2.69. zstd -1 gets
  2.89 from the same coder because of its *modeling*: offsets as log2
  bucket codes plus raw bits, lengths as codes plus raw bits, repeat
  offsets, a 512 KB - 2 MB window. The ratio comes from modeling, not the
  coder.
- Probe (a throwaway 8-stream Huffman decoder, since removed): 8 interleaved LSB-first
  Huffman streams with a packed 2^11 table and a branchless 64-bit refill
  decode Silesia literals at 0.47 ns/symbol (2.0 GB/s of literals) on the
  M1 Max. Literals are 27% of output. Decode budget at ratio 3.2:
  literals ~25 ms + sequences ~18 ms + copies ~26 ms ~= 70 ms for 202 MB,
  ~2.9 GB/s, with partial overlap between passes. Reserve: double-symbol
  Huffman tables (~30% on the literal part).

## Section 1: block layout

Header: version 7, flags, stream sizes, dictionary ID (u32, 0 = none),
checksum, uncompressed length. Block size 256 KB. Window chained across
blocks; `FLAG_CHAIN_RESET` marks independent blocks as in v6.

Five streams, each either entropy-coded or stored raw (one flag bit per
stream, decided per block by coded size vs raw minus 2%):

1. **Literals**: Huffman, 8 interleaved bitstreams, max code length 11.
   Table sent as 256 packed 4-bit lengths, or a "reuse previous block's
   table" flag.
2. **Literal lengths**: tANS over codes; 0-15 direct, 16-35 log2 buckets
   with extra bits in stream 5.
3. **Match lengths**: same scheme; format minimum match 3, the parse
   decides what it emits.
4. **Offsets**: codes 0-2 are repeat offsets (rep0/rep1/rep2 with zstd
   semantics), 3+ are log2 buckets with extra bits. Offsets up to 21 bits
   (2 MB window) in v7 blocks, 23 bits (8 MB) in v8 (see the v8 section
   at the end).
5. **Extra bits**: one raw LSB-first bitstream, 8-way interleaved like
   the others.

Sequence streams 2-4 are 8-way interleaved tANS, one state per
sub-stream, tables sent as normalized counts (<= 64 symbols, 4 KB decode
tables). Every entropy stream carries 8 trailing padding bytes so the
decoder's wide refill never reads outside the block.

Dictionary: dictionary bytes preload the window (offsets may reach into
it) and seed the finder; the header carries the dictionary ID and the
decoder requires the same dictionary. Pre-trained entropy tables are a
later addition through the existing "reuse table" flag, without a format
change.

## Section 2: decoder (`src/v7_decode.rs`, NEON + scalar; AVX2 later)

Three passes per block over L1/L2-resident scratch, no allocation beyond
the output and a once-per-thread scratch:

1. **Entropy pass**: decode streams 2-5 into flat arrays `lit_len`,
   `match_len`, `offset` (u32 each) for every sequence in the block. 8
   tANS states per stream, 4 symbols per state per refill. Repeat-offset
   resolution happens here, sequentially (a three-register chain).
2. **Literal pass**: Huffman-decode all literals of the block into a
   literal buffer (<= 256 KB), 8 streams.
3. **Copy pass**: the existing NEON copy loop (`copy_run`), reading the
   arrays and the literal buffer; bounds checked once per block from the
   totals, offset validated per token.

Rationale: the copy loop is the part already measured fast (2.2
ns/token); passes 1-2 are pure streaming with no dependency on it, so
they run at the spike's ~0.5 ns/symbol. zstd interleaves decoding a
sequence with executing it, so its copies wait on its entropy decoder.

Errors: stream lengths checked against the header before decoding;
invalid Huffman lengths (not a prefix code) or tANS counts (not summing
to the table size) reject the block; the 1M-mutation fuzz is extended to
v7 with tables and streams corrupted individually.

Scratch: two 256 KB buffers plus the three arrays (<= 40K sequences at
256 KB blocks; 480 KB worst case), thread-local, allocated once per
thread. The decoder needs ~1 MB of scratch that v6 did not; that is the
cost of the three-pass design.

## Section 3: compressor (`src/v7_encode.rs`, level `--max`)

**Parse**: zstd -3's "double fast" finder: two hash tables (5-byte and
8-byte hashes), one candidate each, the long one tried first; 2 MB
window; minimum match 4; back-match; skip acceleration on misses. In
addition, the three repeat offsets are tried first at every position (a
4-byte compare each). No lazy matching at this level (that is what zstd
-5 and up do; ~40% compression speed for ~3% ratio). Table memory ~1 MB,
thread-local.

**Emit**: the existing cursors-in-registers path, into the five v7
streams: three u8 code arrays, the extra-bits writer, the literal buffer.

**Encode, per block**: histogram each stream; build tables (Huffman
lengths from `huffman.rs`; tANS normalized counts); store a stream raw
when coding would not beat raw by 2%. Encoding is the interleaved
reverse of the decoder: 8 sub-streams each written backwards (tANS
encodes in reverse), concatenated with a small size table. Table reuse:
when a stream's histogram is close to the previous block's, set the
reuse flag and send no table.

**Cost budget** for >= 0.34 GB/s: parse ~2 ns/byte (the measured ceiling
for this class), histograms and tables ~0.1 ns/byte, entropy encode ~0.5
ns/symbol on ~0.4 symbols/byte ~= 0.2 ns/byte; ~2.3 ns/byte ~= 0.42 GB/s.
The margin is thin and the parse is where it can slip.

**Dictionary**: `compress_with_dict(dict, input)` preloads the window and
seeds the tables; header carries the dictionary ID.

## Section 4: corpus, gates, testing, milestones

**Corpus** (`scripts/download_corpus.sh`, all public, ~1.5 GB): Silesia
and enwik8 (present), plus TPC-H `lineitem` SF-1 as Parquet and CSV, one
hour of GitHub Archive JSON, NASA HTTP server logs, one month of NYC taxi
CSV, a Linux 6.x source tarball (uncompressed) and a `vmlinux`, an
OpenStreetMap PBF extract (small region). Reported as Silesia total plus
a per-file table for the rest, same run against zstd -1, zstd -3 and
liblz4, one pinned core.

**Gates** (all same-run):
- G1 Correctness: round-trip on the whole corpus, x86 and ARM
  bit-identical output, 1M-mutation fuzz on v7.
- G2 Ratio: Silesia >= 3.20, and >= zstd -3 on every non-Silesia file
  individually.
- G3 Decode: >= 3.0 GB/s Silesia, >= 2x zstd -3 per file.
- G4 Compression: >= zstd -3 (0.34 GB/s) Silesia.
- G5 Floors: existing levels' numbers unchanged; multi-thread scaling as
  now; decoder scratch <= 1.5 MB, no per-call allocation.

**Testing**: TDD per component (bit reader/writer, tANS table build and
encode/decode, 8-stream Huffman, sequence codec, repeat offsets,
dictionary), each with a small exact test; round-trip tests on generated
shapes as for the fast and turbo levels; fuzz extended.

**Milestones**, each with a measured number recorded in
`CHANGELOG-BENCH.md` before the next starts:
1. Entropy coders (Huffman-8, tANS-8): standalone, correct, >= 0.5
   ns/symbol.
2. v7 streams and decoder passes 1-3 fed by the current parse:
   correctness and the decode number.
3. Modeling (log2 codes, repeat offsets, extra bits): the ratio number.
4. Double-fast parse, 2 MB window: ratio and compression numbers. Decide
   here whether lazy matching is needed.
5. Dictionary, C ABI, CLI `--max`, corpus and field survey, docs.

Stop rules: if milestone 2 decodes under 2.5 GB/s, add double-symbol
Huffman tables before continuing. If milestone 4 lands under ratio 3.1,
lazy matching goes in and G4 is renegotiated explicitly, not dropped.

## Out of scope

AVX2 port of the v7 decoder (scalar fallback on x86 until then),
pre-trained dictionary entropy tables, levels above `--max`, any change
to format v6.

## Format v8 (v0.3.0)

The same coder and decoder with four changes, all measured on the ultra
level's parse of Silesia (`examples/coder_overhead.rs`); the block header
says `VERSION_V8`, and v7 blocks are still decoded by the same code paths
through a version flag.

1. **8 MB window.** `MAX_OFFSET_BITS` 21 -> 23, `OFF_SYMBOLS` 24 -> 26
   (codes 24 and 25 are the two new log2 buckets). A sequence's extra bits
   are now at most 18 + 17 + 22 = 57: a literal run of a whole block, a
   match below that, an offset below 2^23. The decoder's one-load walk
   holds 57 bits after the sub-byte shift, exactly enough; the encoder
   writes such a sequence in two puts (`put_wide`). Ultra 3.801 -> 3.895,
   max 3.218 -> 3.229. The ultra finder's tree is sized to the input per
   call (64 MB for an 8 MB window, 2 MB for a 256 KB chunk).
2. **Section layout.** Seven 24-bit sub-stream sizes (the eighth is what
   remains), the eight streams back to back, one `PAD` of 8 zero bytes
   at the end of the section instead of one per stream (`bits::Stream`):
   96 bytes per section -> 29. A stream's fast loop may read into the
   next stream's bytes (still inside the section, which is what the load
   bound needs); its own length is the overrun budget of the tail reader.
3. **Length codes.** Direct codes for literal runs to 15 and matches to
   34, then buckets of 1-5 extra bits before the log2 buckets
   (`LL_BITS`/`LL_BASE`, `ML_BITS`/`ML_BASE`: 38 and 54 codes); v7's
   direct-below-16-then-log2 codes are kept for decoding v7 blocks. Worth
   1% on nci and a wash on text: the extra bits it saves, the code
   entropy mostly paid.
4. **Tables.** Literal code lengths as nibbles with runs of unused symbols
   as (0, run - 1): 128 bytes -> ~88. tANS counts as a 4-bit width and the
   bits below the top one: 11 bits each -> 4-7 for the small counts that
   dominate; three tables 167 bytes -> ~93.

Per-block overhead over the order-0 estimate of the parse: 950 bytes
(1.45%) -> 421 (0.85%; nci, at 13 KB per block, 3.3%). Ultra 3.925, max
3.254 on Silesia; decode unchanged. What remains per block: the block
header (32), sub-header (26), five size tables (105), five paddings (40),
the tables when written (~180).

## Format v9: compact framing for small blocks (v0.4.0)

The v8 coding under framing sized for small objects, written for blocks
of at most `COMPACT_MAX` = 32 KB (v8 above that; v7/v8 blocks decode as
before):

- Header (19 bytes): magic, version, flags as one byte, the four lengths
  as u16, checksum. Sub-header (16): coded, reuse, dict id, five u16
  section sizes. One 8-byte padding at the end of the payload; sections
  carry none of their own (a stream's loads may run into the next
  section, the last into the padding).
- Sections: a stream count, then for eight streams seven varint sizes
  and the streams; a section of at most `SINGLE_MAX_SYMBOLS` = 1024
  symbols is one stream (`bits::Framing::Single`): no size table, no
  seven byte-aligned tails, decoded on the clamped per-symbol path
  (slower per symbol, immaterial at that size).

A 4 KB JSON event with a prepared dictionary: 207 bytes of framing
(header 32, sub-header 26, five sections' size tables and paddings) went
to 43; the object 870 -> 676 bytes. Small objects, `examples/small_objects.rs`
(gharchive events, 2,000 per size, dictionaries trained on 2,000 others):

| Object | zstd -3 | zstd -3 + dict | Glyd --max | + Dict | Glyd --ultra + Dict |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 KB | 2.38 | 5.00 | 1.99 | 4.11 | 4.11 |
| 4 KB | 3.56 | 6.52 | 3.25 | 6.06 | 6.22 |
| 16 KB | 4.89 | 7.74 | 4.76 | 7.52 | 7.98 |
| 64 KB | 6.38 | 8.50 | 6.27 | 8.13 | 8.91 |

Before v0.4.0 the same objects with a window-only dictionary compressed
to 2.15 (1 KB) and 3.87 (4 KB). What remains at 1 KB is the framing
still (43 of 249 bytes) and the dictionary content's quality (zstd's
trained content used as Glyd's window: 4.83 against 4.7 at 4 KB before
the framing change).
