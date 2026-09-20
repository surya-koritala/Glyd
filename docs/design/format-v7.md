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
before). Everything a small block does not need is gone from its
framing:

- Block header: one marker byte (`COMPACT_MARKER`, 'G'; the v6-v8 magic
  starts with 'D') in place of the 4-byte magic and 2-byte version, the
  flags as one byte, the uncompressed and payload lengths as varints,
  for a coded block the sequence and literal counts as varints, then the
  4-byte checksum: 12-14 bytes for a block under 16 KB (v8: 32).
- Sub-header: one byte of coded/reuse bits with a flag for a dictionary
  id, the id only when there is one, and the first four section sizes as
  varints (the fifth section runs to the end of the payload): 5-10 bytes
  (v8: 26).
- Sections: eight streams behind seven varint sizes, or, for a section of
  at most `SINGLE_MAX_SYMBOLS` = 1024 symbols, one stream with no size
  table at all (`bits::Framing::Single`). Which of the two a section is
  follows from its symbol count in the block header, so no byte says so.
- No padding on disk. The decoders' loads run past a stream's last byte,
  so the decoder copies a compact block's payload (at most 32 KB) into a
  padded scratch buffer first.
- A literal section under a reused table is coded whenever its bits say
  so (the 64-literal floor applies to fresh tables only).

The single-stream sections decode on their own paths: the three code
streams together (three independent chains, `tans::decode_single3`),
and the walk, literal and code loops take one-load batches inside the
stream's margin (retaken as the stream is read) before the clamped tail.

A 4 KB JSON event with a prepared dictionary: 207 bytes of framing in v8
(header 32, sub-header 26, five sections' size tables and paddings), 48
in the first v9 layout, 21 now; the object 870 -> 676 -> 643 bytes.

## Dictionaries (`src/dict.rs`, v0.4.0)

A `Dict` is content plus entropy tables. The content is the window an
object's first block sees: the max level parses the object in place and
probes the dictionary's own seeded finder tables beside the object's
(`v7_encode::DictTables`); the decoder copies matches from the content
through an extDict path (`copy_seq_from_ext`, `copy_seq_wild_ext`; the
content is kept with 64 zero bytes after it for the wild copies). The
tables (a literal table and the three tANS tables) are what a block that
reuses them writes none of its own: ~90 + 3 x ~25 bytes, most of a
small object's output. The decoder keeps them built (`DecTables`
borrowing the dictionary's), so an object costs no table work; the
encoder keeps the built encoder tables and Huffman codes with them
(`Tables`, shared by `Arc`).

`Dict::train` selects the content by cover, as zstd's fastcover: every
8-byte string in the samples' concatenation is counted in a 2^20-entry
hashed table; the concatenation is cut into as many epochs as the budget
has 1 KB segments; each epoch contributes its best segment, scoring each
distinct string once, and the segment's strings are zeroed so later
picks cover new ones; the segments go in from the end, the earliest
picks (made while every count was whole) nearest the object. The tables
come from a max-level parse of the samples against that content, every
symbol given at least one count. Under Glyd's own compressor this
content matches zstd's trained content on JSON events (6.37 vs 6.32 at
4 KB) and is at or above it on access-log and source-tree objects.

The ultra level with a dictionary (`compress_with_dict_ultra`) prices
its parse from the dictionary's tables (`v7_ultra::Stats::of_tables`,
weight 64 per tANS count) rather than from the object's own few
symbols: 1 KB JSON events 4.77 -> 5.23.

Serialized (`Dict::to_bytes`, "GLYDDICT" v1): the content, the packed
literal lengths, the three count tables. The id is the checksum of that
form; a block names it in its sub-header and the decoder refuses a block
whose id is not the dictionary's.

## Record mode (v0.4.0): typed columns before the level

Byte-level matching finds a repeat and points back at it; on a log or a
table dump most of the redundancy sits in the same field of every
record, thousands of bytes apart and interleaved with the other fields.
Record mode (`src/record.rs`) reorders the text into one stream per
field before the ordinary level compresses it, and rebuilds the bytes
exactly afterwards:

- Shapes recognised (`detect`, on the first megabyte): lines split by
  one delimiter (space, tab or comma) into a constant field count for
  at least 90% of lines (among delimiters consistent on 98%, the one
  splitting finest, so a timestamp's space does not beat a CSV's
  commas); MySQL dumps (`INSERT ... VALUES (...),(...);`), also when a
  unit starts inside a tuple list; JSON objects one per line (90% of
  lines), where a column is a key path (`actor.id`, `commits.[].sha`).
- Column types, chosen per column from its values: integers as zigzag
  varint deltas from the previous row (only canonical decimals, so
  `i64` formatting reproduces them); date-times under a known fixed
  pattern (Common Log Format, ISO 8601 with or without a fraction,
  `YYYY-mm-DD hh:mm:ss[.ffffff]`) as second deltas with the fraction in
  its own stream, the pattern chosen from the first eight values and
  verified to reproduce each, up to a tenth of the values escaped (a
  header line, a malformed field); columns of at most 256 distinct
  values as a dictionary and one byte per value; columns with at most
  one distinct value in three as a dictionary, a 64-deep recency list
  (a byte per value: the position of the value in the list of the last
  64 distinct ones, or an escape) and the escaped ids; columns of
  decimals with more distinct values than that as deltas of the value
  scaled to the column's most places, each value keeping its own places
  ("43.1", "43.10", "43"), other values escaped; everything else as
  newline-separated text.
- Lines that do not fit the shape go to a raw stream in order; for
  dumps the statement text is a frame with a zero byte where each
  record tuple was; for JSON lines the frame is the structure, the keys
  and the text values (text stays where its strings match across
  fields and records; shredding it too cost 13% on GitHub events), with
  a zero where each typed value was and a stream naming each hole's
  column.
- The input is cut into units of 32 MB at line ends; each unit decides
  for itself (a 4 MB trial compressed both ways must favour the
  transform by 5%) and is transformed, compressed and rebuilt on its
  own, so both directions run one unit per core. Input whose first
  4 MB the transform does not pay on takes the plain parallel path
  (units up to 128 MB, the matcher's window).

The envelope (`GLYDRECS`, the units' lengths and kinds, then their
streams) is read by every decoder; the container underneath is
unchanged. Measured (50 MB slices, Glyd `--max` in record mode against
zstd -19 on the raw text): NASA access log 1.98 vs 3.19 MB, ClarkNet
2.51 vs 3.51, enwiki page_props 7.47 vs 8.25, simplewiki pagelinks
2.44 vs 3.60; Wikipedia pageviews 12.9 vs 10.8 (a column of unique
titles gains nothing from the split). JSON events are not record
shaped; their redundancy is inside each record and across the whole
file, where a larger window (128 MB: 23% with zstd `--long`) is the
lever, not columns (shredding measured 6% worse). The transform runs at
~200 MB/s per core and the rebuild at 500-900 MB/s per core.

Telemetry (the extended set of `scripts/download_ext_corpus.sh`, 128 MB
slices, 10 cores, `benchmarks/suite/m1-max-v0.4.0/bench_suite_ext2.*`):
Alibaba cluster machine usage as CSV `--max -r` 12.6x against zstd -3's
4.5x and zstd -19's 6.9x, as JSON lines 54.6x against 15.7x and 28.7x;
NOAA daily weather as CSV 19.6x against 7.0x and 12.0x, as JSON lines
47.0x against 18.7x and 31.6x; NYC taxi trips exported to CSV 8.9x
against 5.6x and 8.4x (its two timestamp columns with fractions were
44% of the image as text); `--ultra -r` 13.7x, 58.9x, 23.1x, 58.0x,
9.4x. A Common Crawl index (JSON after a key and a timestamp, a hash
per line) is not record-shaped and stays plain, 2% under zstd -19: a
hash does not compress, and the ablation in
experiments/structure/json_ablation.py puts 20% of a compressed GitHub
event in its SHAs, 10% in random ids and 30% in free text.

## Long-distance matching (v0.4.0): repeats up to 128 MB back

The block-local finders keep 8 MB of history and, being hash tables
without chains, forget most of it long before that: on GitHub Archive
JSON a repeat 1-8 MB back is found by neither, and the events repeat
whole records across the hour. `zstd -19 --long=27` measured 23% on
such an hour. `src/ldm.rs` adds one pass before the parse of a unit:

- Anchors are content-defined: a position whose 4-byte hash has its
  4 high bits zero (one in 16), so a repeat anchors at the same places
  as its original. The anchors of 16 positions come from one NEON or
  AVX2 step (four multiplies on the shifted words, a mask), their
  positions are extracted without a branch per anchor, then each is
  hashed over its 32 following bytes.
- The table has 2^22 entries of one u32: 27 bits of the position under
  5 bits of hash check. A candidate whose check differs is dropped
  without reading its bytes (97% of candidates on mozilla); the slot is
  prefetched a chunk of 1 KB ahead. Runs of one byte never anchor.
- A candidate's bytes are compared 8 at a time forward and back; a
  match of 32 bytes or more is kept, non-overlapping and in position
  order. The parse (`find_sequences_dfast_far`, `find_sequences_ultra_far`)
  walks the list with a cursor and takes the far match wherever it
  beats the local one, capped at 130 bytes per sequence for a far
  offset (the walk entry's 57 bits hold 18 + 3 + 26 extra bits); the
  remainder follows as a repeat-offset sequence.
- Format v9 offsets are 27 bits (30 offset codes); v8 blocks keep 26.
  A stored table with fewer symbols than its version allows decodes.
- The pass costs 1.2-2 GB/s per core against a max-level parse of
  0.25-0.8 GB/s, so the max level gates it: after 4 MB it stops when
  under 1/64 of the input is inside a repeat, after 16 MB when repeats
  at least 1 MB back cover under 1/32; both checkpoints move to half
  the input when that is sooner. Off for media, Parquet, most SQL dumps
  and mozilla-like binaries (92-96% of the old speed kept); on for text,
  logs and JSON (63-85% of the old speed, 3-16% fewer bytes). The ultra
  level runs it whole.

Measured (one core, M1 Max, first 128 MB): GitHub Archive JSON `--max`
11.63 -> 9.73 MB at 577 MB/s (zstd -3 12.90 MB at 895 MB/s, `zstd -3
--long=27` 10.20 MB at 433); `--ultra` 7.78 MB (zstd -19 8.96, `zstd
-19 --long=27` 7.80). NASA access log `--max` 13.22 -> 11.92 MB at 435
MB/s (`zstd -3 --long=27` 13.63 MB at 383). Silesia `--max` 3.259 ->
3.302 at 247 MB/s (290 before). Refuted on the way: sparser anchors
(one in 32 halves the pass but costs 1.4% on JSON and 2.5% on logs
because 32-64 byte repeats are missed); a 16-byte hash (more false
candidates than the check bits save); prefetching the candidate's
bytes (no gain on M1, dropped).
