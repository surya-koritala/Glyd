# The Glyd formats: a map for decoder writers

Every Glyd output is one of the streams below, told apart by its first
bytes. This page fixes the framing (what a reader must parse to find
the pieces) and points to the documents that fix the coding of each
piece. The reference decoder (`src/`) is normative where the documents
are silent; every release decodes every earlier format
(`tests/format_compat.rs`, `tests/data/`).

Integers are little-endian. A *varint* is 7 bits a byte, low bits
first, the top bit of a byte marking that another follows (`u32`
unless said otherwise; the 64-bit form is the same with more bytes).

## 1. Block streams (the levels)

A plain stream is a sequence of blocks back to back; each block decodes
to at most 256 KB of output. Blocks come in two framings:

**The struct header** (v6, v7, v8): 32 bytes, `u32` magic `0x53494D44`
("SIMD"), then `u16` version, `u16` flags, and the section lengths and
`u32` checksum of the block (`src/format.rs`, `BlockHeader`).

**The compact header** (v9, written for every block since v0.4.0): the
marker byte `0x47`, one flags byte, the uncompressed length and the
payload length as varints, then — unless the flag `RAW` is set — the
sequence count and the literal count as varints, then the `u32`
checksum (CRC32 of the block's uncompressed bytes). The payload follows.

Flags (both framings): `RAW` (the payload is the bytes themselves),
`CHAIN_RESET` (the block starts a new history: a decoder may begin
here, which is how units decode in parallel), `DENSE` / `TURBO` (the
v6 token variants). A stream decodes unit by unit: a unit is a run of
blocks from a `CHAIN_RESET` to the next.

**Payloads.** v6 (the default, `--fast`, `--turbo` levels): LZ tokens
with 32-byte copies, no entropy coding
([docs/design](design/), `src/format.rs`, `src/fallback.rs`). v7/v8/v9
(`--max`, `--ultra`): literals under an 8-way interleaved Huffman code,
sequences (literal length, match length, offset with three repeat
slots) under tANS with packed count tables, extra bits in a separate
section; matches reach 8 MB back within a unit, and far matches (up to
128 MB, from the long-distance matcher) are coded as offsets of up to
27 bits ([design/format-v7.md](design/format-v7.md), `src/v7_format.rs`,
`src/v7_decode.rs`).

## 2. Envelopes (the modes)

An envelope wraps block streams with what the mode needs. All begin
with an 8-byte magic.

| Magic | Mode | Layout after the magic |
| :--- | :--- | :--- |
| `GLYDRECS` | record mode (`-r`) | varint `n_units`; per unit: varint raw length, varint stream length, one byte (1: the stream is a record image, 0: the text itself); then the units' block streams. A record image (`GLYDREC1` inside the decoded stream: mode, delimiter, field count, line count, column types, stream lengths, the column streams) rebuilds the text ([design/format-v7.md](design/format-v7.md#record-mode-v040-typed-columns-before-the-level), `src/record.rs`). |
| `GLYDBASE` | base mode (`--base`) | `u64` base id (length `<< 32 \| CRC32` of the base); varint `n_units`; per unit: varint64 region start in the base, region length, unit length, stream length; then the streams. Each unit's blocks decode with the base's region as history before position 0 ([design/format-v7.md](design/format-v7.md#base-mode-v050-a-version-compressed-against-the-last-one)). |
| `GLYDCOLD` | the cold level (`--cold`) | varint `n_units`; per unit: varint length, varint stream length, `u32` CRC32; then the units' arithmetic-coded streams, each from an empty model ([design/format-v7.md](design/format-v7.md#the-cold-level-context-mixing), `src/cm.rs`). |
| `GLYDPACK` | packs (`--pack`) | varint `n_objects`, varint `index_len`, the index (the objects' lengths as zigzag deltas, itself a `--max` block stream), then one stream of the objects' concatenation (record mode or plain). Object *i* is bytes `[Σ len<i, +len_i)` of the decoded concatenation. |
| `GLYDDICT` | a prepared dictionary (`Dict`) | `u16` version, `u32` content length, the content, then the literal Huffman lengths and three tANS tables; blocks compressed with it carry its id (`src/dict.rs`). |
| `GLYDSHP1` | a shape dictionary (`--shape`) | the kind, the frames (text with a 0 per hole and a column per hole), the columns (type, parameter, seeds), then a `GLYDDICT`; an object's image is a varint row count and flags, per row its frame or the raw line, then every column's values (`src/shape.rs`). |

A decoder that meets a magic it does not know should stop: the
envelopes carry no version byte because each magic *is* the version
(a changed layout gets a new magic).

## 2b. Containers opened (`GLYDDEF2`)

`"GLYDDEF2"`, original length and packed recipe length (varints), the
recipe compressed at the max level (a plain block stream), then the
inner stream of the plain text in any format above. The plain text is
the content of every opened stream, in order, then the side data:
every byte of the object no stream claims (headers, directories,
stored entries, a tar's files), the corrections and the JPEG
streams (2c), in the order the recipe takes them. The recipe
(`src/deflate.rs`) is the segment count and the content's length
(varints), then the segments, each a tag and its fields: 0 kept (varint
length; the bytes from the side); 1 deflate (varint length of the
corrections, from the side; varint length of the text, from the
content); 2 a PNG image stream (2 zlib header bytes, corrections' and
text's lengths as for 1, 4 Adler-32 bytes, then a varint chunk count
and per chunk a varint length and 4 CRC bytes, the recreated zlib
stream being cut into IDAT chunks so); 3 a JPEG stored inside (varint
length of its JPEG stream, from the side); 4 a JPEG under a deflate
stream (varint lengths of the corrections and of the JPEG stream,
both from the side); 5 a container under a deflate stream (varint
length of the corrections, from the side; then the inner container:
its segment count, the varint length of its segments and the
segments, the varint lengths of its content and of its side, taken
from ours); 6 a container stored inside (the inner container,
likewise); 7 (v0.13.3) deflate in chunks: a varint chunk count, per
chunk the varint length of its plain text, the varint length of its
corrections (from the side) and one byte, the bit within a byte at
which its blocks start, then the varint length of the text (from the
content) — the first chunk's corrections carry preflate's parameters,
each chunk's predictor first learns the 32 KB of plain text before it,
and the re-created pieces join by or-ing the byte two share; 8 a
container under a deflate stream in chunks (the chunks as for 7, then
the inner container as for 5); 9 (v0.13.4) deflate opened by Glyd's own
reconstruction (`src/reflate/`): the varint length of its recipe, from
the side, and of its text, from the content — the recipe is a byte of
parameters (the zlib level emulated, Z_FILTERED, Z_FIXED), a varint
chunk count, and per chunk its varint plain length, varint block
count, one byte for the bit within a byte its first block starts at,
and its corrections (varint length, then the arithmetic-coded
decisions of `src/reflate/zlib.rs`); 10 a container under such a
stream (the recipe, then the inner container as for 5); 11 a PNG image
stream opened so (the fields of 2, the recipe in place of the
corrections); 12 a JPEG under such a stream (the fields of 4, the
recipe in place of the corrections). An envelope written by v0.13.4 or
later is `"GLYDDEF3"`; `"GLYDDEF2"` (v0.13.1 to v0.13.3) has the same
recipe with tags 1 to 8 only, and a base such an envelope was made
against is opened the way those versions opened it (preflate, carried
in `third_party/` for that). Containers nest four deep. Recognised containers: gzip
(members back to back), zip (entries from the central directory, or
walked when it does not parse; method 8 opened, stored entries opened
in their own way), tar (ustar entries, regular files opened in their
own way), zlib, PNG, and PDF (every `stream` whose data is a zlib
stream ending, Adler-32 and white space after it, at its
`endstream`). A part of a stream (a unit, a record unit) is never an
envelope.

Earlier envelopes decode: `"GLYDDEFL"` (v0.13.0: the same fields
except that kept bytes and stored JPEGs' Lepton streams sat in the
recipe, each with its varint length, and corrections after their
varint length in the recipe; the plain text held only what the streams
held) and `"GLYDGZIP"` (v0.12.0: original length, recipe length, the
recipe as it is: a member count, per member its header, corrections
(each a varint length then the bytes), the varint length of its plain
text and its 8-byte trailer, then a varint-length tail). v0.12.0 and
v0.13.0 could write such an envelope where a block of a multi-unit
stream was due; a decoder reads the stream around it, the envelope's
inner blocks being those that hold the plain text its recipe takes.

## 2c. JPEG recoded (`GLYDJPEG`)

`"GLYDJPEG"`, original length (varint), then the JPEG stream, which
decodes to the identical JPEG. Since v0.14.0 the stream is Glyd's own
(`src/jpg/`): `"GJPG"`; one byte, 0 when the kept bytes follow as they
are and 1 when compressed at the max level (a plain block stream); the
kept bytes' varint length and the bytes — the JPEG with every scan's
entropy-coded data cut out, so the markers, tables and frame header
come back from them; the scans (varint count), each its odd pad bits
(varint count; per pad a varint marker index and one byte of bits);
the model's streams (varint count, then a varint length each): a
prefix of the first sixteenth of the block rows, then four stripes of
the rest (eight from 10 MB up; fewer when the picture is under 256
block rows; a reader takes any count), each
stripe's contexts starting as the prefix left them, each stream a
range coder (`src/jpg/coder.rs`) over the coefficients of every
component's blocks in its rows (`src/jpg/model.rs`). v0.13.0 to
v0.13.4 wrote Lepton's stream (`lepton_jpeg`, the format of Dropbox's
Lepton; the `jpeg` feature keeps reading it). Baseline JPEGs
(SOF0/SOF1, 8-bit, Huffman) are recoded; a progressive,
arithmetic-coded or 12-bit one stays as it is.

## 3. The store

Not a stream but a directory (`glyd-store/src/lib.rs`): `index` (one text line
per object: id, base id or `-`, depth, raw length, stored length, pack
`id:position` or `-`, name; `D\t<id>` deletes; a later line for an id
replaces an earlier one), `table` (an open-addressing hash table of
fingerprints: `u64` capacity, `u64` count, then 12-byte slots of
`u64` fingerprint and `u32` object id, `0xFFFFFFFF` empty) and
`objects/<id>` (a block stream, a `GLYDBASE` envelope whose base is the
object named in the index, or a `GLYDPACK`). Beside each object sits
`objects/<id>.index`: its index lines (a pack's sidecar carries its
members' lines and its own; a deletion adds a `D` line), so the
directory is rebuilt from the objects alone — the index by
concatenating the sidecars in id order, the table by reading every
object back.
