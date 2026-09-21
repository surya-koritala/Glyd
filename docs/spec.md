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

## 3. The store

Not a stream but a directory (`src/store.rs`): `index` (one text line
per object: id, base id or `-`, depth, raw length, stored length, pack
`id:position` or `-`, name; `D\t<id>` deletes; a later line for an id
replaces an earlier one), `table` (an open-addressing hash table of
fingerprints: `u64` capacity, `u64` count, then 12-byte slots of
`u64` fingerprint and `u32` object id, `0xFFFFFFFF` empty) and
`objects/<id>` (a block stream, a `GLYDBASE` envelope whose base is the
object named in the index, or a `GLYDPACK`).
