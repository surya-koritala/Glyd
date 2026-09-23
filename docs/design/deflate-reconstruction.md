# Deflate reconstruction, Glyd's own

Opening a container means taking every deflate stream inside it apart
into its plain text and whatever it takes to write the identical stream
back. Since v0.12.0 that has been done by preflate-rs (carried in
`third_party/` since v0.13.3). This is the design for doing it in
`src/reflate.rs`, with nothing outside this repository, so the codec has
no dependency, and so the work can be made faster and wider than a
library built for other purposes.

## What a deflate stream is, for this purpose

A sequence of blocks; each block is stored, fixed-Huffman or
dynamic-Huffman; a Huffman block is a list of tokens: literal bytes and
(length, distance) references into the last 32 KB. The plain text is
the tokens applied. The stream is the tokens Huffman-coded, and the
choice of tokens is the encoder's: which match it found, how long, at
which distance, when it preferred a literal, where it ended a block, and
which code lengths it built. Reconstruction is: run the same decisions
over the plain text, and record only where the stream differs.

## Parts

1. **Parser** (`inflate`): bit reader; fixed and dynamic Huffman
   tables; blocks to tokens; the plain text; the bit position where each
   block starts and the code lengths of each dynamic block. Any valid
   stream; a stream that does not parse is left closed. ~600 lines.
2. **Emulator**: zlib's `deflate_fast` (levels 1–3) and `deflate_slow`
   (4–9) over the plain text with the level's `good`, `lazy`, `nice`,
   `chain` and the window, its hash (5-bit shift, 15 bits), its insert
   policy, its block flush rule (the token buffer of 16 K), its match
   limits (258, `MAX_DIST`, the lookahead rule at the end). GNU gzip and
   Info-ZIP are the same family with small differences (parameter
   tables, a slightly different end-of-input rule), taken as variants.
   Seedable: a state can start at any position with the 32 KB before it,
   so chunks run on every core from the first day.
3. **Corrections**: for each token, the emulator's prediction against
   the stream's token. Same: one bit. Different: what differs — literal
   for match or the reverse, a shorter or longer length, a distance that
   is the n-th candidate on the hash chain instead of the first — coded
   with the adaptive binary arithmetic coder already in `src/cm.rs`.
   Block boundaries and the dynamic code lengths are predicted by
   zlib's own rules (`_tr_flush_block`, `build_tree`, `gen_bitlen`)
   and corrected the same way; until the tree predictor is written, the
   code lengths are stored as they are (about 0.5% of the stream).
4. **Detection**: which emulator and level made the stream. The first
   64 KB of tokens are predicted under every candidate; the one with
   the fewest corrections is taken; a stream under which none predicts
   at least 90% of tokens is left closed (that is the 25% rule today).
5. **Recreate**: the emulator again, corrections applied, tokens
   Huffman-coded with the block's code lengths from the bit offset the
   block had. Bit-exact or the open is refused at write time — the
   check every stream gets before its envelope is written.

## Order of work, each step tested against real streams

Streams for the tests come from the tools on the machine and in CI
(`gzip` at every level, Python's `zlib` at every level and strategy,
`zip`, PNG through Python, PDF streams from the corpus), never from a
library in the build; every step's test is a bit-exact round trip and,
where preflate still opens the same stream, a comparison of the
corrections' size and the speed.

1. Parser with tokens, block starts and code lengths; round trip
   through a writer that re-emits the same tokens and trees (no
   prediction yet: this is the "store everything" baseline, and the
   writer is what recreate uses later).
2. zlib `deflate_slow` emulator for level 6, corrections coded, round
   trip on zlib streams; then the level table 4–9, then `deflate_fast`
   1–3, then gzip and Info-ZIP variants.
3. Detection over the candidates.
4. Tree prediction.
5. Seedable state and chunk-parallel open and close.
6. Switch `src/deflate.rs` to `reflate` where it opens a stream at
   least as well as preflate, with preflate as the fallback; measure on
   the container corpus; when the fallback no longer fires on it,
   remove `third_party/preflate-rs`.

What this does not cover, and stays closed: streams from encoders not
emulated (7-Zip's, .NET's, Go's, libdeflate's, zlib-ng's, miniz's; each
is a later variant if it shows up in real buckets), and JPEG, which is
Lepton's road and a separate piece of work to own.
