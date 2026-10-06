# Third-party code carried in this tree

## preflate-rs

`preflate-rs/` is [microsoft/preflate-rs](https://github.com/microsoft/preflate-rs)
0.7.6 (Apache-2.0, see `preflate-rs/LICENSE`), the
library that turns a deflate stream into its plain text plus the
corrections that re-create the stream bit for bit. It is built as the
package `glyd-preflate`, used under the name `preflate_rs`.

`preflate-rs/LICENSE` is upstream's file, byte for byte, from the tag `v0.7.6`
(commit `3bcd33441849a8a25d2b26077128f4e3a88d071b`, the one crates.io's
preflate-rs 0.7.6 was published from). Upstream has no NOTICE file at that tag.

Changes from upstream, all in service of opening and closing one stream
on every core (`src/chunked.rs`):

- `chunked.rs` (new): a stream parsed once, cut into chunks at block
  boundaries, each predicted and re-created by a predictor of its own
  that first learns the 32 KB before the chunk; pieces written from the
  bit offset their blocks had, joined by or-ing the shared byte.
- `deflate/deflate_reader.rs`: `DeflateContents::block_bit_starts`, the
  bit each block starts at.
- `deflate/bit_reader.rs`: `bit_position`.
- `deflate/deflate_writer.rs`: `new_at_bit`, `finish`.
- `preflate_input.rs`: `PlainText::with_prefix`,
  `PreflateInput::from_prefix_start`.
- `token_predictor.rs`: `seed`; `hash_chain.rs` and
  `hash_chain_holder.rs`: `rebase`, so a chain can start at any stream
  position.
- `stream_processor.rs`: `predict_blocks`, `recreate_blocks` and
  `ReconstructionData` visible inside the crate.
- The 26 upstream tests that read sample files not shipped with the
  crate are marked `#[ignore]`; the rest, and `chunked`'s, run with
  Glyd's suite.

And one change that is not about chunks: upstream codes its corrections
with the `cabac` crate (LGPL-3.0-or-later). preflate-rs no longer uses
that crate:

- `bool_coder.rs` (new, Glyd's own code, BSD-3-Clause OR GPL-2.0-only
  like Glyd's codec, not Apache-2.0): the boolean entropy coder of RFC
  6386 section 7 with the adaptive probability context of the
  corrections stream, written from a functional description of that
  stream and the RFC alone. It writes and reads the same bytes as
  `cabac` 0.15.0 did, so every corrections stream an earlier release
  wrote opens, and every one it would have written is the same bytes.
  A unary code or a literal longer than 4096 decisions is refused by its
  reader (the old one looped for ever on some malformed input).
- `cabac_codec.rs`, `stream_processor.rs`, `chunked.rs`: the `use` paths.
- `Cargo.toml`: `cabac` is a dev-dependency, `=0.15.0`, the oracle of
  `src/bool_coder/vs_cabac.rs`, which compares the two coders' streams
  (and preflate's corrections for real deflate streams) in the tests. No
  library or binary links it.

The output for a stream handled whole is unchanged from upstream 0.7.6:
objects written by earlier versions of Glyd, and bases their deltas
were made against, open to the same plain text.

Since v0.13.4 nothing new is written with it: every deflate stream is
opened by Glyd's own reconstruction (`src/reflate/`), and this copy is
here to read what v0.12.0 to v0.13.3 wrote (`GLYDGZIP`, `GLYDDEFL`,
`GLYDDEF2`) and to open the bases their deltas were made against.

## silesia/

Test data, not code, read by the tests of `src/rezstd/` alone. `silesia/moz35k.raw` is 34,972 bytes at offset 156,114 of
`mozilla`, a file of the [Silesia compression corpus](https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia) (Mozilla 1.0
for Tru64 UNIX, tarred): the start of its `mozilla/chrome/en-US.jar`. The jar's files are Mozilla.org code under the
Netscape Public License 1.1, or the Mozilla Public License 1.1 with the GPL 2.0 and LGPL 2.1, as each file's header says
(some have none); the corpus page states no terms of its own. `moz35k.zst` and `moz35k.v152.zst` are the same bytes
compressed by zstd 1.5.5 and 1.5.2 at level 1.
