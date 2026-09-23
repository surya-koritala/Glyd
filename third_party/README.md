# Third-party code carried in this tree

## preflate-rs

`preflate-rs/` is [microsoft/preflate-rs](https://github.com/microsoft/preflate-rs)
0.7.6 (Apache-2.0, see `preflate-rs/LICENSE.txt` and `NOTICE.txt`), the
library that turns a deflate stream into its plain text plus the
corrections that re-create the stream bit for bit. It is built as the
package `glyd-preflate`, used under the name `preflate_rs`.

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

The output for a stream handled whole is unchanged from upstream 0.7.6:
objects written by earlier versions of Glyd, and bases their deltas
were made against, open to the same plain text.
