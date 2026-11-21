# Pack-format fixtures

Small saved checkpoints (a tiny Qwen3: one 128-wide layer, random bf16 weights from a fixed seed) that every glyd must keep loading, bit for bit. They are read by
`glyd-gpu/src/save/pack_compat.rs`, which runs in CI on every merge, on the CPU:

- `committed_packs_verify`: `verify` decodes every pack of each directory and compares each tensor with the sha256 that the release that saved it wrote into `glyd.json`
  (the weights for a lossless layout, the decoded weights for a lossy one).
- `the_current_packer_writes_the_released_files`: this glyd's packer, on the same checkpoint, writes the same bytes as the release's.

| Directory | Written by | Format and layout |
|---|---|---|
| `v0.33.1-mma` | `glyd pack` of release 0.33.1 | glyd-v4, tiered, lossless |
| `v0.33.1-mma12` | the same | glyd-v3, 12-bit, lossless |
| `v0.33.1-a` | the same | glyd-v6, candidate A (three fraction bits), lossy |
| `v0.33.1-i6`, `-i5`, `-i4`, `-i3` | the same | glyd-v7, six, five, four and three-bit block codes (kestrel, swift, wren; i3 has no level) |
| `v1-0.27.0-mma` | glyd 0.27's packer (`glyd_gpu::v1`) | glyd-v1, the tiered layout of glyd 0.20 to 0.27 (glyd-v2 packs are the same words) |

A change to a layout's bytes is a new format version with a new fixture; it must not edit these. To add a release's (the one that introduces a layout):
build its `glyd` and `glyd-gpu`, run `GLYD_PACK_SRC=DIR cargo test -p glyd-gpu write_pack_sources -- --ignored` (it writes the tiny checkpoint to `DIR/src`),
`glyd pack DIR/src tests/data/packs/vX.Y.Z-LAYOUT --layout LAYOUT` for each layout, and add the layout to `LAYOUTS` in `pack_compat.rs` if it is new.

Not here yet: the lossless row layouts and the codebook layouts (`i6r`, `i5r`, `i4r`, `c3r`, `c2r`, `hb`: they need a plan or the Hummingbird codes), the FP8 and
FP4 layouts, and the decode on a GPU (the GPU workflows decode a model; these fixtures are for the CPU decode and the loaders).
