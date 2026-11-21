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
| `main-i6r`, `-i5r`, `-i4r` | this glyd (no release writes them yet) | glyd-v7, the block layout of the same bits, with o_proj and down_proj in the row layout (each row one input's weights) |
| `main-hb` | this glyd, from a synthetic Hummingbird export | glyd-v7 at the level hummingbird: two experts' matrices in codebook rows (c3r at three bits, c2r at two), a dense linear at i6 |
| `v1-0.27.0-mma` | glyd 0.27's packer (`glyd_gpu::v1`) | glyd-v1, the tiered layout of glyd 0.20 to 0.27 (glyd-v2 packs are the same words) |

The release's `glyd pack` wrote the checkpoint's path into each `glyd.json` (`source.repo`); it is `tiny-qwen3` here (nothing reads it).

A change to a layout's bytes is a new format version with a new fixture; it must not edit these. To add a release's (the one that introduces a layout):
build its `glyd` and `glyd-gpu`, run `GLYD_PACK_SRC=DIR cargo test -p glyd-gpu write_pack_sources -- --ignored` (it writes the tiny checkpoint to `DIR/src`),
`glyd pack DIR/src tests/data/packs/vX.Y.Z-LAYOUT --layout LAYOUT` for each layout, and add the layout to `LAYOUTS` in `pack_compat.rs` if it is new.

The GPU job (`gpu-tests.yml`, on the GPU runner) also decodes every directory here with the GPU library: `glyd-gpu verify DIR --device cuda:0`.

The `main-*` ones are what this glyd writes today for layouts no release has: when a release writes one, replace the fixture with the release's (`glyd pack` of it) and keep the name `vX.Y.Z-LAYOUT`.

Not here: FP8 and FP4. They are not saved pack formats: an FP8 or NVFP4 checkpoint is read and packed in one of the layouts above (an FP8 source dequantised to bf16 first: `fp8_matrices_save_as_the_bf16_they_dequantise_to` in `save.rs`), and the engine's `f8` and `f8h` packs are made on the GPU when a model is loaded.
