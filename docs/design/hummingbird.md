# Hummingbird for every mixture of experts

Design for #429. Hummingbird keeps a model's routed experts at 3 and 2 bits a weight by use and the rest of the model as the
release has it. Today it exists for Nemotron 3 Super alone (`src/moe_hb.rs`, `src/hbpack.rs`), from codes a Python script makes
(`research/lossy-2026/nemotron_hbr.py`). This note makes it a format of the MoE kind (`runtime::kinds::Moe`) for every family the
description builder maps (Kimi-K3, DeepSeek-V4, GLM-5.3-Flash, Muse Spark, Qwen3.5/3.8 MoE), with the calibration in the engine.
Written 2026-11-21 against main ca0b6a4b.

Status (2026-11-26): built for the families above: the calibration is `calibrate-experts` (`src/hb_usage.rs`), the packer `pack MODEL OUT --layout hummingbird --usage FILE` (`src/hb_pack.rs`) and the block `src/moe_hbg.rs`; `docs/design/runtime.md` and `glyd-engine/REFERENCE.md` describe what runs. The "Today" and "exists for Nemotron 3 Super alone" lines below are those of 2026-11-21.

## What a hummingbird expert is

An expert of width W (its input: the hidden size, or the latent where the MoE has one) and intermediate size I is stored as
neuron rows of W values, all in the basis of one rotation R per MoE layer:

- `gate_up`: [U, W], U = 2I for a gated MLP (the gate's rows, then the up's) and I otherwise. Row n is R w_n, w_n the release's row.
- `down_t`: [I, W]. Row n is R d_n, d_n = down[:, n] (neuron n's output weights).

A row is its codes (2 or 3 bits, a bitstream), a bf16 scale for each run of 32 values (the run's absmax) and a bf16 codebook of
2^bits entries in units of the scale, fitted to the row (Lloyd-Max of its run-normalised values). That is glyd-gpu's codebook row
layout (`codebook.rs`, tags c2r and c3r) unchanged: it takes any W in runs of 32. Bits a weight: bits + 0.5 + 2^bits 16 / W, so
3.53 and 2.51 at W = 4,096, 3.54 and 2.51 at Kimi-K3's 3,584.

R = (I ⊗ H_b) diag(s): H_b the orthonormal Walsh-Hadamard matrix of order b in natural order, applied to each run of b values,
b the largest power of two that divides W, at most 1,024; s the layer's signs. Ibis already rotates Qwen3.8-Flash-Next's 2,560 in
five blocks of 512 (`qx_cb_rot_in`); Nemotron 3 Super's latent of 1,024 is one block, the case `moe_hb.rs` runs.

| model | W | b | blocks |
| :-- | --: | --: | --: |
| Kimi-K3 (latent) | 3,584 | 512 | 7 |
| DeepSeek-V4-Pro | 7,168 | 1,024 | 7 |
| DeepSeek-V4-Flash, GLM-5.3-Flash | 4,096 | 1,024 | 4 |
| Qwen3.8-Flash-Next | 2,560 | 512 | 5 |
| Nemotron-3.5-Lightning | 2,688 | 128 | 21 |

The signs are drawn from xorshift64* seeded with the FNV-1a hash of `glyd-hb/<layer>` and saved in the directory, which is the
authority (a reader never draws them).

## Running it: one block for every family

`moe::MoeLayer` already holds its routed experts in one of several formats (the packs, or `f8` as an FP8 release has them). Hummingbird
is one more, `hb`, and the layer's forwards (`forward`, `forward_normed`, `forward_weighed` and Kimi-K3's latent path) keep their router,
activation and sum. Only the two products change:

1. x' = R x once a token (a block a run of b values, signs then the butterfly).
2. The first product over the `gate_up` rows of each pair's expert, in the plan's order, into the same [pairs, U] rows the library's
   product gives: <R w, R x> = <w, x>, so the family's activation (SwiGLU with its clamp, Kimi's situ, relu^2, DeepSeek-V4's weighted
   activation with its FP8 rounding) runs on them unchanged.
3. The second product over the `down_t` rows: y'_p = sum_n h_n R d_n, dense over the I neurons (a SwiGLU activation has no zeros;
   relu^2 keeps the sparse product of `moe_rows.rs`).
4. R^T: where the family's sum is linear in the experts' outputs (Kimi-K3, DeepSeek-V4, GLM, Nemotron-H, Qwen) once a token after
   the weighted sum, before anything else is added (Kimi-K3's norm of the sum, the shared expert); where it is not (Muse Spark normalises
   each expert's output before weighing it) once a pair, before the family's own sum kernel.

Nothing in these steps names a family, so a new MoE family gets hummingbird by being described. The CPU reference of the products
(`cpu_hb.rs` generalised to gated experts and blocked rotations) is the kernels' check and the host tier's compute.

## Two bit widths per layer in the expert store

A layer's experts are two groups, one a width, each a stack of experts end to end at its width's stride (as `moe_hb.rs` does now).
A pair goes to its group's kernels by two small tables, `group_of` [E] and `local_of` [E]. In R12's `ExpertStore` (runtime.md §6):

- `Tiers` hold a list of groups per layer, each with its format and its expert bytes; GPU slots are made per group in its own size, the
  layer's budget split by the groups' share of the experts (`moe_hb::split_slots`).
- `LayerPlace` maps an expert to (group, local number) and then to a slot, the host tier or NVMe; the host and NVMe tiers hold each
  group at its real bytes, so a 3-bit expert costs 1.4 times a 2-bit one to copy.
- `Costs` keep a host pair time per format; the copy-or-compute choice is per expert and per format, as the M2 design already says.

Built (`src/hbg_store.rs`): a cached layer's two groups are pools of places on the GPU, and for each call of at most 64 tokens the store
(`ExpertStore::run_groups_at`) reads the router's choices once, copies misses into free places or keeps them for the host, and a small
kernel (`hbg_tables`) points each expert of the call at its place, or at no group where the host computes it. The products and the
plan stay over the layer's own expert numbers, so the activation and sum kernels are the resident layer's. The host computes its pairs
from the host tier as it holds them (`cpu_hbg::store_pairs`) and its sums join the GPU's before R^T (after Muse Spark's normed sum).
Where the computer's memory holds less than all the experts, each group keeps its share in memory and reads the rest from the layer's
expert file in direct reads (`HbgDisk`). A step with such a layer is not captured in a graph.

The planner sizes a layer's experts as 0.4 x 3-bit + 0.6 x 2-bit bytes (the recipe's share) until it has read the directory, then by
the directory's own counts.

## Where it applies

Hummingbird changes the routed experts only, and only where its bits are below the release's experts': MXFP4 (Kimi-K3) and FP4
E2M1 with an E8M0 scale per 32 (DeepSeek-V4) are 4.25 bits, FP8 (GLM-5.3-Flash) 8, bf16 16; the recipe's ~2.9 is below each. The
planner takes the experts' release bits apart from the rest's (`compressed-tensors` MXFP4 and DeepSeek-V4's `expert_dtype` included)
and offers hummingbird where 2.9 is below them. Every other matrix stays as the family runs it at penguin (lossless, or the release's
FP8 exactly), so a hummingbird directory loads with the family's own code for everything but the experts.

A model released at a bit width at or below hummingbird's has nothing to gain and is not offered the level.

## Calibration in the engine

Two steps, both Rust, replacing `nemotron_hbr.py`:

1. `glyd-engine calibrate-experts MODEL OUT.json`: the model runs on the engine (any level and placement that runs it) over the built-in
   calibration corpus (`calib.rs`, chat-templated, 32,768 tokens by default), prompt passes only; each MoE layer's router choices are
   read back and counted per expert. The codec takes each expert's E[x' x'^T] and E[h h^T] where they are given, for GPTQ (`hb_codec::ExpertHessians`); this command does not collect them.
2. `glyd-engine pack MODEL OUT --layout hummingbird --usage OUT.json`: for each MoE layer, the experts ranked by use; the recipe hb40
   puts the 40% most used at 3 bits and the rest at 2 (ties to the lower index; an expert never chosen stays, at 2 bits). Each expert's
   matrices are read from the release exactly (bf16, FP8 with its block scales, FP4 and MXFP4 decoded to f32), rotated, and rounded
   row by row: to the nearest codebook entry, or with GPTQ act-order (over the columns for `gate_up`, over the neurons for `down_t`)
   where Hessians are given and the expert has at least 2,048 tokens. The rest of the release is copied byte for byte.

Nearest rounding needs only the counts, which is what a corpus pass on a single machine gives cheaply. Whether a family needs GPTQ is
the quality bar's question (#235): Nemotron 3 Super shipped with it (+10.3% perplexity at 2.99 bits, KL 0.095).

## The directory

`glyd.json` of format `glyd-hbg/1`, level `hummingbird`, with an `experts` record: the recipe, the rounding, the usage file's tokens and
hash, and each MoE layer's block, file, width, rotation block, sizes, experts' bits, bytes and sha256. It is not a glyd-v7 directory:
glyd-v7 readers take every matrix as packed at a level, and here everything but the experts is the release's.

- the release's tensors but the routed experts, copied with their names and dtypes (BF16, F32, F8_E4M3 and its scales, MXFP4's U8
  blocks and scales), in shards with an index, and the release's config, tokenizer and template files;
- per MoE layer, `experts-BBBB.safetensors` (BBBB the description's block): `glyd_rot_signs` (I8 [W]) and, for each expert e and matrix m
  (`gate_up`, `down_t`), `eEEEE.m.codes`, `.scales` and `.codebook` in the c3r or c2r layout.

A family loader that reads the release reads this directory the same way, and builds the layer's `hb` experts from the expert files
(`HbDir::experts_for`, `MoeLayer::new_hb`).
`hbpack.rs`'s Nemotron 3 Super directories (six-bit elsewhere, from an export) stay readable as they are; new Nemotron-H packs follow
this layout once Nemotron-H is described (R3b).

## Order of the work

1. The codec on the host: rotation, codebook fit, nearest and GPTQ rounding, the recipe; tests against float64 least squares.
2. The packer for described families, from a usage file; a tiny Kimi-K3, GLM and DeepSeek-V4 packed and read back in host tests.
3. The products: the CPU reference, then kernels `hbg_*` for any W in runs of 128 and both widths, and the `hb` format in `MoeLayer`.
4. The corpus pass (router choices read back per MoE layer) on the description executor; the reference backend runs it in host tests.
5. Support rows and the planner's per-part release bits.
6. The expert store's two groups per layer for the described families (R12), then the GPU checks of the testing phase: each family's
   layer against its CPU reference, KL against the release on a rented machine (#235).
