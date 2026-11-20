# Lossless FP8 (e4m3) weights: the format

Written in the former private repo glyd-gpu-private, now part of this repository. Tasks 1, 2 and 5 of `glyd-engine/docs/design/plans/2026-10-05-fp8-lossless.md`: the format chosen by measurement on 2026
checkpoints and its numpy reference (`glyd-gpu-py/glyd_gpu/fp8ref.py`, the byte layout in its docstring), the GPU's pack and unpack, and the
fused product of 1 to 64 tokens with the packed weights (`f8_linear`).

## Result

**64 x 128 patches, each value's sign and 3 mantissa bits kept as they are (4 bits), its exponent coded in levels against the patch's own
ranking of the 16 exponents (a 2-bit digit for ranks 0-2, a 2-bit digit for 3-5, a 3-bit digit for 6-12, then a 4-bit literal), a patch
whose variable part reaches 512 words stored plain.** On 50 samples (571 M values) of the five checkpoints, the format as built
(`fp8ref.pack(w).nbytes()`: heads, parts, padding and the shape counted):

| Repo | bits a value | saved against 8 | worst sample |
| :--- | ---: | ---: | ---: |
| zai-org/GLM-5.3-Flash | 6.700 | 16.3% | 6.727 (15.9%) |
| Qwen/Qwen3.8-27B-FP8 | 6.719 | 16.0% | 6.754 (15.6%) |
| MiniMaxAI/MiniMax-M3-MXFP8 | 6.728 | 15.9% | 6.792 (15.1%) |
| mistralai/Mistral-Small-4-119B-2603 | 6.700 | 16.3% | 6.739 (15.8%) |
| deepseek-ai/DeepSeek-V4.1-Flash | 6.755 | 15.6% | 6.852 (14.4%) |
| all | 6.722 | 16.0% | |

The 15% gate of the spec holds on every repo's mean; DeepSeek's margin is the thinnest (15.6%; one of its ten samples, a shared expert's
w2, is at 14.4%). No sample has a plain patch.

This is layout 2 (Task 5), for the fused product. Layout 1 (Task 1's, below in the study: three levels of 2-bit digits, 4-word heads) was
6.686 bits a value (16.4%). Layout 2 costs 0.036 bits a value more and gives the product what it needs:

- **a row block (16 rows) decodes on its own:** the head (8 words) holds, for row blocks 1-3, the values before them that reach levels 2, 3
  and the literal, so a warp that owns 16 rows of the output reads only its part of each level;
- **a lane's four units of a row block are together** in the stream (stream index 2048 mb + 64 lane + 16 ks + j), so its level-2 and level-3
  digits are one run, found by one warp scan a row block instead of one a unit;
- **level 3 is one 3-bit digit** for ranks 6-12, the literal past them (ranks 13-15): a unit's marked values are read from one window of
  bits, and literals are rare enough that their count and scan run only in a row block that has one;
- **the sign-and-mantissa words are rotated** so that a word masked by `0x87878787` is four values with exponent 0 (and the word rotated by 4
  bits the next four): the exponent is OR'd in, no byte shuffling;
- **the 2-bit digits interleaved by nibble** (value j at bit 4 (j & 7) + 2 (j >> 3)): the digits of a register's four values are one
  selector of a byte permute (`prmt`) into the patch's table of exponents.

## The samples

`fetch_samples.py` reads 10 e4m3 weight tensors of each of the five public repos by HTTP range (no token; layers, experts and rows at
random, seed 0): attention, MLP (the shared experts and dense MLPs) and, where there are any, experts (4 / 3 / 3, else 5 / 5), each a slice of
whole rows, a multiple of 128 of them, 8.4-12.6 MB. 50 samples, 571 M values. Not in git: they are under `$S/fp8data/` (`manifest.json`: repo,
tensor, shape, expert, slice rows, scale tensor and its shape). What it leaves out: the mtp heads, DeepSeek's engram tables (its two
`engram.embed` tensors are lookup tables of 98 GB each, not Linears) and e5m2. DeepSeek-V4.1-Flash's routed experts are FP4 (int8-packed, 278 GB),
so its FP8 samples are attention and shared experts. The scales of the samples: fp32 per 128 x 128 (GLM, Qwen), e8m0 per 32 x 32
(DeepSeek), per 1 x 32 (MiniMax MXFP8), one per tensor or per expert (Mistral).

## What was measured (`fmt_study.py`)

An e4m3 byte is sign (1), exponent (4, bias 7), mantissa (3). The exponent is where the room is: its order-0 entropy is 2.5-2.7 of its 4
bits, and the mantissa is nearly uniform. Every candidate keeps sign and mantissa (4 bits) and codes the exponent; each tile has a 32-bit
pointer to its variable part, and a tile that would take more than 8 bits a value is raw. The tables are bits a value, every header,
escape and raw tile counted, on all 50 samples (the study's cost model; the built row is the real arrays).

**(d) the floors**

| | MiniMax-M3-MXFP8 | Qwen3.8-27B-FP8 | DeepSeek-V4.1-Flash | Mistral-Small-4-119B-2603 | GLM-5.3-Flash | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| order-0 floor of the byte | 6.534 | 6.557 | 6.560 | 6.487 | 6.461 | 6.524 |
| sign + exponent coded, 3 mantissa bits as they are | 6.594 | 6.618 | 6.621 | 6.557 | 6.571 | 6.594 |
| exponent's entropy inside each 64x128 tile (+ 4 bits) | 6.592 | 6.568 | 6.608 | 6.554 | 6.553 | 6.576 |

**(a), (b) the candidates, 64x128 tiles with their own ranking**

| | MiniMax-M3-MXFP8 | Qwen3.8-27B-FP8 | DeepSeek-V4.1-Flash | Mistral-Small-4-119B-2603 | GLM-5.3-Flash | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| (a) window(2): 3 exponents + escape | 7.105 | 7.077 | 7.118 | 7.054 | 7.039 | 7.081 |
| (a) window(3): 7 exponents + escape | 7.097 | 7.114 | 7.127 | 7.106 | 7.116 | 7.112 |
| (b) tiers(2,2): 3 + 3, then a literal | 6.742 | 6.735 | 6.776 | 6.715 | 6.716 | 6.738 |
| (b) tiers(2,3) | 6.845 | 6.830 | 6.863 | 6.811 | 6.801 | 6.832 |
| (b) tiers(1,2,2) | 6.887 | 6.798 | 6.859 | 6.797 | 6.777 | 6.827 |
| (b) tiers(2,1,2) | 6.731 | 6.685 | 6.732 | 6.677 | 6.663 | 6.700 |
| (b) tiers(2,2,2): 3 + 3 + 3, then a literal | 6.674 | 6.673 | 6.707 | 6.654 | 6.652 | 6.673 |
| (b) tiers(2,2,2,=3) | 6.672 | 6.668 | 6.702 | 6.650 | 6.648 | 6.669 |
| (b) tiers(2,2,2,2) | 6.667 | 6.662 | 6.696 | 6.645 | 6.642 | 6.664 |

**(c) where the ranking comes from, tiers(2,2,2), by tile size**

| | MiniMax-M3-MXFP8 | Qwen3.8-27B-FP8 | DeepSeek-V4.1-Flash | Mistral-Small-4-119B-2603 | GLM-5.3-Flash | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 16x128 tiles, ranking: the tile's own | 6.699 | 6.696 | 6.722 | 6.678 | 6.676 | 6.695 |
| 16x128 tiles, ranking: its 128 rows' | 6.682 | 6.724 | 6.723 | 6.665 | 6.671 | 6.695 |
| 16x128 tiles, ranking: the tensor's | 6.682 | 6.738 | 6.728 | 6.665 | 6.675 | 6.699 |
| 64x128 tiles, ranking: the tile's own | 6.674 | 6.673 | 6.707 | 6.654 | 6.652 | 6.673 |
| 64x128 tiles, ranking: its 128 rows' | 6.670 | 6.712 | 6.712 | 6.653 | 6.659 | 6.683 |
| 64x128 tiles, ranking: the tensor's | 6.670 | 6.726 | 6.717 | 6.653 | 6.664 | 6.688 |
| 128x128 tiles, ranking: the tile's own | 6.670 | 6.669 | 6.708 | 6.650 | 6.648 | 6.671 |
| 128x128 tiles, ranking: its 128 rows' | 6.668 | 6.710 | 6.710 | 6.651 | 6.658 | 6.681 |
| 128x128 tiles, ranking: the tensor's | 6.668 | 6.724 | 6.715 | 6.651 | 6.662 | 6.686 |

**(c) the same for window(2) (its base: the tile's own, its 128 rows', the tensor's)**

| | MiniMax-M3-MXFP8 | Qwen3.8-27B-FP8 | DeepSeek-V4.1-Flash | Mistral-Small-4-119B-2603 | GLM-5.3-Flash | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 64x128 tiles, window: the tile's own | 7.105 | 7.077 | 7.118 | 7.054 | 7.039 | 7.081 |
| 64x128 tiles, window: its 128 rows' | 7.104 | 7.144 | 7.129 | 7.057 | 7.053 | 7.101 |
| 64x128 tiles, window: the tensor's | 7.104 | 7.168 | 7.138 | 7.057 | 7.060 | 7.109 |

**the format as built (glyd_gpu/fp8ref.py: 64x128 patches, the ranking of the patch; every head, part and padding byte counted)**: layout 1 (Task 1: tiers(2,2,2), 4-word
heads) and layout 2 (Task 5, the format now: levels 2, 2, 3 and a literal, 8-word heads; `fmt_study.py` prints this row for layout 2)

| | MiniMax-M3-MXFP8 | Qwen3.8-27B-FP8 | DeepSeek-V4.1-Flash | Mistral-Small-4-119B-2603 | GLM-5.3-Flash | all |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| layout 1: fp8ref.pack(...).nbytes() * 8 / values | 6.687 | 6.685 | 6.720 | 6.667 | 6.665 | 6.686 |
| layout 2 | 6.728 | 6.719 | 6.755 | 6.700 | 6.700 | 6.722 |

Reading the tables:

- (a) A fixed k-bit window cannot get near the floor: a patch's 3 commonest exponents hold 73% of its values, its 7 commonest 97%; both windows cost 7.1 bits (11% saved). What
  matters is the second level for the 26% that miss the first: (b) tiers(2,2) is 6.74, three levels 6.67, a fourth level (12 direct
  exponents) 6.66.
- (b) Level widths 2, 2, 2 are the best of the shapes tried (1, 2 and 3 bits at each of up to three levels, and two final-tier variants).
  tiers(2,2,2,=3) (the last level 3-bit digits for what is left, no literal) and tiers(2,2,2,2) gain 0.004 and 0.010 bits for a rank list
  of 16 or a fourth chain of scans in the decode: not taken.
- (c) The ranking of the patch itself is best (6.673 against 6.683 for its 128 rows' and 6.688 for the tensor's, at 64 x 128), by 0.05 on
  Qwen and 0.01 on DeepSeek and GLM, and not at all on MiniMax and Mistral (the tensor's ranking is as good there). A 16 x 128 tile costs
  0.02 bits more than 64 x 128 (its head and pointer are shared by a quarter of the values); 128 x 128 gains 0.003 over 64 x 128. 64 x 128 it is.
- The patches that had to be raw on the samples: none, at the patch's own ranking.

## Why this format

The criteria of the plan: the fewest bits whose decode is a fixed-rate read of each patch, few integer operations a value, and no
data-dependent loop longer than a lane's own spills.

- **Fixed-rate core:** 6 bits a value (4 + the 2-bit digit), the same address arithmetic for every patch. The variable part (levels 2, 3 and
  the literals) is found through the patch's head; the first spill of a lane is a prefix count of the digits before it (a warp scan).
- **Why not fewer levels:** window(3) is a few operations a value but 7.1 bits (11% saved, under the spec's 15%).
- **The head:** 32 bytes a patch (0.031 bits a value): base of its part, its 13 exponents of rank 0-12, n1, n2, n3, the plain flag and the
  row blocks' offsets; two 128-bit loads.
- **The order inside a unit** is the A operand of `mma.sync.m16n8k32` with e4m3, a lane's 16 values in its four registers a0..a3 (and the
  two halves of each register, for `m16n8k16` with bf16, after a byte spread): the product decodes straight into its registers, and the
  unpack writes 4-byte pieces of rows. Compression does not depend on it (the order is a permutation, PERM). Checked bit for bit on an RTX
  4090 (sm_89) and an H100 (sm_90a).
- **Plain patches** (the spill bound): a patch whose variable part would be 512 words (2048 bytes, 2 bits a value: what the coded form saves
  over 8 bits) or more keeps its stream bytes, 6144 in the core and 2048 in its part: no patch takes more than 8192 + 32 bytes. Random bytes
  always reach it; none of the 50 samples' 69,696 patches does.

## The layout

Authoritative: the docstring of `glyd-gpu-py/glyd_gpu/fp8ref.py` (field by field). In short:

- The matrix [O, K] is padded with zero bytes to patches of 64 x 128 (`fp8ref.PATCH`), numbered in row-major order.
- A patch is 4 row blocks mb of 16 rows, each 4 units of 16 x 32 (unit u = 4 mb + ks), a unit 32 lanes of 16 values (lane = 4g + t holds
  rows g and g + 8, columns 4t.. and 16 + 4t..). The stream is the patch's values in the order (mb, lane, ks, j).
- **core** (6144 bytes a patch): per unit 32 words of 2-bit digits (a lane's 16, interleaved by nibble), then 32 lanes x 2 words of signs and
  mantissas (rotated: see the docstring).
- **heads** (8 words a patch): the base of its part in var; the 13 exponents of rank 0-12 as nibbles; n1, n2, n3 (the values that reach
  level 2, level 3, the literal); bit 31 of word 3: plain; words 4-7: the offsets of row blocks 1-3 in each level.
- **var** (words): per patch the 2-bit digits of level 2 (n1), the 3-bit digits of level 3 (n2, a digit may cross a word), then n3 4-bit
  literals, each run padded to a word; then 8 words of zeros.
- The ranking: exponents by count in the patch, commonest first, ties by the higher exponent first, so pack is deterministic and the GPU's
  pack gives the same bytes.

## The GPU kernels

`gpu/glyd_gpu.cu`'s namespace `f8c` (the text between its `f8c begin` and `f8c end` lines) has our own kernels for this layout; the C API (`gpu/glyd_gpu.h`, version 16)
has `glyd_gpu_f8_size_workspace`, `glyd_gpu_f8_size`, `glyd_gpu_f8_pack`, `glyd_gpu_f8_unpack`, `glyd_gpu_f8_unpack_experts`, `glyd_gpu_f8_linear_workspace` and
`glyd_gpu_f8_linear`, and `glyd_gpu/kernels.py` has `F8Packed`, `f8_pack(w)`, `f8_unpack(p, out=None)`, `f8_unpack_into(p, out)`, `f8_unpack_experts(p, E, ids, out)`
and `f8_linear(p, kind, scale, x, out=None)` on torch tensors.

| Kernel | Launch | What it does |
| :--- | :--- | :--- |
| `f8_head_kernel` | one CTA of 256 threads a patch | counts the patch's 16 exponents (shared-memory counts a warp), ranks them, writes its head and the words of var it takes (a plain patch: 512) |
| `f8_scan_kernel` | one CTA of 1,024 threads | the exclusive sum of those words in place (each patch's base in var), the total to the workspace |
| `f8_pack_kernel` | one CTA of 256 threads a patch | the patch through shared memory into the order of the layout, a thread's 32 values (two lanes), the levels' ordinals by two CTA scans, the digits OR'd into a shared copy of its part of var, the core written as 8-byte and 16-byte stores |
| `f8_unpack_kernel` | grid (patches, items), 256 threads | the reverse: head, 8 bytes of digits and 16 of nibbles a thread, the patch's part of var staged in shared memory, level 1 and 2 for all 32 values by byte-permute selectors, levels 3 and the literals for the few marked values, the patch back to rows through shared memory (8 KiB, a swizzle that has no bank conflicts), rows written 16 bytes a thread |
| `f8_linear_kernel<NTL, KIND>` | grid (row groups of 128 rows, splits of K), 256 threads, a warp a row block of 16 rows | Y = X W^T for 1-64 tokens without writing W: the CTA's columns of X staged once in shared memory in the B fragments' order; per patch across its K, the warp's granule (16 rows x 128 columns) decoded straight into the A registers of `mma.sync.m16n8k16` (bf16), 2 products a unit and 8 tokens, NTL tiles of 8 tokens; the scales applied a 128-column block (block), a 32-column unit (mx, ds32) or at the end (tensor, channel); K split over CTAs where the rows alone do not fill the GPU, the last CTA of a row group adding the partial sums in order. `KIND \| F8_A8` (block, tensor, channel; development builds only, `GLYD_F8_LIN_A8=1`): W8A8, X's e4m3 copy (`f8_xq_kernel`) staged in place of bf16 X and the decoded bytes straight into `mma.sync.m16n8k32` e4m3, one product a unit and 8 tokens, each token's scale applied a 128-column block |
| `f8_xq_kernel` | a warp a token's 128 columns | (development builds) W8A8's X: each token's 128 columns as e4m3 of 448 over their largest magnitude (round to nearest even, saturated), in the B fragments' order, and the scale back (fp32) |

`glyd_gpu_f8_size` is the size query: it reads the matrix once, writes `heads`, and waits for the stream (one host read of the total, 8 bytes) to say the words `var` needs; `glyd_gpu_f8_pack`
then fills `core` and `var` (the bytes are `fp8ref.pack`'s, whatever the shape: O and K that are not multiples of the patch, a row's distance that is not K). Unpack is one launch with no
allocation and no wait, so a CUDA graph holds it. `glyd_gpu_f8_unpack_experts` takes a stack of E matrices packed as one [E O, K] (O a multiple of 64) and decodes the matrices
`ids[i]` (int32, on the device, read by the kernel) into `out[i]`: grid y is the item, an id outside [0, E) leaves its place in `out` as it was.

`glyd_gpu_f8_linear` (kind 0: fp32 scales a 128 x 128 block; 1: one fp32; 2: e8m0 a 1 x 32 block, MXFP8; 3: e8m0 a 32 x 32 block; 4: fp32 a row,
compressed-tensors' channel scales) is one launch (two with a development build's W8A8: X's quantization first) with no allocation and no wait; its workspace (the partial sums where K is split) and its `done` counters are the caller's, so a CUDA graph holds it. A warp's
granule: the patch's head (two 16-byte loads), the row block's runs of levels 2 and 3 and its literals copied into the warp's shared buffer (the head's
offsets say where), the lane's first spill at each level by a warp scan, then `f8_lane_unit` a unit: the exponents of levels 1 and 2 by byte permutes
from the patch's table, level 3's marked values from one window of 3-bit digits, the literals (rare) by a loop over the lane's own. The e4m3 bytes
become bf16 by a byte spread, a shift and a mask (exact: every e4m3 value is a bf16 value times 2^120, put back with the scale).

**W8A16 everywhere.** X stays bf16 and the weights are exact, on every GPU and for every kind: the library's only plan in a release build. vLLM's
FP8 GEMMs quantize X to e4m3 (a token and 128 columns for block scales, a token for channel scales, the checkpoint's static input scale for a tensor
scale): their error against the exact product is 15-20 times ours (below). The product is bound by the decode, not by its MMAs (at 64 tokens 16
`mma.sync` a unit of 16 values a lane against some 200 decode instructions), so the e4m3 MMA's double rate buys little. A W8A8 build (X quantized
once a step, the decoded bytes straight into `mma.sync.m16n8k32` e4m3) was measured on the H100 (with another decode change: slower at every token
count) and on the RTX 5090 (`benchmarks/gpu/fp8-speed-consumer-2026-10-06`: 0-17% faster than bf16 X in per-shape sums at 32-64 tokens with at least
2 patches a CTA, but slower on 12 of the 70 shapes, up to 1.58x on a 2048 x 512 matrix, still about 2x vLLM's time at 32-64 tokens on gemma-4, and about 15x bf16 X's error): it stays in development builds
(`GLYD_F8_LIN_A8=1`), never a default.

ptxas (CUDA 13.0.2, `-O3 -Xptxas -v`; 0 bytes of stack and 0 spills for each):

| | sm_89 | sm_90a |
| :--- | ---: | ---: |
| `f8_unpack_kernel` | 48 registers, 10,368 B shared | 48, 11,264 |
| `f8_pack_kernel` | 58, 10,368 | 64, 11,264 |
| `f8_head_kernel` | 90, 640 | 90, 1,024 |
| `f8_scan_kernel` | 44, 128 | 30, 1,024 |

`f8_linear_kernel<NTL, KIND, 0, 0>` (the release build's plan), registers at commit 851cf2e8 (block / tensor / mx / ds32 / channel; the development
build's W8A8 block / tensor / channel after the bar; static shared memory 128 B, the rest dynamic: the warps' slots and the X slots, at most 64 KiB of X
at 64 tokens; `benchmarks/gpu/fp8-speed-consumer-2026-10-06/sass/`):

| NTL (tokens) | sm_89 | sm_120 |
| :--- | :--- | :--- |
| 1 (1-8) | 96 / 96 / 105 / 105 / 96 \| 96 / 96 / 96 | 128 / 127 / 128 / 128 / 127 \| 126 / 124 / 124 |
| 2 (9-16) | 96 / 96 / 96 / 96 / 96 \| 96 / 96 / 96 | 144 / 144 / 146 / 146 / 144 \| 138 / 135 / 135 |
| 4 (17-32) | 118 / 118 / 136 / 132 / 118 \| 96 / 96 / 96 | 168 / 168 / 186 / 182 / 168 \| 161 / 159 / 159 |
| 8 (33-64) | 168 / 170 / 230 / 226 / 170 \| 164 / 164 / 168 | 236 / 236 / 255 / 255 / 234 \| 215 / 216 / 215 |

No spills but sm_120's mx kernel at NTL 8 (4 bytes). Every function of the library outside the FP8 kernels has the same SASS as main's (3b69f0d3)
on sm_89 and on sm_120 (266 functions each, `cuobjdump -sass` compared with addresses and encodings left out).

**Checks** (`gpu/fp8/check_f8.py`; the layout's logic on the CPU, thread for thread, by `check_f8.py --emu`, which `emu.py` and `emu_f8.cpp` build from the same text of `glyd_gpu.cu`):
every case of `test_fp8ref.py` with the GPU's pack and unpack in the reference's place (the GPU's core, heads and var equal the reference's array by array; the unpack, bit for bit), the
50 real samples, unpack into rows apart and pack from rows apart, the list's unpack with ids repeated, out of range and in any order, CUDA graphs holding the unpack and the list's unpack
replayed with other ids and other bytes in the same memory (the capture allocating nothing beyond what torch's own capture of an empty graph does). `f8_linear`: every kind of scales, 1 to
64 tokens, shapes that are not multiples of a patch, plain patches at the first and the last patch (some at words of `var` that are not 16-byte aligned), dense patches (every row block with all
its levels and literals), each product within bf16's rounding of the exact product of the dequantized weights (|y - ref| <= 2^-8 |ref| + 2^-14 |x| |W|^T: a bound, not bit for bit: the sums
are f8_linear's own); on the GPU also the real samples' bytes, the same bits from two runs (K split between CTAs too), activations from 2^-40 to 2^20 of the usual, CUDA graphs at 4, 33 and 64
tokens replayed on other inputs (their capture allocating nothing beyond an empty capture), and the development plans (a step ahead, the spills' pass, W8A8 on compute capability 12.x); in the
emulation (whose loads assert their alignment) also plans that split K over CTAs every way.
Where they ran: pack and unpack bit for bit on an RTX 4090, an RTX 5090 and an H100 (sm_90a, Task 5's run); this tree's `f8_linear` (72872c13; its library is e1970149's) on an RTX 4090 and an RTX 5090, all passed, and in the
emulation; the checks as listed here (dense patches, the decode's paths counted, a step ahead with the spills' pass) at 6228590a on an RTX 5090 and
in the emulation, all passed. compute-sanitizer memcheck and racecheck on a few products of every kind and plan (`check_f8.py --only case_lin_quick`; the CUDA 12.8
toolkit's sanitizer: the CUDA 13.0 wheel's did not start a process on the pods) on the RTX 4090 and the RTX 5090 (on the RTX 5090 also at 6228590a): 0 errors, 0 hazards; layout 1's kernels
in Task 2: 0 hazards, 0 errors. The H100 has not run this tree's `f8_linear`.

**Unpack and pack** (a 7168 x 18432 matrix, Gaussian weights scaled per 128 x 128 block as a block-scaled checkpoint's; 6.686 bits a value, no plain patch; 200 launches back to
back):

| GPU | unpack | e4m3 written | with the packed bytes read | pack (the size query, its wait and the allocations in) |
| :--- | ---: | ---: | ---: | ---: |
| RTX 4090 | 265 us | 498 GB/s | 914 GB/s of 1,008 | 0.5 ms |
| L40S | 380 us | 348 GB/s | 639 GB/s of 864 | 0.6 ms |
| H100 80GB HBM3 | 149 us | 887 GB/s | 1,629 GB/s of 3,350 | 0.5 ms |

On the 4090 the unpack is at the memory's limit; on the L40S and the H100 it is limited by its instructions.

**The fused product against vLLM 0.30's own FP8 GEMM** (`bench_f8.py`: the five models' shapes, TP 1 and 2/4/8 for the big ones, 89 shapes; M = 1, 8, 32, 64; the
real samples' bytes; each runner on 200 MB of copies; one CUDA graph a runner, best of 5; vLLM's kernel as its own selection picks it for a served model). Ours over vLLM's
time, the geometric mean over a model's shapes, and in brackets the shapes where ours is at most vLLM's:

RTX 4090 (vLLM: Marlin W8A16 for block, MXFP8 and DeepSeek's scales; CUTLASS W8A8 for the tensor scale):

| model | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | 1.04 (7/21) | 1.15 (2/21) | 1.60 (1/21) | 2.13 (1/21) |
| Qwen3.8-27B-FP8 | 1.16 (2/7) | 1.22 (0/7) | 1.74 (0/7) | 2.34 (0/7) |
| MiniMax-M3-MXFP8 | 1.06 (5/20) | 1.18 (2/20) | 1.68 (0/20) | 2.17 (0/20) |
| Mistral-Small-4-119B-2603 | 1.05 (5/16) | 1.25 (1/16) | 1.35 (0/16) | 1.79 (0/16) |
| DeepSeek-V4.1-Flash | 0.94 (15/25) | 1.08 (5/25) | 1.58 (0/25) | 2.10 (0/25) |

H100 80GB HBM3 (vLLM: FlashInfer / DeepGEMM W8A8 for block, CUTLASS W8A8 for tensor, Marlin W8A16 for MXFP8 and DeepSeek's):

| model | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | 1.93 (1/21) | 2.14 (0/21) | 2.80 (0/21) | 4.19 (0/21) |
| Qwen3.8-27B-FP8 | 2.94 (0/7) | 3.06 (0/7) | 3.82 (0/7) | 5.71 (0/7) |
| MiniMax-M3-MXFP8 | 1.87 (2/20) | 2.07 (1/20) | 2.85 (0/20) | 3.89 (0/20) |
| Mistral-Small-4-119B-2603 | 1.66 (0/16) | 1.94 (0/16) | 3.04 (0/16) | 4.80 (0/16) |
| DeepSeek-V4.1-Flash | 1.45 (6/25) | 1.67 (5/25) | 2.42 (0/25) | 3.30 (0/25) |

The tables above are Task 5's kernel (2026-10-05). This tree's (`benchmarks/gpu/fp8-speed-consumer-2026-10-06`; the fp8-speed branch's kernel, X
times 2^64, a row's scales; per-shape sums of a model's distinct shapes, ours over vLLM's time at tensor-parallel 1, shapes whose copies may have
sat in L2 left out): RTX 4090, seven 2026 FP8 checkpoints' shapes, 0.94-1.16 at 1 token, 0.96-1.33 at 8, 1.11-1.65 at 32, 1.68-1.96 at 64 (at or
under vLLM's time on Qwen3.8-27B-FP8 and MiniMax-M3-MXFP8 at 1-8 tokens; slower on every shape at 32-64); RTX 5090 (6228590a, against the kernels
stock vLLM selects when started on each checkpoint there), CUTLASS's block-scaled kernel: Qwen3.6-35B-A3B-FP8 1.19, 1.24, 1.96, 2.75 (1.17, 1.25,
2.15, 3.31 at tensor-parallel 2), Qwen3.8-27B-FP8 1.03, 1.04, 1.83, 2.43; CUTLASS W8A8: gemma-4-26B-A4B-it-FP8-dynamic 1.22, 1.34, 2.21, 2.69
(slower in every per-shape sum; per shape at most vLLM's time on 5 of 19 block shapes at 1 token, 3 at 8, none at 32-64). Mistral-Small-4's
per-tensor rows on the RTX 5090 are not a stock comparison (stock vLLM did not start on that checkpoint to confirm its kernel). Against the decode
into a scratch followed by vLLM's kernel (one graph), ours is faster at 1 token on every model and tensor-parallel size measured; at 8 tokens on
the RTX 4090's tensor and row scales and on the RTX 5090's block and row scales, not on every RTX 4090 block, MXFP8 or DeepSeek shape set (up to
1.13, 1.07, 1.00): `glyd_gpu.kernels.F8_LINEAR_TOKENS` holds it by GPU and kind (opt-in data; the RTX 5090's block scales are not entered, since
f8_linear is above stock vLLM's time there).

At most vLLM's time: 46 of 356 on the 4090 (M = 1: 34 of 89, 8: 10, 32: 1, 64: 1), 15 of 356 on the H100 (9, 6, 0, 0). The product is bound by the decode's
instructions: on the H100 a value of a large matrix costs about 1.3 scheduler cycles (Qwen's gate_up, 34816 x 5120, M = 1: 245 us against vLLM's 68 us,
which reads its 178 MB at 2.6 TB/s); to match it the decode would have to be about 3.6 times cheaper. On the 4090, whose memory is a third as fast, it is near
vLLM's at 1-8 tokens. At 32-64 tokens the 4-8 token tiles' accumulators take the registers (114-196 a thread at 8 tiles: one CTA an SM) and the decode's latency
is not hidden. (Task 5's development plan `GLYD_F8_LIN_CTAS=3` is gone: the fp8-speed branch's kernel takes one wave of as many CTAs as the GPU holds.)

Numerics (`bench_f8.py --numerics`: the error against an exact float64 product of the dequantized weights; max of |err| / (|X| |W|^T), and rms(err) / rms(ref)):
ours is 1.7e-3 rms (bf16's rounding of X and Y) for every kind on both GPUs. vLLM's W8A8 kernels (H100's block and tensor, Ada's tensor) are 2.7e-2 to 3.7e-2 rms,
its Marlin W8A16 1.7e-3 to 2.3e-3.

## The names

The wheel's gate (`check_wheel.py`: IDENT_DENY and the words of its METHOD list) refuses method words in the names and messages of every
module of `glyd_gpu/` (tile, tier, escape, mantissa, exponent, raw, sm, ...; docstrings are exempt, they are not compiled). So the plan's
`fp8ref.TILE` is **`fp8ref.PATCH`**, and the code says patch, level, spill (comments and docstrings say tile, tier and escape). The tests
and the benchmarks, which are not in the wheel, say what they like. Task 2's `kernels.py` is under the same gate.

## Reproduce

```
python gpu/fp8/fetch_samples.py DIR                  # 5 repos x 10 tensors, about 570 MB, a minute
python gpu/fp8/fmt_study.py DIR [study.md]           # the tables above (about 20 s); DIR/study.json
GLYD_FP8_SAMPLES=DIR python glyd-gpu-py/tests/test_fp8ref.py     # the reference: every case, and each of the 50 samples bit for bit
python gpu/fp8/check_f8.py --emu --quick             # the kernels' logic on the CPU (emu_f8.cpp from glyd_gpu.cu's f8c text), about 7 minutes
python gpu/fp8/check_f8.py --samples DIR             # on a GPU (GLYD_GPU_LIB: a library of this tree): every check, and the unpack's bandwidth
python gpu/fp8/bench_f8.py --samples DIR --numerics  # on a GPU with vllm 0.30.0: ours against vLLM's FP8 GEMM, every shape and M; bench_f8.json
```

`GLYD_FP8_SAMPLES` defaults to `~/.cache/glyd-fp8-samples`; without samples the real-sample test says it was skipped.
