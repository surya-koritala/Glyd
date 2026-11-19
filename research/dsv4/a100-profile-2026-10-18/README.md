# DeepSeek-V4 and GLM-5.3-Flash step profiles on one A100 (2026-10-18)

Lambda gpu_1x_a100_sxm4 (A100-SXM4-40GB, SM 1410 MHz), about 55 minutes, about $1.85. No H100 had capacity.

Random weights at the releases' widths (`dsv4random --real --layers 3 --experts 32`,
`glmrandom --real --layers 4 --experts 32`). Engine at #159 plus `glmrandom --real`, built for sm_80 on the PC.

The files:

- **`prof-ds.txt`, `prof-glm.txt`:** `dsv4prof`, each stage's time per step without a graph, the kernels, the step with a graph.
- **`moe-speed.txt`:** `moecheck dsv4 glm53 --speed`, the experts' layer at the real shapes, every block layout.
- **`f8moe.txt`:** `f8moecheck` at GLM's shape (288 experts, 4096 x 4096, 8 a token).

## Step time with a graph

| | A100 | 4080 SUPER |
|---|---|---|
| DeepSeek, 3 layers | 3.203 ms | 2.553 ms |
| GLM, 4 layers | 4.186 ms | 4.06 ms |

## What the A100 shows that the 4080 did not

1. **Small kernels are 2-3x slower than on the 4080.** They are bound by clock and serial depth, not bytes.
   - `v4_hc_fin`: 32 us (4080: 13.5)
   - `v4_attn_out`: 40 us (16)
   - `moe_logits`: 22 us (9)
   - `v4_fp8`: 9.7 us (3.2)

   The A100 clocks at 1.41 GHz against the 4080's 2.55. This is why the 4x H100 DeepSeek run (0.61 ms a layer) was no faster per layer than the 4080.
2. **Without a graph the A100 loses 2.1 ms a step to launch gaps (5.34 vs 3.20 ms).** The 4080 loses 0.55 ms.
3. **The experts' products at one token reach 22-38% of the A100's 1,555 GB/s.**
   - Split-K (`GLYD_GPU_MOE_SLOTS` 864 to 3456) makes it slower, so the limit is decode work a block, not the grid.
   - The FP8 experts' product reaches 734 GB/s (47%); on the 4080 it reached 665 of 736 (90%).
4. **The exact FP4 layouts are the slowest at one token here.**

   | Layout | 1 token | 64 tokens |
   |---|---|---|
   | FP4, packed scales | 218.6 us | 5574 us |
   | FP4 | 202.8 us | 4554 us |
   | Int5 | 181.8 us | 4713 us |
   | Int4 | 173.7 us | 4430 us |

   The FP4 decode (two byte tables, a merge, the signs) costs more integer work a byte than the A100's clock gives at that bandwidth. On the 4080 the packed-scale layout was the fastest.

## Next

- **A cheaper FP4 decode:** one byte table of twice the magnitude (0 to 12) into the bf16 mantissa, one fused multiply-add by the scale, the signs by XOR.
- **Fewer and shallower small kernels:** hyper-connections, router logits, the activations' FP8.
- **An H100 profile when Lambda has capacity.**
