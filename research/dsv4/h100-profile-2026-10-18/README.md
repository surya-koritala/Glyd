# DeepSeek-V4, GLM-5.3-Flash and Gemma 4 12B penguin on one H100 SXM (2026-10-18)

Lambda gpu_1x_h100_sxm5 (H100 80GB HBM3), about 30 minutes, about $2.15. Engine at main after #162 plus the scores butterfly,
built for sm_90 on the PC.

The files:

- **`prof-ds.txt`, `prof-glm.txt`:** `dsv4prof` at the releases' widths (random weights, 3 and 4 layers, 32 experts).
- **`moe-speed.txt`:** `moecheck dsv4 glm53 --speed`.
- **`f8moe.txt`:** `f8moecheck` at GLM's shape.
- **`prodbench-gemma.txt`:** the penguin products of glyd/gemma-4-12B-penguin against the copy rate.

## Step time with a graph

| | H100 SXM | 4080 SUPER | A100 |
|---|---|---|---|
| DeepSeek, 3 layers | 1.791 ms | 2.536 ms | 3.203 ms |
| GLM, 4 layers | 2.527 ms | 4.06 ms | 4.186 ms |

DeepSeek is about 0.48 ms a layer on the H100. The layer's bytes (about 200 MB) take about 0.06 ms at 3.35 TB/s, so it runs at about 12% of the bytes ceiling.

## Where the H100's time goes

1. **About 30 kernels a layer.** Each one costs a few microseconds of launch, ramp and tail even in a graph. That alone is more than the layer's 60 us of bytes.
   - The largest, per call: `v4_hc_fin` 20 us, `v4_attn_out` 27, `v4_lin32` 18, `v4_hc_proj` 13.6, `v4_attn_scores` 14.
   - DeepSeek's experts' stage is 213 us a layer, its hyper-connections 151 (without a graph).
2. **The products run at well below the copy rate (3,076 GB/s here).**
   - **Dense penguin products (Gemma 12B):** q/k/v 44%, o 40%, gate_up 63%, down 55% (GLY-334).
   - **Block-layout experts at one token:** 20-35%.

     | Layout | 1 token | GB/s | Share of copy rate |
     |---|---|---|---|
     | int5 | 119.2 us | 863 | 28% |
     | int4 | 112.2 us | 749 | 24% |
     | FP4 | 115.7 us | 687 | 22% |
     | FP4, packed scales | 125.4 us | 606 | 20% |

   - **FP8 experts at GLM's shape:** 1,673 GB/s (54%).
3. **The exact FP4 layouts are slower than int4 and int5 here as on the A100.** Their decode is the cost: FP4 with packed scales at 4 tokens takes 318 us against int4's 252.

## What reaching 95% takes on the H100

Tuning each kernel will not get there; the launches alone exceed the budget. It needs:

- **Fewer kernels a layer.** Fuse the small ones (hyper-connections with their norms, the FP8 rounding of activations into the kernels that make them, routing into one kernel), then a persistent kernel for a layer, or for the decode step.
- **Hopper-native products.** TMA loads, a deeper pipeline (more bytes in flight a warp), the decode overlapped with wgmma, for the penguin, block and FP8 layouts at 1-64 tokens.
- **A cheaper FP4 decode**, or FP4 held as int5 on datacenter GPUs, where speed matters more than its 23% of bytes.
