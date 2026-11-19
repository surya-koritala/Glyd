# 2x H100 SXM: PR #166 checked, a kernel timeline, and the real DeepSeek-V4-Flash on two GPUs (2026-10-19)

Lambda gpu_2x_h100_sxm5 (NVLink, NV18), 15:27-15:46 Eastern, about $2.80. Branch hbm-prefetch at 2a828e35 (with main #167 and the
layout H experts fix), engine and library for sm_80 and sm_90a built on the PC. `sess166h.sh` in the scratchpad ran it.

## Checks (all as on the 4080)

- dsv4check, layouts H and C and split over the two GPUs: 11 of 24, 2.4437 (the digits of every run before).
- f8moecheck, H and C, uniform and `--peaked` bytes: 0.45-0.49 bf16 steps, the same for both layouts.
- moecheck: every stage within its bounds. glmdyn (GLM, dynamic against the old engine), one GPU and split: 48 of 48, 0.0000.
- musecheck on the bounded model (all 16 experts, q scale 1): 75 of 80, 0.0547 decode, 0.0469 prompt (the 4080 and main the same).

## The real model on two GPUs

DeepSeek-V4-Flash (deepseek-ai/DeepSeek-V4-Flash, 149 GB download), split 22 | 21 layers:

- weights 73.36 + 70.22 GiB; 74.16 and 71.03 of 79.18 GiB in use. The release (148.6 GiB) leaves too little room on two.
- "The capital of France is" -> " Paris. It is one of the most famous cities in the world. Paris is known for its beautiful architecture, art, and culture."
- **15.31 ms a token (65.3 tok/s)**; on 4x H100 on 10-18: 26.4 ms (37.9 tok/s).
- Score, the same 517 tokens: perplexity 2.242 (mean log-probability -0.8072); 4x H100 on 10-18 2.2363, vLLM 2.2398, 8x A100 2.222.
- Load 288-292 s.

About 9.7 GB of weights a token: 2.9 ms at one H100's 3.35 TB/s (the stages run in turn), so 19% of that; 9.5% of both GPUs' bandwidth.

## Steps with a graph (3 and 4 layers, random weights at the releases' widths)

| ms a step | layout C | layout H | 10-18 |
|---|---|---|---|
| DeepSeek, 3 layers | 1.571 / 1.574 | **1.400 / 1.404** | 1.791 |
| GLM, 4 layers | 2.460 / 2.469 | **2.074 / 2.080** | 2.527 |

## Where a DeepSeek layer's time goes (Nsight Systems, the last decode step, H100 0)

120 kernels, 1,439 us: 1,300 us in kernels, 140 us between them (median gap 0.42 us). The first layer, about 290 us:

| | us | |
|---|---|---|
| FP8 products (q up, o low rank, o, shared expert) | ~123 | ~110 MB: ~0.9 TB/s, 27% of the bandwidth |
| FP4 experts (gate/up 45, down 26) | ~71 | ~77 MB: ~31% |
| lossless products (q down, kv) | ~21 | |
| hyper-connections (hc_fin 11 us twice) | ~32 | |
| attention, indexer, rounding, routing | ~40 | |

The layer's ~200 MB take 60 us at the bandwidth. The products are the lever (about 125 us of the gap), then the small kernels and
the gaps (~85 us).
