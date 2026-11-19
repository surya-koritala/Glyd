# One H100 SXM: where DeepSeek-V4-Flash's and Kimi-K3's decode steps go (2026-10-22)

Lambda gpu_1x_h100_sxm5 (us-south-2), 00:02-00:19 EDT, about $1.20. Main after #222 (shared expert beside the routed ones), built on the PC.
DeepSeek-V4-Flash at the release's widths on random weights (`dsv4random --real`: 3 layers, 32 experts). Kimi-K3's first two real layers
(`hubslice`, 24.03 GB in 147 s: 163 MB/s with glyd's downloader on this link). `launch.sh`, `job.sh`, `inner.sh`; everything in `out/`.

## Step times with a graph

| | H100 SXM | 4080 SUPER |
|---|---|---|
| DeepSeek, 3 layers + head | 1.242 / 1.247 ms (1.791 on Oct 18) | 2.360 ms |
| Kimi-K3, 2 layers + head, experts on the GPU | 2.770 / 2.773 ms | |
| Kimi-K3, the same with 318 places of 896 (8x A100's plan), 200 steps | 4.374 ms | |

The cached Kimi run copied at 52.7 GB/s (PCIe 5; the A100 box did 24.8).

The library's one-token variants on DeepSeek's step: GLYD_GPU_DEPTH=2 1.218 ms, GLYD_GPU_STAGES=4 1.230, =8 1.234, =8 with
GLYD_GPU_STAGE_BLOCKS=3 1.223, GLYD_GPU_PF=2 1.234. At most 2%: the products are not waiting on single loads.

## The kernels of one decode step (Nsight Compute 2025.3.1, clocks locked, so a little long)

`ncu-all-ds.txt`, `ncu-all-kimi.txt`: every kernel's time; `ncu-ds.txt`, `ncu-kimi.txt`: instructions, bytes, issue and occupancy of
the products; `moe-details.txt`: the FP4 experts' kernel in full.

DeepSeek, a layer, the largest:

| Kernel | Bytes | Time | Of 3.35 TB/s |
|---|---|---|---|
| `mma_moe_kernel<Fp4R>` gate/up (6 experts) | 51.2 MB | 53.7 us | 28% |
| the same, down | 25.6 MB | 31.9 us | 24% |
| `f8b_lin1_kernel`, dense FP8 | 33.7 MB | 22-23.6 us | 43% |
| `f8b_lin1_kernel`, dense FP8 | 8.4 MB | 11 us | 23% |
| `v4_hc_fin` (one block, twice a layer) | | 15.6 us | |
| small penguin products (8-128 blocks) | | 14 us each | |

Once a token: the head 393 us (1.06 GB raw, penguin), `argmax_bf16` 15 us and `v4_hc_head` 16 us, both on one block.

The FP4 experts' kernel: 114 registers a thread, so 2 blocks an SM (25% theoretical occupancy, 20% achieved); 3.3 warps a
scheduler, 0.9 eligible, none eligible in 47-51% of cycles; 12 lane instructions a byte (the A100's figure). It waits for want of warps,
not of issue slots: fewer registers (3-4 blocks an SM) before a leaner decode.

Kimi-K3, its MoE layer (experts on the GPU), ~886 us against ~480 us of bytes:

| Kernel | Time | Of 3.35 TB/s |
|---|---|---|
| q/k/v product (penguin, 528 MB raw) | 380 us | ~29% |
| output gate (176 MB raw) | 130 us | ~28% |
| o product (176 MB raw) | 80 us | ~45% |
| experts, gate/up (16 FP4R) | 155 us | ~35% |
| experts, down | 85 us | ~31% |
| shared experts gate/up, down | 77, 44 us | ~47%, ~42% |
| `kimi_res_mix` (one block), twice | 33 us each | |
| `kimi_rms2` (one block), three times | 14.5 us each | |
| `moe_select32` (one block) | 25 us | |

The one-block kernels are ~190 us of the layer, about 17 ms of a token over 93 layers. The head: 874 us (2.35 GB raw), `argmax_bf16` 19 us.

## What to do, largest first

1. Kimi: the depth mix, the norm and the select over several blocks (bit for bit: the same per-thread order, block_sum's tree).
2. The penguin products at one token on the H100 (Kimi's q/k/v at 29%; GLY-334).
3. The FP4 experts' kernel: registers for 3-4 blocks an SM, then the loop (GLY-349 step 2).
4. DeepSeek's one-block kernels (hc_fin's Sinkhorn off the critical path, argmax, hc_head) and the small penguin products' grids.
