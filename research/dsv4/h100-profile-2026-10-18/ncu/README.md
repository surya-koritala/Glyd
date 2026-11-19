# Nsight Compute on the one-token products (H100 SXM), and an occupancy test (H100 PCIe), 2026-10-18

Nsight Compute 2026.3.1, installed from NVIDIA's CUDA repository on the rented machine.

| Kernel (H100 SXM, one token) | Registers | Occupancy (theoretical / achieved) | DRAM throughput | No eligible warp |
|---|---|---|---|---|
| `mma_moe_kernel<Int5>` (DeepSeek's experts) | 113 | 25 / 20% | 34-41% | 53% |
| `f8_linear_kernel` grouped (GLM's FP8 experts) | 125 | 25 / 24% | 49-51% | 63% |
| `step_kernel` (Gemma 12B penguin, dense) | 121 | 25 / 21% | 30-45% | 50% |

The top stall: waiting on L1TEX loads, 48% of the 6.8 cycles between two issued instructions. Nsight's own summary: "the kernel is
memory-latency-bound, which is not necessarily memory-bandwidth-bound; prefer latency-hiding changes such as ILP, unrolling, or pipelining."
The experts' kernel also runs 384 blocks at 2 an SM: 1.45 waves, the second half full.

## Occupancy test (H100 PCIe; `../occupancy-pcie`)

Registers were capped with a `GLYD_GPU_OCC` build so that 3 or 4 blocks fit an SM: 80 and 64 registers, no spills. All builds passed
moecheck 140 of 140.

| One token | 2 blocks (as built) | 3 | 4 |
|---|---|---|---|
| int5 experts (DeepSeek) | 144 us | 189 | 277 |
| FP8 experts (GLM's shape) | 1,264 GB/s | 1,330 | 1,174 |
| Gemma penguin q/k/v | 55% of copy | 48% | 43% |
| DeepSeek 3-layer step | 2.24 ms | 2.31 | 2.60 |

More warps did not hide the latency. Fewer registers each cost the warps their loads in flight (less unrolling), which matters more.
Not kept, apart from a possible 3 for the FP8 kernel (+5%).

## Next

Each step's loads include the token's activations from L2 beside the weights, so a step waits on two long loads. Keep the activations
in shared memory (one row of 4,096 bf16 is 8 KB) so a step waits on the weights alone, with two or three steps ahead.
