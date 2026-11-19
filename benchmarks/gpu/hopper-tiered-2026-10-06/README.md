# The tiered layout's products of 1-64 tokens on Hopper: a TMA and wgmma kernel of its own (2026-10-06) — not run yet

**Status: built and compiled, not measured.** The H100 job below was refused before a pod was made: from 01:27 EDT the Runpod
launcher refuses every data-center GPU (H100, H200, B200, A100) and every pod name outside the quantized sessions' list (the
owner's limits for the last credits). Spend for this record: $0. Nothing here is a default: the new route is off until measured.

## Why

On an H100 SXM the tiered layout (10.8 bits a weight) took 1.15x the 12-bit layout's GPU time for Qwen3.5-9B's products at 1 token,
1.19x at 8, 1.30x at 16, 2.07x at 32 and 2.47x at 64, its output layer 1.45-2.40x (`tiered-speed-2026-10-06`, `h100-1-routes-v029`;
the same from the L2 cache as from memory: bound by the decode, not by memory). So Hopper keeps the 12-bit layout (21% smaller, not
33%). The tiered layout's kernels there are the step kernel (1-16 tokens) and the 2/4-m-tile `mma_gemm` (17-64): each warp loads
a step's block offset, then the block (two dependent round trips to memory a step, one step ahead), decodes it and multiplies by
mma.sync; at 33-64 tokens one block of 8 warps an SM (254 registers) and 32 mma.sync a step.

## What this branch adds (gpu/glyd_gpu.cu)

`tiered_tma_kernel` (sm_90a), the route WG for the tiered layout, mma12_tma_kernel's plan for its steps:

- A TMA warp copies each stage (4 steps, 64 columns) into a ring of up to 8 slots: X's tile (tensor map, 128-byte swizzle), the 4
  steps' fixed parts (5,120 bytes), the bytes of their blocks that the short form of the decode reads (each step's window and
  tail: from 128 before the first one's end to 128 past the last one's start, up to 2,048 bytes; more: read from global memory),
  and block_base at the 4 steps. Every global load of the decode is made stages ahead; none waits in a consumer.
- Consumer warpgroups, a row block each: warp k decodes step k from shared memory (`exponents_h`, `pairs`: the step kernel's
  decode, unchanged; a flagged step by the general `exponents`, inline), puts its 1,024 weights into one tile of the stage's 64 rows
  by 64 columns in X's 128-byte swizzle, and wgmma multiplies the tile by X's tile (A and B from shared memory, the 12-bit TMA
  kernel's descriptor for both). A weight is decoded once for all of the step's tokens. A variant keeps A in registers (the trade
  of fragments between the 4 warps, then mma12_tma_kernel's wgmma).
- The split of the work over the blocks and the sums out are mma12_tma_kernel's (`sum_out12`): with two warpgroups and one block
  an SM the products are the 12-bit TMA kernel's, in the same order, so the two give the same bits on the same matrix (the check).
- Shapes (GLYD_GPU_HT_V, dev builds): 0 two warpgroups, one block an SM, A from shared memory (default); 1 three warpgroups; 2 one
  warpgroup, 3 blocks an SM; 3 one, 4 an SM; 4 two warpgroups, A in registers; 5 two warpgroups, 2 blocks an SM.
- Route: off by default (`tiered_wg_max()` 0 until measured); GLYD_GPU_HT_MAX=64 (dev builds) asks for it at 1-64 tokens, K a
  multiple of 64; glyd_gpu_mma_linear's route 3 given takes it (an x not 16-byte aligned: the route GEMM). `kernels.mma_gemm_x3`
  takes a tiered pack by that route (GLinear's Hopper step path, once the route is on).
- `gpu/check_hopper_tiered.py`: the new route against the 12-bit TMA kernel bit for bit (shapes 0 and 4), against fp32, the flags
  (every 2nd-64th step and random shares: the general decode), the blocks' bytes from global memory (GLYD_GPU_HT_BB=0), the same
  bits every run, the other shapes against fp32, the routes; on the packers' test matrices (K a multiple of 64) and random ones.

## Static numbers (CUDA 13.0.2 nvcc, sm_90a; not measured)

| | registers (16 / 32 / 64 tokens) | spills |
| :--- | ---: | :--- |
| shape 0 (2 warpgroups, A from shared memory) | 83 / 96 / 128 | none |
| shape 1 (3 warpgroups) | 83 / 102 / 128 | none |
| shape 2 (1 warpgroup, 3 blocks an SM) | 80 / 96 / 128 | none |
| shape 3 (1 warpgroup, 4 blocks an SM) | 80 / 96 / 96 | 36 / 92 bytes at 64 |
| shape 4 (2 warpgroups, A in registers) | 111 / 128 / 162 | none |
| shape 5 (2 warpgroups, 2 blocks an SM) | 80 / 96 / 96 | 36 / 92 bytes at 64 |

No wgmma serialization (ptxas C751x) in any shape. The consumer's loop (shape 0, flagged steps' path left out): 470 instructions a
stage a warp, that is per 1,024 weights for any number of tokens (ALU 236, IMAD 99, shared memory 60, shuffles 11, wgmma 4); the
step kernel's: 355 a step (ALU 196, IMAD 74) at 1-8 tokens, 2.4-2.6x that a weight's tokens at 33-64. On an H100 (4 ALU
half-rate pipes an SM) the ALU floor is about 118 clk a step of 1,024 weights, against the 12-bit layout's measured 169-200 at 1
token and 176-213 at 64 (`h100-1-routes-v029`) and the memory floor at bf16's measured read rate, about 129. Whether the kernel
gets near that floor is what the job answers.

## The job (one H100 SXM, ~55 min, ~$3.5)

`job/ht_job.sh` (the tree at the run's COMMIT as `ht_src.tar`; `layouts.py`, `fetch_tensors.py` and `summarize.py` are the
tiered-speed track's, as of its 8d03f7b1): build for sm_90a, SASS, the self-test, check_step.py, check_hopper_tiered.py,
check_capi.py; layouts.py on a full-attention and a linear-attention layer of Qwen3.5-9B and -27B, a sliding and a global layer
of Gemma-4-12B and -31B, Qwen3.5-9B's output layer, 1-64 tokens, every shape, the old routes beside it; the fastest shape then
for check_vllm.py --quick on Qwen3.5-9B and `vllm serve` Qwen3.5-9B under Glyd (12-bit, tiered, 12-bit again) at 1, 8, 32 and 64
users, 1,024 tokens in and 256 out. No bf16 run. The new route failing its checks: no vllm and no serve.

```sh
S=<the launcher's scratchpad>; T=$(mktemp -d); git archive HEAD gpu glyd-gpu-py | tar -x -C $T; cp -r benchmarks/gpu/hopper-tiered-2026-10-06/job $T/job
git rev-parse --short HEAD > $T/COMMIT; tar -C $T -cf ht_src.tar gpu glyd-gpu-py job COMMIT
GPU="NVIDIA H100 80GB HBM3" MAX_MIN=70 NAME=htier1 UPLOAD="$PWD/ht_src.tar" bash tools/runpod/rp_try.sh benchmarks/gpu/hopper-tiered-2026-10-06/job/ht_job.sh
```

What decides the defaults after it: the route on for the tiered layout on Hopper where the job measures it at least as fast as
the step kernel and mma_gemm at every token count; `best_layout` tiered on Hopper only where it is at least as fast as the 12-bit
layout at every token count measured (the prompts too: 65-1024 tokens stay on the decode for cuBLAS, 1.18-2.2x the 12-bit
layout's time in `h100-1-routes-v029`, which this kernel does not change) and end to end at 1-64 users.
