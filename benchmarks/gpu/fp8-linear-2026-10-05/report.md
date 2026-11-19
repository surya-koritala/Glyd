# Task 5 report: the fused FP8 product (f8_linear)

Worktree `/Users/suryakoritala/glyd-fp8`, branch `fp8-lossless` (glyd-gpu-private). Commits of this task, in order:
`42e6999f` (layout 2 and the first fused product, bench_f8), `ae1ce476` (bench with vLLM's choice of kernel, numerics), `c43379fe`
(the lane decode by raw byte permutes: the kernel kept), `f9281de4` (pipelined variant), `493d0846` (W8A8 on Hopper), `46a53845`
(staged decode), `9a003de6` (back to `c43379fe`'s kernel, W8A16 everywhere, the capture check fixed), `3bac5386` (Ada launch fix,
plan variant to measure), `d0fd96e2` (README).

## Status

BLOCKED on the speed gate; everything else done. The fused product is built for every scale layout, exact (W8A16), correct on Ada and
Hopper and in the emulation, with 0 spills. The gate (ours at most vLLM 0.30's FP8 GEMM at every model shape and M = 1, 8, 32, 64) fails:
RTX 4090 46 of 356 (close at 1-8 tokens, 1.35-2.34x at 32-64 by model), H100 15 of 356 (1.45-5.71x by model). On the H100 the decode's instruction
cost makes the gate unreachable for this format (section 6). Task 6's whole-model runs wait on this gate (plan, owner rulings of
2026-10-05), so this needs the owner's decision.

## 1. Design

### The format: layout 2 (a change of format)

The fused product needs a warp that owns 16 output rows to decode just those rows of a patch, with no CTA-wide scan. Layout 1
(Task 1) could not: its level runs were unit-major across the whole patch, so a row block's first spill at each level needed
the patch's earlier row blocks decoded or scanned. Layout 2 (fp8ref.py's docstring is authoritative; `gpu/fp8/README.md`
summarizes it):

- stream order (mb, lane, ks, j): row block mb, then a lane's four units of that row block together;
- heads of 8 words: base, sym[0..12] (13 ranks), n1, n2, n3, plain flag, and for row blocks 1-3 the counts before them at
  levels 2, 3 and the literal, so a row block's runs start where the head says;
- level 1: 2-bit digit (ranks 0-2, 3 = spill); level 2: 2-bit (ranks 3-5, 3 = spill); level 3: 3-bit (ranks 6-12, 7 = spill);
  literal: 4 bits (ranks 13-15) - literals are rare enough that their count and scan run only in a row block that has one;
- the d1 digits interleaved by nibble (a register's four digits are one `prmt` selector into the patch's exponent table);
- the sign+mantissa words rotated so that `w & 0x87878787` is four values with exponent 0, `rotl(w, 4) & 0x87878787` the next four;
- plain patch if its variable part reaches 512 words or it has 1024+ literals; 8 guard words after var.

Everything that depends on the layout was updated and is bit for bit: `fp8ref.py` (pack/unpack), `tests/test_fp8ref.py`
(layout test from an independent docstring decoder, parts contiguity, guard zeros, plain threshold 511/512, all-zero patch, every
case passing, the 50 real samples bit for bit), the GPU pack/head/scan/unpack (`f8c` in `gpu/glyd_gpu.cu`), `check_f8.py`, the
CPU emulation (`emu.py`, `emu_f8.cpp`), the C header and bindings (heads `[patches x 8]`), README.

Sizes, layout 2, `fp8ref.pack(w).nbytes()` on the 50 samples (571 M values; heads, parts, padding, shape counted):

| Repo | bits a value | saved vs FP8 | worst sample |
| :--- | ---: | ---: | ---: |
| zai-org/GLM-5.3-Flash | 6.700 | 16.3% | 6.727 (15.9%) |
| Qwen/Qwen3.8-27B-FP8 | 6.719 | 16.0% | 6.754 (15.6%) |
| MiniMaxAI/MiniMax-M3-MXFP8 | 6.728 | 15.9% | 6.792 (15.1%) |
| mistralai/Mistral-Small-4-119B-2603 | 6.700 | 16.3% | 6.739 (15.8%) |
| deepseek-ai/DeepSeek-V4.1-Flash | 6.755 | 15.6% | 6.852 (14.4%) |
| all | 6.722 | 16.0% | |

Every model's mean stays at least 15% under FP8 (the rule for a format change). Layout 1 was 6.686 (16.4%); layout 2 costs 0.036
bits a value (mostly the 16 extra head bytes, 0.016, and per-run padding). DeepSeek's worst single sample (a shared expert's w2)
is 14.4% (it was 14.8% in layout 1, also under 15%): the gate is on the model's mean, as in Task 1. No sample has a plain patch.

### The kernel: `f8_linear_kernel<NTL, KIND>` (kept: `c43379fe`'s)

Y [M, N] = X [M, K] W^T, M 1-64, W never written out, X and Y bf16, sums fp32.

- Grid (row groups of 128 rows, splits of K), 256 threads, a warp a row block (16 rows), NTL tiles of 8 tokens (1, 2, 4, 8 for
  M up to 8, 16, 32, 64).
- The CTA stages its columns of X once in shared memory, in the B fragments' order (rows padded 64 bytes against bank conflicts).
- Per patch of its K range, a warp decodes its granule (16 rows x 128 columns): head by two 16-byte loads; the row block's runs of
  levels 2, 3 and literals copied into the warp's shared buffer (464 words); warp scans for the lane's first spill at each level;
  then `f8_lane_unit` per unit: level 1 and 2 exponents by `prmt` from the head's table, level 3 marks from one window of 3-bit
  digits, literals by a loop over the lane's own (rare); OR into the rotated sign/mantissa words: e4m3 bytes in the mma A order.
- e4m3 to bf16 by a byte spread (`prmt`), a shift and a mask (value x 2^-120, exact), then 2 `mma.sync.m16n8k16` (bf16, fp32
  accumulate) a unit and token tile.
- Scales: block (fp32 128x128): a per-granule temporary accumulator scaled once; mx (e8m0 1x32) and ds32 (e8m0 32x32): per unit
  of 32 columns; tensor: once at the end, with the 2^120.
- K split over CTAs where the rows alone do not fill the GPU (plan: about two CTAs an SM, the X tile at most 64 KiB); partial sums
  [splits][M][N] fp32 in the caller's workspace, the last CTA of a row group (atomic counter, left zero) adds them in split order:
  the same bits every run.
- C API (version stays 16): `glyd_gpu_f8_linear_workspace(N, K, M, &bytes)`, `glyd_gpu_f8_linear(core, heads, var, N, K, kind, scale,
  lds, x, ldx, M, y, ldy, workspace, bytes, done, stream)`; one launch, nothing allocated or waited for (a CUDA graph holds it).
  Header, `_lib.py` (`f8_linear`, workspace and done counters kept per stream), `kernels.py` `f8_linear(p, kind, scale, x, out=None)`
  with kind one of block / tensor / mx / ds32, Rust declarations. No method words in identifiers or strings (check_wheel's
  `ident_problems` on kernels.py, _lib.py, fp8ref.py: none); `test_gpu.py::test_c_header` passes.

### W8A16 on both GPUs, and why

Measured numerics (bench_f8 --numerics: error against an exact float64 product of the dequantized weights; max = largest
|err| / (|X| |W|^T), rms = rms(err) / rms(ref)), H100 (run 3):

| model | scales | vLLM kernel | vLLM max | vLLM rms | ours max | ours rms |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | block | FlashInferFp8DeepGEMMDynamicBlockScaledKernel | 8.1e-03 | 3.7e-02 | 6.3e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | block | FlashInferFp8DeepGEMMDynamicBlockScaledKernel | 4.8e-03 | 3.7e-02 | 4.5e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | mx | MarlinMxfp8LinearKernel | 4.8e-04 | 1.7e-03 | 4.8e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | tensor | CutlassFP8ScaledMMLinearKernel | 1.5e-02 | 2.7e-02 | 1.5e-03 | 1.7e-03 |
| DeepSeek-V4.1-Flash | ds32 | MarlinMxfp8LinearKernel | 1.2e-03 | 1.7e-03 | 1.2e-03 | 1.7e-03 |

RTX 4090 (run 7; vLLM there: Marlin W8A16 for block, MXFP8 and DeepSeek; CUTLASS W8A8 for the tensor scale):

| model | scales | vLLM kernel | vLLM max | vLLM rms | ours max | ours rms |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | block | MarlinFP8ScaledMMLinearKernel | 7.1e-04 | 2.3e-03 | 6.3e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | block | MarlinFP8ScaledMMLinearKernel | 4.3e-04 | 2.3e-03 | 3.6e-04 | 1.7e-03 |
| MiniMax-M3-MXFP8 | mx | MarlinMxfp8LinearKernel | 5.1e-04 | 1.7e-03 | 5.1e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | tensor | CutlassFP8ScaledMMLinearKernel | 1.4e-02 | 2.7e-02 | 1.5e-03 | 1.7e-03 |
| DeepSeek-V4.1-Flash | ds32 | MarlinMxfp8LinearKernel | 1.3e-03 | 1.7e-03 | 1.3e-03 | 1.7e-03 |

On the H100, vLLM's FP8 GEMM for fp32-scaled checkpoints (FlashInfer/DeepGEMM block, CUTLASS tensor) is W8A8 (X quantized to e4m3 a
token and 128 columns, or a token): its error is about 20 times ours. Ours is W8A16 (X kept bf16, weights exact): within bf16's
rounding of the exact product. For MXFP8 and DeepSeek's e8m0 scales vLLM uses Marlin (W8A16), the same error as ours.

The product is bound by the decode's instructions, not its MMAs (at 64 tokens 16 `mma.sync` a lane's unit of 16 values against a
decode of a few hundred instructions), so an e4m3 MMA's double rate buys little. A W8A8 build (X quantized once a step by its own
kernel, the decoded bytes straight into `mma.sync.m16n8k32` e4m3) was built and checked against the exact quantized product
(`493d0846`), and measured on the H100 only together with the staged decode (`46a53845`, run 5): slower at every token count. It was
removed with the staged decode (`9a003de6`). Decision: W8A16 on Ada and Hopper.

## 2. Variants measured (H100), and what was kept

Geometric mean of our time over the 89 shapes (us), H100:

| Run | Commit | Kernel | M=1 | M=8 | M=32 | M=64 | gate |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | :--- |
| 3 | c43379fe | lane decode, raw prmt, one level-3 window, literal scan only where present | 21.2 | 24.0 | 38.0 | 64.7 | 15 / 356 |
| 4 | f9281de4 | + pipelined: heads in shared memory, X ring of 2 patches, cp.async var/core a patch ahead | 21.9 | 24.6 | 40.5 | 60.7 | 15 / 356 |
| 5 | 46a53845 | staged decode (marks and spills dense, byte-permute gather) + W8A8 for block/tensor | 22.4 | 25.7 | 50.5 | 102.6 | 13 / 356 |

Run 3's kernel is the fastest at 1-32 tokens and is the one committed (`9a003de6` restores its device code byte for byte; the
H100 table below is its measurement). Run 1 measured vLLM only (its kernel choice then wrong: fixed in the bench); run 2's
bench stopped at FlashInfer's JIT link (`-lnvrtc`, fixed in the job).

## 3. Registers and spills (CUDA 13.0.2, -O3 -Xptxas -v, Docker)

0 bytes stack, 0 spills for every f8 kernel, both arches. `f8_linear_kernel<NTL, KIND>` registers (block / tensor / mx / ds32):

| NTL | sm_89 | sm_90a |
| :--- | :--- | :--- |
| 1 | 75 / 72 / 79 / 72 | 75 / 72 / 77 / 79 |
| 2 | 80 / 77 / 80 / 79 | 80 / 76 / 83 / 80 |
| 4 | 92 / 86 / 126 / 114 | 92 / 87 / 119 / 112 |
| 8 | 142 / 114 / 192 / 180 | 121 / 114 / 196 / 182 |

Static shared memory 128 B (sm_89) / 1,024 B (sm_90a); dynamic: the X tile (8 NTL rows of 256 B a patch, at most 64 KiB) and 14.5 KiB
of warp buffers, at most 77 KiB. unpack 48 regs (10,368 B / 11,264 B shared), pack 58 / 64 (10,368 / 11,264), head 90 / 90
(640 / 1,024), scan 44 / 30 (128 / 1,024). The library's other 242 functions: same ptxas lines as before this task on both
arches (diffed against the pre-task build); only the format's pack and head kernels changed (layout 2).

## 4. Correctness

`gpu/fp8/check_f8.py` (on a GPU: the library of the tree; `--emu`: the f8c text of `glyd_gpu.cu` compiled for the CPU and run thread by thread):

| Where | Code | Result |
| :--- | :--- | :--- |
| CPU emulation (`--emu --quick`, Mac) | 9a003de6 | ALL BIT FOR BIT: every test_fp8ref case through the emulated pack/unpack, strides, experts list, dense patches, `f8_linear` 32 products (every kind; plans: the library's, all K in one CTA, one patch a CTA, 8 tiles x a third of K) within bounds |
| RTX 4090, run 7 | 3bac5386 | ALL BIT FOR BIT (check exit 0): every case, the 50 real samples bit for bit (sizes as in section 1), strides, experts, dense, `f8_linear` 52 products within bf16's rounding of the exact product (worst 3.42e-3 of sum |x w|), two runs equal, graph replay on other inputs; case_graph (capture allocates no more than an empty capture) ok; unpack 265 us |
| L40S, run 6 | 9a003de6 | all ok except case_linear: "f8_linear: invalid argument" (the 99 KiB shared-memory limit Ada refuses: fixed in 3bac5386 and passed on the 4090); case_graph ok (the capture check fix works) |
| H100, run 3 | c43379fe (device code identical to the final) | every case ok, `f8_linear` 52 products within bounds (worst 3.42e-3); case_graph failed then on the check's own bug (torch allocates 1 KiB per capture for its generator's state, which the check counted as ours; fixed in 9a003de6) |

Also: `tests/test_fp8ref.py` all pass (layout 2; real samples bit for bit on the Mac); `test_gpu.py::test_c_header` passes; check_wheel's identifier
check on kernels.py, _lib.py, fp8ref.py finds nothing; numerics in section 1 (ours within bf16's rounding on every kind and both GPUs).

Not run: compute-sanitizer (racecheck, memcheck) on layout 2's kernels and on `f8_linear` (Task 2 ran it on layout 1's); the H100 has not
run the final commit's host code (3bac5386 changed only the shared-memory limit the launch asks for, 99 KiB to 77 KiB, and added the
development-only plan variant; the device code is c43379fe's).

## 5. Speed: ours vs vLLM 0.30's own FP8 GEMM

`gpu/fp8/bench_f8.py`: the five models' real shapes (TP 1, and 2/4/8 for the big ones: 89 shapes), M = 1, 8, 32, 64; weights from
the real samples (tiled), each runner on 200 MB of copies so neither reads from L2; one CUDA graph a runner, 20 replays, best of 5.
vLLM's kernel is the one its own selection picks for a served model of that checkpoint (`init_fp8_linear_kernel` /
`init_mxfp8_linear_kernel` with a model config, as `vllm serve` builds it).

Per-model summaries (ours / vLLM, geometric mean; in brackets the shapes at most 1):

RTX 4090:

| model | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | 1.04 (7/21) | 1.15 (2/21) | 1.60 (1/21) | 2.13 (1/21) |
| Qwen3.8-27B-FP8 | 1.16 (2/7) | 1.22 (0/7) | 1.74 (0/7) | 2.34 (0/7) |
| MiniMax-M3-MXFP8 | 1.06 (5/20) | 1.18 (2/20) | 1.68 (0/20) | 2.17 (0/20) |
| Mistral-Small-4-119B-2603 | 1.05 (5/16) | 1.25 (1/16) | 1.35 (0/16) | 1.79 (0/16) |
| DeepSeek-V4.1-Flash | 0.94 (15/25) | 1.08 (5/25) | 1.58 (0/25) | 2.10 (0/25) |

H100:

| model | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | 1.93 (1/21) | 2.14 (0/21) | 2.80 (0/21) | 4.19 (0/21) |
| Qwen3.8-27B-FP8 | 2.94 (0/7) | 3.06 (0/7) | 3.82 (0/7) | 5.71 (0/7) |
| MiniMax-M3-MXFP8 | 1.87 (2/20) | 2.07 (1/20) | 2.85 (0/20) | 3.89 (0/20) |
| Mistral-Small-4-119B-2603 | 1.66 (0/16) | 1.94 (0/16) | 3.04 (0/16) | 4.80 (0/16) |
| DeepSeek-V4.1-Flash | 1.45 (6/25) | 1.67 (5/25) | 2.42 (0/25) | 3.30 (0/25) |

Every shape (vLLM us / ours us; **ok** where ours is at most vLLM's). The 4090's "rows with errors" are the plan variant's (section 6), not the
default product's.

### RTX 4090 (run 7, commit 3bac5386): NVIDIA GeForce RTX 4090 (sm_89)

Gate: 46 of 356 at most vLLM's time; by tokens: M=1 34/89, M=8 10/89, M=32 1/89, M=64 1/89; worst ratio 2.83, geometric mean 1.40

vLLM's kernels: GLM-5.3-Flash (block): MarlinFP8ScaledMMLinearKernel; Qwen3.8-27B-FP8 (block): MarlinFP8ScaledMMLinearKernel; MiniMax-M3-MXFP8 (mx): MarlinMxfp8LinearKernel; Mistral-Small-4-119B-2603 (tensor): CutlassFP8ScaledMMLinearKernel; DeepSeek-V4.1-Flash (ds32): MarlinMxfp8LinearKernel

| model | tensor | tp | N x K | M=1 vLLM / ours | M=8 | M=32 | M=64 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | qkv_a | 1 | 2048 x 4096 | 13.3 / 13.6 | 13.3 / 15.2 | 14.7 / 20.4 | 18.9 / 35.1 |
| GLM-5.3-Flash | q_b | 1 | 16384 x 1536 | 29.5 / 28.7 **ok** | 29.5 / 30.8 | 33.2 / 52.1 | 37.8 / 72.0 |
| GLM-5.3-Flash | o | 1 | 4096 x 16384 | 74.7 / 87.0 | 74.9 / 89.7 | 76.7 / 133.3 | 81.9 / 231.5 |
| GLM-5.3-Flash | mlp_gate_up | 1 | 24576 x 4096 | 116.0 / 134.3 | 117.4 / 141.1 | 122.2 / 202.9 | 141.6 / 338.7 |
| GLM-5.3-Flash | mlp_down | 1 | 4096 x 12288 | 56.8 / 52.2 **ok** | 56.7 / 54.5 **ok** | 58.6 / 107.6 | 62.5 / 149.6 |
| GLM-5.3-Flash | expert_gate_up | 1 | 4096 x 4096 | 20.9 / 21.2 | 20.9 / 22.5 | 22.4 / 28.7 | 25.6 / 63.9 |
| GLM-5.3-Flash | expert_down | 1 | 4096 x 2048 | 12.0 / 13.2 | 12.0 / 14.4 | 13.3 / 19.5 | 16.1 / 33.5 |
| GLM-5.3-Flash | q_b | 2 | 8192 x 1536 | 17.0 / 16.9 **ok** | 17.0 / 18.2 | 17.0 / 23.6 | 19.1 / 41.7 |
| GLM-5.3-Flash | o | 2 | 4096 x 8192 | 38.8 / 39.1 | 38.8 / 41.0 | 40.3 / 81.9 | 44.1 / 111.3 |
| GLM-5.3-Flash | mlp_gate_up | 2 | 12288 x 4096 | 57.1 / 74.0 | 57.1 / 74.4 | 63.0 / 96.9 | 75.9 / 152.9 |
| GLM-5.3-Flash | mlp_down | 2 | 4096 x 6144 | 29.9 / 28.8 **ok** | 29.9 / 30.5 | 31.4 / 50.6 | 34.5 / 74.2 |
| GLM-5.3-Flash | expert_down | 2 | 4096 x 1024 | 7.5 / 8.9 | 7.5 / 10.0 | 8.8 / 14.7 | 11.7 / 24.5 |
| GLM-5.3-Flash | q_b | 4 | 4096 x 1536 | 9.9 / 12.0 | 9.9 / 13.7 | 11.2 / 22.0 | 14.0 / 39.6 |
| GLM-5.3-Flash | mlp_gate_up | 4 | 6144 x 4096 | 30.6 / 29.3 **ok** | 30.5 / 31.9 | 31.7 / 57.7 | 36.6 / 86.4 |
| GLM-5.3-Flash | mlp_down | 4 | 4096 x 3072 | 16.6 / 17.0 | 16.6 / 18.4 | 18.1 / 23.8 | 21.2 / 42.3 |
| GLM-5.3-Flash | expert_gate_up | 4 | 1024 x 4096 | 10.8 / 9.1 **ok** | 10.8 / 11.2 | 13.4 / 18.9 | 20.1 / 32.8 |
| GLM-5.3-Flash | expert_down | 4 | 4096 x 512 | 5.3 / 6.4 | 5.3 / 7.4 | 6.5 / 11.4 | 9.0 / 17.2 |
| GLM-5.3-Flash | q_b | 8 | 2048 x 1536 | 7.6 / 8.1 | 7.6 / 9.5 | 9.4 / 15.4 | 13.4 / 27.1 |
| GLM-5.3-Flash | mlp_gate_up | 8 | 3072 x 4096 | 16.5 / 17.4 | 16.4 / 19.6 | 18.9 / 29.7 | 23.3 / 56.3 |
| GLM-5.3-Flash | expert_gate_up | 8 | 512 x 4096 | 12.7 / 6.7 **ok** | 12.7 / 8.8 **ok** | 16.9 / 15.9 **ok** | 29.1 / 26.0 **ok** |
| GLM-5.3-Flash | expert_down | 8 | 4096 x 256 | 3.2 / 4.7 | 3.3 / 5.9 | 4.3 / 11.5 | 6.7 / 19.0 |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 82.0 / 93.4 | 82.3 / 97.5 | 88.1 / 149.5 | 102.0 / 225.9 |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 37.2 / 35.5 **ok** | 37.1 / 38.6 | 38.6 / 78.5 | 43.8 / 89.4 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 92.3 / 112.8 | 93.1 / 117.8 | 100.8 / 160.4 | 112.0 / 281.2 |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 59.3 / 71.2 | 59.2 / 75.2 | 64.7 / 105.4 | 79.6 / 161.3 |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 37.1 / 35.1 **ok** | 37.2 / 38.1 | 38.5 / 74.7 | 43.9 / 100.0 |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 194.5 / 245.4 | 195.9 / 256.4 | 209.9 / 355.1 | 227.4 / 624.2 |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 98.8 / 147.4 | 99.0 / 153.6 | 102.6 / 172.2 | 112.6 / 298.1 |
| MiniMax-M3-MXFP8 | qkv | 1 | 9216 x 6144 | 64.7 / 80.9 | 64.7 / 84.2 | 72.2 / 108.2 | 72.1 / 173.8 |
| MiniMax-M3-MXFP8 | o | 1 | 6144 x 8192 | 58.1 / 71.9 | 58.1 / 76.4 | 60.1 / 108.2 | 65.8 / 159.7 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 1 | 24576 x 6144 | 175.1 / 206.1 | 176.6 / 218.6 | 183.7 / 307.5 | 194.3 / 541.4 |
| MiniMax-M3-MXFP8 | mlp_down | 1 | 6144 x 12288 | 85.9 / 132.4 | 86.1 / 137.5 | 89.1 / 155.6 | 97.5 / 250.5 |
| MiniMax-M3-MXFP8 | expert_gate_up | 1 | 6144 x 6144 | 44.4 / 47.2 | 44.5 / 50.2 | 46.3 / 82.6 | 52.1 / 106.7 |
| MiniMax-M3-MXFP8 | expert_down | 1 | 6144 x 3072 | 24.1 / 25.2 | 24.1 / 27.8 | 25.3 / 44.0 | 30.9 / 58.6 |
| MiniMax-M3-MXFP8 | index_q | 1 | 512 x 6144 | 15.4 / 8.4 **ok** | 15.4 / 11.1 **ok** | 18.1 / 21.2 | 29.9 / 38.3 |
| MiniMax-M3-MXFP8 | qkv | 2 | 4608 x 6144 | 34.2 / 35.1 | 34.2 / 36.8 | 35.6 / 68.9 | 42.1 / 88.6 |
| MiniMax-M3-MXFP8 | o | 2 | 6144 x 4096 | 31.0 / 30.1 **ok** | 31.0 / 32.6 | 32.4 / 58.1 | 38.0 / 88.2 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 2 | 12288 x 6144 | 85.8 / 131.1 | 86.3 / 135.9 | 93.9 / 156.4 | 98.4 / 235.4 |
| MiniMax-M3-MXFP8 | expert_gate_up | 2 | 3072 x 6144 | 23.7 / 24.4 | 23.7 / 27.0 | 26.2 / 47.3 | 31.5 / 60.5 |
| MiniMax-M3-MXFP8 | expert_down | 2 | 6144 x 1536 | 13.6 / 14.8 | 13.6 / 16.5 | 14.6 / 28.5 | 18.8 / 50.2 |
| MiniMax-M3-MXFP8 | qkv | 4 | 2304 x 6144 | 19.2 / 20.1 | 19.2 / 22.0 | 21.1 / 30.3 | 26.2 / 57.6 |
| MiniMax-M3-MXFP8 | o | 4 | 6144 x 2048 | 17.1 / 17.9 | 17.2 / 20.1 | 18.5 / 29.0 | 23.0 / 53.6 |
| MiniMax-M3-MXFP8 | expert_gate_up | 4 | 1536 x 6144 | 15.5 / 15.3 **ok** | 15.6 / 17.0 | 17.3 / 23.6 | 22.1 / 43.2 |
| MiniMax-M3-MXFP8 | expert_down | 4 | 6144 x 768 | 8.6 / 9.7 | 8.6 / 11.6 | 9.7 / 21.9 | 14.0 / 37.4 |
| MiniMax-M3-MXFP8 | qkv | 8 | 1152 x 6144 | 14.0 / 12.7 **ok** | 14.0 / 14.7 | 15.7 / 22.4 | 22.2 / 39.6 |
| MiniMax-M3-MXFP8 | o | 8 | 6144 x 1024 | 10.3 / 11.8 | 10.3 / 12.7 | 11.1 / 18.4 | 15.6 / 32.1 |
| MiniMax-M3-MXFP8 | expert_gate_up | 8 | 768 x 6144 | 13.3 / 10.8 **ok** | 13.3 / 12.7 **ok** | 15.4 / 20.7 | 23.9 / 37.8 |
| MiniMax-M3-MXFP8 | expert_down | 8 | 6144 x 384 | 6.5 / 7.3 | 6.5 / 8.7 | 6.2 / 15.3 | 9.4 / 26.3 |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 10.4 / 10.9 | 10.5 / 12.5 | 13.3 / 18.7 | 20.4 / 32.8 |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 8.2 / 8.9 | 8.2 / 10.0 | 12.0 / 14.7 | 13.6 / 21.8 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 5.4 / 5.6 | 5.4 / 6.9 | 10.8 / 12.4 | 11.9 / 20.2 |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 21.6 / 20.7 **ok** | 21.7 / 22.2 | 25.7 / 28.3 | 27.7 / 63.1 |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 12.8 / 12.9 | 12.8 / 14.1 | 16.6 / 19.3 | 18.3 / 33.4 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 5.8 / 6.5 | 5.8 / 7.5 | 8.4 / 11.8 | 10.8 / 17.9 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 3.5 / 4.6 | 3.6 / 5.9 | 7.3 / 11.3 | 8.4 / 18.9 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 12.7 / 12.9 | 12.8 / 14.4 | 15.5 / 20.3 | 21.0 / 35.0 |
| Mistral-Small-4-119B-2603 | q_b | 4 | 1024 x 1024 | 4.7 / 4.6 **ok** | 4.8 / 5.7 | 7.6 / 10.0 | 10.2 / 16.0 |
| Mistral-Small-4-119B-2603 | kv_b | 4 | 1536 x 256 | 3.4 / 4.5 | 3.4 / 5.8 | 6.0 / 11.2 | 7.3 / 18.7 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 4 | 1024 x 4096 | 10.0 / 9.1 **ok** | 10.1 / 11.2 | 13.0 / 18.9 | 20.3 / 29.8 |
| Mistral-Small-4-119B-2603 | expert_down | 4 | 4096 x 512 | 6.2 / 6.4 | 6.2 / 7.4 | 9.9 / 11.5 | 11.6 / 17.3 |
| Mistral-Small-4-119B-2603 | q_b | 8 | 512 x 1024 | 4.5 / 4.5 **ok** | 4.5 / 5.5 | 7.5 / 9.8 | 9.9 / 15.8 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | 768 x 256 | 3.3 / 4.5 | 3.4 / 5.8 | 6.0 / 11.2 | 7.1 / 18.7 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 8 | 512 x 4096 | 9.7 / 6.7 **ok** | 9.8 / 8.7 **ok** | 12.8 / 15.9 | 20.2 / 25.9 |
| Mistral-Small-4-119B-2603 | expert_down | 8 | 4096 x 256 | 3.9 / 4.7 | 4.0 / 6.0 | 8.3 / 11.5 | 9.6 / 19.1 |
| DeepSeek-V4.1-Flash | wq_a | 1 | 1280 x 5120 | 12.8 / 12.2 **ok** | 12.8 / 14.2 | 14.4 / 21.7 | 20.4 / 38.5 |
| DeepSeek-V4.1-Flash | wkv | 1 | 512 x 5120 | 13.7 / 7.9 **ok** | 13.7 / 10.3 **ok** | 16.0 / 19.6 | 26.3 / 35.6 |
| DeepSeek-V4.1-Flash | wq_b | 1 | 32768 x 1280 | 47.9 / 44.0 **ok** | 48.0 / 45.1 **ok** | 50.5 / 90.2 | 55.1 / 129.7 |
| DeepSeek-V4.1-Flash | wo_a | 1 | 8192 x 4096 | 39.8 / 39.8 **ok** | 39.8 / 41.6 | 40.1 / 73.8 | 44.1 / 106.1 |
| DeepSeek-V4.1-Flash | wo_b | 1 | 5120 x 8192 | 49.0 / 50.8 | 49.0 / 53.2 | 51.0 / 87.7 | 56.8 / 132.9 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 1 | 4096 x 1280 | 9.0 / 11.2 | 9.0 / 12.2 | 10.2 / 19.6 | 12.9 / 35.4 |
| DeepSeek-V4.1-Flash | shared_w13 | 1 | 4608 x 5120 | 29.2 / 28.9 **ok** | 29.3 / 30.6 | 30.4 / 60.7 | 36.1 / 84.8 |
| DeepSeek-V4.1-Flash | shared_w2 | 1 | 5120 x 2304 | 17.0 / 16.8 **ok** | 16.9 / 19.1 | 17.6 / 28.3 | 21.4 / 50.7 |
| DeepSeek-V4.1-Flash | wq_b | 2 | 16384 x 1280 | 25.5 / 25.3 **ok** | 25.5 / 26.9 | 29.4 / 47.0 | 28.4 / 64.7 |
| DeepSeek-V4.1-Flash | wo_a | 2 | 4096 x 4096 | 21.3 / 21.6 | 21.3 / 23.0 | 22.8 / 29.7 | 25.8 / 65.6 |
| DeepSeek-V4.1-Flash | wo_b | 2 | 5120 x 4096 | 26.3 / 25.5 **ok** | 26.3 / 27.9 | 27.7 / 53.2 | 33.8 / 78.3 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 2 | 2048 x 1280 | 7.2 / 7.8 | 7.2 / 9.7 | 8.6 / 17.5 | 12.8 / 31.5 |
| DeepSeek-V4.1-Flash | shared_w13 | 2 | 2304 x 5120 | 16.6 / 16.9 | 16.6 / 19.5 | 18.6 / 30.0 | 22.7 / 55.1 |
| DeepSeek-V4.1-Flash | shared_w2 | 2 | 5120 x 1152 | 10.5 / 11.3 | 10.5 / 12.7 | 11.0 / 20.8 | 14.9 / 35.7 |
| DeepSeek-V4.1-Flash | wq_b | 4 | 8192 x 1280 | 14.9 / 16.0 | 14.9 / 17.3 | 15.1 / 23.1 | 18.7 / 41.5 |
| DeepSeek-V4.1-Flash | wo_a | 4 | 2048 x 4096 | 13.5 / 13.4 **ok** | 13.5 / 14.9 | 15.0 / 21.1 | 18.9 / 36.1 |
| DeepSeek-V4.1-Flash | wo_b | 4 | 5120 x 2048 | 15.0 / 16.3 | 15.0 / 18.6 | 16.2 / 27.3 | 20.9 / 50.0 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 4 | 1024 x 1280 | 6.5 / 5.2 **ok** | 6.5 / 7.1 | 7.9 / 14.7 | 12.1 / 25.2 |
| DeepSeek-V4.1-Flash | shared_w13 | 4 | 1152 x 5120 | 12.6 / 11.7 **ok** | 12.6 / 13.5 | 14.5 / 21.2 | 20.8 / 37.9 |
| DeepSeek-V4.1-Flash | shared_w2 | 4 | 5120 x 576 | 7.1 / 8.1 | 7.1 / 9.6 | 12.6 / 16.1 | 11.8 / 27.3 |
| DeepSeek-V4.1-Flash | wo_a | 8 | 1024 x 4096 | 11.6 / 9.3 **ok** | 11.6 / 11.4 **ok** | 13.5 / 19.1 | 20.1 / 33.3 |
| DeepSeek-V4.1-Flash | wo_b | 8 | 5120 x 1024 | 9.3 / 11.0 | 9.3 / 11.9 | 10.1 / 17.7 | 14.9 / 31.5 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 8 | 512 x 1280 | 8.3 / 4.8 **ok** | 8.3 / 6.7 **ok** | 10.1 / 14.3 | 17.3 / 24.6 |
| DeepSeek-V4.1-Flash | shared_w13 | 8 | 576 x 5120 | 14.0 / 8.3 **ok** | 14.0 / 10.9 **ok** | 16.4 / 20.7 | 26.5 / 36.3 |
| DeepSeek-V4.1-Flash | shared_w2 | 8 | 5120 x 288 | 7.1 / 6.4 **ok** | 7.7 / 7.7 | 12.9 / 13.3 | 11.7 / 21.3 |

62 rows with errors, e.g. GLM-5.3-Flash q_b: variant: RuntimeError: f8_linear: invalid argument

### H100 (run 3, kernel c43379fe: the committed device code): NVIDIA H100 80GB HBM3 (sm_90)

Gate: 15 of 356 at most vLLM's time; by tokens: M=1 9/89, M=8 6/89, M=32 0/89, M=64 0/89; worst ratio 7.30, geometric mean 2.52

vLLM's kernels: GLM-5.3-Flash (block): FlashInferFp8DeepGEMMDynamicBlockScaledKernel[FlashInferFp8BlockScaledMMKernel<32<=DeepGemmFp8BlockScaledMMKernel]; Qwen3.8-27B-FP8 (block): FlashInferFp8DeepGEMMDynamicBlockScaledKernel[FlashInferFp8BlockScaledMMKernel<32<=DeepGemmFp8BlockScaledMMKernel]; MiniMax-M3-MXFP8 (mx): MarlinMxfp8LinearKernel; Mistral-Small-4-119B-2603 (tensor): CutlassFP8ScaledMMLinearKernel; DeepSeek-V4.1-Flash (ds32): MarlinMxfp8LinearKernel

| model | tensor | tp | N x K | M=1 vLLM / ours | M=8 | M=32 | M=64 |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| GLM-5.3-Flash | qkv_a | 1 | 2048 x 4096 | 10.0 / 14.6 | 9.9 / 16.2 | 11.4 / 23.9 | 11.1 / 34.3 |
| GLM-5.3-Flash | q_b | 1 | 16384 x 1536 | 13.1 / 31.0 | 13.3 / 32.9 | 15.5 / 54.5 | 15.6 / 65.5 |
| GLM-5.3-Flash | o | 1 | 4096 x 16384 | 30.6 / 79.1 | 31.0 / 83.8 | 33.9 / 131.8 | 36.7 / 210.0 |
| GLM-5.3-Flash | mlp_gate_up | 1 | 24576 x 4096 | 39.1 / 105.1 | 39.9 / 110.2 | 43.3 / 164.3 | 43.8 / 243.2 |
| GLM-5.3-Flash | mlp_down | 1 | 4096 x 12288 | 24.7 / 64.7 | 24.8 / 66.6 | 26.9 / 105.4 | 29.0 / 145.9 |
| GLM-5.3-Flash | expert_gate_up | 1 | 4096 x 4096 | 11.3 / 24.5 | 11.3 / 26.0 | 13.3 / 34.9 | 13.6 / 71.2 |
| GLM-5.3-Flash | expert_down | 1 | 4096 x 2048 | 7.9 / 14.6 | 8.0 / 16.0 | 10.1 / 23.2 | 10.0 / 33.2 |
| GLM-5.3-Flash | q_b | 2 | 8192 x 1536 | 9.3 / 19.5 | 9.3 / 20.9 | 11.3 / 28.0 | 11.2 / 38.6 |
| GLM-5.3-Flash | o | 2 | 4096 x 8192 | 18.0 / 42.9 | 17.8 / 45.0 | 20.0 / 87.2 | 21.1 / 119.7 |
| GLM-5.3-Flash | mlp_gate_up | 2 | 12288 x 4096 | 22.4 / 71.1 | 22.9 / 74.2 | 25.1 / 91.2 | 25.4 / 139.0 |
| GLM-5.3-Flash | mlp_down | 2 | 4096 x 6144 | 14.5 / 33.7 | 14.5 / 35.4 | 16.7 / 45.8 | 17.3 / 71.6 |
| GLM-5.3-Flash | expert_down | 2 | 4096 x 1024 | 6.4 / 10.0 | 6.4 / 11.3 | 8.2 / 17.8 | 8.3 / 26.9 |
| GLM-5.3-Flash | q_b | 4 | 4096 x 1536 | 7.4 / 14.7 | 7.4 / 17.2 | 9.4 / 29.5 | 9.2 / 44.5 |
| GLM-5.3-Flash | mlp_gate_up | 4 | 6144 x 4096 | 13.5 / 35.0 | 13.5 / 37.7 | 16.1 / 52.2 | 16.2 / 83.9 |
| GLM-5.3-Flash | mlp_down | 4 | 4096 x 3072 | 9.7 / 19.4 | 9.7 / 21.2 | 11.7 / 29.1 | 11.8 / 39.6 |
| GLM-5.3-Flash | expert_gate_up | 4 | 1024 x 4096 | 9.2 / 10.3 | 9.1 / 12.7 | 11.1 / 23.6 | 11.1 / 38.5 |
| GLM-5.3-Flash | expert_down | 4 | 4096 x 512 | 5.6 / 8.2 | 5.6 / 9.5 | 7.3 / 15.3 | 7.2 / 22.5 |
| GLM-5.3-Flash | q_b | 8 | 2048 x 1536 | 6.9 / 9.9 | 7.0 / 12.0 | 8.5 / 21.4 | 8.2 / 34.2 |
| GLM-5.3-Flash | mlp_gate_up | 8 | 3072 x 4096 | 10.6 / 20.1 | 10.5 / 22.9 | 12.6 / 35.9 | 12.7 / 54.6 |
| GLM-5.3-Flash | expert_gate_up | 8 | 512 x 4096 | 9.1 / 8.5 **ok** | 9.1 / 10.8 | 10.9 / 21.6 | 10.8 / 36.0 |
| GLM-5.3-Flash | expert_down | 8 | 4096 x 256 | 5.1 / 8.2 | 5.1 / 9.8 | 6.5 / 17.2 | 6.6 / 27.4 |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 29.8 / 86.7 | 30.3 / 90.5 | 33.6 / 133.4 | 34.4 / 197.9 |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 16.1 / 45.5 | 16.4 / 50.2 | 19.8 / 80.5 | 19.8 / 91.8 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 33.0 / 85.4 | 33.4 / 90.2 | 36.6 / 136.0 | 37.6 / 216.8 |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 23.6 / 69.0 | 23.9 / 71.5 | 26.5 / 100.4 | 27.0 / 156.5 |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 15.8 / 43.3 | 16.1 / 45.7 | 18.8 / 71.3 | 19.1 / 114.5 |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 68.1 / 244.8 | 68.9 / 258.2 | 70.7 / 288.4 | 72.1 / 452.2 |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 37.9 / 116.6 | 38.5 / 121.5 | 43.8 / 149.1 | 46.2 / 273.2 |
| MiniMax-M3-MXFP8 | qkv | 1 | 9216 x 6144 | 26.4 / 84.1 | 26.6 / 86.2 | 31.7 / 103.5 | 40.2 / 215.4 |
| MiniMax-M3-MXFP8 | o | 1 | 6144 x 8192 | 24.5 / 68.1 | 24.6 / 72.4 | 28.9 / 102.2 | 39.1 / 219.3 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 1 | 24576 x 6144 | 59.0 / 156.0 | 59.3 / 162.2 | 64.5 / 245.3 | 76.9 / 525.8 |
| MiniMax-M3-MXFP8 | mlp_down | 1 | 6144 x 12288 | 33.8 / 108.3 | 33.8 / 113.9 | 39.9 / 142.8 | 53.1 / 295.0 |
| MiniMax-M3-MXFP8 | expert_gate_up | 1 | 6144 x 6144 | 19.9 / 56.3 | 19.9 / 60.4 | 22.9 / 86.9 | 31.6 / 152.2 |
| MiniMax-M3-MXFP8 | expert_down | 1 | 6144 x 3072 | 13.0 / 31.2 | 12.9 / 34.0 | 14.6 / 57.1 | 20.5 / 76.4 |
| MiniMax-M3-MXFP8 | index_q | 1 | 512 x 6144 | 20.7 / 10.2 **ok** | 20.8 / 13.7 **ok** | 23.9 / 28.2 | 34.5 / 53.3 |
| MiniMax-M3-MXFP8 | qkv | 2 | 4608 x 6144 | 17.5 / 44.1 | 17.5 / 45.9 | 19.0 / 70.1 | 26.7 / 125.5 |
| MiniMax-M3-MXFP8 | o | 2 | 6144 x 4096 | 15.5 / 34.6 | 15.4 / 37.5 | 17.3 / 52.7 | 24.2 / 119.5 |
| MiniMax-M3-MXFP8 | mlp_gate_up | 2 | 12288 x 6144 | 32.9 / 106.1 | 33.2 / 110.9 | 37.7 / 140.2 | 46.4 / 269.3 |
| MiniMax-M3-MXFP8 | expert_gate_up | 2 | 3072 x 6144 | 13.8 / 29.1 | 13.7 / 32.2 | 15.0 / 47.1 | 21.2 / 79.0 |
| MiniMax-M3-MXFP8 | expert_down | 2 | 6144 x 1536 | 10.0 / 18.9 | 10.1 / 21.7 | 11.0 / 38.3 | 15.8 / 66.8 |
| MiniMax-M3-MXFP8 | qkv | 4 | 2304 x 6144 | 13.0 / 24.2 | 13.0 / 26.6 | 14.5 / 38.2 | 20.5 / 71.8 |
| MiniMax-M3-MXFP8 | o | 4 | 6144 x 2048 | 11.1 / 20.4 | 11.0 / 23.1 | 12.2 / 35.4 | 17.2 / 66.1 |
| MiniMax-M3-MXFP8 | expert_gate_up | 4 | 1536 x 6144 | 13.0 / 19.1 | 12.9 / 20.9 | 14.3 / 29.3 | 20.3 / 54.3 |
| MiniMax-M3-MXFP8 | expert_down | 4 | 6144 x 768 | 7.4 / 11.9 | 7.3 / 14.4 | 9.6 / 28.9 | 13.6 / 51.7 |
| MiniMax-M3-MXFP8 | qkv | 8 | 1152 x 6144 | 13.8 / 14.7 | 13.9 / 17.0 | 15.7 / 28.0 | 22.1 / 52.6 |
| MiniMax-M3-MXFP8 | o | 8 | 6144 x 1024 | 9.3 / 14.2 | 9.3 / 15.5 | 10.2 / 22.4 | 14.1 / 41.2 |
| MiniMax-M3-MXFP8 | expert_gate_up | 8 | 768 x 6144 | 16.0 / 14.2 **ok** | 15.9 / 16.6 | 18.1 / 27.7 | 25.8 / 51.4 |
| MiniMax-M3-MXFP8 | expert_down | 8 | 6144 x 384 | 7.5 / 9.5 | 7.5 / 11.3 | 5.7 / 19.6 | 7.2 / 35.2 |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 7.5 / 13.9 | 7.5 / 15.5 | 8.9 / 22.9 | 8.8 / 33.3 |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 5.9 / 9.9 | 6.0 / 11.3 | 6.3 / 17.6 | 6.3 / 26.7 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 5.1 / 8.2 | 5.1 / 9.9 | 5.3 / 17.1 | 5.5 / 27.1 |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 10.5 / 23.7 | 10.7 / 25.5 | 11.5 / 33.8 | 10.8 / 69.3 |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 7.4 / 14.4 | 7.5 / 15.8 | 7.8 / 22.7 | 7.7 / 32.7 |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 5.4 / 8.3 | 5.4 / 9.7 | 5.8 / 15.6 | 5.8 / 24.2 |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 4.8 / 7.8 | 4.8 / 9.6 | 5.0 / 16.8 | 5.1 / 27.1 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 8.0 / 14.4 | 8.0 / 16.0 | 9.6 / 23.6 | 9.3 / 33.9 |
| Mistral-Small-4-119B-2603 | q_b | 4 | 1024 x 1024 | 5.2 / 8.0 | 5.3 / 9.4 | 5.5 / 15.5 | 5.5 / 24.1 |
| Mistral-Small-4-119B-2603 | kv_b | 4 | 1536 x 256 | 3.8 / 6.9 | 3.9 / 8.5 | 4.1 / 15.7 | 4.2 / 25.6 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 4 | 1024 x 4096 | 7.1 / 10.2 | 7.3 / 12.7 | 7.2 / 23.5 | 7.3 / 38.2 |
| Mistral-Small-4-119B-2603 | expert_down | 4 | 4096 x 512 | 5.3 / 8.2 | 5.3 / 9.6 | 5.6 / 15.3 | 5.7 / 22.3 |
| Mistral-Small-4-119B-2603 | q_b | 8 | 512 x 1024 | 4.5 / 6.9 | 4.6 / 8.2 | 4.6 / 14.5 | 4.8 / 23.3 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | 768 x 256 | 3.8 / 6.8 | 3.9 / 8.4 | 4.0 / 15.3 | 4.0 / 25.3 |
| Mistral-Small-4-119B-2603 | expert_gate_up | 8 | 512 x 4096 | 6.8 / 8.5 | 7.0 / 11.0 | 6.9 / 21.5 | 7.1 / 35.9 |
| Mistral-Small-4-119B-2603 | expert_down | 8 | 4096 x 256 | 4.9 / 8.1 | 5.0 / 9.7 | 5.1 / 17.0 | 5.3 / 27.1 |
| DeepSeek-V4.1-Flash | wq_a | 1 | 1280 x 5120 | 12.6 / 14.8 | 12.5 / 17.1 | 14.1 / 27.7 | 19.9 / 51.3 |
| DeepSeek-V4.1-Flash | wkv | 1 | 512 x 5120 | 18.6 / 10.2 **ok** | 18.6 / 13.5 **ok** | 21.1 / 27.4 | 30.8 / 51.5 |
| DeepSeek-V4.1-Flash | wq_b | 1 | 32768 x 1280 | 20.5 / 51.6 | 20.9 / 54.9 | 20.0 / 77.9 | 24.2 / 176.4 |
| DeepSeek-V4.1-Flash | wo_a | 1 | 8192 x 4096 | 17.1 / 41.5 | 17.2 / 44.1 | 17.8 / 72.5 | 24.2 / 147.5 |
| DeepSeek-V4.1-Flash | wo_b | 1 | 5120 x 8192 | 21.6 / 55.1 | 22.0 / 59.3 | 24.8 / 92.4 | 34.0 / 190.1 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 1 | 4096 x 1280 | 6.9 / 14.2 | 7.0 / 16.4 | 8.2 / 25.9 | 10.9 / 46.2 |
| DeepSeek-V4.1-Flash | shared_w13 | 1 | 4608 x 5120 | 15.8 / 38.4 | 15.8 / 40.1 | 16.6 / 60.6 | 23.3 / 123.6 |
| DeepSeek-V4.1-Flash | shared_w2 | 1 | 5120 x 2304 | 10.5 / 20.0 | 10.4 / 22.7 | 12.1 / 35.6 | 17.0 / 66.4 |
| DeepSeek-V4.1-Flash | wq_b | 2 | 16384 x 1280 | 11.1 / 26.5 | 11.1 / 28.5 | 13.8 / 45.7 | 18.3 / 83.5 |
| DeepSeek-V4.1-Flash | wo_a | 2 | 4096 x 4096 | 11.5 / 24.6 | 11.5 / 26.3 | 12.5 / 35.3 | 16.9 / 87.7 |
| DeepSeek-V4.1-Flash | wo_b | 2 | 5120 x 4096 | 14.5 / 30.6 | 14.6 / 33.7 | 15.6 / 47.3 | 21.8 / 103.6 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 2 | 2048 x 1280 | 8.3 / 9.9 | 8.3 / 12.6 | 9.6 / 23.9 | 13.5 / 44.4 |
| DeepSeek-V4.1-Flash | shared_w13 | 2 | 2304 x 5120 | 12.1 / 20.5 | 12.0 / 24.5 | 13.5 / 40.2 | 18.9 / 75.4 |
| DeepSeek-V4.1-Flash | shared_w2 | 2 | 5120 x 1152 | 8.7 / 14.6 | 8.7 / 16.6 | 9.7 / 26.4 | 13.8 / 48.1 |
| DeepSeek-V4.1-Flash | wq_b | 4 | 8192 x 1280 | 9.1 / 19.2 | 9.1 / 21.3 | 8.4 / 35.4 | 11.2 / 61.0 |
| DeepSeek-V4.1-Flash | wo_a | 4 | 2048 x 4096 | 10.2 / 14.7 | 10.2 / 16.3 | 11.6 / 24.3 | 15.8 / 44.7 |
| DeepSeek-V4.1-Flash | wo_b | 4 | 5120 x 2048 | 11.1 / 20.0 | 11.3 / 22.7 | 11.6 / 35.3 | 16.3 / 65.8 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 4 | 1024 x 1280 | 8.5 / 8.3 **ok** | 8.6 / 11.1 | 9.9 / 22.8 | 14.2 / 38.5 |
| DeepSeek-V4.1-Flash | shared_w13 | 4 | 1152 x 5120 | 13.4 / 14.7 | 13.3 / 17.0 | 15.1 / 27.6 | 21.5 / 51.2 |
| DeepSeek-V4.1-Flash | shared_w2 | 4 | 5120 x 576 | 7.2 / 9.9 | 7.2 / 11.8 | 13.1 / 20.4 | 18.8 / 37.9 |
| DeepSeek-V4.1-Flash | wo_a | 8 | 1024 x 4096 | 12.9 / 10.4 **ok** | 12.9 / 12.9 **ok** | 15.1 / 23.8 | 21.0 / 44.2 |
| DeepSeek-V4.1-Flash | wo_b | 8 | 5120 x 1024 | 9.1 / 14.4 | 9.0 / 15.6 | 9.8 / 22.5 | 13.6 / 40.9 |
| DeepSeek-V4.1-Flash | indexer_wq_b | 8 | 512 x 1280 | 13.2 / 7.8 **ok** | 13.3 / 10.5 **ok** | 15.5 / 21.9 | 22.3 / 37.7 |
| DeepSeek-V4.1-Flash | shared_w13 | 8 | 576 x 5120 | 19.0 / 10.2 **ok** | 18.9 / 13.5 **ok** | 21.3 / 27.5 | 31.1 / 51.7 |
| DeepSeek-V4.1-Flash | shared_w2 | 8 | 5120 x 288 | 9.3 / 8.3 **ok** | 10.2 / 10.1 **ok** | 15.5 / 17.7 | 21.1 / 28.7 |

## 6. Where the gate passes and fails, and why

**H100: fails.** 15 of 356 (M=1 9/89, M=8 6/89, M=32 0, M=64 0), geometric mean 2.52, worst 7.30. Ours wins only where vLLM's kernel is
latency-bound (MiniMax index_q 512 x 6144: 0.49 at M=1; DeepSeek wkv 512 x 5120: 0.52; small TP8 slices). On large matrices the product is
instruction-bound: Qwen gate_up 34816 x 5120 at M=1 is 245 us against vLLM's 68 us (vLLM reads its 178 MB at 2.6 TB/s); 178.3 M values in
245 us on 132 SMs x 4 schedulers at about 1.75 GHz is about 1.3 scheduler cycles a value. To match vLLM the decode would have to be about 3.6
times cheaper; the common path alone (digit select, the OR into the rotated sign/mantissa word, the bf16 spread, the MMA feed) is several
integer instructions a value before any spill (27% of values spill at least once). Three restructurings measured on the H100 (pipelined,
staged, W8A8) were slower than run 3's kernel at 1-32 tokens (the pipelined one 6% faster at 64, all still 3-6x over vLLM there). My conclusion: with this format, no tuning of the decode reaches vLLM's FP8 GEMM on the
H100 at large shapes; the H100 gate cannot pass. (ncu was not available: Runpod refuses the performance counters, ERR_NVGPUCTRPERM.)

**RTX 4090: fails, but close at 1-8 tokens.** 46 of 356 (M=1 34/89, M=8 10/89, M=32 1/89, M=64 1/89), geometric mean 1.40, worst 2.83. By
model at M=1: 0.94 (DeepSeek) to 1.16 (Qwen); at M=8: 1.08-1.25; at M=32: 1.35-1.74; at M=64: 1.79-2.34. vLLM there is Marlin W8A16 at
about 0.9 TB/s; our decode on Ada keeps up with the 4090's third-as-fast memory at 1-8 tokens on many shapes. At 32-64 tokens the token tiles'
accumulators take the registers (NTL 8: 114-196 registers a thread, one CTA of 8 warps an SM; NTL 4: 86-126, two) and the decode's latencies
are not hidden: from M=8 to M=64 our time grows 2.6x (geometric mean; 1.7-3.7x) where vLLM's grows 1.5x (1.1-2.4x).

**The plan variant (GLYD_F8_LIN_CTAS=3, run 7).** It sizes a CTA's K range so that 3 CTAs (2 at NTL 4, 1 at NTL 8) fit an SM's shared memory
and picks the split by waves of CTAs. Where it changed the plan and ran: 8% faster at M=1, 4% at M=8, 25% at M=32, 16% at M=64 (best 0.60 of
the default's time); 11 shapes went from over vLLM's time to under it (e.g. GLM mlp_gate_up TP2 M=1: 1.29 to 0.97; MiniMax o M=1: 1.24 to
0.98), one went over (GLM q_b TP2 M=1: 1.00 to 1.03). With it, 49 of the 294 products it ran on were at most vLLM's. But the library refused it
("f8_linear: invalid argument") on 62 of 356 products, mostly the largest shapes (GLM o/gate_up/down TP1, all of Qwen's big ones, MiniMax
qkv/mlp): the default plan never failed there, the same kernel ran, and I did not find the cause from the logs (the bench's variant branch
prints no traceback). It stays off by default and is listed below.

## 7. Pods

All on Runpod through `scratchpad/rp_run.sh`, names `glyd-bench-f8f-*`, one at a time, each deleted at the end.

| # | Name | GPU | Minutes | What it answered |
| :--- | :--- | :--- | ---: | :--- |
| 1 | f8f-h1 | H100 80GB HBM3 | 5 | vLLM's times (its kernel choice wrong: no model config; fixed) |
| 2 | f8f-h2 | H100 | 5 | first fused kernel: checks pass, unpack 171 us; bench stopped at FlashInfer's link (fixed) |
| 3 | f8f-h3 | H100 | 9 | kernel c43379fe: checks, full bench with numerics (the H100 table) |
| 4 | f8f-h4 | H100 | 9 | pipelined variant: slower at 1-32 tokens; ncu not permitted on Runpod |
| 5 | f8f-h5 | H100 | 10 | staged decode + W8A8: slower |
| 6 | f8f-a1 | L40S | 7 | Ada: check found the launch bug (99 KiB limit refused on Ada); partial bench; stopped early |
| 7 | f8f-a2 | RTX 4090 | 21 | Ada: every check bit for bit, the full table, the plan variant (14 of the 21 minutes were the host's slow pip install) |

Plus failed creates (Runpod graphql errors, no pod made). Total about 66 pod minutes.

Why pod 6 (past the 5th): no previous run measured Ada; Ada's balance of memory bandwidth to issue rate (about 0.9 TB/s against
H100's 3.35 TB/s, a similar issue rate) is where the fused decode can match a memory-bound GEMM, and Ada's sm_89 binary of the
product had never run. Estimate before it: from run 3's H100 cycles a value, about 0.9x vLLM's Marlin time at 1-8 tokens on large
shapes, 1.3-2x at 32-64.

Why pod 7: pod 6's check found that every launch needing more than 48 KiB of dynamic shared memory failed on Ada (the kernel asked
the driver for 99 KiB, which Ada refuses once the kernel's static shared memory is added; Hopper allows 227 KiB, so runs 2-5 never
saw it). Its bench therefore had no Ada time for those plans and no Ada correctness for them: it could not answer the question. Pod
7 ran the fixed library, and in the same job a plan variant chosen from pod 6's partial table (where 3 CTAs fit an SM, ratios
0.93-0.97 at M=1; where 1-2 fit, 1.0-1.25).

## 8. What is left

- **A decision on the gate** (owner): the H100 cannot reach vLLM's time with this format's decode (section 6); the 4090 is within 0-25% at
  1-8 tokens and 35-134% over at 32-64. Options as I see them: (a) use the fused product only where it is at most vLLM's (Ada, few tokens,
  measured shapes) and vLLM's own GEMM on unpacked weights elsewhere, which costs the memory saving where it is used; (b) accept slower
  GEMMs for 16% less weight memory (more KV cache), judged on a whole-model run; (c) research a cheaper-to-decode format, which Task 1's
  study says cannot keep 15% (a fixed window is 11%).
- On Ada: the occupancy-aware plan (GLYD_F8_LIN_CTAS) helps where it ran; find why it was refused on 62 products, then make it the default
  if a full run confirms. M=32/64: fewer registers at NTL 4/8 (the block kind's per-granule temporaries, the mx/ds32 per-unit ones), or
  splitting the token tiles across warps sharing a decoded granule.
- compute-sanitizer racecheck/memcheck on layout 2's kernels and `f8_linear`.
- The unpack on the L40S runs at 348 GB/s written (instruction-bound; the 4090 and the H100 are at 498 and 887): it could use the product's
  row-block decode.
- MoE routed experts (a grouped product over expert ids), vLLM integration (Task 6), B200 and A100 (the owner's new data-center scope) are
  not in this task.
- The pods of runs 1-7 are all deleted; `rp_pod.sh list` at the end showed only another engineer's pod (glyd-bench-f8a7).
