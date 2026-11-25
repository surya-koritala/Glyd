# V8a: NVIDIA kernel libraries where they are faster

Status: the decision table per operation (#255), with the numbers measured so far and, where nothing is measured, the test that decides it.
AMD's libraries (hipBLASLt, aiter, composable_kernel) are V8b (#447, M5b).

## The rule

Glyd's own kernels stay where they read Glyd's formats in place (the lossless and lossy packed layouts, the 12-bit KV pages, the lossless FP8
format): no library reads them, and a library behind them first needs the weights or pages decoded to bf16, which costs traffic the fused kernel
does not (decoded first: 1.51 + 2 + 2 = 5.51 bytes a weight for the 12-bit layout against bf16's 2 and the fused kernel's 1.51,
`benchmarks/gpu/prompt-speed-2026-10-06`). A library takes an operation where, on that GPU and at that size, it is measured faster than Glyd's
own on the same work, including any decode it needs. The choice is made once, per operation and GPU, in code: for the dense products `gpu_ops::dense_choice`; for the packed products it is the
route table (`glyd-engine/src/fold.rs`, folded from the GPU library by R17 and checked against it by `foldcheck`), plus the executor's rule on
top of it (`Dyn::decodes`).

## The table (NVIDIA)

| operation | Glyd's own | library | measured | decision |
| :--- | :--- | :--- | :--- | :--- |
| packed product, 1-64 tokens (every decode step) | the step product (`mma`, `mma12`, block layouts), Hopper's TMA + wgmma kernel for the 12-bit layout from 17 tokens | none reads the layouts; cuBLAS only on the decoded bf16 matrix | Qwen3-8B on an RTX 4080 SUPER: 59.7 tokens/s at one user, 90% of the weights-read ceiling (`rtx4080s-qwen3-8b-engine-2026-11-20`) | own, every GPU |
| packed product, prompt, 12-bit layout | the prompt kernel (ROUTE_BIG); Ampere/Ada's staged product to 64 tokens (128 on an A100) | cuBLAS on the matrix decoded first (ROUTE_DECODE), or decoded ahead beside the products before it (ROUTE_AHEAD) | A100: decoded from 769 tokens; L4: from 2,560 (0.4-21% slower to 2,304, 1-3% faster at 2,560, 7-49% at 4,096-8,192: `l4-routes-2026-09-29`); GeForce Ada ahead from 1,793, A10 from 640, L40S from 2,048; Hopper: wgmma to 1,024, then decoded; the 12-bit layout's prompt kernel 2-4% over bf16 cuBLAS from 512 tokens on an RTX 5090 (`tiered-speed-2026-10-06`) | the route table (`fold::route`) |
| packed product, prompt, tiered and lossy layouts | the prompt kernel | cuBLAS on the decoded matrix | L4: decoded from 896 (4-5% slower at 768, 9-11% faster at 896, 17-47% at 3,072-8,192: `l4-routes-2026-09-29`); GeForce Ada ahead from 513, A10 from 512, L40S from 1,024; Hopper: decoded from 65 tokens (1.18-2.2x the 12-bit layout's time there, `hopper-tiered-2026-10-06`); A100, Ampere GeForce, Blackwell: no crossover in the table | the route table; **and** the executor decodes these layouts for cuBLAS from 512 rows where the table has no crossover for them, for the families whose descriptions take the decoded way. A100-SXM4-40GB, Qwen3-8B at penguin (tiered), prompt tokens/s fused against decoded for cuBLAS (a run of 2026-11-24, `GLYD_ENGINE_DECODE_FROM`): 493 rows 5,839 against 8,093 (1.39x), 995 rows 6,338 against 9,800 (1.55x), 4,003 rows 6,210 against 10,724 (1.73x); at 995 rows the products take 135.9 ms fused against 80.4 ms decoded (18.9 ms of it the decode). The threshold of 512 rows leaves a 493-row prompt fused; below 493 rows not measured |
| 12-bit prompt, split SMs (opt-in) | the matrix decoded ahead on SMs set apart, cuBLAS on the rest | cuBLAS | A100 SXM Qwen3-8B pass 0.876-0.971x at 769-4,096 tokens, 14B 0.845-0.947x; GH200 Qwen3-32B 0.909-0.952x; H100 SXM Qwen3-14B 0.893-0.938x at 2,048-8,192 (`option2-2026-09-29`) | the route SPLIT where the caller asks (the engine does not) |
| dense bf16 product (a decoded matrix; bf16 layers) | `gemm3`, `gemm_silu` (Kern::pick_tile) | cuBLAS (`ffi::Blas`, `gpu_ops::Gemm`); cuBLASLt not integrated | RTX 4080 SUPER, Qwen3-8B 2,000-token pass: own 353 ms (93 TFLOPS, 89% of the card's fp32-accumulate peak) against vLLM's 357 ms on cuBLAS (`rtx4080s-gemm-tiles-2026-11-20`). A100-SXM4-40GB, Qwen3-8B's prompt with its matrices decoded: cuBLAS 8,109 / 9,853 / 10,751 prompt tokens/s at 498 / 1,001 / 4,003 tokens against own 7,375 / 9,481 / 10,567 (cuBLAS 10%, 4%, 2% faster; decode the same; `a100-device-seam-2026-11-24`). Other GPUs not measured | cuBLAS where the machine has it, but on Ada (compute capability 8.9) own; own on Ampere and Hopper where cuBLAS is absent (`gpu_ops::dense_choice`, which the executor asks; `GLYD_ENGINE_GEMM=blas` or `own` forces one in a development build). The A100 keeps cuBLAS, as measured |
| prompt attention, heads of 64 and 128, bf16 or 12-bit pages | `attn_pf_t` (FlashAttention-2's loop, GQA packed; 12-bit pages unpacked to bf16 first) | FlashInfer prefill (Apache-2.0, header-only) | A100 before this kernel: a 4,009-token pass 115 ms in attention (41 TFLOPS); with it: not measured; FlashInfer: not measured | own until measured: FlashInfer's prefill instantiated for D 64/128 bf16 pages, compiled into the fatbin, timed against `attn_pf_t` on the same rows (F1a's row: Llama, D 64 and 128) |
| prompt attention, heads of 256 and 512 | the library's prefill (12-bit pages read in place) | FlashInfer prefill (its plan takes heads to 512) | not measured | own over 12-bit pages (no library reads them); over bf16 pages own until measured, as for 64 and 128 |
| decode attention, 12-bit pages | the library's `kv_decode` (pages decoded into the MMA operands, split plan, fixed merge order) | none reads the pages | RTX 4080 SUPER: 12-bit pages decode at bf16 pages' speed (PR #547) | own |
| decode attention, bf16 pages | `kv_decode_bf16` (the same kernel and plan) | FlashInfer batch decode, FlashAttention's paged decode | not measured | own until measured; bits are the 12-bit path's where the pages hold the same values, which a library would not keep |
| MLA (DeepSeek-V4, Kimi-K3, GLM) | the executor's MLA op | FlashMLA, CUTLASS MLA (Hopper and Blackwell builds), FlashInfer MLA | not measured | own on Ampere and Ada (no FlashMLA kernel there); Hopper and Blackwell: not measured |
| grouped experts, packed layouts | `mma_moe`, `mma12_moe` (the library), the expert store's placed product | CUTLASS grouped GEMM, Marlin MoE: neither reads the layouts | A100 and H100 records: `a100-moe-v028-2026-10-03`, `h100-moe-v028-2026-10-03` (own only) | own |
| FP8 (the lossless FP8 format) | `f8_linear`, `f8_moe` (mma.sync e4m3, SM89+) | CUTLASS FP8 blockwise (plain FP8 only), DeepGEMM (compiles at run time: unsuitable) | `fp8-linear-2026-10-05`, `fp8-kernels-2026-10-05` (own only) | own: the format is Glyd's |
| sampling (top-k candidates) | `ModelStep::top`, the top-k kernels | FlashInfer sampling | not measured | own |
| Blackwell (SM100, SM120) dense, FP8, NVFP4 products | mma.sync (no tcgen05, no block-scaled FP4 MMA) | CUTLASS's tcgen05 kernels, FlashInfer | not measured: no Blackwell GPU on the runner | not decided; M3 (V7a #254, F6 #181) |

## Integrating a library

- **Loaded at run time** (cuBLAS today): the engine opens `libcublas.so.13` or `.12` from the machine (`GLYD_CUBLAS` to name one; `GLYD_ENGINE_BLAS=off`
  to refuse it) and never ships it, so it adds nothing to the bundle and needs no NOTICE entry. cuBLASLt would come the same way (`libcublasLt.so`),
  behind `gpu_ops::Gemm`.
- **Compiled in** (FlashInfer, CUTLASS: header-only, Apache-2.0 and BSD-3-Clause): instantiations for the shapes the table takes, built into the
  engine's fatbin by `build.rs`, with a NOTICE entry each. Budget: the release fatbin per CUDA version stays within the library's 24 MB
  (`one-kernel-build.md`, Limits); each instantiation's size and compile time are measured when it is added.
- **Not adopted**: FlashAttention 4 (its compiler is under an NVIDIA EULA and runs in Python), DeepGEMM (compiles at run time).

## What is left

- Not measured, by row: the dense products on GPUs but the RTX 4080 SUPER and the A100, the A100 tiered prompt crossover below 493 rows, FlashInfer prefill and decode
  (needs the compiled-in integration first), MLA on Hopper and Blackwell, sampling, Blackwell's products.
- The A100's decoded prompt spends 18.9 ms a pass decoding the matrices (32% of a 498-token pass with cuBLAS): a decode beside the products before it
  (ROUTE_AHEAD, as GeForce Ada, the A10 and the L40S take) is the next measurement there.
- The L4 rows of the table are the 2026-09-29 measurement; a rerun is due with the executor's paths (the old measurement ran the Python package).
