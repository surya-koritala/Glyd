# The tiered layout against the 12-bit one on 2026 models: an RTX 5090 and an H100 SXM (2026-10-06)

Runpod pods, one job each (`job/ts_job.sh`, its source tar of the tree at the commit in each run's `COMMIT`), vLLM 0.30.0 and its
PyTorch 2.13 (CUDA 13.0), the library built on the pod for its GPU alone (sm_120 or sm_90a). The matrices are the models' own,
fetched by byte range (`gpu/fetch_tensors.py`): a full-attention layer (3) and a linear-attention layer (0) of Qwen3.5-9B and
Qwen3.5-27B, a sliding-window layer (0) and a global layer (5) of Gemma-4-12B-it and Gemma-4-31B-it, Qwen3.5-9B's output layer, and
Qwen3.5-35B-A3B's experts of layer 3 (256, top 8). q, k, v and gate, up are joined as vLLM joins them; the linear-attention layer's
in_proj_qkv and in_proj_z as one.

`gpu/layouts.py` times each matrix in both layouts and bf16 (cuBLAS) under CUDA graphs over a ring of copies four times the L2 cache,
each layout by its route on that GPU (the vLLM plugin's: the library's route for M, the matrix decoded for cuBLAS where that route
says so) and by the other routes; every pack is unpacked and compared bit for bit, every product against fp32 (within 1.5e-2 of the
largest output) and twice (the same bits). `--hot` adds the time from the L2 cache where a pack fits half of it. `summary.txt` of
each run sums a step through the model's layers (each matrix counted as often as the model has it) and gives the ratios;
`sass.txt` the instruction mix of the dense kernels (`cuobjdump -sass`, `gpu/experimental/tiered_fast/sassstat.py`),
`log/build.txt` their registers and spills (`ptxas -v`). Runpod's pods give no access to the GPU's counters (Nsight Compute:
`ERR_NVGPUCTRPERM`), so what binds a kernel is read from its time from memory against its time from the L2 cache.

| Run | GPU | Tree | What |
| :--- | :--- | :--- | :--- |
| `rtx5090-1-routes-v029` | RTX 5090 | 2933f7ed (v0.29.0's kernels) | both layouts by every route, 1-2048 tokens, checks |
| `h100-1-routes-v029` | H100 80GB HBM3 (SXM) | 2933f7ed | the same |
| `rtx5090-2-shapes` | RTX 5090 | a964308e | the tiered layout by the stream-K kernel in 4 shapes for 1-64 tokens (`mid1`-`mid4`), 3 prompt shapes (`big5`-`big7`) |
| `h100-2-shapes` | H100 SXM | a964308e | `mid1`-`mid4` |
| `rtx5090-3-short-decode` | RTX 5090 | f194a920 | the step kernel's short decode in every tiered kernel (`base`; `gen` the general decode out of line) |
| `rtx5090-4-shapes` | RTX 5090 | c7d26810 | 9-32 tokens by the 2-m-tile product (`mt2`), the step kernel at 3 blocks an SM (`bl3`), `cand` (2-m-tile 9-32, stream-K 33-64), 2 more prompt shapes |
| `rtx5090-5-final` | RTX 5090 | 8cbc6ae0 (this branch's routes) | checks (self-test, check_step, check_capi, check_vllm --quick on Qwen3.5-9B), the matrices; vllm serve at 1-64 users cut short (below) |

## RTX 5090: a step's products, the tiered layout's GPU time over the 12-bit layout's

| Model | 1 | 8 | 16 | 32 | 48 | 64 tokens | prompt 128 | 512 | 2048 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| v0.29.0's routes: Qwen3.5-9B | 0.94 | 0.99 | 1.06 | 0.96 | 1.10 | 1.17 | 1.07 | 1.03 | 1.03 |
| Qwen3.5-27B | 0.94 | 0.96 | 1.00 | 0.94 | 1.19 | 1.24 | 1.07 | 1.03 | 1.04 |
| Gemma-4-12B | 0.96 | 1.02 | 1.10 | 0.97 | 1.14 | 1.22 | 1.09 | 1.03 | 1.03 |
| Gemma-4-31B | 0.93 | 0.95 | 0.98 | 0.96 | 1.13 | 1.22 | 1.07 | 1.03 | 1.04 |
| this branch: Qwen3.5-9B | 0.93 | 0.99 | 0.96 | 0.96 | 0.84 | 0.80 | 1.06 | 1.02 | 1.02 |
| Qwen3.5-27B | 0.92 | 0.95 | 0.95 | 0.93 | 0.88 | 0.81 | 1.06 | 1.02 | 1.01 |
| Gemma-4-12B | 0.96 | 1.01 | 0.99 | 0.98 | 0.83 | 0.80 | 1.09 | 1.02 | 1.03 |
| Gemma-4-31B | 0.92 | 0.95 | 0.95 | 0.95 | 0.90 | 0.86 | 1.07 | 1.02 | 1.03 |

(`rtx5090-1-routes-v029` and `rtx5090-5-final`, their `layouts-*.json`.)

- **What binds it.** At 1 token the tiered layout reads 1.2-1.5 TB/s of its packs against bf16's 1.4-1.7 (the 12-bit layout
  1.2-1.7): 444-554 clk a step of 1,024 weights at the median 2,842 MHz, from the L2 cache 0.53-0.62 of that time: mostly memory.
  At 16 tokens the step kernel (blocks of 4 warps, 32 sums a thread: 128 registers and 4 bytes of spill on sm_120) is 0.71-0.75
  from the L2 cache, and at 48-64 tokens the 4-m-tile product (254 registers, one block of 8 warps an SM) 0.91-0.97: bound by
  their instructions, where the 12-bit layout's lighter decode was not.
- **The stream-K kernel** (the prompt kernel's producer and consumer warps, stream-K, its consumers multiplying only the m-tiles
  that hold tokens) for 1-64 tokens in blocks of 64: with 8 producer warps and 4 consumers (`mid1`) 0.80-0.86x at 64 tokens and
  0.84-0.91x at 48, 1.00-1.05x at 16, 1.03-1.12x at 1-8; with 4 producers (`mid2`) 0.89-0.97x at 64; with blocks of 2 consumers
  and 4 producers (`mid3`, one block an SM by its registers) 1.19-1.35x at 64; 8 producers by 8 consumers of 4 row blocks (`mid4`) 0.82-0.90x at 64.
  The number of producer warps an SM is what counts. Kept as an opt-in route: `mid1` from 33 tokens (below).
- **9-32 tokens by the 2-m-tile product** (`mt2`): 0.95-1.00x at 16 (the step kernel 0.98-1.10x), the same as before at 17-32.
  The step kernel in blocks of which 3 an SM (`bl3`, no spill) 0.98-1.07x at 16. Kept as an opt-in route: `mt2` from 9 tokens (below).
- **The step kernel's short decode elsewhere** (`rtx5090-3-short-decode`): the prompt kernel and the stream-K kernel no faster
  (within 1%), and its general decode out of line made the 4-m-tile product 1.34-1.53x (`base` there): not kept (reverted).
- **Prompts.** The prompt kernel's tiered producers cost 1-3% from 512 tokens and 6-9% at 128 against the 12-bit layout's (the
  12-bit layout itself is 2-4% over bf16's cuBLAS from 512 tokens). No shape tried was faster: 8 producers and 8 consumers (`big5`), GeForce Ada's
  8 consumers of 256 tokens (`big6`, 1.01x at 512-1024 on Qwen3.5-9B, 1.03-1.17x elsewhere and 1.7-2.0x at 128), the 4/4/3/2
  stream-K (`big7`), 8 consumers of 256 by two row blocks (`big9`), GeForce Ada's 12-bit shape (`big10`). The matrix decoded for
  cuBLAS: 1.06-1.12x at 2048.
- **Qwen3.5-9B's output layer** (248,320 x 4,096): 1.06x at 1 token, 1.08x at 8, 0.92-0.95x at 16-64 (`rtx5090-5-final`).
- **The experts** (Qwen3.5-35B-A3B layer 3, `moe-*.txt`): 0.92-1.00x at 1-1,024 tokens a step (1.004x at 1).

## The final run's checks (`rtx5090-5-final`, this branch's routes)

- The self-test (every bf16 bit pattern in both layouts, bit for bit; the prompt and mid products 1-1,100 tokens within 1e-2, the
  same every run), `check_step.py` (6,878 checks), `check_capi.py` (7,324 calls of the library and of PyTorch's JIT build of the
  same source, bit for bit, all identical; 260,052 routes as main's rule) and every `layouts.py` product (against fp32, twice):
  pass.
- `check_vllm.py --quick Qwen/Qwen3.5-9B`: 15 of 18 pass, among them every pack bit for bit (211), each layer's product within
  1e-2 of F.linear and the same bits every run, both layouts against bf16 (top-1 0.9968 tiered, 0.9948 12-bit; bf16 eager's
  0.9955), exact mode bf16's bits (8 of 8). The 3 that fail say the embedding is not packed (one table pack, the LM head's, where
  2 are expected; exact mode lists `language_model.model.embed_tokens` as vLLM's own): the plugin's coverage of Qwen3.5, not these
  kernels (both layouts alike).
- The memory, as vLLM reports it for Qwen3.5-9B: weights 12.69 GiB tiered, 13.89 GiB 12-bit, 17.66 GiB bf16; KV cache 359,568,
  322,080 and 197,472 tokens.
- **vllm serve was cut short**: the pod was deleted from outside the job after bf16's 1-user pass (91.2 tokens/s, first token 120
  ms, 10.53 ms a token), at about 05:28 UTC, during the day's Runpod budget limits; the tiered and 12-bit servers did not run. So
  there is no end-to-end comparison of the layouts on this GPU, and `best_layout` is not changed by this work.
- **Opt-in.** Without that comparison the routes of this branch (`mt2` from 9 tokens, `mid1` from 33, on compute capability 12) are
  not the default: the library takes them for the route `GLYD_GPU_ROUTE_GEMM_CC12` alone, which the vLLM plugin gives a Linear's
  products where `--additional-config '{"glyd": {"cc12": true}}'` asks; every default route is v0.29.0's. The stream-K kernel's
  skip of the m-tiles past the tokens is in `mid1`'s shape alone (GeForce Ada's prompts, not measured with it, as before).

## H100 SXM

| Model | 1 | 8 | 16 | 32 | 64 tokens | prompt 128 | 512 | 2048 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.5-9B | 1.12 | 1.15 | 1.26 | 2.08 | 2.48 | 2.21 | 1.42 | 0.95 |
| Qwen3.5-27B | 1.05 | 1.17 | 1.26 | 1.98 | 2.33 | 2.42 | 1.42 | 0.94 |
| Gemma-4-12B | 1.21 | 1.25 | 1.36 | 2.04 | 2.39 | 2.21 | 1.40 | 0.94 |
| Gemma-4-31B | 1.06 | 1.17 | 1.27 | 1.97 | 2.33 | 2.47 | 1.47 | 0.94 |

The tiered layout is bound by its decode at every count: at 1 token 197-244 clk a step on the large matrices against the 12-bit
layout's 168-200, the same from the L2 cache as from memory (the 4,096 x 4,096 matrices 258 against 275), its packs read at
1.3-1.8 TB/s against the 12-bit layout's 1.4-2.4 and bf16's 2.0-2.8. The step kernel's hot loop at 1-8 tokens is 709 instructions for two steps
(about 355 a step; 121 registers, no spill; at 9-16 tokens 128 registers and 52 bytes of spill loads); the 12-bit layout takes its TMA and wgmma kernel from 17 tokens, where the tiered
layout's 2- and 4-m-tile products take 2.0-2.5x its time and the stream-K shapes 1.9-2.2x (`h100-2-shapes`). Output layer 1.45x at
1 token. Experts 1.14-1.50x. At 2048 tokens both layouts decode for cuBLAS and the tiered one is 0.94-0.95x. Nothing tried here
brings the tiered layout to the 12-bit one's speed on Hopper; that needs a decode of fewer instructions a weight (the design of
2026-10-04 measured the same gap from the other side).
