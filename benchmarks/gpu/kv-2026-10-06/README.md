# The lossless KV cache at head size 256 and on hybrid models: Qwen3.5-9B on an RTX 4090 (2026-10-06)

Private record. The branch `kv-2026`: the lossless KV cache (`kv: lossless`) for Qwen3.5 / Qwen3.6 / Qwen3.8 (head size 256, Gated DeltaNet layers beside full attention).
Two Runpod pods (RTX 4090, Ada sm_89, 24 GB), vLLM 0.30.0, torch 2.13.0+cu130, the library built on the pod for sm_89; the model `Qwen/Qwen3.5-9B` (2026-02), Glyd's weights
(`--quantization glyd`, fraction 1, the text model only: `--language-model-only`). A third pod was stopped by the owner's budget pause (below).

| Run | Pod | Commit | What |
| :-- | :-- | :-- | :-- |
| [run1](https://github.com/surya-koritala/Glyd/tree/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run1) | kv26a, 26 min, $0.32 | 070eaf0e | the plugin's tests, the kernels' checks, head size 128's bits against main's library, the decode kernel at head size 256 against FlashAttention 2, Qwen3.5-9B with vLLM's cache and kv lossless (its tables made on the first start), verify under churn, `vllm serve` at 1, 8, 32, 64 requests at once and all at once ([job.sh](job.sh)) |
| [run2](https://github.com/surya-koritala/Glyd/tree/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run2) | kv26b, 12 min, $0.15 | 2ad2e80c (main merged) | kv lossless against vLLM's cache at the same block size (704), vLLM's cache twice; the kernels' checks with main's library and the branch's, twice each ([job2.sh](job2.sh)) |
| run3 | kv26c, 8 min, $0.10 | 9c6c01aa | stopped: at 05:27 UTC the owner paused bf16 GPU work to save the last Runpod credits (`rp_budget/ALLOW`), and the pod was deleted during its first engine start. Not retried ([job3.sh](job3.sh): the KV cache's tokens in a fixed 6 GiB at 8,192 and 32,768 tokens, and vLLM's own greedy spread when only the batch changes) |

Spend: $0.58 of the track's $25.

## How it is held

- **A head of 256 channels is two of the page's heads of 128** (channels 0-127, then 128-255): the page format, the write, unpack and release kernels are unchanged (a page of 4 KV heads of 256 is a page of 8 heads of 128:
  49,408 bytes, as Qwen3-8B's); the C API takes D = 256 and maps it (`kv_heads(H, D)`). New: the decode kernel's paired mode (a page on two warps, each its half's tiles, the halves' scores summed in shared memory
  before the softmax: both warps hold the same sums), the gather into the staging cache with a head's 256 channels whole, the staging put at D = 256, the merge of the partials 128 channels at a time.
- **Hybrid models.** The linear-attention layers' states stay vLLM's, in the one pool of blocks. vLLM sizes the attention block from Glyd's page: 704 tokens (44 pages of 16) where bf16's is 528. The scheduler maps
  the status rings' pages to blocks and blocks to pages (block_size / 16); every block a step takes from the free pool has its pages cleared at the start of the step's `execute_model` (vLLM's worker, wrapped),
  before vLLM zeroes the step's new attention blocks and copies Mamba states (a block's pages may hold the reserves' chunks and slabs of an earlier life: the zeroed page would no longer name them).
- **Calibration** counts a head of 256 as two of 128 (tables `[2 H, 128]`); the reserves and the admission's predictor work in the pages' heads.
- `kv: auto` keeps vLLM's cache for head size 256 and linear-attention layers ("not measured faster than vLLM's own yet"); `kv: lossless` holds them.

## Results

**Every value exact.**
- The kernels' checks ([run1/kernels.out](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run1/kernels.out)): 15 of 15 at head size 256 pass: every 16-bit pattern as an exception (~2% of 5.2 M values) back bit for bit in the page's arena and in the escape pool's
  chunks (its books agree with the pages'), raw pages with heavy tokens bit for bit (the raw pool's books agree), the staging cache [blocks, 16, 4, 512] bit for bit, the staging put, decode attention within
  0.0044 of an fp32 reference at 1 to 8 query heads a KV head and 1, 2, 4, 16 KV heads, with chunks, raw pages and fixed splits; a sequence's bits its own alone, batched and padded; the merge in the kernel or
  apart and the next page ahead or not the same bits; a CUDA graph's replay the bits of a call. Head size 128's 81 checks: 80 pass in run1, 81 of 81 in both branch runs of run2.
- The one head-128 failure is not this branch's: "the escape pool: 64 chunks ... 65 chunks taken at most" (the high-water status word over-counts by one when a chunk freed by a page's losing warp is taken again
  before the count is lowered). main's library failed it in both of run2's runs ([run2/kernels-main-1.out](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run2/kernels-main-1.out), [-2](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run2/kernels-main-2.out)), the branch's passed it in both.
- Head size 128 unchanged: the outputs of unpack, gather and 36 decode configurations (G 1, 4, 8; the library's splits and 1, 7, 64; merge in the kernel or apart; next page ahead or not) are the same bits with
  main's library and the branch's ([run1/bits.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run1/bits.txt): 38 of 38 outputs; the pool's bytes and status words differ as its chunks are taken in a racing order, in two runs of either library alike).
- `verify` (a bf16 shadow of every layer's cache, every write and every context read back compared) under churn: 48 requests of 200 to 3,000 tokens on a cache of 48 blocks, blocks passing between the cache
  groups, requests preempted: 4,326,752,256 values compared, 145,648 pages, no difference; 28,996 pages cleared before reuse ([run1/hybrid-verify.out](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run1/hybrid-verify.out)).
- Against vLLM's own cache at the same block size (704: a hybrid model's prompt steps and prefix hits follow the block size) ([run2/compare-stock704-kv.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run2/compare-stock704-kv.txt)):
  - prompt logprobs **17,906 of 17,906 bit for bit** (7 prompts of 64 to 7,000 tokens, chunked prefill in steps of 2,048: the later chunks read the earlier ones' pages);
  - prefix hits: the same tokens cached (2,816 in each of 6 requests), the first tokens' 5 logprobs **6 of 6 bit for bit**;
  - greedy decoding (Glyd's decode kernel, not FlashAttention's sums): 4 of 13 sequences the same to the end, 527 of 1,280 tokens before the first that differs; vLLM's cache run twice: 13 of 13 the same
    ([run2/compare-stock704-stock704b.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run2/compare-stock704-stock704b.txt)). vLLM's own spread when only the batch changes was run3's, not measured.
  - At vLLM's default block (528) the prompts are stepped otherwise and nothing is comparable bit for bit (run1: 723 of 17,906; the prefix hits cache 2,640 tokens against 2,816).

**More tokens in the KV cache.**

| | Blocks | Block | KV cache tokens (at 8,192) | Requests of 8,192 at once |
| :-- | --: | --: | --: | --: |
| vLLM's cache (`kv_hybrid.py`, run1) | 416 | 528 | 154,903 | 18.91 |
| kv lossless (run1) | 403 | 704 | 183,409 | 22.39 |
| vLLM's cache (`vllm serve`, run1) | 403 | 528 | 150,062 | 18.32 |
| kv lossless (`vllm serve`, run1) | 390 | 704 | 177,493 | 21.67 |

1.18x the tokens at `--gpu-memory-utilization 0.9` (kv lossless's reserves, 46 MiB and 81 MiB, inside the same memory). The pools a start made varied with the start (vLLM's cache at block 704 got 312
blocks on the pod's first start and 387 on its third in run2): run3's fixed 6 GiB was to take that out; not measured.

**Speed** (`vllm serve`, random 1,024 tokens in, 256 out, `--max-num-seqs 128`, one pass each, run1 [summary-serve.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run1/summary-serve.txt)):

| Load | vLLM's cache req/s | kv lossless req/s | output tok/s | TTFT median ms | TPOT median ms |
| :-- | --: | --: | :-- | :-- | :-- |
| 1 at once | 0.291 | 0.292 | 74.5 / 74.7 | 144.3 / 151.2 | 12.91 / 12.84 |
| 8 | 1.713 | 1.716 | 438.4 / 439.3 | 629.5 / 641.6 | 15.77 / 15.65 |
| 32 | 3.733 | 3.800 | 955.6 / 972.8 | 749.8 / 734.2 | 30.06 / 29.52 |
| 64 | 4.551 | 4.684 | 1,165.1 / 1,199.2 | 986.1 / 757.9 | 50.24 / 48.95 |
| 256 at once | 4.211 | 4.645 (1.10x) | 1,077.9 / 1,189.2 | 23,509 / 22,358 | 60.16 / 59.33 |

Tokens a second at least vLLM's at every load; the time to the first token 5% and 2% longer at 1 and 8 requests at once (one pass each). So `auto` stays off for these models: kv lossless is opt-in.

**The decode kernel alone** (one layer of Qwen3.5-9B: 4 KV heads of 256, 4 query heads each; in a CUDA graph; [run1/bench256.json](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run1/bench256.json)): 1.35x to 1.56x faster than FlashAttention 2 at every shape
(1x1,024 9.9 against 15.4 us; 1x8,192 34.9 / 52.2; 8x4,096 115.0 / 157.4; 64x1,024 218.1 / 298.9; 32x4,096 430.8 / 580.4); the write of a step's token 2.4 to 5.8 us a layer against vLLM's 1.4 to 1.6.

**First start:** the tables at 8,192 positions in 189 s (the model's load 63 s, the text 2 s and 2 s, the vocabulary's 248,320 rows 121 s); worst kind of text left out K 0.0103%, V 0.0311%; no token needs the
second reserve ([run1/tables.txt](https://github.com/surya-koritala/Glyd/blob/7d9e9be6f33395f39421304b33c9f8481afa5962/benchmarks/gpu/kv-2026-10-06/run1/tables.txt)).

## Not done

- **Gemma 4** keeps vLLM's cache (refused with why): its 40 sliding-window layers (head 256, window 1,024) need the decode kernel to start at the window and vLLM's sliding-window manager's frees, and its 8 global
  layers are head 512 with one KV head shared as K and V (16 query heads to it, past the kernel's 8).
- The long tiers (65,536 and 131,072) and the H100 run of the kv-newer plan.
