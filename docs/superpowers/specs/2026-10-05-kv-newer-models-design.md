# The lossless KV cache for newer models: head size 256, hybrid attention, long context

Private design (glyd-gpu-private). Owner's scope, 2026-10-05: Qwen3.8-style models plus long context; sliding window later.

## Goal

`kv: auto` holds the KV cache of Qwen3.8-style models in vLLM 0.30, exactly as it holds Qwen3's today:

- **Head size 256.** The full-attention layers of Qwen3.8 (and Qwen3.5) have head size 256; the kernels take 128 only.
- **Hybrid models.** 48 of Qwen3.8-27B's 64 layers are linear attention (Gated DeltaNet). Their state stays vLLM's; Glyd holds the 16 full-attention layers' cache.
- **Long context.** Today's limit is 40,960 tokens. It becomes 131,072, through 65,536, so that the gain reaches the request lengths where it exists.

Not in scope: sliding window (Gemma 4), the linear-attention state itself (fp32, rewritten every step), and windows over 131,072.

## Why long context matters here

vLLM gives a hybrid model one block pool. A block is as big as one linear-attention state, 3.1 MB a layer for Qwen3.8-27B, which is 784 tokens of bf16 attention cache. Each request holds:

- one block per linear-attention group, three groups in all;
- as many attention blocks as its tokens need.

With Glyd's 12-bit pages an attention block holds about 1,040 tokens, and the gain follows from that:

| Request length | bf16 blocks | Glyd blocks | More requests at once |
| :--- | ---: | ---: | ---: |
| 1,280 tokens | 2 + 3 | 2 + 3 | 1.00x |
| 8,192 tokens | 11 + 3 | 8 + 3 | 1.27x |
| 32,768 tokens | 42 + 3 | 32 + 3 | 1.29x |

These are estimates, to be measured.

## Design

1. **Kernels for D = 256.** `glyd_gpu_kv_write`, `_release`, `_unpack`, `_gather`, `_work_put` and `_decode` take D as a template parameter: 128 or 256, both compiled.
   - A page is the same format with longer tiles: 16 H header bytes, then the directory and arena, then H × 12,288 bytes of K and V tiles (6,144 at D = 128).
   - The decode kernel at D = 256 holds twice the accumulators per query head. Its CTA shape is re-tuned on the L4 and H100 (Qwen3.8: 6 query heads to a KV head, within MAX_GROUP 8).
   - Prompt rows keep running vLLM's FlashAttention on the bf16 staging cache; FA2 and FA3 take head size 256.
   - The C API version goes to 16, because D = 256 is a new accepted value. Each function still refuses any other D.
2. **Hybrid models in the plugin.**
   - The refusal of linear-attention and state-space layers is lifted. Glyd's backend serves the full-attention layers only; vLLM's own backends serve the rest.
   - vLLM 0.30 already sizes a hybrid model's attention block through the backend's `customize_spec`. It also takes the kernel block, 16 tokens, from `get_supported_kernel_block_sizes`, so a manager block of about 1,040 tokens reaches the backend as 16-token pages. The plugin's code that assumes one KV cache group or one block size (`vllm_sched`, `release`, `_CACHES`, the reserves' shares) is generalized: Glyd's blocks are the attention group's.
   - **Recycled memory.** vLLM shares a block's bytes between groups, so an attention page can hold a linear-attention state's bytes:
     - The write kernel already clears a page at its first token and checks ownership in the pools. A recycled page is therefore safe at write.
     - The danger is a release that comes after a block was handed to a linear-attention group: clearing it would zero a live state. Rule: for a hybrid model, every block handed over is released in the next step's metadata build, uncapped. A block that cannot be released then is not handed back to the free queue until it is.
     - Tested with a stress run that churns blocks between groups.
   - **The reserves** (escape and raw pools) are sized from the attention layers' share of the pool, not the whole pool.
3. **Calibration (kvcal).** It hooks the full-attention layers only, makes tables for D = 256, and runs hybrid models; the first-start time is logged as today.
4. **Long context.**
   - `LONG_BANDS` (65,536 and 131,072) leave the `GLYD_KV_LONG=1` gate and become tiers like the rest. The calibration of a long tier runs in steps (CHUNKED and STEP exist).
   - Its time on a 27B model on an H100 is measured and logged. If it is over 10 minutes, the long tiers are made on demand, the first time a request goes past 40,960 tokens, not at start.
   - Where the GPU cannot hold a tier's calibration beside the weights, `auto` keeps vLLM's cache with a line that says why, as today.
5. **`auto`.** It turns on for these models on the A100, L4, H100 and GH200 only once the checks below pass on them. Otherwise it keeps vLLM's cache with the reason.

## Checks: the release gate for this work

- **Exact:**
  - `kv_stress` at D = 256 and on a hybrid skeleton: 0 values that differ.
  - `verify` mode (the bf16 shadow), with no error, over a long run that churns blocks between groups.
  - Prompt logprobs exactly vLLM's (`kv_agree`) on a hybrid model with chunked prefill and prefix hits.
- **Speed:** decode tokens a second at D = 256 not below vLLM's own cache at any batch measured on the L4 and the H100. Where it is below, `auto` stays off for D = 256 on that GPU.
- **No regressions:** the head-128 checks (Qwen/WebWorld-8B, a 2026 model of Qwen3's architecture, on the L4) pass.
- **The H100 run,** Qwen3.8-27B against bf16 and Glyd without the KV cache: KV tokens, and requests a second at 1k, 8k and 32k-token requests. The target is at least 1.2x requests at once at 8k and 32k, with every value exact.
- **The dev models** on the L4: the tiny and real-shape Qwen3.8 skeletons from the feasibility work, made from the real config.

## Costs

- **The L4 dev box** for all kernel and plugin work.
- **One H100 run** of about 40 minutes: about $2-4 on Runpod or Lambda, asked before launch.
