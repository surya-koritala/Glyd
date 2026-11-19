# Prompt speed, 2026-10-06

What was seen (`ceiling/results-ts5090a`, `results-tsh100a`, Qwen3.5-9B's products, private main): a prompt's products are slower than
bf16's with both layouts. H100 SXM, a pass at 128 / 512 / 1,024 / 2,048 tokens: 12-bit 6.74 / 13.08 / 22.99 / 46.99 ms, tiered
14.87 / 18.57 / 27.08 / 44.62, bf16 5.67 / 9.55 / 18.26 / 36.09. RTX 5090: 12-bit 10.60 / 35.58 / 69.55 / 136.31, tiered 11.35 /
36.48 / 71.45 / 140.42, bf16 10.75 / 34.47 / 66.88 / 131.18. The goal: a prompt's products at least as fast as bf16's on an H100 and
an RTX 5090, no worse on an RTX 4090, bit for bit.

## The ideas by bytes and instructions, before any run

Qwen3.5-9B's pass multiplies 6.91 G weights (Qwen3.5-27B 24.3 G, gemma-4-31B-it 29.3 G). Per weight:

| way | DRAM bytes a weight | note |
| :-- | --: | :-- |
| bf16 (cuBLAS) | 2 | |
| 12-bit, fused | 1.51 | decoded again for every block of tokens |
| tiered, fused | 1.35 | |
| decoded first, then cuBLAS | 1.51 + 2 + 2 = 5.51 (tiered 5.35) | 3.5 bytes more than bf16, whatever the tokens |

- **Decoded first** costs Qwen3.5-9B 24 GB more traffic a pass: at least 7.2 ms on an H100 SXM (3.35 TB/s), 13.5 ms on an RTX 5090
  (1.79 TB/s), 23 ms on an RTX 4090 (1.01 TB/s). Measured before: 10.9-11.4 ms on the H100, 13.4 ms on the 5090. It can never reach
  bf16 on its own.
- **Decoded ahead on a second stream, beside the product before** (the idea to evaluate): the decode is memory-bound and the products
  past ~512 tokens are compute-bound (bf16 at 2,048 tokens: 784 TFLOP/s on the H100, 216 on the 5090), so the decode fits in the
  products' time *if it gets SM room*:
  - **H100**: cuBLAS's Hopper kernels hold a whole SM (registers and shared memory); nothing of another stream fits beside them. The decode
    then takes SMs from the products: the route SPLIT does that with green contexts, measured on an H100 SXM at +2.7% to +12.8% over
    bf16 at 8,192 to 2,048 tokens (Qwen3-14B, `option2-2026-09-29`), and the GPU runs at its 700 W cap, where the decode's extra DRAM
    traffic costs clocks. Expected: no decode-to-buffer way reaches bf16 on Hopper.
  - **GeForce (5090, 4090)**: cuBLAS's blocks leave room for a few warps of another kernel an SM (model.py's Early is built on that).
    Issue slots: 6.91 G weights x ~3 instructions / 32 lanes = 0.65 G warp instructions, spread over 170 SMs x 4 schedulers at 2.8 GHz:
    0.34 ms of issue a scheduler, under 1% of a pass from 512 tokens on. Bandwidth: 24 GB over the pass is 185 GB/s at 2,048 tokens
    (10% of the 5090's), 700 GB/s at 512 (39%, on top of cuBLAS's own ~400 GB/s). Expected: near bf16 from about 1,000 tokens, short of
    it at 512 and below; at 128 tokens a pass (10.75 ms on the 5090) is shorter than the decode at full bandwidth (13.5 ms), so short
    prompts stay fused (the 5090's 12-bit fused pass at 128 tokens is already 1.4% faster than bf16's).
- **One decode a layer shared by merged projections**: under vLLM, q, k, v and gate, up are one matrix each already (QKVParallelLinear,
  MergedColumnParallelLinear), and Qwen3.5's DeltaNet input projection is one matrix: each is decoded once now. Nothing to gain.
- **Buffers within the memory saved**: decoding ahead takes two buffers of the largest matrix: 0.40 GB for Qwen3.5-9B (it saves 3.8 GB
  in 12 bits), 0.71 GB for Qwen3.5-27B, 0.92 GB for gemma-4-31B-it; vLLM's profile counts them (made at the first prompt).
- **The threshold a GPU where each way wins**: measured below, a pass at each length.

## What was run

One batched job a GPU (`job/ps_job.sh`, `job/ps_summary.py`), the tree at the commit in each `COMMIT`:

- `gpu/prompt_pass.py MODEL` for Qwen3.5-9B, Qwen3.5-27B and gemma-4-31B-it (their configs' shapes and layer types; random weights of a
  trained matrix's spread), 128 / 512 / 1,024 / 2,048 / 4,096 / 8,192 tokens, both layouts: bf16 (timed just before each layout's
  ways), `glyd` (the vLLM plugin's op as it runs today), `decode` (decoded first), `ahead:N` (decoded ahead, the plugin's
  development path `GLYD_AHEAD_PROMPT`, N warps an SM for the decode, 0 the library's own launch), `glinear` (glyd.gpu's GLinear: the
  decode ahead of GeForce Ada, the route SPLIT of an H100 SXM). Checks before each timing: packs bit for bit; `decode` and `ahead`
  bf16's bits; `glyd` and `glinear` within 1.5e-2 of fp32.
- `vllm serve Qwen/Qwen3.5-9B` bf16, Glyd (its default layout and routes) and Glyd with `GLYD_AHEAD_PROMPT=1` (on the 5090 also
  `GLYD_DEC_MIN=513`: 12-bit prompts past 512 tokens decoded, so decoded ahead), each with the same flags (`--language-model-only
  --max-model-len 9216 --max-num-seqs 16 --gpu-memory-utilization 0.9`, `VLLM_USE_FLASHINFER_SAMPLER=0`): one user, 8 prompts of 1,000
  random tokens and 5 of 8,000, 8 tokens out, `vllm bench serve`'s time to first token, after warm-up prompts.

## What ran, and what did not

The three pods (H100 SXM `psh100a`, RTX 5090 `ps5090a`, RTX 4090 `ps4090a`, launched 05:12-05:13 UTC) were deleted at about 05:18
UTC, 6.5 minutes in, when the Runpod budget stop put this work on the launcher's block list ("stopped to save the last Runpod
credits"); the launcher refuses it since, so it was not run again. Done before the stop: the build and the library's self-test on all
three; `prompt_pass.py` on Qwen3.5-9B whole on the H100 and the 5090, and Qwen3.5-27B in 12 bits on the H100 to 4,096 tokens (8,192 in
`h100-sxm/pass-Qwen3.5-27B.txt`, GLinear's row missing). Not done: gemma-4-31B-it, Qwen3.5-27B on the 5090, everything on the 4090
(past its self-test), and every `vllm serve` first-token measurement (1,000 and 8,000 tokens). Spend: $0.56 (the budget ledger's
uptime: H100 383 s at $3.49/h, 5090 395 s at $0.99/h, 4090 383 s at $0.74/h).

## Results

A pass through the layers' products over bf16's, bf16 timed just before each layout's ways (`*/summary.txt`, `*/pass-*.txt`). Every
check passed: the packs bit for bit, `decode` and `ahead` bf16's bits on every product, `glyd` and `glinear` within 3.5e-3 of fp32.
`ahead`: the best of its three launches.

Qwen3.5-9B, RTX 5090:

| tokens | 12-bit glyd | decode | ahead | glinear | tiered glyd | decode | ahead | glinear |
| --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 128 | 0.995 | 2.280 | 2.367 | 0.992 | 1.084 | 2.189 | 2.267 | 1.066 |
| 512 | 1.046 | 1.396 | 1.275 | 1.029 | 1.078 | 1.359 | 1.254 | 1.072 |
| 1,024 | 1.029 | 1.193 | 1.088 | 1.027 | 1.065 | 1.180 | 1.092 | 1.063 |
| 2,048 | 1.022 | 1.101 | 1.039 | 1.024 | 1.058 | 1.093 | 1.044 | 1.060 |
| 4,096 | 1.019 | 1.052 | 1.022 | 1.022 | 1.059 | 1.049 | 1.024 | 1.058 |
| 8,192 | 1.023 | 1.028 | 1.013 | 1.029 | 1.067 | 1.026 | 1.013 | 1.068 |

Qwen3.5-9B, H100 SXM (`ahead` here is `ahead:0`; with 2 or 4 warps an SM it took 1.25-28x bf16's time):

| tokens | 12-bit glyd | decode | ahead | glinear | tiered glyd | decode | ahead | glinear |
| --: | --: | --: | --: | --: | --: | --: | --: | --: |
| 128 | 1.215 | 3.113 | 3.065 | 1.227 | 2.710 | 2.703 | 2.641 | 2.714 |
| 512 | 1.396 | 2.152 | 2.112 | 1.298 | 2.022 | 2.008 | 1.976 | 1.987 |
| 1,024 | 1.228 | 1.446 | 1.418 | 1.194 | 1.393 | 1.386 | 1.459 | 1.390 |
| 2,048 | 1.297 | 1.371 | 1.265 | 1.296 | 1.258 | 1.346 | 1.252 | 1.257 |
| 4,096 | 1.172 | 1.252 | 1.177 | 1.174 | 1.148 | 1.204 | 1.132 | 1.159 |
| 8,192 | 1.069 | 1.087 | 1.073 | 1.083 | 1.061 | 1.087 | 1.069 | 1.073 |

Qwen3.5-27B, H100 SXM, 12 bits: glyd 1.118 / 1.308 / 1.252 / 1.256 / 1.143 / 1.064 at 128 / 512 / 1,024 / 2,048 / 4,096 / 8,192
tokens; glinear (the route SPLIT from 2,048) 1.142 / 1.309 / 1.249 / 1.139 / 1.090; decoded ahead 3.328 / 2.060 / 1.556 / 1.264 /
1.124 / 1.059.

- **Nothing is at least as fast as bf16 at every prompt length, on either GPU: no default route changed.** The 5090's 12-bit default
  is at bf16's speed at 128 tokens (0.995x) and 1.9-4.6% slower from 512 to 8,192.
- **Decoding ahead hides about half of the decode on the 5090** (1,024 tokens: +8.8% over bf16 against +19.3% decoded first) and
  beats the default only at 8,192 tokens in 12 bits (1.013x against 1.023x) and from 2,048 tiered (1.044 / 1.024 / 1.013x against
  1.058 / 1.059 / 1.067x). The counts' "near bf16 from about 1,000 tokens" did not hold: decoded ahead is still 8.8-9.2% over bf16
  at 1,024 tokens and 1.3% at 8,192 (why, its traffic or the power cap beside cuBLAS, is not measured apart here).
- **On the H100 the decode hardly runs beside cuBLAS**, as the counts said: decoded ahead is at most 8% faster than decoded first
  (2,048-4,096 tokens) and never near bf16; with few warps an SM it runs on those warps alone and takes 1.25-28x. In 12 bits (the
  H100's layout) the best way at each length is 1.06-1.31x bf16: the fused kernel to 1,024 tokens, decoded first or ahead past it,
  GLinear's SPLIT for Qwen3.5-27B. The 12-bit decode runs at about 2.1 TB/s there (11.5 ms of a Qwen3.5-9B pass at 128 tokens; the
  tiered one 9.3 ms), under the HBM's 3.35.
- What is left is the products' kernels against cuBLAS, not the choice of route: the 5090's prompt kernel's 2-5% in 12 bits (6-8% tiered), the H100's
  wgmma kernel's 12-40% and its decode past 1,024 tokens.

Findings, not acted on: the vLLM plugin never takes SPLIT, which on the H100 cut Qwen3.5-27B's pass from
1.256x to 1.139x at 2,048 tokens and 1.143x to 1.090x at 4,096 (GLinear), still short of bf16; the 12-bit decode on Hopper at the
HBM's rate would take about 3-4 ms off a Qwen3.5-9B pass past 1,024 tokens by the counts, still short of bf16.
