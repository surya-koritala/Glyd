# vLLM with `--quantization glyd`: mixtures of experts on an L4 (M4, 2026-09-30)

The plugin's MoE method (`GlydMoEMethod`, 61825a1) on the dev L4. It covers the experts packed at load, their grouped
products, and exact mode's experts decoded for vLLM's own Triton MoE kernel. Then M4's job for two GPUs, run here on
one: the steps that need no second GPU.

- **Machine:** the AWS dev L4 (g6.4xlarge: 24 GB, 72 W; 16 vCPUs), driver 595.91.07.
- **Software:** vLLM 0.30.0 (torch 2.13.0+cu130, transformers 5.17.0), and the library from main's v0.25.1 built for
  sm_89.
- **Model:** ibm-granite/granite-3.1-3b-a800m-instruct, from the box's cache. It has 32 layers, each with 40 experts
  (8 a token), hidden 1,536, and 512 an expert. vLLM runs its bf16 experts by its Triton MoE kernel.

## First runs (`smoke/`: `moe_smoke.sh`, `moe_exact.sh`)

- **Glyd tiered, eager, verified:**
  - 64 Linears packed and verified, and 32 MoE layers' experts. The router's gate (40 rows) stays bf16.
  - Every product was within 3.79e-3 of its reference: the experts' against their matrices decoded, in float32. A
    second call gave the same bits.
- **Glyd 12-bit, compiled with CUDA graphs:** the same layer checks.
- **Exact, eager:** bf16 eager's tokens and logprobs, 8 of 8 prompts bit for bit.

## check_vllm.py --quick: all 15 passed (`check-granite-3.1-3b-a800m-instruct/`, 677 s)

| Against vLLM's bf16 | Glyd tiered | Glyd 12-bit |
| :--- | ---: | ---: |
| Packs decoded to their weights, bit for bit | 64 and 32 MoE layers' experts | 64 and 32 |
| Worst product against its reference, 1-4,096 tokens | 3.79e-3 | 3.79e-3 |
| Top-1 / \|Δ\| on bf16's continuation (bf16 eager's: 0.9883 / 1.16e-2) | 0.9909 / 1.09e-2 | 0.9883 / 1.17e-2 |
| KV cache at 0.85 (bf16: 185,280 tokens) | 213,568 tokens | 202,176 tokens |

- **Compile cache:** a graph each for bf16 and the two layouts; each layout started again loaded its own.
- **Exact, eager:** bf16 eager's tokens, logprobs and continuation, bit for bit (8 of 8).
- **Exact compiled:** refused. With inductor's deterministic mode, compiled bf16's bits (8 of 8, and the
  continuation).
- **Fused compiled in that mode:** the same bits across a restart on its graphs.

## M4's job on one GPU (`job-dryrun/`, `m4dry.sh`)

- **`VJ_TP=2`:** stopped at once, "VJ_TP 2, but 1 GPUs here" (`tp-on-one-gpu.txt`).
- **`VJ_STEPS=moebench,profile`** with granite, Qwen3-1.7B and short rates: every step ran, and results/DONE was written,
  in 706 s.
  - **moebench on granite:** bf16's KV cache 211,968 tokens, Glyd's (tiered) 240,352; 16 prompts at 1 request a
    second, 32 at once.
  - **profile on Qwen3-1.7B:** a step's GPU time by kind, bf16 against Glyd, at decode steps of 1, 8 and 32 sequences
    and prompts of 512 and 2,048 tokens (`results/profile/`).

Tensor parallelism itself, over two GPUs, waits for the instance.
