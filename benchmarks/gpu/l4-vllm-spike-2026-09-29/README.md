# vLLM with `--quantization glyd`: the M1 spike on an L4 (2026-09-29)

This was a first test of the question: does Glyd's product op run inside vLLM's torch.compile and CUDA graphs? It
also measured what the packed weights do to vLLM's KV cache.

## Setup

- **Machine:** AWS g6.4xlarge, NVIDIA L4 (23 GB).
- **Software:** vLLM 0.30.0 with torch 2.13.0+cu130 and transformers 5.17.0, in a venv of its own.
- **Library:** built from main (v0.25.0).
- **Plugin:** `bindings/python/glyd/gpu/vllm_plugin.py` as committed with these logs, loaded from its
  `vllm.general_plugins` entry point.

## Scripts

- `spike.py` runs `LLM(..., quantization="glyd" or bf16, gpu_memory_utilization=0.85, max_model_len=4096)`. It
  generates greedily, 64 tokens for each of 8 prompts. It records the tokens, their logprobs and the KV cache's blocks.
  It then times 256 tokens at 1, 8 and 32 sequences.
- `spike_run.sh` runs the whole set, each run in a process and a compile cache of its own. At the end it starts `vllm
  serve --quantization glyd` and sends one request.
- `compare.py` compares each Glyd run with its bf16 run.
- `peak.sh` measures vLLM's own "Peak GPU memory after loading weights" for Qwen3-8B, in bf16 and in Glyd.

## Results

Everything is in `results/`. The comparison is `results/compare.txt`.

**Qwen3-1.7B:**

| Run | Tokens/s at 1 / 8 / 32 sequences | KV cache |
| :--- | :--- | ---: |
| bf16, CUDA graphs | 68.8 / 514.8 / 1755.2 | 128,064 tokens |
| Glyd tiered | 80.6 / 606.6 / 1981.9 | 141,584 tokens |
| Glyd 12-bit | 82.0 / 619.0 / 2089.8 | 136,304 tokens |

**Qwen3-8B:**

| Run | Tokens/s at 1 / 8 / 32 sequences | KV cache | Weights | Load peak |
| :--- | :--- | ---: | ---: | ---: |
| bf16 | 16.7 / 126.1 / 448.2 | 11,056 tokens | 15.27 GiB | 15.27 GiB |
| Glyd tiered | 21.5 / 166.0 / 549.5 | 40,368 tokens | 11.64 GiB | 12.03 GiB |

**Exact mode** (`GLYD_EXACT=1`) with `--enforce-eager` gives bf16 eager's tokens and logprobs, bit for bit, on all 8
prompts. With CUDA graphs it does not.

**Default-mode tokens against bf16** (Qwen3-1.7B, tiered, both with CUDA graphs):

- tokens the same: [17, 64, 64, 64, 64, 40, 64, 42] of 64;
- mean |Δ logprob| 9.7e-3.

**bf16 against itself, CUDA graphs against eager:**

- tokens the same: [13, 64, 12, 64, 0, 64, 64, 64] of 64;
- mean |Δ logprob| 1.08e-2.

**`q17-glyd-mma12-sharedcache`:** the 12-bit layout started on the tiered run's compile cache stops at a compiled
input-size check.

**Glyd eager** runs at 0.85-0.91x bf16 eager. That is Python and ctypes per product. CUDA graphs take it away.
