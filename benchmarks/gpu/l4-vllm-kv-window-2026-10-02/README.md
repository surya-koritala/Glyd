# The lossless KV cache at a model's whole window: the first start, and a 36,000-token prompt (2026-10-02)

Qwen3-8B, its weights packed as `glyd run` serves them (11.8 GiB), vLLM 0.30.0, an NVIDIA L4 (24 GB), eager; `kv: lossless` against vLLM's own cache with the same weights.
At a model's first start the cache is set up once for that model, for the longest context it will serve (`--max-model-len`): 4,096, 8,192, 16,384, 32,768 or 40,960 tokens (Qwen3's own window). The set-up is kept in `~/.cache/glyd/kv`.

- **The set-up for 40,960 tokens took 217 s on the L4** at a first start (the model's load 32 s of it). The file is 3.9 MB.
- **Through the compiled wheel** (glyd-gpu as the release builds it), `GLYD_KV=lossless glyd run --context 8192` on a fresh cache directory made the set-up for 8,192 tokens in 130 s, the KV cache on, at its first start. `glyd run` (one user's chat) keeps vLLM's own cache unless `GLYD_KV` says otherwise, because of that first start; `glyd serve` and `vllm serve --quantization glyd` hold the cache on an A100, an L4, an H100 and a GH200 by default.
- **A 36,000-token prompt** (this repository's documents, one request; the same weights and the same number of cache blocks on both sides): all 35,999 prompt logprobs are bit for bit vLLM's (the largest difference 0), the first generated token's logprob is, and the 64 greedy tokens are the same for the first 19 (20 of 64 in all: a decode step's attention is not bit-equal to vLLM's, as in every record of this cache). The prompt took 20.993 s with vLLM's cache and 21.83 s with kv (1.04x as long); a decode step at that depth 67.459 ms and 62.045 ms (1.087x faster).
- **A window past 40,960 tokens** keeps vLLM's own cache: `kv: auto` falls back to it with one line that says why, and `kv: lossless` serves the window on vLLM's cache too, with the same line.

## Files

`first-start-log.txt` (the first start's log lines, the descriptions of what each step does removed), `long-stock.json`, `long-kv.json` and `long-compare.txt` (the 36,000-token prompt).
