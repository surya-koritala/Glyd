# The KV cache's tables for a model's whole window, and a 36,000-token prompt through them: Qwen3-8B on an L4 (2026-10-02)

Qwen3-8B, its weights packed as `glyd run` serves them (11.8 GiB), vLLM 0.30.0, the dev L4 (24 GB), eager; `kv: lossless` against vLLM's own cache with the same weights.
The tables are the per-model figures the KV cache is made from, calibrated once on vLLM itself at a model's first start, for the longest context it will serve (`--max-model-len`): 4,096, 8,192, 16,384, 32,768 or 40,960 positions (Qwen3's own window).

- **Making the tables for 40,960 positions took 217 s on the L4** at a first start: the model's load 32 s, the bundled text run through it as two sequences of 40,959 tokens 44 s and 45 s, the model's whole vocabulary 95 s. The file is 3.9 MB (3.8 MB for 4,096 positions, 3.9 for 16,384). Of the values of the worst kind of text, 0.0156% of the keys and 0.0189% of the values fall outside 12 bits (the limit is 0.15%); no token of the vocabulary needs the second reserve.
- **The same through the compiled wheel** (the branch's own CI build, kv-vllm merged on gpu-package-v027): `glyd run Qwen/Qwen3-8B` on a fresh cache directory, with the KV cache on, made those tables as its first start (216 s of the 435 s the command took, the window left at the model's own 40,960 tokens), answered, and a second run with the tables kept took 88 s. Since then `glyd run` (one user's chat) keeps vLLM's own cache unless `GLYD_KV` says otherwise, because of that first start; `glyd serve` and `vllm serve --quantization glyd` hold the cache on an A100, an L4 and an H100 by default. Plain `glyd run` is ready in 39 s with no tables; `GLYD_KV=lossless glyd run --context 8192` on a fresh cache directory made the tables for 8,192 positions in 130 s (ready in 2 m 49 s); `GLYD_KV=auto glyd run` with the 40,960-position tables kept was ready in 40 s, and `glyd serve` in 45 s.
- **A 36,000-token prompt** (this repository's documents, one request, `kv_long.py`; the same weights and the same number of cache blocks on both sides, prompts in steps of 1,024 tokens): all 35,999 prompt logprobs are bit for bit vLLM's (the largest difference 0), the first generated token's logprob is, and the 64 greedy tokens are the same for the first 19 (20 of 64 in all: a decode step's attention is not bit-equal to vLLM's, as in every record of this cache). The prompt took 21.0 s with vLLM's cache and 21.8 s with kv (1.04x as long); a decode step at that depth 67.5 ms and 62.0 ms (1.087x faster).

## Tables that do not reach the window

A key's exponents drift with its position, so the tables are made over sequences as long as the window. Every value is stored exactly whatever the tables, so a window past them changes how much of the cache's room the values that do not fit 12 bits take, never a value. To see how much, the same 40,960-position calibration was run with the exponents counted only from the first 16,384 positions of each sequence (what tables made for 16,384 positions would hold) and the values outside 12 bits counted at every depth. The extra storage a 16-token block needs past its heads' first entries, in entries (mean / 99.9th percentile / largest):

| depth | tables that reach it | tables for 16,384 positions |
| :--- | :--- | :--- |
| 0k to 4k | 0.22 / 13 / 131 | 0.20 / 14 / 123 |
| 4k to 8k | 0.12 / 10 / 16 | 0.12 / 8 / 18 |
| 8k to 12k | 0.16 / 12 / 18 | 0.13 / 8 / 16 |
| 12k to 16k | 0.18 / 12 / 19 | 0.15 / 11 / 19 |
| 16k to 20k | 0.37 / 26 / 37 | 0.54 / 28 / 37 |
| 20k to 24k | 0.33 / 20 / 29 | 2.11 / 53 / 63 |
| 24k to 28k | 0.25 / 12 / 16 | 5.22 / 75 / 90 |
| 28k to 32k | 0.62 / 17 / 32 | 8.10 / 94 / 109 |
| 32k to 36k | 0.77 / 19 / 32 | 10.88 / 110 / 115 |
| 36k to 40k | 0.65 / 15 / 29 | 12.30 / 109 / 114 |

With tables that reach the depth nearly every block's need stays small at every depth (99.9% of them under 26 entries); with tables that stop at 16,384, the blocks past 20,000 positions need 10 to 20 times as much, up to 115 entries, which the cache's first reserve (1.6% of it) has to take. How much of that reserve it uses was not measured; the held-out rate with those tables is 0.0644% for the keys (0.0156% with tables that reach), still under the limit. So a window past the tables costs capacity at the depths it is not calibrated for and nothing in values; the cache therefore does not serve a window its tables do not reach: `kv: auto` keeps vLLM's own cache for it and `kv: lossless` serves it on vLLM's own cache, each with one line that says why.

## Files

`tables-40960.json` and `tables-40960-exponents-from-first-16384.json` (the tables' figures, by depth), `first-start-tables-40960-log.txt` and `tables-40960-exponents-from-first-16384-log.txt` (the calibrations' log lines), `long-stock.json`, `long-kv.json` and `long-compare.txt` (the 36,000-token prompt). The tools are in [`gpu/vllm/`](../../../gpu/vllm): `kv_long.py`, and `python -m glyd.gpu.kvcal`.
