# Qwen3-30B-A3B on an H100: serving a mixture of experts, Glyd against bf16 (2026-10-03)

`vllm serve --quantization glyd` against vLLM's own bf16 path on Qwen/Qwen3-30B-A3B (128 experts, 8 a token): vLLM 0.30.0, torch 2.13 with CUDA 13.0 (driver 580.126.20), one NVIDIA H100 80GB HBM3 (compute capability 9.0, SXM; 26 CPUs; Lambda `gpu_1x_h100_sxm5`).
Glyd is v0.28.0's mixture-of-experts work as it stood on the run's day, with no setting changed by hand; the run was not repeated on the release's final tree.
Four server starts in one run (bf16, Glyd with the lossless KV cache, Glyd, bf16 again), each measured with `vllm bench serve`; the tables are the run's own (`serve/summary.txt`).

**The baseline is vLLM's own bf16 path with its default, untuned fused-MoE configuration.** The bf16 server's log says `Using default MoE config. Performance might be sub-optimal!`: vLLM has no tuned file for E=128, N=768 on this GPU (`serve/bf16/serve-bf16.txt.gz`). A tuned bf16 would be faster than the baseline here; that was not measured. Name the baseline wherever these numbers are quoted.

## What to know

- **The run is comparable.** Every mode's saturated pass ran at a median SM clock of 1,980 MHz (the H100's own top), at most 60 C, with no thermal slowdown from the driver, and the two bf16 passes, the first and the last of the four, are 1.7% apart in requests a second (12.32 and 12.53) (`serve/clocks.txt`).
- **Saturated, 256 requests at once, 1,024 tokens in and 256 out** (`vllm bench serve`, 0.9 of the GPU): bf16 12.32 requests a second, **Glyd 13.13 (1.07x)**, **Glyd with the lossless KV cache 17.10 (1.39x)**; against the mean of the two bf16 passes (12.43) 1.06x and 1.38x. Output tokens a second 3,155, 3,361 and 4,378. The first token comes after 6,920, 3,853 and 3,437 ms (0.56x and 0.50x). Each token takes 27.2, 45.9 and 43.7 ms (1.69x and 1.61x as long): Glyd has room for more requests at once, 155 and 245 sequences in a decode step on average against bf16's 88.
- **One request a second** (64 requests): the same 0.97 to 0.98 requests a second; the first token 62, 56 and 60 ms (0.90x and 0.97x), each token 6.6, 6.3 and 6.3 ms (0.95x).
- **Memory:** the weights take 56.88 GiB in bf16, 44.65 GiB with Glyd and 45.41 GiB with the lossless KV cache's tables; the KV cache holds 118,720, 251,696 and 331,008 tokens (2.12x and 2.79x bf16's) at the same 0.9, and only the last holds all 256 requests (327,680 tokens) at once.
- **Where Glyd is behind: the prompt steps.** Of the saturated pass's 14.3 s with the KV cache, 5.7 s (40%) are 32 prompt steps, which take 22 us a token against bf16's 15 (1.47x); its 255 decode steps take 8.6 s against bf16's 15.7 s (`serve/steps.txt`). Layer by layer, Glyd's experts take 0.90x to 1.04x of vLLM's bf16 layer's time at 16 to 232 tokens a step (a decode step), 1.08x to 1.15x at 384 to 768, and 1.38x, 1.73x, 1.97x and 2.01x at 1,152, 2,048, 4,096 and 8,192 (prompt steps), each the best of the settings tried, which the code's own choice is within 0.9% of.
- **Every check passes:** 84 of 84 (the model's layer 0 against an fp32 product at 1 to 4,096 tokens a step, uneven loads, a layer with exceptions by the dozen) (`steps.txt`).
- **The attention products stay bf16 for a mixture of experts' model on an H100** (the plugin's default; the router too): Glyd's own product for them is slower than cuBLAS's there.

## Serving, each mode against bf16 (`vllm bench serve`, random dataset, 1,024 tokens in, 256 out, seeds apart per pass)

`glyd-picked` is Glyd's weights with vLLM's own KV cache, `glyd-picked+kv` with the lossless KV cache (its tables for the model are made at the first start, 78 s here), `bf16-2` the bf16 server started again at the end.

| Rate (req/s) | Mode | Requests/s | vs bf16 | First token (ms) | vs bf16 | Each token (ms) | vs bf16 | SM clock, median (MHz) | Highest temperature (C) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.97 | 1.00x | 62 | 1.00x | 6.6 | 1.00x | 1980 | 44 |
| 1 | glyd-picked | 0.98 | 1.01x | 56 | 0.90x | 6.3 | 0.95x | 1980 | 44 |
| 1 | glyd-picked+kv | 0.97 | 1.00x | 60 | 0.97x | 6.3 | 0.95x | 1980 | 47 |
| 1 | bf16-2 | 0.97 | 1.00x | 63 | 1.02x | 6.6 | 1.00x | 1980 | 44 |
| inf | bf16 | 12.32 | 1.00x | 6920 | 1.00x | 27.2 | 1.00x | 1980 | 58 |
| inf | glyd-picked | 13.13 | 1.07x | 3853 | 0.56x | 45.9 | 1.69x | 1980 | 60 |
| inf | glyd-picked+kv | 17.10 | 1.39x | 3437 | 0.50x | 43.7 | 1.61x | 1980 | 57 |
| inf | bf16-2 | 12.53 | 1.02x | 6951 | 1.00x | 27.1 | 1.00x | 1980 | 58 |

## The clocks of each pass

| Mode | Pass | Requests/s | SM clock median (MHz) | lowest | Highest temperature (C) | Power median (W) | Seconds |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1 |  | 1980 | 1980 | 44 | 288 | 81 |
| bf16 | inf | 12.32 | 1980 | 1635 | 58 | 577 | 21 |
| bf16-2 | 1 |  | 1980 | 1980 | 44 | 309 | 80 |
| bf16-2 | inf | 12.53 | 1980 | 1485 | 58 | 572 | 20 |
| glyd-picked | 1 |  | 1980 | 1980 | 44 | 306 | 80 |
| glyd-picked | inf | 13.13 | 1980 | 1650 | 60 | 579 | 19 |
| glyd-picked+kv | 1 |  | 1980 | 1980 | 47 | 291 | 80 |
| glyd-picked+kv | inf | 17.10 | 1980 | 1845 | 57 | 569 | 14 |

## Where the saturated pass's time went (the servers' iteration lines)

| Mode | Steps | Seconds | Prompt steps | s | ms a step | us a token | Decode steps | s | ms a step | us a token | Decode steps' mean sequences | Highest KV usage |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 778 | 20.3 | 58 | 4.6 | 79 | 15 | 720 | 15.7 | 21.8 | 259 | 88 | 100% |
| glyd-picked | 520 | 18.9 | 40 | 6.7 | 168 | 22 | 480 | 12.2 | 25.4 | 204 | 155 | 100% |
| glyd-picked+kv | 287 | 14.3 | 32 | 5.7 | 178 | 22 | 255 | 8.6 | 33.9 | 141 | 245 | 98% |
| bf16-2 | 778 | 20.0 | 58 | 4.6 | 79 | 15 | 720 | 15.4 | 21.3 | 254 | 89 | 100% |

## Not measured here

- A tuned bf16 baseline (vLLM's tuning script for E=128, N=768 on this GPU).
- Other prompt and output lengths (1,024 in and 256 out only), other models, tensor parallel, an H200 or a PCIe H100.
- Prompt-heavy mixes: the layer-by-layer figures above put the gap at 1,152 to 8,192 tokens a step.

## Files

- `serve/`: a directory a mode (`bf16`, `glyd-picked`, `glyd-picked+kv`, `bf16-2`) with each pass's `vllm bench serve` output and result (`*.json`), the mode's summary, and the nvidia-smi samples of each pass (`smi-*.csv`) and of the whole server start (`smi-reasons.csv`: temperature, SM clock, power and the driver's slowdown reasons a second); the bf16 servers' logs (`serve-bf16.txt.gz`, vLLM's own). `summary.txt` has the tables, `clocks.txt` the clocks and `steps.txt` the iteration lines' table.
- `steps.txt`: the run's steps with their times; `machine.txt`, `env.txt`.
