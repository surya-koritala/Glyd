# Qwen3-16B-A3B on an A100: serving a mixture of experts, Glyd against bf16 (2026-10-03)

`vllm serve --quantization glyd` against vLLM's own bf16 path on kalomaze/Qwen3-16B-A3B: vLLM 0.30.0, torch 2.13 with CUDA 13.0 (driver 580.126.20), one NVIDIA A100-SXM4-40GB (compute capability 8.0, 40 GB, 400 W; 30 CPUs, AMD EPYC 7J13; Lambda `gpu_1x_a100_sxm4`).
Glyd is v0.28.0's mixture-of-experts work as it stood on the run's day, with no setting changed by hand; the run was not repeated on the release's final tree.
Three server starts in one run (bf16, Glyd with the lossless KV cache, bf16 again), each measured with `vllm bench serve`; the tables are the run's own (`serve/summary.txt`).

**The serve model is `kalomaze/Qwen3-16B-A3B`**: Qwen3-30B-A3B with half its experts (64 of 128 a layer pruned by their routing statistics; Apache-2.0): the same 48 layers, hidden 2,048, intermediate 768, 8 experts a token and the same attention, 16.03B parameters, 29.87 GiB in bf16. Qwen3-30B-A3B itself (57 GiB) does not fit this GPU in bf16. The layer-by-layer table below is Qwen3-30B-A3B's own layer 0 (128 experts), as the H100 record has it.

**The baseline is vLLM's own bf16 path with its default, untuned fused-MoE configuration**: its log says `Using default MoE config. Performance might be sub-optimal!` and names the file it did not find, `E=64,N=768,device_name=NVIDIA_A100-SXM4-40GB.json` (`serve/bf16/serve-bf16.txt.gz`). A tuned bf16 would be faster than the baseline here; that was not measured. Name the baseline wherever these numbers are quoted.

## What to know

- **The run is comparable** (`serve/clocks.txt`): every saturated pass at a median SM clock of 1,410 MHz (the A100's top), at most 64 C, no slowdown from the driver in any server's run, and the two bf16 passes, the first and the last of three, 4.4% apart (4.12 and 3.94 requests a second).
- **Saturated, 256 requests at once, 1,024 tokens in and 256 out**: bf16 4.12 requests a second (3.94 the second time), **Glyd with the lossless KV cache 6.18 (1.50x; 1.57x against the second bf16 pass)**. The first token comes after 27,100 and 14,233 ms (0.53x); each token takes 30.9 and 62.9 ms (2.04x as long). The gain is room: the weights take 29.87 and 23.97 GiB, the KV cache holds 40,288 and 144,896 tokens (3.60x), and a decode step carries 32 and 103 sequences on average.
- **One request a second** (64 requests): 0.94 and 0.95 requests a second, each token 13.9 and 12.3 ms (0.88x), and **the first token 102 and 110 ms: 1.08x later** (p99 401 and 175 ms).
- **Every check passes:** 162 of 162 (the real layer against an fp32 product at 1 to 4,096 tokens a step, uneven loads, a layer with exceptions by the dozen) (`steps.txt`).
- **The server of an A100 cuts a step at 2,048 tokens** (vLLM's default for this GPU; the H100's is 8,192), so no prompt step here has more: they take 15.7 s of bf16's 61.6 s saturated pass and 22.0 s of Glyd's 40.9 s (133 ms a step, 72 us a prompt token against 52: 1.38x), and Glyd's decode steps 30.5 ms with 103 sequences, 380 us a token against bf16's 758 (`serve/steps.txt`).
- **Layer by layer, Glyd's experts against vLLM's bf16 layer** (the table below): 0.75x at 16 tokens a step, 0.96x to 1.12x from 32 to 512 (1.01x at 232), 1.22x at 768, 1.67x at 1,152, 1.40x at 1,536, 1.58x at 2,048, and 1.71x, 1.70x and 1.75x at 3,072, 4,096 and 8,192 tokens.
- **The attention products and the router stay bf16** on an A100 (the plugin's default); the weights above include them as bf16.

## Serving, each mode against bf16 (`vllm bench serve`, random dataset, 1,024 tokens in, 256 out)

`glyd-picked+kv` is Glyd's weights with the lossless KV cache (its tables for the model made at the first start, 107 s), `bf16-2` the bf16 server started again at the end, to show the machine's drift. 64 prompts at 1 a second, then 256 at once.

| Rate (req/s) | Mode | Requests/s | vs bf16 | First token (ms) | vs bf16 | Each token (ms) | vs bf16 | SM clock, median (MHz) | Highest temperature (C) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.94 | 1.00x | 102 | 1.00x | 13.9 | 1.00x | 1410 | 49 |
| 1 | glyd-picked+kv | 0.95 | 1.01x | 110 | 1.08x | 12.3 | 0.88x | 1410 | 50 |
| 1 | bf16-2 | 0.94 | 1.00x | 103 | 1.01x | 14.1 | 1.01x | 1410 | 50 |
| inf | bf16 | 4.12 | 1.00x | 27100 | 1.00x | 30.9 | 1.00x | 1410 | 64 |
| inf | glyd-picked+kv | 6.18 | 1.50x | 14233 | 0.53x | 62.9 | 2.04x | 1410 | 56 |
| inf | bf16-2 | 3.94 | 0.96x | 28024 | 1.03x | 32.2 | 1.04x | 1410 | 64 |

## The clocks of each pass

| Mode | Pass | Requests/s | SM clock median (MHz) | lowest | Highest temperature (C) | Power median (W) | Seconds |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 1 |  | 1410 | 1095 | 49 | 200 | 92 |
| bf16 | inf | 4.12 | 1410 | 1410 | 64 | 264 | 61 |
| bf16-2 | 1 |  | 1410 | 1095 | 50 | 200 | 90 |
| bf16-2 | inf | 3.94 | 1410 | 1410 | 64 | 265 | 65 |
| glyd-picked+kv | 1 |  | 1410 | 1095 | 50 | 196 | 89 |
| glyd-picked+kv | inf | 6.18 | 1410 | 1410 | 56 | 284 | 41 |

## Where the saturated pass's time went (the servers' iteration lines)

| Mode | Steps | Seconds | Prompt steps | s | ms a step | us a token | Decode steps | s | ms a step | us a token | Decode steps' mean sequences | Highest KV usage |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 2076 | 61.6 | 171 | 15.7 | 92 | 52 | 1905 | 45.9 | 24.1 | 758 | 32 | 100% |
| glyd-picked+kv | 784 | 40.9 | 166 | 22.0 | 133 | 72 | 618 | 18.9 | 30.5 | 380 | 103 | 100% |
| bf16-2 | 2077 | 64.6 | 173 | 16.0 | 92 | 52 | 1904 | 48.7 | 25.6 | 805 | 32 | 100% |

## The experts' layer by tokens a step, GPU time in a CUDA graph (Qwen3-30B-A3B's layer 0, microseconds)

vLLM's own bf16 layer (its default fused-MoE configuration) against Glyd's, as the code runs it.

| Tokens a step | bf16 (vLLM's layer) | Glyd | Glyd / bf16 |
| ---: | ---: | ---: | ---: |
| 16 | 616 us | 464 us | 0.75x |
| 32 | 763 us | 732 us | 0.96x |
| 64 | 857 us | 890 us | 1.04x |
| 128 | 919 us | 920 us | 1.00x |
| 232 | 947 us | 960 us | 1.01x |
| 384 | 975 us | 1,033 us | 1.06x |
| 512 | 996 us | 1,111 us | 1.12x |
| 768 | 1,155 us | 1,413 us | 1.22x |
| 1,152 | 1,209 us | 2,014 us | 1.67x |
| 1,536 | 1,266 us | 1,776 us | 1.40x |
| 2,048 | 1,544 us | 2,435 us | 1.58x |
| 3,072 | 1,998 us | 3,413 us | 1.71x |
| 4,096 | 2,477 us | 4,200 us | 1.70x |
| 8,192 | 4,456 us | 7,816 us | 1.75x |

## Not measured here

- A tuned bf16 baseline (vLLM's tuning script for E=64, N=768 on this GPU).
- Other prompt and output lengths (1,024 in and 256 out only), other models, tensor parallel, an 80 GB A100.

## Files

- `serve/`: a directory a mode (`bf16`, `glyd-picked+kv`, `bf16-2`) with each pass's `vllm bench serve` output and result (`*.json`), the mode's summary, and the nvidia-smi samples of each pass (`smi-*.csv`) and of the whole server start (`smi-reasons.csv`: temperature, SM clock, power and the driver's slowdown reasons a second); the bf16 servers' logs (`serve-bf16.txt.gz`, vLLM's own). `summary.txt` has the tables, `clocks.txt` the clocks and `steps.txt` the iteration lines' table.
- `steps.txt`: the run's steps with their times; `machine.txt`, `env.txt`.
