# Qwen3-30B-A3B on an H100: MoE serving with the H100's settings built in, against bf16 (2026-10-03)

Qwen/Qwen3-30B-A3B (128 experts, 8 a token), bf16 weights, vLLM 0.30.0, torch 2.13 with CUDA 13.0 (driver 580.126.20), one NVIDIA H100 80GB HBM3 (compute capability 9.0, SXM; 26 CPUs; Lambda `gpu_1x_h100_sxm5`).
The tree is moe-v028 a045def3 on v0.27.0's sources (private/v027 0880bfda): the settings an H100 takes, by the name the driver gives it, are in the code, and no tuning variable is set for any Glyd run here.
One run of the job `job/moe28b_job.sh` (the run called moe28c): the experts' checks, `vllm bench serve` in four server starts (bf16, Glyd with the lossless KV cache, Glyd, bf16 again), the layer sweep and the dense products.
The tables below are the job's own (`moe28c/summary.txt`); the files are listed at the end.

**The baseline is vLLM's own bf16 path with its default, untuned fused-MoE configuration.** The bf16 server's log says `Using default MoE config. Performance might be sub-optimal!`: vLLM has no tuned file for E=128, N=768 on this GPU (`moe28c/serve/bf16/serve-bf16.txt.gz`). A tuned bf16 would be faster than the baseline here; that was not measured. Name the baseline wherever these numbers are quoted.

## What to know

- **The run is comparable.** Every mode's saturated pass ran at a median SM clock of 1,980 MHz (the H100's own top), at most 60 C, with no thermal slowdown from the driver (power cap in 6 to 7 s of each server's 227 to 311 s, hardware and software thermal slowdown in none), and the two bf16 passes, the first and the last of the four, are 1.7% apart in requests a second (12.32 and 12.53). The job's thermal guard (`moe28c/serve/clocks.txt`) says COMPARABLE.
- **Saturated, 256 requests at once, 1,024 tokens in and 256 out** (`vllm bench serve`, 0.9 of the GPU): bf16 12.32 requests a second, **Glyd 13.13 (1.07x)**, **Glyd with the lossless KV cache 17.10 (1.39x)**; against the mean of the two bf16 passes (12.43) 1.06x and 1.38x. Output tokens a second 3,155, 3,361 and 4,378. The first token comes after 6,920, 3,853 and 3,437 ms (0.56x and 0.50x). Each token takes 27.2, 45.9 and 43.7 ms (1.69x and 1.61x as long): Glyd has room for more requests at once, 155 and 245 sequences in a decode step on average against bf16's 88.
- **One request a second** (64 requests): the same 0.97 to 0.98 requests a second; the first token 62, 56 and 60 ms (0.90x and 0.97x), each token 6.6, 6.3 and 6.3 ms (0.95x).
- **Memory**: the weights take 56.88 GiB in bf16, 44.65 GiB with Glyd and 45.41 GiB with the lossless KV cache's tables; the KV cache holds 118,720, 251,696 and 331,008 tokens (2.12x and 2.79x bf16's) at the same 0.9, and only the last holds all 256 requests (327,680 tokens) at once.
- **Where Glyd is behind: the prompt steps.** Of the saturated pass's 14.3 s with the KV cache, 5.7 s (40%) are 32 prompt steps, which take 22 us a token against bf16's 15 (1.47x); its 255 decode steps take 8.6 s against bf16's 15.7 s. The layer sweep says the same: the experts' layer (best variant; the built-in rule is within 0.9% of the best tile) takes 0.90x to 1.04x of vLLM's bf16 layer's time at 16 to 232 tokens a step (a decode step), 1.08x to 1.15x at 384 to 768, and 1.38x, 1.73x, 1.97x and 2.01x at 1,152, 2,048, 4,096 and 8,192 (prompt steps).
- **Every check passes**: 84 of 84 (the model's layer 0 against an fp32 product at 1 to 4,096 tokens a step with each wgmma tile forced, the products from 17 tokens a step, and a refused product made an error; uneven loads; a layer with exceptions by the dozen), the worst difference from the fp32 product, as the check prints it, 3.1e-03 (gate and up) and 1.1e-02 (down) at the real layer; the wgmma workspace's test passes (`moe28c/check/`).
- **The attention products stay bf16 for a mixture of experts' model on an H100** (the plugin's default; the router too): the dense table shows Glyd's best route taking 1.0x to 2.4x the time of cuBLAS for the qkv, o and router products at 1 to 512 tokens a step.

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

| Mode | Steps | Count | Seconds | Mean |
| :--- | :--- | ---: | ---: | ---: |
| bf16 | decode steps of <64 sequences | 253 | 4.4 s | 17.3 ms a step |
| bf16 | decode steps of 64-127 sequences | 467 | 11.3 s | 24.1 ms a step |
| glyd-picked | decode steps of <64 sequences | 241 | 4.1 s | 17.0 ms a step |
| glyd-picked | decode steps of 64-127 sequences | 8 | 0.2 s | 25.3 ms a step |
| glyd-picked | decode steps of 128-191 sequences | 6 | 0.2 s | 32.0 ms a step |
| glyd-picked | decode steps of 192-255 sequences | 225 | 7.7 s | 34.3 ms a step |
| glyd-picked+kv | decode steps of <64 sequences | 8 | 0.1 s | 16.8 ms a step |
| glyd-picked+kv | decode steps of 64-127 sequences | 8 | 0.2 s | 24.5 ms a step |
| glyd-picked+kv | decode steps of 128-191 sequences | 8 | 0.2 s | 31.1 ms a step |
| glyd-picked+kv | decode steps of 192-255 sequences | 26 | 0.9 s | 34.8 ms a step |
| glyd-picked+kv | decode steps of 256+ sequences | 205 | 7.2 s | 34.9 ms a step |
| bf16-2 | decode steps of <64 sequences | 253 | 4.0 s | 15.8 ms a step |
| bf16-2 | decode steps of 64-127 sequences | 467 | 11.4 s | 24.4 ms a step |

## The experts' layer by tokens a step, GPU time in a CUDA graph (Qwen3-30B-A3B's layer 0, microseconds)

vLLM's own bf16 layer (its default fused-MoE configuration) against the built-in settings (`new`); `nt16` to `nt128` are the wgmma tile forced to that size, `off` the wgmma products turned off (the other kernels).
The last column is the plugin's decoded route (the experts decoded, then vLLM's layer on them), which the plugin takes from 8,192 tokens a step on an H100.

| Tokens a step | bf16 (vLLM's layer) | v027 | new | best variant | best / bf16 | the plugin's decoded route* / bf16 |
| ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| 16 | 272 us | - | 247 us | nt32 246 us | 0.90x | 3.46x |
| 24 | 330 us | - | 339 us | new 339 us | 1.03x | 3.45x |
| 32 | 362 us | - | 381 us | nt16 377 us | 1.04x | 3.46x |
| 48 | 394 us | - | 401 us | new 401 us | 1.02x | 3.43x |
| 64 | 404 us | - | 414 us | nt16 414 us | 1.02x | 3.42x |
| 128 | 424 us | - | 421 us | nt16 421 us | 0.99x | 3.34x |
| 232 | 432 us | - | 438 us | nt32 438 us | 1.01x | 3.32x |
| 384 | 442 us | - | 482 us | nt64 480 us | 1.08x | 3.27x |
| 512 | 451 us | - | 506 us | nt64 502 us | 1.12x | 3.24x |
| 768 | 477 us | - | 548 us | new 548 us | 1.15x | 3.12x |
| 1,152 | 502 us | - | 693 us | nt128 692 us | 1.38x | 3.03x |
| 2,048 | 607 us | - | 1,052 us | new 1,052 us | 1.73x | 2.71x |
| 4,096 | 940 us | - | 1,856 us | nt128 1,850 us | 1.97x | 2.15x |
| 8,192 | 1,713 us | - | 3,453 us | new 3,453 us | 2.01x | 1.67x |

The wgmma tile the built-in rule takes against each forced tile (us a layer):

| Tokens | rule's tile | new | nt16 | nt32 | nt64 | nt128 | best tile | the rule vs best | off (the other kernels) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 16 | 247 | 246 | 246 | 247 | 247 | 32 | +0.4% | 246 |
| 24 | 16 | 339 | 349 | 354 | 372 | 439 | 16 | -2.7% | 391 |
| 32 | 16 | 381 | 377 | 391 | 405 | 465 | 16 | +0.9% | 423 |
| 48 | 16 | 401 | 406 | 419 | 439 | 497 | 16 | -1.2% | 488 |
| 64 | 16 | 414 | 414 | 421 | 448 | 516 | 16 | +0.1% | 498 |
| 128 | 16 | 421 | 421 | 426 | 454 | 512 | 16 | +0.1% | 515 |
| 232 | 32 | 438 | 535 | 438 | 463 | 532 | 32 | +0.2% | 538 |
| 384 | 32 | 482 | 802 | 481 | 480 | 558 | 64 | +0.6% | 574 |
| 512 | 64 | 506 | 977 | 627 | 502 | 577 | 64 | +0.8% | 610 |
| 768 | 64 | 548 | 1,369 | 857 | 551 | 630 | 64 | -0.6% | 708 |
| 1,152 | 128 | 693 | 1,945 | 1,208 | 915 | 692 | 128 | +0.1% | 1,203 |
| 2,048 | 128 | 1,052 | 3,281 | 1,905 | 1,269 | 1,053 | 128 | -0.1% | 1,850 |
| 4,096 | 128 | 1,856 | 6,330 | 3,607 | 2,377 | 1,850 | 128 | +0.3% | 3,161 |
| 8,192 | 128 | 3,453 | 12,391 | 6,991 | 4,452 | 3,470 | 128 | -0.5% | 5,781 |

the rule's layer time is at most 0.9% over the best forced tile at the tokens measured

## The attention and router products against cuBLAS, one call, in a CUDA graph (microseconds)

Qwen3-30B-A3B's qkv (5,120 x 2,048), o (2,048 x 4,096) and router (128 x 2,048) products. The routes: `decode` the matrix unpacked, then cuBLAS; `default` the library's own pick for that many tokens; `gemm` the small-M kernel (to 64 tokens); `wg` the wgmma kernel; each with its time over cuBLAS's in brackets, and "best / bf16" the fastest of them over cuBLAS.

| Matrix (O x K) | Tokens a step | bf16 (cuBLAS) | decode | default | gemm | wg | best / bf16 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 x 2048 | 1 | 5.3 us | 8.1 (1.54) | 7.7 (1.46) | 7.7 (1.46) | 9.7 (1.84) | 1.46x |
| 128 x 2048 | 8 | 5.2 us | 8.1 (1.57) | 8.1 (1.56) | 8.0 (1.55) | 9.8 (1.88) | 1.55x |
| 128 x 2048 | 16 | 5.3 us | 8.2 (1.55) | 8.9 (1.66) | 8.9 (1.66) | 9.9 (1.85) | 1.55x |
| 128 x 2048 | 32 | 5.5 us | 8.4 (1.53) | 10.8 (1.95) | 11.9 (2.15) | 10.8 (1.95) | 1.53x |
| 128 x 2048 | 64 | 5.5 us | 8.5 (1.55) | 12.0 (2.18) | 17.2 (3.12) | 12.1 (2.19) | 1.55x |
| 128 x 2048 | 128 | 5.6 us | 8.6 (1.54) | 15.7 (2.82) | - | 15.6 (2.80) | 1.54x |
| 128 x 2048 | 232 | 5.6 us | 8.6 (1.54) | 20.5 (3.67) | - | 20.3 (3.62) | 1.54x |
| 128 x 2048 | 512 | 5.8 us | 8.8 (1.52) | 20.9 (3.61) | - | 20.7 (3.58) | 1.52x |
| 2048 x 4096 | 1 | 10.8 us | 23.7 (2.19) | 11.0 (1.01) | 11.0 (1.01) | 11.8 (1.09) | 1.01x |
| 2048 x 4096 | 8 | 10.8 us | 23.7 (2.20) | 11.8 (1.09) | 11.7 (1.09) | 11.8 (1.10) | 1.09x |
| 2048 x 4096 | 16 | 10.8 us | 24.0 (2.22) | 13.1 (1.21) | 13.1 (1.21) | 11.9 (1.10) | 1.10x |
| 2048 x 4096 | 32 | 10.9 us | 24.1 (2.21) | 12.7 (1.17) | 20.7 (1.89) | 12.7 (1.17) | 1.17x |
| 2048 x 4096 | 64 | 11.2 us | 24.5 (2.19) | 14.2 (1.27) | 30.4 (2.71) | 14.1 (1.26) | 1.26x |
| 2048 x 4096 | 128 | 10.7 us | 25.5 (2.38) | 19.8 (1.85) | - | 20.3 (1.89) | 1.85x |
| 2048 x 4096 | 232 | 11.4 us | 26.5 (2.33) | 33.5 (2.94) | - | 32.9 (2.88) | 2.33x |
| 2048 x 4096 | 512 | 17.5 us | 33.3 (1.90) | 35.4 (2.02) | - | 35.4 (2.02) | 1.90x |
| 5120 x 2048 | 1 | 10.0 us | 25.7 (2.57) | 15.0 (1.50) | 14.9 (1.49) | 12.5 (1.24) | 1.24x |
| 5120 x 2048 | 8 | 10.0 us | 24.8 (2.48) | 15.6 (1.56) | 15.5 (1.55) | 12.5 (1.25) | 1.25x |
| 5120 x 2048 | 16 | 9.9 us | 24.8 (2.49) | 17.0 (1.71) | 16.9 (1.70) | 12.6 (1.27) | 1.27x |
| 5120 x 2048 | 32 | 10.1 us | 25.0 (2.47) | 13.3 (1.31) | 22.4 (2.21) | 13.4 (1.32) | 1.31x |
| 5120 x 2048 | 64 | 10.2 us | 25.5 (2.52) | 14.7 (1.44) | 31.8 (3.13) | 14.6 (1.44) | 1.44x |
| 5120 x 2048 | 128 | 10.1 us | 26.3 (2.59) | 18.6 (1.84) | - | 18.6 (1.84) | 1.84x |
| 5120 x 2048 | 232 | 11.1 us | 30.6 (2.75) | 27.7 (2.49) | - | 26.9 (2.42) | 2.42x |
| 5120 x 2048 | 512 | 17.9 us | 36.5 (2.04) | 39.9 (2.23) | - | 39.9 (2.23) | 2.04x |

## Not measured here

- A tuned bf16 baseline (vLLM's tuning script for E=128, N=768 on this GPU).
- Other prompt and output lengths (1,024 in and 256 out only), other models, tensor parallel, an H200 or a PCIe H100.
- The prompt steps' share of other mixes: the layer sweep puts the gap at 1,152 to 8,192 tokens a step.

## Files

- `job/`: the job (`moe28b_job.sh`, run as `bash ~/moe28b_job.sh` on a fresh instance with `moe28b_src.tar`, the tree's `gpu/` and `glyd-gpu-py/` at a045def3 and `job/`) and the scripts that make its tables (`sweep_summary.py`, `tile_verdict.py`, `ncu_summary.py`).
- `moe28c/summary.txt`: the job's summary, the tables above and the KV cache's lines; `steps.txt` its steps with times; `machine.txt`, `machine-short.txt`, `env.txt`, `COMMIT`, `job.log`.
- `moe28c/serve/`: a directory a mode (`bf16`, `glyd-picked`, `glyd-picked+kv`, `bf16-2`) with each pass's `vllm bench serve` output and result (`*.json`), the server's log (`serve-*.txt.gz`: gunzip it for `job/sweep_summary.py steps`, which reads the iteration lines), the nvidia-smi samples of each pass (`smi-*.csv`) and of the whole server start (`smi-reasons.csv`: temperature, SM clock, power and the driver's slowdown reasons a second); `clocks.txt`, `steps.txt` and `kv-notes.txt` beside them.
- `moe28c/check/check.txt`: the 84 checks; `moe28c/sweep/`: the layer sweep (`layer.json`, `layer.txt`, `summary.txt`); `moe28c/dense/`: the dense products; `moe28c/log/`: the downloads, the environment, the thermal preflight (42 s of matmuls on the idle GPU: 67 C, 698 W, no thermal slowdown).

## Nsight Compute on the grouped gate and up product (a separate run, 2026-10-03; `prof/`)

The job's prof part (`M28_PARTS=prof`, job 8d7f35cb, tree moe-a100 9267523e, the same H100 SXM at its locked 1.44 GHz under ncu 2025.3.1) captured `mma12_moe_wg_kernel` for Qwen3-30B-A3B's layer 0 at 232 and 8,192 tokens a step. Both captured launches are the gate and up product (the check calls it once for each of 12 routings before it calls the down), so the down kernel is not in it; the job now makes the check's calls once, and its next prof run captures both. `prof/summary.txt` has the metrics; `prof/details-232.txt` and `details-8192.txt` ncu's own report; `raw-*.csv.gz` every metric; `sass-*.csv.gz` the stall samples by SASS instruction (`ncu --page source --print-source sass`, exported from the reports afterwards); the reports (15 and 19 MB) are not kept. What it says is in `docs/design/moe-big-steps.md`: the kernel is latency-bound (tensor pipe 10% and 34%, DRAM 54% and 17%, 9 warps an SM, an instruction every 7 cycles a warp), and its stage takes 3.3 times the tensor work.

| | 232 tokens (tile of 32) | 8,192 tokens (tile of 128) |
| :--- | ---: | ---: |
| Duration at 1.44 GHz | 341.8 us | 1.75 ms |
| DRAM throughput / tensor pipe / issue slots | 53.8% / 10.3% / 31.5% | 16.6% / 34.4% / 29.2% |
| Warp instructions | 76.7 M | 379.8 M |
| Stalls | wait 25%, long scoreboard 24%, selected 13%, short scoreboard 13% | wait 27%, long scoreboard 18%, selected 13%, short scoreboard 13% |
