# vLLM with `--quantization glyd` on an L4, a set of prompts for each pass (2026-10-03)

[`l4-vllm-m5-2026-09-30`](../l4-vllm-m5-2026-09-30) again: `vllm bench serve` on Qwen3-8B, bf16 and then Glyd on the dev L4, with each pass's prompts new to the server. The published run's passes all
ran at `--seed 0`, so a later pass repeated an earlier pass's prompts, and vLLM's prefix cache (on, as deployments run it) served those it still held. In its Glyd server's log the prefix cache hit
rate stays 0.0% through the rate-0.25 pass, rises to 48.3% during the rate-1 pass (which repeated the 32 prompts of the pass before) and never rises again; its bf16
server's stays 0.0%. Here each pass's seed is its place in the list of rates (0 to 3), the same for both modes, and the summary prints each server's highest prefix cache hit rate.

## Setup

- **Machine:** AWS g6.4xlarge: NVIDIA L4 (24 GB, 72 W cap, 2,040 MHz at most), driver 595.91.07, 16 vCPUs.
- **Software:** as the published run: vLLM 0.30.0 (torch 2.13.0+cu130, transformers 5.17.0), v0.25.1's library built for sm_89, the plugin at 61825a1.
- **Run:** each mode's server started twice on a new, empty compile cache (the cold start noted, the second measured); `--gpu-memory-utilization 0.9`, `--max-model-len 4096`, the random dataset, 1,024 tokens in
  and 256 out; 32 prompts at 0.25 requests a second, 64 at 1, 128 at 4 and 256 at once, seeds 0 to 3. bf16 ran at 00:33 UTC and Glyd at 00:55 (the published run: back to back).
- **Glyd's layout:** the smallest layout (`mma`), as the published run.
- **Temperature:** both modes' servers started at 50 C (the published run: 53 C and 61 C) and ran at up to 84-85 C at saturation.

## Results (`bench-Qwen3-8B/`)

| | Weights | KV cache, warm | Max concurrency at 4,096 | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 15.27 GiB | 27,024 tokens | 6.60x | 19,760 tokens |
| Glyd `mma` | 11.83 GiB | 51,040 tokens (1.89x) | 12.46x | 35,488 tokens (1.80x) |

The servers' highest prefix cache hit rate (their logs, every 10 s): bf16 0.1%, glyd 0.2% (the published run: 0.0% and 48.3%).

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | bf16 | 0.22 | 56.5 | 474 / 936 | 70.1 / 73.5 | 32 | 1620 MHz, 78 C |
| 0.25 | glyd | 0.23 | 58.1 | 541 / 1116 | 54.1 / 58.2 | 32 | 1290 MHz, 78 C |
| 1 | bf16 | 0.67 | 172.7 | 2323 / 7745 | 104.4 / 124.7 | 64 | 1425 MHz, 81 C |
| 1 | glyd | 0.80 | 203.7 | 739 / 1473 | 85.3 / 97.8 | 64 | 1200 MHz, 71 C |
| 4 | bf16 | 0.75 | 192.7 | 55807 / 119443 | 109.5 / 182.1 | 128 | 1365 MHz, 84 C |
| 4 | glyd | 1.06 | 270.5 | 32478 / 66240 | 143.8 / 234.9 | 128 | 1155 MHz, 84 C |
| inf | bf16 | 0.76 | 193.7 | 151874 / 309861 | 110.9 / 182.0 | 256 | 1410 MHz, 85 C |
| inf | glyd | 1.05 | 268.6 | 108495 / 221260 | 150.7 / 239.0 | 256 | 1140 MHz, 84 C |

Every request completed. While it worked, Glyd's kernels ran at a median 1,140-1,290 MHz, bf16's at 1,365-1,620 (nvidia-smi's samples).

**Glyd against bf16** (the runs' JSON means):

| Rate (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) |
| :--- | ---: | ---: | ---: | ---: |
| 0.25 | 1.03x (0.22 to 0.23) | +14% (474 to 541 ms) | -23% (70.1 to 54.1 ms) | -22% |
| 1 | 1.18x (0.67 to 0.80) | -68% (2,323 to 739 ms) | -18% (104.4 to 85.3 ms) | -22% |
| 4 | 1.40x (0.75 to 1.06) | -42% (55,807 to 32,478 ms) | +31% (109.5 to 143.8 ms) | -17% |
| inf | 1.39x (0.76 to 1.05) | -29% (151,874 to 108,495 ms) | +36% (110.9 to 150.7 ms) | -18% |

## Against the published run

| Rate (req/s) | Requests/s, published to here | TTFT mean, published to here | TPOT mean, published to here |
| :--- | :--- | :--- | :--- |
| 0.25 | 1.02x to 1.03x | +16% to +14% | -21% to -23% |
| 1 | 1.15x to 1.18x | **-85% to -68%** (3,944 to 580 ms; 2,323 to 739 ms) | -13% to -18% |
| 4 | 1.32x to 1.40x | -36% to -42% | +39% to +31% |
| inf | 1.33x to 1.39x | -25% to -29% | +42% to +36% |

- The first pass (nothing to repeat in either run) reproduces the published one to within 6% in every column.
- The published rate-1 pass of Glyd repeated the 32 prompts of its rate-0.25 pass and its prefix cache held them: Glyd's first token was 580 ms there, 739 ms here, and each token 89.2 ms against 85.3 ms.
  bf16's first token at that rate is 2,323 ms here against the published 3,944 ms, with its requests a second (0.67 against 0.66) and each token (104.4 ms against 102.9) the same within 2%: bf16 queues at
  that rate, and a queue's first token moves with the arrival times, which the seed sets. The first token at 1 request a second is therefore 68% sooner than bf16's, not 85%.
- In the published Glyd log the hit rate never rises during the rate-4 and saturated passes, so those passes served nothing from the cache. Saturated: Glyd 1.05 requests a second against the published
  0.99, bf16 0.76 against 0.75: 1.39x against 1.33x.

## Files

- `bench-Qwen3-8B.txt` and `bench-Qwen3-8B-glyd.txt`: the benchmark's console for each mode.
- `bench-Qwen3-8B/`: per mode and rate, vLLM's result JSON and console output and nvidia-smi's samples each second of the rate; each server's log and KV cache lines, cold and warm; `summary.txt`.
