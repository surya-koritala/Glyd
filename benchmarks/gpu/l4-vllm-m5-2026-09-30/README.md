# vLLM with `--quantization glyd` on an L4: bf16 and Glyd in one session (2026-09-30)

`vllm bench serve` on Qwen3-8B, bf16 and then Glyd, back to back on the dev L4, as M3's job runs it on the other GPUs.
It replaces the L4's earlier pair, whose Glyd run came within the hour after bf16's.

## Setup

- **Machine:** AWS g6.4xlarge: NVIDIA L4 (24 GB, 72 W cap, 2,040 MHz at most), driver 595.91.07, 16 vCPUs.
- **Software:**
  - vLLM 0.30.0 (torch 2.13.0+cu130, transformers 5.17.0), in a venv of its own.
  - v0.25.1's library, built for sm_89.
  - The plugin at 61825a1.
- **Run (`m5_l4.sh`):** `bench_serve.sh` under the box's lock, with `WARM=1`:
  - each mode's server started twice on a new, empty compile cache (the cold start noted, the second measured);
  - `--gpu-memory-utilization 0.9`, the random dataset, 1,024 tokens in and 256 out;
  - 32 prompts at 0.25 requests a second, 64 at 1, 128 at 4 and 256 at once.
- **Glyd's layout:** the one `best_layout` picks on an L4, tiered.
- **Temperature:** each mode waited up to 180 s for the GPU to cool to 50 C. bf16's measured server started at 53 C,
  and Glyd's at 61 C. Both then ran at up to 83-85 C.

## Results (`bench-Qwen3-8B/`)

| | Weights | KV cache, warm | Requests of 1,280 tokens at once | KV cache, cold |
| :--- | ---: | ---: | ---: | ---: |
| bf16 | 15.27 GiB | 27,024 tokens | 21.1 | 19,760 tokens |
| Glyd tiered | 11.83 GiB | 51,040 tokens (1.89x) | 39.9 | 35,488 tokens (1.80x) |

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | ITL median / p99 (ms) | Hottest |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.25 | bf16 | 0.22 | 56.5 | 490 / 958 | 70.2 / 73.7 | 66.7 / 340.0 | 78 C |
| 0.25 | glyd | 0.23 | 57.9 | 568 / 1,184 | 55.4 / 59.2 | 50.9 / 439.3 | 82 C |
| 1 | bf16 | 0.66 | 168.0 | 3,944 / 14,030 | 102.9 / 118.5 | 84.1 / 402.4 | 83 C |
| 1 | glyd | 0.75 | 193.0 | 580 / 1,735 | 89.2 / 127.7 | 71.9 / 779.6 | 83 C |
| 4 | bf16 | 0.75 | 191.9 | 55,619 / 120,233 | 110.5 / 178.1 | 84.8 / 653.4 | 85 C |
| 4 | glyd | 0.99 | 252.6 | 35,499 / 73,389 | 154.1 / 250.8 | 98.6 / 810.1 | 84 C |
| inf | bf16 | 0.75 | 192.0 | 153,308 / 312,944 | 111.7 / 184.1 | 84.8 / 660.3 | 85 C |
| inf | glyd | 0.99 | 254.5 | 114,859 / 233,774 | 158.9 / 251.0 | 97.9 / 806.9 | 83 C |

Every request completed. In the servers' logs (every 10 s), with requests waiting, bf16 ran a median of 23 requests at
once and Glyd 41; at most 25 and 48. At the L4's 72 W cap, while it worked, Glyd's kernels ran at a median 915-1,170 MHz,
bf16's at 1,350-1,590 (nvidia-smi's samples; `summary.txt`).

**Glyd against bf16:**

| Rate (req/s) | Requests/s | TTFT mean | TPOT mean | Request's time (E2E mean) | ITL median / p99 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| 0.25 | 1.02x (0.22 to 0.23) | +16% (490 to 568 ms) | -21% (70.2 to 55.4 ms) | -20% | -24% / +29% |
| 1 | 1.15x (0.66 to 0.75) | -85% (3,944 to 580 ms) | -13% (102.9 to 89.2 ms) | -23% | -15% / +94% |
| 4 | 1.32x (0.75 to 0.99) | -36% (55,619 to 35,499 ms) | +39% (110.5 to 154.1 ms) | -11% | +16% / +24% |
| inf | 1.33x (0.75 to 0.99) | -25% (153,308 to 114,859 ms) | +42% (111.7 to 158.9 ms) | -15% | +16% / +22% |

- **Wins:**
  - 1.89x the KV cache: 40 requests of 1,280 tokens at once against 21.
  - Past bf16's saturation (from 1 request a second), 1.15-1.33x the requests a second, and each request finished in
    11-23% less time.
  - At low load, each token 21% sooner.
- **Losses:**
  - At 0.25 requests a second, the first token 16% later.
  - At saturation, each token 39-42% later, with about twice the requests a step.
  - The steps with a prompt in them (ITL p99): 22-94% longer.

## The earlier pair

`../l4-vllm-m2-2026-09-29` measured Glyd (`bench-Qwen3-8B-l4routes/`, the same library's L4 routes) within the hour
after bf16 (`bench-Qwen3-8B-low/`, `bench-Qwen3-8B-warm/`), on the same L4. It ran cooler then (at most 71-80 C, against
78-85 C here), and gave:

- the same KV cache (1.89x);
- 1.39x the requests a second saturated;
- at low load, the first token +17% and each token -23%;
- saturated, the first token -28% and each token +36%.

This session's pair replaces it in the docs.

## Files

- `m5_l4.sh`: the run; `run.log`, its console; `bench-Qwen3-8B.txt`, `bench_serve.sh`'s.
- `bench-Qwen3-8B/`:
  - vLLM's result JSONs and console output;
  - each server's log and KV cache, cold and warm;
  - nvidia-smi's samples each second of a rate;
  - `summary.txt`.
