# The lossless KV cache and Glyd's weights, served on an L4: `vllm bench serve` (2026-10-01; serving re-measured 2026-10-03)

Qwen3-8B, vLLM 0.30.0, an NVIDIA L4 (24 GB), through the compiled package, vLLM's default mode (compiled, CUDA graphs).
Four servers, one each: `bf16` is vLLM's own (bf16 weights, vLLM's KV cache); `kv` is bf16 weights with the lossless KV cache; `glyd@1` is Glyd's weights with vLLM's own KV cache; `kv+glyd` is both.

## What to know

- **Saturated (256 requests at once), the lossless cache alone serves 1.23x the requests a second** (0.93 against 0.76) with 1.29x the KV tokens (34,912 against 27,024); Glyd's weights alone 1.39x (1.05) with 2.02x the KV tokens (54,464); **both together 1.59x (1.20) with 2.64x the KV tokens (71,248 against 27,024)**. Each token then takes 1.06x, 1.41x and 1.55x as long, and the first token 0.81x, 0.70x and 0.58x of bf16's.
- **At 1 request a second** (64 requests): 1.11x (`kv`), 1.20x (`glyd@1`) and 1.22x (`kv+glyd`) the requests a second, the mean first token after 2,171, 698 and 680 ms against bf16's 5,861 ms (bf16 completed 0.65 requests a second, so its requests queue).
- **One user at a time** (8 requests): with the lossless cache a token takes 60.2 ms against bf16's 60.6 (0.99x) and the first token 333 ms against 331 (1.00x); with Glyd's weights a token takes 47.9 ms (`glyd@1`) and 47.6 ms (`kv+glyd`), 0.79x and 0.78x, and the first token 431 and 432 ms (1.30x).
- **The prefix cache's hit rate is 0.0% on all four servers, in each of the 195 lines their logs give** (`bench/hit-*.txt`, `bench/summary.txt`): every pass draws prompts of its own.

## `vllm bench serve` (`bench/`)

Random dataset, 1,024 tokens in and 256 out, `--ignore-eos`, 0.9, `--max-model-len 4096`, prefix caching and chunked prefill on, async scheduling on (vLLM's defaults); one server a mode, started twice and the second measured; 64 requests at 1 a second, 256 at once, 8 one at a time. Each pass draws prompts of its own (the seed 4 plus the pass's place: 4, 5 and 6), and the servers' logs give the prefix cache's hit rate. The "started cold" lines of `bench/summary.txt` are not cold starts: vLLM's compile cache had been filled by the first run, so a mode's first and second starts show the same cache sizes. Glyd's weights are 11.38 GiB.

| Rate (req/s) | Mode | Requests/s | Output tokens/s | TTFT mean / p99 (ms) | TPOT mean / p99 (ms) | Completed | SM clock, temperature |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 0.65 | 166.4 | 5861 / 16710 | 100.7 / 112.9 | 64 | 1500 MHz, 74 C |
| 1 | glyd@1 | 0.78 | 200.3 | 698 / 1126 | 102.8 / 132.4 | 64 | 1215 MHz, 72 C |
| 1 | kv | 0.72 | 184.6 | 2171 / 8144 | 101.4 / 114.3 | 64 | 1500 MHz, 74 C |
| 1 | kv+glyd | 0.80 | 203.7 | 680 / 1133 | 96.7 / 124.9 | 64 | 1185 MHz, 76 C |
| inf | bf16 | 0.76 | 193.5 | 151326 / 310359 | 111.0 / 182.8 | 256 | 1380 MHz, 83 C |
| inf | glyd@1 | 1.05 | 269.6 | 105717 / 224786 | 156.3 / 253.4 | 256 | 1080 MHz, 84 C |
| inf | kv | 0.93 | 237.8 | 122692 / 254981 | 117.7 / 198.9 | 256 | 1410 MHz, 83 C |
| inf | kv+glyd | 1.20 | 307.9 | 87962 / 197773 | 172.4 / 268.1 | 256 | 1140 MHz, 83 C |
| one | bf16 | 0.06 | 16.2 | 331 / 339 | 60.6 / 60.7 | 8 |  |
| one | glyd@1 | 0.08 | 20.3 | 431 / 438 | 47.9 / 48.0 | 8 |  |
| one | kv | 0.06 | 16.3 | 333 / 344 | 60.2 / 60.3 | 8 |  |
| one | kv+glyd | 0.08 | 20.4 | 432 / 441 | 47.6 / 47.7 | 8 |  |

Each against `bf16`:

| Rate (req/s) | Mode | Weights (GiB) | KV cache (tokens) | Requests/s | TTFT mean (ms) | TPOT mean (ms) |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| 1 | bf16 | 15.27 | 27,024 | 0.65 | 5,861 | 100.7 |
| 1 | glyd@1 | 11.38 (0.75x) | 54,464 (2.02x) | 0.78 (1.20x) | 698 (0.12x) | 102.8 (1.02x) |
| 1 | kv | 15.36 (1.01x) | 34,912 (1.29x) | 0.72 (1.11x) | 2,171 (0.37x) | 101.4 (1.01x) |
| 1 | kv+glyd | 11.49 (0.75x) | 71,248 (2.64x) | 0.80 (1.22x) | 680 (0.12x) | 96.7 (0.96x) |
| inf | bf16 | 15.27 | 27,024 | 0.76 | 151,326 | 111.0 |
| inf | glyd@1 | 11.38 (0.75x) | 54,464 (2.02x) | 1.05 (1.39x) | 105,717 (0.70x) | 156.3 (1.41x) |
| inf | kv | 15.36 (1.01x) | 34,912 (1.29x) | 0.93 (1.23x) | 122,692 (0.81x) | 117.7 (1.06x) |
| inf | kv+glyd | 11.49 (0.75x) | 71,248 (2.64x) | 1.20 (1.59x) | 87,962 (0.58x) | 172.4 (1.55x) |
| one | bf16 | 15.27 | 27,024 | 0.06 | 331 | 60.6 |
| one | glyd@1 | 11.38 (0.75x) | 54,464 (2.02x) | 0.08 (1.25x) | 431 (1.30x) | 47.9 (0.79x) |
| one | kv | 15.36 (1.01x) | 34,912 (1.29x) | 0.06 (1.01x) | 333 (1.00x) | 60.2 (0.99x) |
| one | kv+glyd | 11.49 (0.75x) | 71,248 (2.64x) | 0.08 (1.26x) | 432 (1.30x) | 47.6 (0.78x) |

The 1-a-second pass follows the arrival times, so it moves between seeds.

## A first run with one repeated prompt (`bench/seed0/`)

The first run on the same machine and package used seed 0 for every pass (`bench/seed0/`): one of its 328 prompts appears twice, so the servers' logs show hits for it, at most 1.1% to 1.2% and 0.3% over the whole run, which the summary flags as above 1%. It gave `kv` 0.72 against 0.67 requests a second at 1 a second (1.08x) and a mean first token of 3,195 ms for bf16 where the run above has 5,861 ms. Its saturated rows are the run above's within 1.4% and its one-user rows within 0.7%: `kv` 0.93 against 0.76 requests a second saturated, `kv+glyd` 1.19 against `glyd@1`'s 1.05 (1.13x), the one user's token 60.2, 60.6, 47.8 and 47.9 ms.

## Files

`bench/` (each mode's result JSON, the prefix cache's hit-rate lines of each server's log (`hit-*.txt`), nvidia-smi's clock and temperature (`smi-*.csv`) and the summary; `seed0/`: the first run's JSON, hit-rate lines and summary). Result files are the harness's own JSON; logs that carry the plugin's internal diagnostics are not part of this record.
