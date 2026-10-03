# The lossless KV cache on an A100: a decode step and serving (2026-10-02; serving 2026-10-03)

Qwen3-8B, bf16 weights, vLLM 0.30.0, an A100 SXM4 40 GB (Lambda), vLLM's default mode (torch.compile, FULL decode graphs), `kv: lossless` against vLLM's own cache ("stock" below). The kernels' checks of the job: 81 of 81 pass here.
In the step job the KV cache holds 177,392 tokens against vLLM's 136,592 (1.299x) at the same memory utilization (0.9).
The same measurements on an L4: [`l4-vllm-kv-step-2026-10-02`](../l4-vllm-kv-step-2026-10-02).
The serving section below is a later job on the same machine type (`kvs1a`: its `env`, `build` and `serve` steps).

## One request: decode tokens a second

128 new tokens a request, in three rounds with the two processes alternating (stock first, kv first, stock first).

| batch x tokens | stock | kv | kv / stock, each round | median |
| :--- | ---: | ---: | :--- | ---: |
| 1 x 8,192 | 71.4 (14.00 ms a step) | 73.7 (13.57 ms a step) | 1.031, 1.032, 1.032 | **1.032x** |
| 1 x 1,024 | 75.3 (13.27 ms a step) | 76.7 (13.04 ms a step) | 1.018, 1.018, 1.018 | **1.018x** |

## A decode step (the profile)

Milliseconds a step (36 layers): the step's wall time, and the GPU's time by kind, in the default mode.

| batch x tokens | stock a step | kv a step | kv / stock | attention: stock -> kv | cache write: stock -> kv |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1 x 8,192 | 14.05 | 13.65 | 1.029x | 1.532 -> 1.031 | 0.112 -> 0.184 |
| 1 x 1,024 | 13.34 | 13.17 | 1.013x | 0.769 -> 0.534 | 0.112 -> 0.186 |
| 3 x 8,192 | 15.88 | 14.86 | 1.069x | 3.342 -> 2.330 | 0.108 -> 0.193 |
| 32 x 1,024 | 17.51 | 15.96 | 1.097x | 4.380 -> 2.779 | 0.116 -> 0.218 |

The write is the one part of the step that kv does slower; the attention more than pays for it at every shape.

## Serving, 1,024 tokens in and 256 out (`kvs1a`)

`vllm bench serve`, random dataset, 1,024 tokens in and 256 out, `--ignore-eos`, 0.9, `--max-model-len 4096`, prefix caching, chunked prefill and async scheduling as vLLM has them; a server a mode, started twice and the second measured; 64 requests at 1 a second, 256 at once, 8 one at a time; every pass draws prompts of its own; vLLM 0.30.0.

| load | stock | kv | kv / stock |
| :--- | ---: | ---: | :--- |
| 1 request a second | 0.95 requests/s, 14.7 ms a token, first token 113 ms | 0.95, 14.3 ms, 114 ms | 1.00x, 0.97x, 1.01x |
| saturated (256 at once) | 5.92 requests/s, 63.5 ms a token, first token 15,319 ms | 7.62, 59.0 ms, 12,408 ms | **1.29x**, 0.93x, 0.81x |
| one user | 0.28 requests/s, 13.4 ms a token, first token 91 ms | 0.29, 13.2 ms, 94 ms | 1.02x, 0.98x, 1.04x |

The KV cache holds 141,040 tokens with vLLM's and 183,424 with `kv` (1.30x); the model's memory is 15.64 against 15.27 GiB (1.02x). The prefix cache's hit rate in the servers' logs is at most 0.9% (vLLM's cache) and 0.8% (`kv`), and 0.3% over each whole run: one of the 328 prompts appears twice, which skips at most one of 328 prefills (`kvs1a/serve-1k/hit-*.txt`, `summary.txt`).

## Files

`kvs1a/` (the serving job: `steps.txt`, `env.txt`, `serve-1k/` with each run's `vllm bench serve` output and result, the prefix cache's hit-rate lines (`hit-*.txt`) and the summary), `tps/tps-batch1.json` (each round's tokens a second), `profile/` (stock and kv). Prefix hits on this machine type: [`a100-vllm-kv-prefix-2026-10-02`](../a100-vllm-kv-prefix-2026-10-02).
