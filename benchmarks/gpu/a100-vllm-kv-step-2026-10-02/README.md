# A decode step on an A100: one request, the profile, the kernels and serving (2026-10-02; serving 2026-10-03)

Qwen3-8B, bf16 weights, vLLM 0.30.0, an A100 SXM4 40 GB (108 SMs; Lambda), vLLM's default mode (torch.compile, FULL decode graphs), `kv: lossless` against vLLM's own cache; kv-vllm 591a37bc, the job's `env`, `build`, `test`, `decode`, `tps` and `profile` steps (the kernel checks: 81 of 81 pass here).
The KV cache holds 177,392 tokens against vLLM's 136,592 (1.299x) at the same memory utilization (0.9).
The serving section below is a later job on the same machine type (`kvs1a`, kv-vllm d766885b, the job's `env`, `build` and `serve` steps).

## One request: decode tokens a second

128 new tokens a request, in three rounds with the two processes alternating (stock first, kv first, stock first).

| batch x tokens | stock | kv | kv / stock, each round | median |
| :--- | ---: | ---: | :--- | ---: |
| 1 x 8,192 | 71.4 (14.00 ms a step) | 73.7 (13.57 ms a step) | 1.031, 1.032, 1.032 | **1.032x** |
| 1 x 1,024 | 75.3 (13.27 ms a step) | 76.7 (13.04 ms a step) | 1.018, 1.018, 1.018 | **1.018x** |

## A decode step (the profile)

Milliseconds a step (36 layers), the step's wall time and the GPU's time by kind, in the default mode (`kv_step_profile.py --graphs`). Stock's attention is FlashAttention's split-KV kernel and its combine kernel; its cache write is vLLM's own.

| batch x tokens | stock a step | kv a step | kv / stock | attention: stock -> kv | cache write: stock -> kv |
| :--- | ---: | ---: | ---: | :--- | :--- |
| 1 x 8,192 | 14.05 | 13.65 | 1.029x | 1.532 -> 1.031 | 0.112 -> 0.184 |
| 1 x 1,024 | 13.34 | 13.17 | 1.013x | 0.769 -> 0.534 | 0.112 -> 0.186 |
| 3 x 8,192 | 15.88 | 14.86 | 1.069x | 3.342 -> 2.330 | 0.108 -> 0.193 |
| 32 x 1,024 | 17.51 | 15.96 | 1.097x | 4.380 -> 2.779 | 0.116 -> 0.218 |

The write is the one part of the step that kv does slower (0.07 to 0.10 ms a step more); the attention more than pays for it at every shape.

## The kernels alone: one layer, in a CUDA graph (microseconds)

| batch x tokens | decode: kv (the five repeats' range) | FlashAttention 2 | FlashAttention 2 / kv | write: kv | write: vLLM |
| :--- | :--- | ---: | ---: | ---: | ---: |
| 1 x 1,024 | 12.19 (12.11-12.24) | 17.05 | 1.40x | 5.27 | 2.53 |
| 1 x 2,048 | 13.95 (13.95-14.03) | 23.60 | 1.69x | 5.09 | 2.53 |
| 1 x 4,096 | 20.43 (20.35-20.53) | 30.59 | 1.50x | 4.15 | 2.53 |
| 1 x 8,192 | 31.87 (30.26-32.18) | 43.78 | 1.37x | 3.76 | 2.53 |
| 2 x 2,048 | 21.30 (20.04-21.35) | 33.25 | 1.56x | 4.20 | 2.56 |
| 4 x 8,192 | 93.64 (87.32-94.28) | 119.48 | 1.28x | 6.22 | 2.53 |
| 8 x 4,096 | 86.58 (85.91-92.57) | 118.73 | 1.37x | 6.53 | 2.56 |
| 32 x 1,024 | 90.32 (90.16-90.47) | 144.23 | 1.60x | 8.76 | 3.02 |
| 32 x 8,192 | 634.73 (573.90-640.87) | 977.18 | 1.54x | 8.24 | 2.61 |
| 64 x 8,192 | 1268.38 (1143.17-1268.79) | 1611.03 | 1.27x | 10.83 | 2.69 |
| 32 x 32,768 | 2233.73 (2233.14-2472.19) | 3867.88 | 1.73x | 8.14 | 2.64 |
| 128 x 4,096 | 1268.81 (1224.09-1269.43) | 1511.22 | 1.19x | 11.52 | 3.15 |
| 256 x 1,024 | 644.28 (586.37-647.07) | 783.00 | 1.22x | 17.72 | 3.66 |

kv's decode times for the larger batches vary by up to 12% between the repeats of a call (the range); FlashAttention's vary by under 1%. The largest difference of the two outputs on the same keys and values (finite inputs) is in `decode/graph-bench.json`.

## The same job before the last change to the decode attention

At kv-vllm 32795af3 on the same machine type (the job `kvs`): one request of 8,192 tokens 1.022, 1.025 and 1.026x in three rounds; 4 x 8,192 1.066x and 32 x 1,024 1.103x (one round each; `tps/`). One request of 1,024 tokens was not run then: the kernels' times predicted 0.998x (the attention saves 0.08 ms a step and the write costs 0.10 ms more than vLLM's), and the run after the change measured 1.018x.
Between the two runs the decode attention's configuration for a batch of one or two requests was changed: its outputs for those batches differ in the last bits; for every other batch they are the same bits.
32 x 8,192 and 64 x 8,192 do not fit either cache on 40 GB.

## Serving, 1,024 tokens in and 256 out (`kvs1a`)

`vllm bench serve`, random dataset, 1,024 tokens in and 256 out, `--ignore-eos`, 0.9, `--max-model-len 4096`, prefix caching, chunked prefill and async scheduling as vLLM has them; a server a mode, started twice and the second measured; 64 requests at 1 a second, 256 at once, 8 one at a time; every pass draws prompts of its own; vLLM 0.30.0.

| load | stock | kv | kv / stock |
| :--- | ---: | ---: | :--- |
| 1 request a second | 0.95 requests/s, 14.7 ms a token, first token 113 ms | 0.95, 14.3 ms, 114 ms | 1.00x, 0.97x, 1.01x |
| saturated (256 at once) | 5.92 requests/s, 63.5 ms a token, first token 15,319 ms | 7.62, 59.0 ms, 12,408 ms | **1.29x**, 0.93x, 0.81x |
| one user | 0.28 requests/s, 13.4 ms a token, first token 91 ms | 0.29, 13.2 ms, 94 ms | 1.02x, 0.98x, 1.04x |

The KV cache holds 141,040 tokens with vLLM's and 183,424 with `kv` (1.30x); the model's memory is 15.64 against 15.27 GiB (1.02x). The prefix cache's hit rate in the servers' logs is at most 0.9% (vLLM's cache) and 0.8% (`kv`), and 0.3% over each whole run: one of the 328 prompts appears twice, which skips at most one of 328 prefills (`kvs1a/serve-1k/hit-*.txt`, `summary.txt`).

## Files

`kvs1a/` (the serving job: `steps.txt`, `machine-short.txt`, `env.txt`, `serve-1k/` with each run's `vllm bench serve` output and result, the prefix cache's hit-rate lines (`hit-*.txt`) and the summary), `tps/` (each run's decode tokens a second; the first pass and the one-request rounds of the earlier job beside the latest), `profile/` (stock and kv), `decode/graph-bench.json` (the kernels, eager loop and graph timings, and the largest difference of the outputs). The tools are in [`gpu/vllm/`](../../../gpu/vllm): `kv_tps.py`, `kv_step_profile.py`, `kv_decode_bench.py`. Prefix hits on this machine type: [`a100-vllm-kv-prefix-2026-10-02`](../a100-vllm-kv-prefix-2026-10-02).
