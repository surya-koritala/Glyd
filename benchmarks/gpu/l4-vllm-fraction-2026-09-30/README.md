# vLLM with `--quantization glyd` and `fraction` on an L4 (2026-09-30)

The plugin's `fraction` option packs only that share of a model's decoder layers and leaves the others as vLLM runs them
(`0` is bf16, `1` every layer, as before). Runs on the dev L4: a smoke on Qwen3-0.6B, `check_vllm.py --quick --fraction
0.5` on Qwen3-8B and `--brief` on granite-3.1-3b-a800m-instruct (a mixture of experts), values the option refuses, and
`vllm bench serve` on Qwen3-8B at fractions 0, 0.5 and 1 against bf16. They are sanity numbers for the option, not the
GH200's answer: the GH200 job is prepared and not run. `h100-sxm/` (2026-10-01) is the sweep on an H100 SXM, Qwen3-32B
served saturated at fractions 0, 0.25, 0.5, 0.75 and 1 against bf16: the first Hopper run of the option, from the
v0.26.0 candidate (its own README).

## Setup

- **Machine:** AWS g6.4xlarge: NVIDIA L4 (23,034 MiB, 72 W cap, 2,040 MHz at most), driver 595.91.07, 16 vCPUs.
- **Software:**
  - vLLM 0.30.0 (torch 2.13.0+cu130, transformers 5.17.0), in a venv of its own.
  - v0.25.1's library, built for sm_89.
  - The plugin at 0449a9c (`vllm_plugin.py`, `check_vllm.py`, `bench_serve.sh` and `bench_summary.py` as that commit has
    them).
- **Runs:** each under the box's lock, one after another (`chain.sh` and `chain2.sh`, which queued each on the lock as the
  one before it ended): `smoke/smoke.sh`, `checks/l4_check.sh`, `l4_sweep.sh`, `bad-fraction/bad_fraction.sh`.
  `checks/test_vllm.txt` is `test_vllm.py` on the same box with vLLM installed and no GPU visible: 9 of 9.
- **The check's fix:** the granite run is at the commit after 0449a9c that changed `check_vllm.py`'s layer count (above);
  the other runs used 0449a9c's.

## Smoke: Qwen3-0.6B, eager (`smoke/`)

One server each, `--enforce-eager`, `gpu_memory_utilization` 0.3, three prompts of 24 tokens greedy; the fraction given as
`--additional-config`'s `{"glyd": {"fraction": F}}`.

| | Layers packed | KV cache blocks | Tokens and logprobs against bf16's |
| :--- | :--- | ---: | :--- |
| bf16 | | 2,923 | |
| fraction 0 | 0 of 28 | 2,923 | bit for bit |
| fraction 0.5 | 14 of 28 (1, 3, 5, ... 27) | 2,986 | 2 of 3 prompts the same |
| fraction 1 | 28 of 28 | 3,063 | 2 of 3 prompts the same |

The options in effect, fraction included, were in each server's `additional_config` (the compile cache's key), with the
packs' digest: empty at 0, and a different one at 0.5 and at 1.

## Check: `check_vllm.py --quick --fraction 0.5` on Qwen3-8B, all 18 passed (`checks/`, 19 minutes)

The 15 checks of a `--quick` run on Glyd at fraction 0.5, tiered and 12-bit, and three more:

- Every pack (72: 18 layers' qkv, o, gate_up and down) decoded to its weights bit for bit, in both layouts.
- Every layer's product within 4.15e-3 (tiered) and 3.80e-3 (12-bit) of F.linear on its matrix decoded, the same bits
  every run.
- Top-1 agreement with bf16 on its 1,536-token continuation 0.9916 (tiered) and 0.9929 (12-bit), bf16 eager's 0.9922.
- **The layers:** 18 of 36 packed, which are the rule's (layer i where floor((i + 1) f) > floor(i f): 1, 3, 5, ... 35),
  each with all four of its Linears, the other 18 with vLLM's own method. Checked in both layouts.
- Exact eager: bf16 eager's tokens, logprobs and prompt_logprobs bit for bit (8 of 8 prompts, the continuation too).
- Exact compiled, inductor deterministic: compiled bf16's bits (8 of 8, the continuation too), and exact compiled without
  the mode refused with Glyd's message.
- Fused, compiled, inductor deterministic: the same bits from one run to the next, the second loading the first's graphs.
- The compile cache: a graph for each of bf16, tiered and 12-bit on one cache, each loaded again.
- **Fraction 0, eager:** nothing packed, and bf16 eager's tokens, logprobs and prompt_logprobs bit for bit (8 of 8, the
  continuation too), with the same KV cache (17,568 tokens each).

The KV cache of these runs (0.85 utilization, 4,096 tokens): bf16 11,056 tokens, fraction 0.5 tiered 22,064 and 12-bit
19,936.

## Check: `check_vllm.py --brief --fraction 0.5` on granite-3.1-3b-a800m-instruct, all 7 passed (`checks/`, 4 minutes)

A mixture of experts (32 layers, each with attention Linears and 40 experts), Glyd in the layout the GPU's best (tiered):

- 16 of 32 layers packed (1, 3, 5, ... 31): 32 Linears and the experts of 16 layers (`moe: 16`), every pack decoded to its
  weights bit for bit, every product (the experts' too) within 3.38e-3 of its matrix decoded.
- The other 16 layers, their Linears and their experts, are vLLM's own methods (`UnquantizedLinearMethod`,
  `UnquantizedFusedMoEMethod`); the layers packed are the rule's, each with all its Linears and its experts.
- Top-1 agreement with bf16 on the 1,536-token continuation 0.9896 (bf16 eager's 0.9883).
- Exact eager: bf16 eager's tokens, logprobs and prompt_logprobs bit for bit.
- Fraction 0, eager: nothing packed, bf16 eager's bits, the same KV cache (194,560 tokens each).
- KV cache of the Glyd run (0.85 utilization): 198,368 tokens against bf16's 185,280.

A first run of this check (`checks/check-granite-3.1-3b-a800m-instruct-f0.5-first-run.txt`) failed its layer check and
nothing else: granite builds its router gate, a Linear in every layer, without a quantization config, so it is vLLM's own
method in a packed layer too, and the check counted it as a layer left alone. The check now counts only Linears a
quantization config was given to (`check_vllm.py`'s `_layers`); the plugin was not changed, and the second run is this one.

## A fraction Glyd refuses (`bad-fraction/`)

`vllm serve Qwen/Qwen3-0.6B --quantization glyd` with `--additional-config '{"glyd": {"fraction": F}}'` for F = 1.5, -0.25
and "half", and with `GLYD_FRACTION` 2 and abc: each stopped in the engine's process, before any worker, in 10-20 seconds,
with exit 1 and `glyd: fraction 1.5: a number from 0 to 1 (the share of the decoder layers packed)` (the value as given)
as the error.

## Bench: Qwen3-8B, bf16 and fractions 0, 0.5 and 1 (`bench-Qwen3-8B/`)

M5's run (`../l4-vllm-m5-2026-09-30`) with two more modes in between bf16 and fraction 1, all back to back in one session:
`bench_serve.sh` with `WARM=1`, so each server started twice on a new compile cache and the second measured; the tiered
layout (the L4's); `--gpu-memory-utilization 0.9`; the random dataset, 1,024 tokens in and 256 out; 64 prompts at 1
request a second and 256 at once. Each mode waited up to 180 s for the GPU to cool to 50 C; the measured servers started at
54 C (bf16), 62, 60 and 59 C, and the GPU ran at up to 85 C.

| Mode | Weights | KV cache | Requests/s, at once | TTFT at 1 a second | TPOT at 1 a second | TTFT, at once | TPOT, at once |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 15.27 GiB | 27,024 tokens | 0.75 | 3,351 ms | 100.2 ms | 152,281 ms | 111.2 ms |
| fraction 0 | 15.27 GiB | 27,024 tokens (1.00x) | 0.76 (1.00x) | 3,539 ms (1.06x) | 100.8 ms (1.01x) | 151,872 ms (1.00x) | 111.0 ms (1.00x) |
| fraction 0.5 | 13.65 GiB | 37,904 tokens (1.40x) | 0.96 (1.27x) | 1,095 ms (0.33x) | 99.9 ms (1.00x) | 122,428 ms (0.80x) | 125.4 ms (1.13x) |
| fraction 1 | 11.83 GiB | 51,040 tokens (1.89x) | 1.04 (1.38x) | 772 ms (0.23x) | 95.6 ms (0.95x) | 109,789 ms (0.72x) | 152.4 ms (1.37x) |

`summary.txt` has every rate's percentiles, the GPU's clock and temperature, and each mode's cold start (KV cache 19,760,
19,760, 22,336 and 35,488 tokens).

- **Fraction 0 is bf16.** The same weights and KV cache (27,024 tokens warm, 19,760 cold), the same requests a second
  (0.755 against 0.754 at once) and each token within 1% (100.8 against 100.2 ms at 1 a second, 111.0 against 111.2 at once). Its
  first token at 1 a second was 6% later (3,539 against 3,351 ms, p99 13,253 against 13,084): bf16 is already queueing
  there, 0.66 requests a second completed at 1 offered.
- **Fraction 1 reproduces M5's Glyd:** the KV cache to the token (51,040 warm, 35,488 cold), and 1.04 requests a second
  saturated against M5's 0.99 (bf16 0.75 in both), each token at once 1.37x bf16's against M5's 1.42x.
- **Fraction 0.5** packed 18 of 36 layers: 89% of bf16's weights, 1.40x its KV cache, 1.27x its requests a second at once,
  the first token at 1 a second 67% sooner, and each token as fast as bf16's at 1 a second, 13% slower at once. Of fraction
  1's gain over bf16 in requests a second it took 71% (0.955 against bf16's 0.754 and fraction 1's 1.038); of fraction 1's
  extra time per token at once, 34% (125.4 against 111.2 and 152.4 ms).
- Glyd's kernels ran at lower SM clocks at the L4's 72 W cap, the lower the more layers were packed: a median 1,275 and
  1,155 MHz at 1 a second for fractions 0.5 and 1, against bf16's 1,485 (`summary.txt`).

## Files

- `bench-Qwen3-8B.txt`, `bench-Qwen3-8B/`: the console and the results: per mode `serve-MODE[-cold].txt` (the server's log),
  `kv-MODE[-cold].txt` (its weights, KV cache and the plugin's line), `MODE-rateR.json`, `bench-MODE-rateR.txt`,
  `smi-MODE-rateR.csv`, then `summary.txt`.
- `checks/`: for each model `check-MODEL-f0.5.txt` (the console) and `check-MODEL-f0.5/` (`report.txt`, each run's JSON and
  log), the granite check's first run's console, `l4_check.sh`, `test_vllm.txt`.
- `bad-fraction/`: each refused start's output.
- `smoke/`: `smoke.py`, `smoke.sh` and each fraction's JSON (tokens, logprobs, the layers packed, `additional_config`) and
  log.
- `l4_sweep.sh`, `chain.sh`, `chain2.sh`: how the bench ran, and the runs in turn.
- `h100-sxm/`: the H100 SXM's sweep: its README, the job (`budget_job.sh`) and its results.
