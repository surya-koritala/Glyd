# A mixture of experts' two routes in vLLM, by tokens a step, on an L4 (2026-09-30)

granite-3.1-3b-a800m-instruct under `--quantization glyd` (tiered, the L4's layout). Each MoE layer holds 40 experts,
routes each token to 8, and has hidden 1,536 and intermediate 512. It has two ways to run a layer's experts:

- **Grouped:** the library's grouped products on the packs.
- **Decoded:** the experts its tokens are routed to are decoded into a scratch buffer, then run by vLLM's own Triton
  MoE kernel, as exact mode runs them.

This directory measures where the decoded route is the faster, per layer and per step, sets the plugin's threshold by
it, and serves with it.

## Per layer (`moe_routes.py`: `routes-*.json`)

The model's first MoE layer, T random tokens each routed to 8 experts at random, the GPU time of a call (the median of
20, after a warm-up). bf16's is vLLM's Triton kernel on the bf16 experts, from a bf16 run.

| Tokens a step | Grouped (ms) | Decoded (ms) | Decoded against grouped | bf16's layer (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.156 | 0.446 | 2.87x | 0.352 |
| 8 | 0.578 | 1.813 | 3.14x | 0.850 |
| 64 | 0.843 | 2.272 | 2.70x | 1.020 |
| 256 | 1.204 | 2.397 | 1.99x | 1.141 |
| 512 | 1.421 | 2.520 | 1.77x | 1.217 |
| 768 | 2.043 | 2.692 | 1.32x | 1.315 |
| 1,024 | 2.755 | 2.844 | 1.03x | 1.444 |
| 1,152 | 2.991 | 2.890 | 0.97x | 1.511 |
| 1,280 | 3.324 | 3.049 | 0.92x | 1.656 |
| 1,536 | 4.034 | 3.251 | 0.81x | 2.001 |
| 2,048 | 5.209 | 3.941 | 0.76x | 2.564 |
| 4,096 | 8.932 | 6.201 | 0.69x | 4.564 |
| 8,192 | 17.032 | 11.043 | 0.65x | 8.660 |

- **The decoded route is the faster from 1,152 tokens a step.** It decodes nearly every expert once a call, a cost
  that pays only over many tokens.
- **Grouped against bf16's layer:**
  - to 128 tokens a step, the grouped products take less time than bf16's layer;
  - from 256, more (1.06x at 256, 2.0x at 8,192).
  - The decoded route never beats bf16's layer: it runs the same kernel after a decode.
- **The two routes' outputs** differ by at most 6.9e-3 (relative), as any two summation orders do.

## Per step (`profile_steps.py`: `steps-*.json`)

A step's GPU time, ms, vLLM compiled with CUDA graphs: decode steps of B sequences, and a prompt of M tokens alone.
"Grouped" and "decoded" run one route throughout (`GLYD_MOE_DECODE_MIN=-1` and `=1`); "routed" is the plugin's default,
decoded from 1,152 tokens.

| Step | M | bf16 | Glyd, grouped | Glyd, decoded | Glyd, routed |
| :--- | ---: | ---: | ---: | ---: | ---: |
| decode | 1 | 8.47 | 7.24 | 13.41 | 7.26 |
| decode | 8 | 20.16 | 15.76 | 48.03 | 15.77 |
| decode | 32 | 25.59 | 22.56 | 63.50 | 22.57 |
| decode | 64 | 28.36 | 28.92 | 70.36 | 29.01 |
| decode | 128 | 33.02 | 34.78 | 76.89 | 34.58 |
| decode | 256 | 38.30 | 47.29 | 84.13 | 47.24 |
| prompt | 512 | 36.86 | 52.44 | 84.49 | 51.57 |
| prompt | 1,024 | | | | 88.94 |
| prompt | 2,048 | 106.00 | 172.06 | 157.72 | 157.62 |
| prompt | 4,096 | 227.52 | 351.27 | 287.20 | 284.90 |

The routed default takes each step's faster route. A prompt of 2,048 tokens takes 8% less GPU time than grouped
throughout, and 4,096 tokens 19% less. Decode steps are as grouped throughout, and from 64 sequences they are still
slower than bf16's (47.2 against 38.3 ms at 256).

## Serving (`bench-routed/`, `bench-grouped/`)

`vllm bench serve`, warm with the cold start noted, `--gpu-memory-utilization 0.9`, 1,024 tokens in and 256 out. There
were 64 prompts at 1 request a second, 128 at 4 and 256 at once, and each mode had one run.

| Rate (req/s) | bf16 | Glyd, routed | Glyd, grouped throughout |
| :--- | :--- | :--- | :--- |
| 1 | 0.92 req/s; TTFT 105 ms; TPOT 19.4 ms | 0.94; 134; 16.1 | 0.93; 134; 16.1 |
| 4 | 3.13; 155; 46.2 | 3.04; 236; 52.0 | 3.06; 206; 50.9 |
| inf | 5.10; 9,365; 106.7 | 4.38; 9,229; 140.6 | 4.27; 9,383; 145.4 |

- **KV cache:** bf16 211,968 tokens, Glyd routed 237,648 (1.12x), grouped throughout 240,352. The scratch buffer of one
  layer's experts decoded costs 1.1% of Glyd's KV cache here.
- **Routed against grouped throughout:**
  - 1.03x the requests a second saturated, each token 3% sooner;
  - the same at 1 request a second;
  - at 4 a second, the first token later (236 against 206 ms, one run each).
- **Glyd against bf16 at saturation:** 0.86x the requests a second, each token 32% later. This is the decode steps'
  cost from 64 sequences, which the route does not touch: the grouped products at those batches are the work left.

## Files

- `m8_moe.sh`: the per-layer runs and the per-step profiles.
  - `m8_moe.log` is its first try, whose profiles stopped on a token id outside granite's vocabulary; the fix is
    `profile_steps.py` drawing ids from the model's vocabulary. `m8_moe2.log` is the second try.
  - The finer per-layer runs (`routes-*-fine.*`) were run by hand, with the same script and `TOKENS` set.
- `m8_route.sh`: `test_vllm.py`, the routed profile and the two benches.
- `routes-*.json`, `steps-*.json`, `bench-*`: the results; `summary.txt` in each bench regenerated by
  `bench_summary.py`.
