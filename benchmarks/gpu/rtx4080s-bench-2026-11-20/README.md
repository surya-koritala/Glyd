# glyd bench on the RTX 4080 SUPER: the gate's loads in five sessions, and how much they move (2026-11-20)

The reference result of `glyd-engine bench` ([docs/bench.md](../../../docs/bench.md)) for the RTX 4080 SUPER, and the measurement the regression gate's noise band stands on. Machine: glyd-pc (AMD Ryzen 9 7950X3D, RTX 4080 SUPER 16 GB,
driver 595.91.07, 736 GB/s by the spec sheet; the GPU runs at its boost clocks under load, 2,670 MHz SM and 11,251 MHz memory in every session; clocks are not locked: no root here). Model: Qwen3-8B lossless, 10.3 GiB in the engine,
a ceiling of 66.5 tokens/s a user (736 GB/s over those bytes, KV reads not counted). Program: `glyd-engine bench Qwen/Qwen3-8B --suite gate --runs 5 --weights-gib 10.3` of main at 0.33.0, a development build
(the CI build), a `glyd-engine serve` started for each session, the CI image (nvidia/cuda 13.0.3), the GPU lock of glyd-pc held for each. The computer's load average was 2 to 4 (CI runners on glyd-pc may add CPU load); no session was
marked busy by the harness (the GPU's utilization before each: 0%), though a CI job's container was up for two of them.

## The five sessions

`session-1.json` to `session-5.json` are the files `bench` wrote (`glyd-bench/1`). The median of each session's five runs, and how far the sessions are from one another:

| load | metric | session 1 | 2 | 3 | 4 | 5 | spread of the sessions (sd / mean) | spread of one session's runs |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| short x1 | decode tok/s a user (% of ceiling) | 60.04 (90%) | 60.03 | 60.06 | 60.07 | 60.05 | 0.02% | 0.06% |
| short x8 | decode tok/s a user (% of ceiling) | 58.84 (88%) | 58.84 | 58.78 | 58.86 | 58.84 | 0.05% | 0.06% |
| short x8 | total tok/s | 231.9 | 231.9 | 231.7 | 231.9 | 231.9 | 0.03% | 0.10% |
| 1k x1 | decode tok/s a user (% of ceiling) | 59.41 (89%) | 59.38 | 59.39 | 59.42 | 59.40 | 0.02% | 0.02% |
| 4k x1 | decode tok/s a user (% of ceiling) | 57.73 (87%) | 57.71 | 57.72 | 57.74 | 57.73 | 0.02% | 0.03% |
| 1k x1 | prefill tok/s | 5,128 | 5,134 | 5,127 | 5,127 | 5,130 | 0.06% | 0.32% |
| 4k x1 | prefill tok/s | 5,907 | 5,898 | 5,889 | 5,881 | 5,894 | 0.16% | 0.46% |
| short x1 | first token s | 0.021 | 0.021 | 0.021 | 0.021 | 0.021 | 0.38% | 0.81% |
| 1k x1 | first token s | 0.194 | 0.194 | 0.194 | 0.194 | 0.194 | 0.07% | 0.05% |
| 4k x1 | first token s | 0.679 | 0.680 | 0.681 | 0.681 | 0.681 | 0.13% | 0.16% |
| short x8 | first token s | 1.146 | 1.149 | 1.146 | 1.145 | 1.149 | 0.16% | 0.18% |

Decode is the same to 0.02-0.05% from one session to the next, in the runs of one session 0.02-0.06%; prefill and the times move 0.1-0.4%. The gate's noise band (z = 3 on these spreads, a floor of 0.3%) is therefore
**0.3% for decode, 0.3-1.2% for prefill and the first token** (the first token of a short prompt, 21 ms, and its prefill rate are held to wider bands: they are quantized to milliseconds). The gate between two of these sessions
(`bench --compare session-1.json session-2.json --history` the other three) is within the band at every figure, and fails the same pair with a synthetic 5% slowdown on all 16 figures
(`--synthetic 0.95`).

## What the numbers say about the server

- Decode is 87-90% of the ceiling at the loads of the gate, as in [rtx4080s-bar-2026-11-20](../rtx4080s-bar-2026-11-20) (59.9 tok/s at short x1, 57.7 at 4k x1: the same figures to 0.3%).
- **The first token of 8 short prompts at once is 1.146 s** (median of the users; 1.15 in all five sessions and in the earlier Python client's `load.py` against the same server: 1.147). The bar's record of the same load today
  is 0.064 s (the same program at the commit it was measured at). A first token that takes as long as a third of the replies' decoding is the server holding some of the eight requests until others finish, not a rate: this is
  the gate's to catch and the first thing it found. The total tokens/s at the load is the same (232 against 233 in the bar).

## The full suite, once

`full.json`: `bench --suite full --runs 2 --warmup 0 --max-tokens 64`, the loads R8 lists (decode at 1, 8, 64 and 256 at once, prefill at 1K, 4K and 32K), on the same server with its default context: see the file for each load, and `errors`
for the ones this server does not take (a 32,000-token prompt past its context; 64 and 256 at once wait for a place: R11).

## Files

| File | What |
| :--- | :--- |
| `session-1.json` ... `session-5.json` | the gate's loads, five runs each, one server each, one after another |
| `full.json` | the full suite once |
