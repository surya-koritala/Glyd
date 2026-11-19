# A100: where the one-token products wait, and what they cost in instructions (2026-10-19)

Lambda gpu_1x_a100_sxm4 (40 GB), 31 minutes, about $1.03. Nsight Compute 2026.3.1 with source counters (per-instruction stall samples,
`ncu/*-stalls.txt`; the full reports were not kept). Nsight locks the clocks: SM 1.21 GHz, so durations here are longer than in runs.

## Instructions a byte of weights

| Product (one token) | Model | Instructions issued | Bytes read | Lane instructions a byte |
|---|---|---|---|---|
| `mma_moe_kernel<Fp4R>` gate/up | DeepSeek experts | 21.30 M | 51.2 MB | 13.3 |
| same, after the fix below | | 19.22 M | 51.2 MB | 12.0 |
| `mma_moe_kernel<Fp4R>` down | DeepSeek experts | 11.15 M | 25.6 MB | 13.9 |
| `step_kernel` (tiered, penguin) | Gemma 4 12B | 12.09 M | ~44 MB | ~8.8 |
| `f8_linear_kernel`, grouped | GLM experts (glmrandom --real) | 83.13 M | ~105 MB | ~25 |
| `f8_linear_kernel` | DeepSeek dense FP8 | 21.51 M | ~27 MB | ~25 |

(Bytes for the tiered and FP8 products from DRAM throughput, calibrated on the FP4R product's known 51.2 MB.)

What a GPU can issue a byte it reads (SMs x 4 schedulers x clock x 32 lanes / bandwidth), at most:

| GPU | Lane instructions a byte |
|---|---|
| RTX 4080 SUPER | 35 |
| A100 40GB | 12.5 |
| A100 80GB | 9.6 |
| H100 SXM | 9-10 |
| B200 | ~4.6 |

The integer pipe issues at half rate on A100 and H100, so the practical figure is lower for integer-heavy decodes (Nsight: 10-12% of
samples "math", the pipe busy).

So, with these decodes, at best: the tiered (penguin) product can reach the H100's bandwidth; FP4R about 75% of it; FP8 about 35-40%.
The earlier H100 FP8 figure (49-51% of bandwidth, `../h100-profile-2026-10-18/ncu`) was f8moecheck's random weights, which the format
stores plain (no decode): not representative. GLM (FP8 throughout) needs a cheaper FP8 decode on HBM GPUs.

## Where the time goes

- FP4R: 30% of all stall samples on one instruction, the first use of the step's weight load (long scoreboard). The loop is 205
  instructions a step: about 96 the FP4 decode, 8 tensor-core products, and the rest addresses, the step counter (64-bit), the
  activations' loads and a division by the steps a row block (`Fp4R::block`: 24 instructions, issued whether or not the
  power-of-two branch took it).
- FP8: no single wait; fixed-latency dependencies 24%, the CTA's barrier a step 9%, the integer pipe 12%. Instruction-bound.
- Tiered: shared-memory and shuffle waits (short scoreboard) 19%, loads 12%, barrier 8%. Latency-bound, not instruction-bound.

## The fix: Fp4R's division as a multiply-high

`step / ks` as `(t + (n - t) / 2) >> sh`, t = umulhi(n, mg) (fp4r_fmt; checked against n / ks for 170 M pairs, every even ks to 4096
and more). Same digits on the random DeepSeek model with the library before it (dsv4check: 11 of 24, 2.4437; int5 the same); moecheck
within bounds.

| Experts' layer, one token (us) | before | after |
|---|---|---|
| FP4R (packed scales) | 174.3 / 174.3 | **167.5 / 167.4** |
| FP4R, two steps ahead | 177.5 / 177.6 | 167.0 / 167.0 |
| FP4, two steps ahead | 158.5 / 158.5 | 158.1 / 158.3 |
| FP4, one step ahead | 166.3 / 166.2 | 166.2 / 166.3 |

FP4 two steps ahead is 4.8% faster in all four runs: it is now the default on HBM GPUs (FP4R stays at one: no gain, 10 more
registers). DeepSeek's 3-layer step: 2.915-2.921 ms with the fix; before, 2.71 and 2.93 in two runs of the same build (too noisy to
compare at that level).

## Next

1. A cheaper FP8 decode for HBM GPUs (GLM, DeepSeek's dense matrices): the current one is ~25 lane instructions a byte.
2. FP4R's loop: per-row-block addresses once, 32-bit step counters, the activations from shared memory (about 75 of the 205
   instructions a step are not the decode).
3. The tiered product's waits (latency, not instructions).
