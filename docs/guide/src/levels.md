# Choosing a level

A level is how a model's weights are stored on the GPU. You pick one with a tag: `glyd-engine run Qwen/Qwen3.5-9B:swift`.
With no tag, `run` uses penguin, and refuses a model that has none. `fit` lists every level and picks the first that fits:
lossless if it fits, then the highest lossy level that does, then experts in RAM for a mixture of experts. `--lossy` is the
same as `:kestrel`.

| Level | Bits a weight | Lossless | Where |
| :--- | :--- | :--- | :--- |
| penguin | about 10.8 | yes: the same weights, bit for bit | most families |
| kestrel | 6.5 | no | Qwen3.5, Nemotron-H, Gemma 4 |
| swift | 5.5 | no | the same |
| wren | 4.5 | no | the same; a development build, or a quantization-aware checkpoint |
| hummingbird | 2 to 4 on the experts | no | Nemotron 3 Super, pre-packed |

Which families run at which level is on the [Models](models.md) page. A level a family does not run is refused.

## What each costs

Quality is the KL divergence of the model's next-token distribution from bf16's, on WikiText-2 text (253,952 scored
positions), and how often its top token is bf16's. Lower KL is closer. The comparison point is the FP8 checkpoint most
people would run instead.

**Qwen3.8-27B, A100 80 GB, one user** (two pods; quality from one study)

| Weights | GiB | KL, mean | Top token as bf16 | Perplexity | Decode tokens/s |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 16 bits | 0 | 100% | 7.2226 | |
| FP8 checkpoint, vLLM 0.30.0 | 28.025 | 0.00465 | 97.1% | 7.2410 | 47.45 |
| kestrel | 20.479 | 0.00513 | 96.7% | 7.2665 | 45.03 |
| swift | 17.351 | 0.01408 | 94.4% | 7.3058 | 47.78 |
| wren (four-bit block codes, no calibration) | not recorded | 0.05335 | 88.9% | 7.4184 | not recorded |

- kestrel is 1.10 times FP8's KL with 27% less weight memory, and decodes at 0.95 of FP8's speed on this card.
  Sources: [int6-l40-2026-10-09](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/gpu/int6-l40-2026-10-09) (A100 PCIe pod),
  [int5-a100-2026-10-09](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/gpu/int5-a100-2026-10-09) (A100 SXM4 pod, a different one, so the speeds are not one run's comparison).
- The wren row is the study's four-bit block codes at 4.16 bits, the nearest measurement to the engine's 4.5-bit layout,
  from [kld-q56-2026-10-09](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/weights/kld-q56-2026-10-09).
  Four bits with no calibration change 11% of top tokens; a quantization-aware checkpoint gives wren back exactly.
- These runs are from October 2026, v0.29.3 builds. They have not been re-run on main.

**Penguin**: KL 0 by construction, the weights are the same. Qwen3-8B on an RTX 4080 SUPER 16 GB: 10.3 GiB against bf16's
15.3 GiB, 59.7 tokens/s, 2026-11-20 on a main build
([log](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/gpu/rtx4080s-qwen3-8b-engine-2026-11-20)).

**Hummingbird and wren on Nemotron 3 Super (120B-A12B, a mixture of experts)**

| Level | Machine | Weights | Decode tokens/s | Quality |
| :--- | :--- | :--- | ---: | :--- |
| wren | A100 80 GB | 65.18 GiB on the GPU | 71.1 | KL 0.0133, four-bit experts on held-out text (includes a 0.0047 floor) |
| wren | RTX 4090 24 GB, experts in RAM | 59.1 GiB of experts in RAM | 11.95 | the same layout |
| hummingbird | RTX 4090 24 GB, experts in RAM | 20.7 GiB on the GPU, 39.2 GiB of experts in RAM | 32.39 | KL 0.0946, perplexity +10.3% on held-out text |

Logs: [nh-run-check-wren-2026-10-10](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/gpu/nh-run-check-wren-2026-10-10),
[nh-offload-4090-2026-10-10](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/gpu/nh-offload-4090-2026-10-10),
[nh-hb-4090-2026-10-10](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/gpu/nh-hb-4090-2026-10-10). On GSM8K and MMLU-Pro
questions the hummingbird run scored 147 of 150 and 62 of 80; bf16 Nemotron was not run on them, so the gap to bf16 is not
measured ([card](https://github.com/surya-koritala/Glyd/tree/main/benchmarks/cards/2026-10-10/qwen38-27b-swift-vs-nemotron-hb)).

## How to choose

1. Run `glyd-engine fit MODEL`. It lists every level with its size on your GPU and says which fit.
2. Take penguin if it fits. Nothing changes.
3. If not, take the highest level that fits. Going down a level costs more quality than the one above it: on the 27B, kestrel
   to swift is 2.7 times the KL, swift to four-bit block codes 3.8 times.
4. For code and long tool loops, stay at kestrel or above where you can.

`fit` sizes include the context. For Qwen3.8-27B on an RTX 4090 preset it picks swift at 17.8 GiB and 32,768 tokens, and lists
wren at 14.6 GiB. Those sizes and the ~75 tokens/s are estimates from memory bandwidth, not runs.
