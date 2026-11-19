# A100: FP8 layout H against the 6.75-bit layout (2026-10-19)

Lambda gpu_1x_a100_sxm4 (40 GB), 17 minutes, about $0.56. Branch hbm-prefetch at c3cdccba (engine and library for sm_80 and sm_90a).
GLYD_F8_LAYOUT=h is layout H (the default on HBM GPUs), c the 6.75-bit layout. Models: dsv4random (`rand`; `ds3`: 3 layers, 32 experts, real-shaped)
and glmrandom (`glm4`: 4 layers, 32 experts, FP8).

**Outputs.** dsv4check, both layouts: 11 of 24 steps, largest logit difference 2.4437 (the same digits as every run before). The dev build
unpacks every FP8 matrix it packs and compares it byte for byte.

**FP8 products at one token** (Nsight, SM locked at 1.21 GHz):

| | 6.75-bit | H | |
|---|---|---|---|
| GLM experts, gate/up (us) | 317.4 | **211.3** | -33% |
| GLM experts, down (us) | 165.8 | **114.4** | -31% |
| GLM shared expert (us) | 52.0 | **36.8** | -29% |
| DeepSeek dense matrix (us) | 92.2 / 93.3 | **61.0 / 63.8** | -32% |
| instructions, GLM gate/up | 83.1 M | 54.5 M | -34% |
| DRAM throughput, GLM gate/up | 22% | 37% | |

**Steps with a graph** (ms, two runs each):

| | 6.75-bit | H |
|---|---|---|
| DeepSeek, 3 layers | 2.863 / 2.915 | **2.570 / 2.560** |
| GLM, 4 layers | 3.907 / 3.877 | 3.843 / 3.842 |

(The GLM toy's step is mostly its BF16 attention and small kernels.) Weights: DeepSeek 2.784 -> 2.813 GiB, GLM 5.012 -> 5.283 GiB.

FP8 is still ~17 lane instructions a byte in layout H (an A100 issues 12.5): next, the product loop's own instructions (a quarter of the total) and the
conversion to bf16.
