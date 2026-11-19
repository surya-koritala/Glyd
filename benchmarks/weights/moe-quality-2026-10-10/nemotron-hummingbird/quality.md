## Routed experts at 2-4 bits: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 @ 2dc98e2a

Held-out text: 24 sequences of 2048 tokens (49,152 tokens, 6 domains), 49,128 next-token positions scored. Reference: the bf16 model as released. KL divergence is taken over its top 128 tokens and the rest of the vocabulary as one bucket (a lower bound on the full divergence). Only the routed experts of the 40 MoE layers are rounded (512 each, 5,505,024 weights an expert); the Mamba-2 and attention layers, the latent projections, the shared experts, the routers, embedding and head stay as released. Every layer of a variant sees the layers before it already rounded.

bf16 model: perplexity 3.021; its top 128 tokens leave out 1.12% of the probability on average. The layer-by-layer pass with the model's own experts reproduces its forward to KL 0.0e+00 (top-1 agreement 1.0000).

| item | variant | what | bits/weight | experts GB | KL from bf16, mean (s.e. over sequences) | KL from control | KL p99 | top-1 = bf16 % | top-1 = control % | perplexity | vs bf16 | minutes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | bf16 | the model as released | 16.00 | 225.5 | 0 | | 0 | 100 | | 3.021 | | 0.6 |
| 0 | ref | control: the bf16 experts through the rounding path unrounded (the identity: must reproduce the model) | 16.000 | 225.5 | 0.0047 (0.0006) |  | 0.0379 | 99.01 |  | 3.021 | -0.02% | 0.6 |
| 7 | hlm2g | every expert hlm2 with GPTQ act-order (hlm3g at 2 bits) | 2.544 | 35.8 | 0.1532 (0.0304) | 0.1533 | 2.7152 | 91.05 | 90.98 | 3.548 | +17.43% | 13.0 |
| 8 | hb40 | Hummingbird, mixed by usage: the 40% most-used experts of each layer hlm3g, the rest hlm2g | 2.961 | 41.7 | 0.0831 (0.0171) | 0.0831 | 1.5417 | 93.96 | 93.91 | 3.286 | +8.76% | 15.1 |
| 9 | hb20 | Hummingbird, mixed by usage: the 20% most-used experts of each layer int4, the rest hlm2g | 2.934 | 41.3 | 0.1022 (0.0199) | 0.1024 | 1.9248 | 93.07 | 93.03 | 3.364 | +11.34% | 11.7 |

The control (ref) is the experts as released, run through the same path unrounded: its KL 0.0047 from the bf16 model is the floor of that column (the path sums the experts in float32, the model's own kernels in bf16). The columns 'from control' score every other variant against the control instead: what rounding the weights alone costs.

### KL divergence from bf16, mean, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref | 0.0047 | 0.0072 | 0.0023 | 0.0085 | 0.0050 | 0.0030 | 0.0024 |
| hlm2g | 0.1532 | 0.4061 | 0.0584 | 0.1930 | 0.1013 | 0.1173 | 0.0434 |
| hb40 | 0.0831 | 0.2109 | 0.0368 | 0.1215 | 0.0459 | 0.0621 | 0.0215 |
| hb20 | 0.1022 | 0.2553 | 0.0427 | 0.1503 | 0.0572 | 0.0778 | 0.0297 |

### KL divergence from the control (ref), mean, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| hlm2g | 0.1533 | 0.4064 | 0.0588 | 0.1933 | 0.1004 | 0.1169 | 0.0439 |
| hb40 | 0.0831 | 0.2111 | 0.0372 | 0.1220 | 0.0451 | 0.0618 | 0.0217 |
| hb20 | 0.1024 | 0.2560 | 0.0431 | 0.1512 | 0.0567 | 0.0772 | 0.0302 |

### top-1 agreement with bf16, %, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref | 99.01 | 98.30 | 99.57 | 98.42 | 99.27 | 98.91 | 99.56 |
| hlm2g | 91.05 | 82.17 | 96.64 | 86.71 | 94.06 | 90.00 | 96.69 |
| hb40 | 93.96 | 88.52 | 97.20 | 90.71 | 96.52 | 93.16 | 97.64 |
| hb20 | 93.07 | 86.94 | 97.14 | 88.81 | 96.07 | 92.16 | 97.31 |

### top-1 agreement with the control, %, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| hlm2g | 90.98 | 81.99 | 96.64 | 86.44 | 94.14 | 90.00 | 96.68 |
| hb40 | 93.91 | 88.51 | 97.24 | 90.49 | 96.48 | 93.15 | 97.58 |
| hb20 | 93.03 | 86.77 | 97.22 | 88.59 | 96.03 | 92.32 | 97.24 |

### perplexity, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 3.021 | 4.135 | 1.676 | 7.135 | 2.687 | 3.557 | 1.609 |
| ref | 3.021 | 4.143 | 1.674 | 7.101 | 2.693 | 3.558 | 1.610 |
| hlm2g | 3.548 | 6.307 | 1.770 | 8.867 | 3.033 | 3.924 | 1.692 |
| hb40 | 3.286 | 5.028 | 1.730 | 8.291 | 2.831 | 3.738 | 1.649 |
| hb20 | 3.364 | 5.331 | 1.737 | 8.577 | 2.887 | 3.784 | 1.669 |

### The most-used experts (profile from the calibration text)

| share of a layer's experts | per layer | calibration routed tokens covered % | held-out routed tokens covered % |
|---:|---:|---:|---:|
| 20% | 102 | 32.98 | 32.73 |
| 40% | 205 | 56.18 | 56.03 |

### Weight error (relative RMSE of the rounded experts against the bf16 original, all layers)

| variant | spec | weights | relative RMSE |
|---|---|---:|---:|
| ref | ref | 112,742,891,520 | 0.0000 |
| hlm2g | hlm2g | 112,742,891,520 | 0.4396 |
| hb40 | hlm2g | 67,601,694,720 | 0.4458 |
| hb40 | hlm3g | 45,141,196,800 | 0.2165 |
| hb20 | hlm2g | 90,282,393,600 | 0.4425 |
| hb20 | int4 | 22,460,497,920 | 0.0915 |

GPTQ (hlm2g): 20388 expert-layers rounded with GPTQ, 92 fell back to nearest (fewer than 2048 of the 294,912 calibration tokens routed to them).

GPTQ (hb40): 20397 expert-layers rounded with GPTQ, 83 fell back to nearest (fewer than 2048 of the 294,912 calibration tokens routed to them).

GPTQ (hb20): 16315 expert-layers rounded with GPTQ, 85 fell back to nearest (fewer than 2048 of the 294,912 calibration tokens routed to them).
