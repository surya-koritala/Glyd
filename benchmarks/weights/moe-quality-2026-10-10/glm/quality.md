## Routed experts at 2-4 bits: zai-org/GLM-5.3-Flash @ eb9eb208

Held-out text: 24 sequences of 2048 tokens (49,152 tokens, 6 domains), 49,128 next-token positions scored. Reference: the FP8 model as released. KL divergence is taken over its top 128 tokens and the rest of the vocabulary as one bucket (a lower bound on the full divergence). Only the routed experts of the 42 MoE layers are rounded (288 each, 25,165,824 weights an expert); attention, shared experts, dense layers, routers, embedding and head stay as released. Every layer of a variant sees the layers before it already rounded.

FP8 model: perplexity 3.858; its top 128 tokens leave out 1.33% of the probability on average. The layer-by-layer pass with the model's own experts reproduces its forward to KL 0.0e+00 (top-1 agreement 1.0000).

| item | variant | what | bits/weight | experts GB | KL from FP8, mean (s.e. over sequences) | KL from control | KL p99 | top-1 = FP8 % | top-1 = control % | perplexity | vs FP8 | minutes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | FP8 | the model as released | 8.00 | 304.5 | 0 | | 0 | 100 | | 3.858 | | 1.2 |
| 0 | ref | control: the FP8 experts dequantised to bf16, not rounded (what this measurement cannot beat) | 8.002 | 304.5 | 0.0438 (0.0102) |  | 0.6124 | 95.52 |  | 3.870 | +0.31% | 1.6 |
| 5 | mix13 | mixed by usage: the 13% most-used experts of each layer int4, the rest hlm3 (the GLM on 24 GB + 128 GB plan) | 3.665 | 139.5 | 0.1523 (0.0259) | 0.1533 | 2.3071 | 89.87 | 89.91 | 3.975 | +3.05% | 2.7 |
| 1 | int4 | every expert int4 (symmetric codes, mse-fit bf16 scale per run of 32) | 4.500 | 171.2 | 0.0934 (0.0192) | 0.0950 | 1.4082 | 92.57 | 92.77 | 3.866 | +0.22% | 4.0 |
| 3 | hlm3 | every expert hlm3: rotation + per-row codebook at 3 bits, rounded to nearest | 3.542 | 134.8 | 0.1752 (0.0292) | 0.1770 | 2.6005 | 88.94 | 89.04 | 4.020 | +4.21% | 2.4 |
| 2 | int3 | every expert int3 (as int4, 3 bits) | 3.500 | 133.2 | 0.2128 (0.0354) | 0.2124 | 3.1525 | 87.98 | 88.03 | 3.988 | +3.38% | 4.0 |
| 6 | cold50 | the coldest 50% of each layer's experts hlm2, the other 50% int4 (the aggressive plan) | 3.511 | 133.6 | 0.2247 (0.0369) | 0.2267 | 3.4791 | 87.55 | 87.50 | 3.979 | +3.15% | 3.3 |
| 4 | hlm3g | every expert hlm3 with GPTQ act-order on routed-token Hessians (experts with few calibration tokens fall back to RTN) | 3.542 | 134.8 | 0.1654 (0.0288) | 0.1651 | 2.6827 | 89.81 | 89.83 | 4.071 | +5.54% | 24.6 |

The control (ref) is the FP8 experts dequantised to bf16 and run through the same path: its KL 0.0438 from the FP8 model is the floor of that column (the FP8 model's kernels also round each activation to FP8; a Glyd engine would not). The columns 'from control' score every other variant against the control instead: what rounding the weights alone costs.

### KL divergence from FP8, mean, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref | 0.0438 | 0.0264 | 0.0108 | 0.1422 | 0.0394 | 0.0132 | 0.0309 |
| mix13 | 0.1523 | 0.1952 | 0.0526 | 0.3890 | 0.1258 | 0.0787 | 0.0725 |
| int4 | 0.0934 | 0.0846 | 0.0264 | 0.2727 | 0.0864 | 0.0411 | 0.0493 |
| hlm3 | 0.1752 | 0.2341 | 0.0528 | 0.4296 | 0.1561 | 0.0931 | 0.0856 |
| int3 | 0.2128 | 0.2788 | 0.0608 | 0.5259 | 0.1869 | 0.1207 | 0.1038 |
| cold50 | 0.2247 | 0.2543 | 0.1209 | 0.5606 | 0.1839 | 0.1150 | 0.1134 |
| hlm3g | 0.1654 | 0.2349 | 0.0597 | 0.4235 | 0.1062 | 0.0910 | 0.0774 |

### KL divergence from the control (ref), mean, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| mix13 | 0.1533 | 0.1926 | 0.0520 | 0.3948 | 0.1237 | 0.0773 | 0.0794 |
| int4 | 0.0950 | 0.0833 | 0.0255 | 0.2896 | 0.0856 | 0.0402 | 0.0458 |
| hlm3 | 0.1770 | 0.2316 | 0.0526 | 0.4344 | 0.1590 | 0.0929 | 0.0913 |
| int3 | 0.2124 | 0.2791 | 0.0625 | 0.5193 | 0.1876 | 0.1216 | 0.1042 |
| cold50 | 0.2267 | 0.2555 | 0.1232 | 0.5718 | 0.1808 | 0.1149 | 0.1142 |
| hlm3g | 0.1651 | 0.2347 | 0.0593 | 0.4300 | 0.1036 | 0.0920 | 0.0707 |

### top-1 agreement with FP8, %, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref | 95.52 | 95.68 | 98.31 | 90.17 | 95.05 | 96.76 | 97.17 |
| mix13 | 89.87 | 86.72 | 95.70 | 81.24 | 90.29 | 90.71 | 94.57 |
| int4 | 92.57 | 91.52 | 97.13 | 85.11 | 92.38 | 93.26 | 96.01 |
| hlm3 | 88.94 | 85.28 | 95.49 | 80.19 | 88.94 | 89.67 | 94.06 |
| int3 | 87.98 | 84.42 | 95.54 | 78.31 | 87.62 | 88.20 | 93.77 |
| cold50 | 87.55 | 84.78 | 92.90 | 76.92 | 88.42 | 89.12 | 93.16 |
| hlm3g | 89.81 | 85.98 | 95.59 | 80.42 | 91.27 | 90.32 | 95.26 |

### top-1 agreement with the control, %, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| mix13 | 89.91 | 86.69 | 95.69 | 81.61 | 90.17 | 90.74 | 94.55 |
| int4 | 92.77 | 91.73 | 97.26 | 85.76 | 92.22 | 93.54 | 96.08 |
| hlm3 | 89.04 | 85.53 | 95.69 | 80.62 | 88.53 | 90.01 | 93.84 |
| int3 | 88.03 | 84.45 | 95.70 | 78.60 | 87.45 | 88.20 | 93.80 |
| cold50 | 87.50 | 84.77 | 92.83 | 77.05 | 88.41 | 89.04 | 92.89 |
| hlm3g | 89.83 | 85.99 | 95.70 | 80.57 | 91.08 | 90.35 | 95.30 |

### perplexity, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| FP8 | 3.858 | 2.928 | 1.724 | 14.188 | 7.713 | 2.891 | 2.063 |
| ref | 3.870 | 2.953 | 1.727 | 14.160 | 7.735 | 2.876 | 2.089 |
| mix13 | 3.975 | 3.276 | 1.737 | 14.103 | 7.948 | 2.980 | 2.076 |
| int4 | 3.866 | 3.031 | 1.730 | 13.679 | 7.645 | 2.914 | 2.089 |
| hlm3 | 4.020 | 3.341 | 1.748 | 15.157 | 7.557 | 3.034 | 2.080 |
| int3 | 3.988 | 3.485 | 1.755 | 13.774 | 7.408 | 3.108 | 2.074 |
| cold50 | 3.979 | 3.463 | 1.853 | 13.399 | 6.987 | 3.117 | 2.120 |
| hlm3g | 4.071 | 3.451 | 1.761 | 15.402 | 7.743 | 3.041 | 2.067 |

### The most-used experts (profile from the calibration text)

| share of a layer's experts | per layer | calibration routed tokens covered % | held-out routed tokens covered % |
|---:|---:|---:|---:|
| 13% | 37 | 26.42 | 25.81 |
| 50% | 144 | 68.43 | 67.72 |

### Weight error (relative RMSE of the rounded experts against the FP8 original, all layers)

| variant | spec | weights | relative RMSE |
|---|---|---:|---:|
| ref | ref | 304,405,807,104 | 0.0018 |
| mix13 | hlm3 | 265,298,116,608 | 0.1678 |
| mix13 | int4 | 39,107,690,496 | 0.0913 |
| int4 | int4 | 304,405,807,104 | 0.0913 |
| hlm3 | hlm3 | 304,405,807,104 | 0.1678 |
| int3 | int3 | 304,405,807,104 | 0.1957 |
| cold50 | hlm2 | 152,202,903,552 | 0.3357 |
| cold50 | int4 | 152,202,903,552 | 0.0913 |
| hlm3g | hlm3g | 304,405,807,104 | 0.2180 |

GPTQ (hlm3g): 11868 expert-layers rounded with GPTQ, 228 fell back to nearest (fewer than 2048 of the 294,912 calibration tokens routed to them).
