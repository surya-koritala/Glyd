## Routed experts at 2-4 bits: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 @ 2dc98e2a

Held-out text: 24 sequences of 2048 tokens (49,152 tokens, 6 domains), 49,128 next-token positions scored. Reference: the bf16 model as released. KL divergence is taken over its top 128 tokens and the rest of the vocabulary as one bucket (a lower bound on the full divergence). Only the routed experts of the 40 MoE layers are rounded (512 each, 5,505,024 weights an expert); the Mamba-2 and attention layers, the latent projections, the shared experts, the routers, embedding and head stay as released. Every layer of a variant sees the layers before it already rounded.

bf16 model: perplexity 3.021; its top 128 tokens leave out 1.12% of the probability on average. The layer-by-layer pass with the model's own experts reproduces its forward to KL 0.0e+00 (top-1 agreement 1.0000).

| item | variant | what | bits/weight | experts GB | KL from bf16, mean (s.e. over sequences) | KL p99 | top-1 = bf16 % | perplexity | vs bf16 | minutes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| | bf16 | the model as released | 16.00 | 225.5 | 0 | 0 | 100 | 3.021 | | 0.6 |
| 0 | ref | control: the bf16 experts through the rounding path unrounded (the identity: must reproduce the model) | 16.000 | 225.5 | 0.0047 (0.0006) | 0.0379 | 99.01 | 3.021 | -0.02% | 0.6 |
| 5 | mix13 | mixed by usage: the 13% most-used experts of each layer int4, the rest hlm3 (GLM's split, to compare; this model's 24 GB plan holds 32%) | 3.706 | 52.2 | 0.0273 (0.0037) | 0.4147 | 96.75 | 3.101 | +2.65% | 1.4 |
| 1 | int4 | every expert int4 (symmetric codes, mse-fit bf16 scale per run of 32) | 4.500 | 63.4 | 0.0133 (0.0015) | 0.1884 | 97.85 | 3.044 | +0.76% | 1.9 |
| 3 | hlm3 | every expert hlm3: rotation + per-row codebook at 3 bits, rounded to nearest | 3.587 | 50.6 | 0.0326 (0.0046) | 0.5041 | 96.33 | 3.123 | +3.38% | 1.2 |
| 2 | int3 | every expert int3 (as int4, 3 bits) | 3.500 | 49.3 | 0.0445 (0.0062) | 0.7516 | 95.68 | 3.159 | +4.58% | 1.9 |
| 6 | cold50 | the coldest 50% of each layer's experts hlm2, the other 50% int4 (the aggressive plan) | 3.522 | 49.6 | 0.0564 (0.0084) | 0.9280 | 95.02 | 3.177 | +5.17% | 1.6 |
| 4 | hlm3g | every expert hlm3 with GPTQ act-order on routed-token Hessians (experts with few calibration tokens fall back to RTN) | 3.587 | 50.6 | 0.0310 (0.0057) | 0.5127 | 96.73 | 3.105 | +2.79% | 13.6 |

The control (ref) is the experts as released, run through the same path unrounded: its KL 0.0047 from the bf16 model is the check that the path reproduces the model (0 up to the order of float additions), so the KL of every other variant is what rounding the weights alone costs.

### KL divergence from bf16, mean, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref | 0.0047 | 0.0072 | 0.0023 | 0.0085 | 0.0050 | 0.0030 | 0.0024 |
| mix13 | 0.0273 | 0.0527 | 0.0101 | 0.0314 | 0.0342 | 0.0253 | 0.0100 |
| int4 | 0.0133 | 0.0239 | 0.0058 | 0.0169 | 0.0160 | 0.0118 | 0.0054 |
| hlm3 | 0.0326 | 0.0648 | 0.0116 | 0.0341 | 0.0455 | 0.0285 | 0.0113 |
| int3 | 0.0445 | 0.0873 | 0.0151 | 0.0429 | 0.0628 | 0.0432 | 0.0159 |
| cold50 | 0.0564 | 0.1110 | 0.0222 | 0.0801 | 0.0500 | 0.0525 | 0.0226 |
| hlm3g | 0.0310 | 0.0790 | 0.0135 | 0.0361 | 0.0249 | 0.0242 | 0.0082 |

### top-1 agreement with bf16, %, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref | 99.01 | 98.30 | 99.57 | 98.42 | 99.27 | 98.91 | 99.56 |
| mix13 | 96.75 | 94.74 | 98.90 | 95.33 | 96.95 | 95.97 | 98.63 |
| int4 | 97.85 | 96.57 | 99.21 | 96.97 | 98.06 | 97.26 | 99.02 |
| hlm3 | 96.33 | 94.06 | 98.62 | 95.09 | 96.32 | 95.40 | 98.47 |
| int3 | 95.68 | 93.10 | 98.46 | 94.93 | 95.22 | 94.24 | 98.16 |
| cold50 | 95.02 | 91.94 | 98.07 | 92.71 | 96.14 | 93.62 | 97.64 |
| hlm3g | 96.73 | 93.86 | 98.55 | 95.63 | 97.63 | 96.09 | 98.61 |

### perplexity, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 3.021 | 4.135 | 1.676 | 7.135 | 2.687 | 3.557 | 1.609 |
| ref | 3.021 | 4.143 | 1.674 | 7.101 | 2.693 | 3.558 | 1.610 |
| mix13 | 3.101 | 4.333 | 1.691 | 7.343 | 2.800 | 3.628 | 1.628 |
| int4 | 3.044 | 4.182 | 1.683 | 7.129 | 2.733 | 3.591 | 1.616 |
| hlm3 | 3.123 | 4.429 | 1.693 | 7.371 | 2.813 | 3.647 | 1.637 |
| int3 | 3.159 | 4.451 | 1.702 | 7.444 | 2.893 | 3.702 | 1.646 |
| cold50 | 3.177 | 4.544 | 1.712 | 7.840 | 2.737 | 3.720 | 1.657 |
| hlm3g | 3.105 | 4.418 | 1.691 | 7.421 | 2.770 | 3.606 | 1.619 |

### The most-used experts (profile from the calibration text)

| share of a layer's experts | per layer | calibration routed tokens covered % | held-out routed tokens covered % |
|---:|---:|---:|---:|
| 13% | 67 | 23.66 | 23.42 |
| 50% | 256 | 65.96 | 65.85 |

### Weight error (relative RMSE of the rounded experts against the bf16 original, all layers)

| variant | spec | weights | relative RMSE |
|---|---|---:|---:|
| ref | ref | 112,742,891,520 | 0.0000 |
| mix13 | hlm3 | 97,989,427,200 | 0.1673 |
| mix13 | int4 | 14,753,464,320 | 0.0915 |
| int4 | int4 | 112,742,891,520 | 0.0915 |
| hlm3 | hlm3 | 112,742,891,520 | 0.1673 |
| int3 | int3 | 112,742,891,520 | 0.1955 |
| cold50 | hlm2 | 56,371,445,760 | 0.3350 |
| cold50 | int4 | 56,371,445,760 | 0.0915 |
| hlm3g | hlm3g | 112,742,891,520 | 0.2213 |

GPTQ (hlm3g): 20399 expert-layers rounded with GPTQ, 81 fell back to nearest (fewer than 2048 of the 294,912 calibration tokens routed to them).
