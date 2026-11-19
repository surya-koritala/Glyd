## Routed experts at 2-4 bits: nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 @ 2dc98e2a

Held-out text: 24 sequences of 2048 tokens (49,152 tokens, 6 domains), 49,128 next-token positions scored. Reference: the bf16 model as released. KL divergence is taken over its top 128 tokens and the rest of the vocabulary as one bucket (a lower bound on the full divergence). Only the routed experts of the 40 MoE layers are rounded (512 each, 5,505,024 weights an expert); the Mamba-2 and attention layers, the latent projections, the shared experts, the routers, embedding and head stay as released. Every layer of a variant sees the layers before it already rounded.

bf16 model: perplexity 3.021; its top 128 tokens leave out 1.12% of the probability on average. The layer-by-layer pass with the model's own experts reproduces its forward to KL 0.0e+00 (top-1 agreement 1.0000).

| item | variant | what | bits/weight | experts GB | KL from bf16, mean (s.e. over sequences) | KL from control | KL p99 | top-1 = bf16 % | top-1 = control % | perplexity | vs bf16 | minutes |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| | bf16 | the model as released | 16.00 | 225.5 | 0 | | 0 | 100 | | 3.021 | | 0.6 |
| 0 | ref | control: the bf16 experts through the rounding path unrounded (the identity: must reproduce the model) | 16.000 | 225.5 | 0.0047 (0.0006) |  | 0.0379 | 99.01 |  | 3.021 | -0.02% | 0.6 |
| 10 | hbr40 | Hummingbird, neuron rows, mixed by usage: the 40% most-used experts of each layer 3 bits, the rest 2; GPTQ act-order on both matrices, one rotation of the latent a layer (the export's recipe) | 2.988 | 42.1 | 0.0946 (0.0180) | 0.0943 | 1.7239 | 93.38 | 93.39 | 3.334 | +10.34% | 23.2 |
| 8 | hb40 | Hummingbird, mixed by usage: the 40% most-used experts of each layer hlm3g, the rest hlm2g | 2.961 | 41.7 | 0.0831 (0.0171) | 0.0831 | 1.5417 | 93.96 | 93.91 | 3.286 | +8.76% | 15.9 |
| 11 | hbr40p | hbr40 with a rotation of the latent for each expert (what sharing the layer's costs) | 2.988 | 42.1 | 0.0968 (0.0186) | 0.0968 | 1.8391 | 93.43 | 93.37 | 3.344 | +10.70% | 21.6 |

The control (ref) is the experts as released, run through the same path unrounded: its KL 0.0047 from the bf16 model is the floor of that column (the path sums the experts in float32, the model's own kernels in bf16). The columns 'from control' score every other variant against the control instead: what rounding the weights alone costs.

### KL divergence from bf16, mean, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref | 0.0047 | 0.0072 | 0.0023 | 0.0085 | 0.0050 | 0.0030 | 0.0024 |
| hbr40 | 0.0946 | 0.2352 | 0.0478 | 0.1277 | 0.0524 | 0.0790 | 0.0253 |
| hb40 | 0.0831 | 0.2109 | 0.0368 | 0.1215 | 0.0459 | 0.0621 | 0.0215 |
| hbr40p | 0.0968 | 0.2448 | 0.0460 | 0.1288 | 0.0514 | 0.0832 | 0.0264 |

### KL divergence from the control (ref), mean, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| hbr40 | 0.0943 | 0.2358 | 0.0474 | 0.1278 | 0.0511 | 0.0784 | 0.0255 |
| hb40 | 0.0831 | 0.2111 | 0.0372 | 0.1220 | 0.0451 | 0.0618 | 0.0217 |
| hbr40p | 0.0968 | 0.2451 | 0.0459 | 0.1292 | 0.0507 | 0.0831 | 0.0266 |

### top-1 agreement with bf16, %, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| ref | 99.01 | 98.30 | 99.57 | 98.42 | 99.27 | 98.91 | 99.56 |
| hbr40 | 93.38 | 87.80 | 96.78 | 90.25 | 96.19 | 91.78 | 97.48 |
| hb40 | 93.96 | 88.52 | 97.20 | 90.71 | 96.52 | 93.16 | 97.64 |
| hbr40p | 93.43 | 87.13 | 97.04 | 90.23 | 96.41 | 92.07 | 97.69 |

### top-1 agreement with the control, %, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| hbr40 | 93.39 | 87.80 | 96.91 | 90.11 | 96.25 | 91.88 | 97.39 |
| hb40 | 93.91 | 88.51 | 97.24 | 90.49 | 96.48 | 93.15 | 97.58 |
| hbr40p | 93.37 | 87.03 | 97.09 | 90.13 | 96.31 | 92.12 | 97.55 |

### perplexity, by domain

| variant | all | wiki | code | chat | math | multi | reason |
|---|---:|---:|---:|---:|---:|---:|---:|
| bf16 | 3.021 | 4.135 | 1.676 | 7.135 | 2.687 | 3.557 | 1.609 |
| ref | 3.021 | 4.143 | 1.674 | 7.101 | 2.693 | 3.558 | 1.610 |
| hbr40 | 3.334 | 5.221 | 1.750 | 8.341 | 2.874 | 3.785 | 1.656 |
| hb40 | 3.286 | 5.028 | 1.730 | 8.291 | 2.831 | 3.738 | 1.649 |
| hbr40p | 3.344 | 5.264 | 1.736 | 8.443 | 2.851 | 3.813 | 1.668 |

### The most-used experts (profile from the calibration text)

| share of a layer's experts | per layer | calibration routed tokens covered % | held-out routed tokens covered % |
|---:|---:|---:|---:|
| 40% | 205 | 56.18 | 56.03 |

### Weight error (relative RMSE of the rounded experts against the bf16 original, all layers)

| variant | spec | weights | relative RMSE |
|---|---|---:|---:|
| ref | ref | 112,742,891,520 | 0.0000 |
| hbr40 | hbr2g | 67,601,694,720 | 0.4997 |
| hbr40 | hbr3g | 45,141,196,800 | 0.2302 |
| hb40 | hlm2g | 67,601,694,720 | 0.4458 |
| hb40 | hlm3g | 45,141,196,800 | 0.2165 |
| hbr40p | hbp2g | 67,601,694,720 | 0.4996 |
| hbr40p | hbp3g | 45,141,196,800 | 0.2302 |

GPTQ (hbr40): 20398 expert-layers rounded with GPTQ, 82 fell back to nearest (fewer than 2048 of the 294,912 calibration tokens routed to them).

GPTQ (hb40): 20397 expert-layers rounded with GPTQ, 83 fell back to nearest (fewer than 2048 of the 294,912 calibration tokens routed to them).

GPTQ (hbr40p): 20397 expert-layers rounded with GPTQ, 83 fell back to nearest (fewer than 2048 of the 294,912 calibration tokens routed to them).
