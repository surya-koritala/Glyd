| Format | Checkpoints | Samples (with a CM) | Stored | Order 0 | Best static context | CM code length | Layout estimate a tensor | Fixed-rate layout | Best general compressor (its samples: stored / static / CM) | Room on its own bytes: static / CM | Order 0 over samples: min / median / max |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | ---: | ---: |
| bf16 | 9 | 90 (90) | 16.000 | 10.542 | **10.458** | 10.466 | 10.810 | 10.810 | 10.518 (zpaq -m5; 16.000 / 10.481 / 10.493) | 34.6% / 34.6% | 9.356 / 10.495 / 13.098 |
| bf16 holding FP8 values | 1 | 10 (10) | 16.000 | 7.686 | **7.670** | 7.676 | 10.682 | 10.682 | 6.508 (zpaq -m5; 16.000 / 6.483 / 6.500) | 52.1% / 52.0% | 6.483 / 6.497 / 10.655 |
| FP8, 128x128 block scales | 2 | 18 (18) | 8.001 | 6.562 | **6.514** | 6.510 | 6.726 | 6.726 | 6.555 (zpaq -m5; 8.001 / 6.533 / 6.521) | 18.6% / 18.6% | 6.486 / 6.544 / 6.771 |
| FP8, a scale a tensor | 1 | 9 (9) | 8.000 | 6.480 | **6.475** | 6.484 | 6.700 | 6.700 | 6.498 (zpaq -m5; 8.000 / 6.477 / 6.487) | 19.1% / 18.9% | 6.473 / 6.477 / 6.506 |
| FP8, a scale a row | 1 | 9 (9) | 8.007 | 6.541 | **6.510** | 6.522 | 6.804 | 6.801 | 6.566 (zpaq -m5; 8.006 / 6.531 / 6.541) | 18.7% / 18.5% | 6.486 / 6.503 / 6.617 |
| MXFP8 | 1 | 9 (9) | 8.250 | 6.527 | **6.496** | 6.500 | 6.957 | 6.760 | 6.551 (zpaq -m5; 8.250 / 6.503 / 6.505) | 21.3% / 21.2% | 6.499 / 6.504 / 6.555 |
| NVFP4 | 4 | 30 (30) | 4.500 | 4.192 | **4.154** | 4.144 | 4.500 | 4.260 | 4.193 (zpaq -m5; 4.500 / 4.157 / 4.141) | 7.7% / 7.9% | 3.933 / 3.968 / 3.974 |
| MXFP4 | 3 | 22 (22) | 4.250 | 3.774 | **3.720** | 3.719 | 4.250 | 4.055 | 3.770 (zpaq -m5; 4.250 / 3.726 / 3.723) | 12.5% / 12.5% | 3.671 / 3.752 / 3.895 |
| INT4 GPTQ | 1 | 9 (9) | 4.156 | 3.522 | **3.511** | 3.517 | 4.156 | 3.712 | 3.553 (zpaq -m5; 4.156 / 3.505 / 3.510) | 15.5% / 15.4% | 3.448 / 3.473 / 3.480 |
| INT4 AWQ | 2 | 18 (18) | 4.606 | 4.261 | **4.189** | 4.239 | 4.606 | 4.431 | 4.220 (zpaq -m5; 4.606 / 4.158 / 4.210) | 9.1% / 8.0% | 3.610 / 3.864 / 3.896 |
| INT4 compressed-tensors | 2 | 18 (18) | 4.309 | 3.747 | **3.726** | 3.723 | 4.309 | 3.959 | 3.736 (zpaq -m5; 4.314 / 3.707 / 3.701) | 13.5% / 13.6% | 3.428 / 3.600 / 3.798 |
| INT8 W8A8 | 1 | 9 (9) | 8.003 | 6.998 | **6.966** | 6.972 | 8.003 | 7.220 | 6.960 (zpaq -m5; 8.002 / 6.943 / 6.946) | 13.0% / 12.9% | 6.921 / 7.021 / 7.068 |
| GGUF Q4_K_M | 1 | 10 (10) | 5.047 | 4.793 | **4.746** | 4.745 | 5.047 | 5.021 | 5.287 (zpaq -m5; 5.394 / 5.094 / 5.099) | 6.0% / 6.0% | 3.859 / 4.863 / 7.529 |
| GGUF Q8_0 | 1 | 10 (10) | 8.500 | 7.904 | **7.887** | 7.885 | 8.500 | 8.435 | 8.082 (zpaq -m5; 8.500 / 7.875 / 7.870) | 7.2% / 7.2% | 7.585 / 7.671 / 7.690 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 7 | 11 (11) | 8.051 | 6.552 | **6.508** | 6.498 | 6.795 | 6.757 | 6.565 (zpaq -m5; 8.074 / 6.520 / 6.510) | 19.2% / 19.3% | 6.480 / 6.546 / 6.618 |
| 16-bit parts of quantized checkpoints | 18 | 18 (18) | 16.000 | 10.307 | **10.284** | 10.265 | 10.855 | 10.855 | - | 35.7% / 35.8% | 6.547 / 10.490 / 10.564 |

| Format | Whole checkpoints: off at the static floor (range) | off at the CM (range) | off at the layout estimate (range; quantized checkpoints: their 16-bit and FP8 parts only) | off at the fixed-rate layouts | room left: static / CM |
| :--- | ---: | ---: | ---: | ---: | ---: |
| bf16 | 34.6% (34.4-34.8%) | 34.5% (34.3-34.7%) | 32.1% (22.8-33.5%) | 32.1% | **2.5 / 2.4 pts** |
| bf16 holding FP8 values | 59.3% (59.3%) | 59.2% (59.2%) | 33.4% (33.4%) | 33.4% | **25.9 / 25.8 pts** |
| FP8, 128x128 block scales | 20.5% (19.4-21.8%) | 20.6% (19.5-21.9%) | 18.0% (16.8-19.4%) | 18.0% | **2.5 / 2.6 pts** |
| FP8, a scale a tensor | 19.5% (19.5%) | 19.3% (19.3%) | 16.7% (16.7%) | 16.7% | **2.8 / 2.7 pts** |
| FP8, a scale a row | 21.0% (21.0%) | 20.9% (20.9%) | 17.6% (17.6%) | 17.7% | **3.3 / 3.2 pts** |
| MXFP8 | 21.4% (21.4%) | 21.4% (21.4%) | 15.9% (15.9%) | 18.3% | **5.5 / 5.5 pts** |
| NVFP4 | 17.5% (12.1-24.7%) | 17.7% (12.7-24.7%) | 12.3% (6.0-21.2%) | 15.4% | **5.2 / 5.4 pts** |
| MXFP4 | 14.6% (14.2-17.1%) | 14.6% (14.1-17.1%) | 3.4% (2.4-6.5%) | 7.4% | **11.2 / 11.2 pts** |
| INT4 GPTQ | 21.1% (21.1%) | 21.0% (21.0%) | 10.1% (10.1%) | 17.4% | **11.0 / 10.9 pts** |
| INT4 AWQ | 18.0% (16.7-19.0%) | 17.2% (15.9-18.3%) | 8.4% (4.4-11.6%) | 10.5% | **9.5 / 8.8 pts** |
| INT4 compressed-tensors | 22.8% (22.1-24.2%) | 23.3% (22.1-25.5%) | 12.6% (11.8-14.1%) | 18.4% | **10.2 / 10.7 pts** |
| INT8 W8A8 | 26.3% (26.3%) | 26.2% (26.2%) | 20.6% (20.6%) | 24.3% | **5.7 / 5.6 pts** |
| GGUF Q4_K_M | 5.9% (5.9%) | 5.9% (5.9%) | 0.0% (0.0%) | 0.5% | **5.9 / 5.9 pts** |
| GGUF Q8_0 | 7.2% (7.2%) | 7.2% (7.2%) | 0.0% (0.0%) | 0.8% | **7.2 / 7.2 pts** |

| Format | Samples | Stored | Order 0 | Fields apart | Best position context | Best neighbour context | Matching tensor | Best single context | CM: order 0 | + position, scale | + neighbours | + matching tensor | CM's share of the values |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 27 | 16.000 | 10.505 | 10.552 | 10.481 | 10.503 | 10.512 | 10.481 | 10.493 | 10.490 | 10.493 | 10.493 | 0.540 |
| bf16 holding FP8 values | 3 | 16.000 | 6.496 | 6.543 | 6.483 | 6.495 | 6.496 | 6.483 | 6.499 | 6.497 | 6.499 | 6.500 | 0.500 |
| FP8, 128x128 block scales | 6 | 8.000 | 6.582 | 6.621 | 6.533 | 6.572 | 6.577 | 6.533 | 6.546 | 6.519 | 6.521 | 6.521 | 0.600 |
| FP8, a scale a tensor | 3 | 8.000 | 6.484 | 6.532 | 6.477 | 6.483 | 6.484 | 6.477 | 6.483 | 6.484 | 6.487 | 6.487 | 0.562 |
| FP8, a scale a row | 3 | 8.000 | 6.572 | 6.613 | 6.529 | 6.569 | 6.572 | 6.529 | 6.548 | 6.535 | 6.538 | 6.538 | 1.000 |
| MXFP8 | 3 | 8.000 | 6.524 | 6.568 | 6.484 | 6.521 | 6.524 | 6.484 | 6.527 | 6.483 | 6.485 | 6.485 | 0.500 |
| NVFP4 | 9 | 4.000 | 3.959 | 3.982 | 3.928 | 3.957 | 3.960 | 3.928 | 3.959 | 3.911 | 3.911 | 3.911 | 0.554 |
| MXFP4 | 8 | 4.000 | 3.759 | 3.867 | 3.706 | 3.753 | 3.759 | 3.706 | 3.759 | 3.704 | 3.703 | 3.703 | 0.833 |
| INT4 GPTQ | 3 | 4.000 | 3.461 | - | 3.452 | 3.460 | 3.461 | 3.452 | 3.462 | 3.454 | 3.455 | 3.455 | 1.000 |
| INT4 AWQ | 6 | 4.000 | 3.871 | - | 3.801 | 3.864 | 3.874 | 3.801 | 3.871 | 3.866 | 3.856 | 3.856 | 0.528 |
| INT4 compressed-tensors | 6 | 4.000 | 3.586 | - | 3.564 | 3.584 | 3.586 | 3.564 | 3.583 | 3.558 | 3.558 | 3.558 | 0.506 |
| INT8 W8A8 | 3 | 8.000 | 6.976 | - | 6.942 | 6.975 | 6.976 | 6.942 | 6.946 | 6.942 | 6.944 | 6.945 | 0.500 |
| GGUF Q4_K_M | 3 | 4.877 | 4.722 | - | 4.682 | 4.716 | 4.315 | 4.682 | 4.722 | 4.688 | 4.686 | 4.686 | 0.561 |
| GGUF Q8_0 | 3 | 8.000 | 7.660 | - | 7.645 | 7.659 | 7.648 | 7.645 | 7.661 | 7.639 | 7.640 | 7.640 | 0.604 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 4 | 8.000 | 6.570 | 6.610 | 6.512 | 6.564 | 6.575 | 6.512 | 6.542 | 6.499 | 6.501 | 6.501 | 0.564 |

| Format | Samples | Field | Stored bits | CM bits (all contexts) | Order-0 entropy of the field alone |
| :--- | ---: | :--- | ---: | ---: | ---: |
| bf16 | 90 | exponent | 8 | 2.560 | 2.631 |
| bf16 | 90 | sign | 1 | 0.991 | 1.000 |
| bf16 | 90 | mantissa | 7 | 6.914 | 6.959 |
| bf16 holding FP8 values | 10 | exponent | 8 | 2.561 | 2.570 |
| bf16 holding FP8 values | 10 | sign | 1 | 0.996 | 1.000 |
| bf16 holding FP8 values | 10 | mantissa | 7 | 4.119 | 4.163 |
| FP8, 128x128 block scales | 18 | exponent | 4 | 2.572 | 2.622 |
| FP8, 128x128 block scales | 18 | sign | 1 | 1.000 | 1.000 |
| FP8, 128x128 block scales | 18 | mantissa | 3 | 2.938 | 2.981 |
| FP8, a scale a tensor | 9 | exponent | 4 | 2.554 | 2.551 |
| FP8, a scale a tensor | 9 | sign | 1 | 1.000 | 1.000 |
| FP8, a scale a tensor | 9 | mantissa | 3 | 2.931 | 2.978 |
| FP8, a scale a row | 9 | exponent | 4 | 2.584 | 2.603 |
| FP8, a scale a row | 9 | sign | 1 | 1.000 | 1.000 |
| FP8, a scale a row | 9 | mantissa | 3 | 2.936 | 2.979 |
| MXFP8 | 9 | exponent | 4 | 2.550 | 2.573 |
| MXFP8 | 9 | sign | 1 | 1.000 | 1.000 |
| MXFP8 | 9 | mantissa | 3 | 2.934 | 2.984 |
| NVFP4 | 30 | magnitude | 3 | 2.920 | - |
| NVFP4 | 30 | sign | 1 | 0.993 | 1.000 |
| MXFP4 | 22 | magnitude | 3 | 2.756 | - |
| MXFP4 | 22 | sign | 1 | 0.943 | 0.996 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 11 | exponent | 4 | 2.563 | 2.609 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 11 | sign | 1 | 0.996 | 1.000 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 11 | mantissa | 3 | 2.934 | 2.980 |
| 16-bit parts of quantized checkpoints | 18 | exponent | 8 | 2.720 | 2.563 |
| 16-bit parts of quantized checkpoints | 18 | sign | 1 | 0.989 | 0.999 |
| 16-bit parts of quantized checkpoints | 18 | mantissa | 7 | 6.556 | 6.829 |

| Checkpoint | Format | Stored GB | Samples by group | Layout estimate | Fixed-rate layouts | Order 0 | Static floor | CM | Room left: static / CM (points) |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| zai-org/GLM-5.3-BF16 | bf16 | 1,506.7 | bf16 expert 6, bf16 attn 1, bf16 mlp 1, bf16 lm_head 1, bf16 embed 1 | 33.5% | 33.5% | 34.5% | 34.6% | 34.5% | 1.1 / 1.0 |
| nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16 | bf16 | 247.2 | bf16 expert 5, bf16 attn 1, bf16 mtp 1, bf16 mlp 1, bf16 embed 1, bf16 lm_head 1 | 22.8% | 22.8% | 29.2% | 34.5% | 34.3% | 11.8 / 11.5 |
| google/gemma-4-31B-it | bf16 | 62.5 | bf16 mlp 6, bf16 attn 2, bf16 embed 1, bf16 vision 1 | 32.7% | 32.7% | 34.1% | 34.6% | 34.5% | 1.9 / 1.8 |
| meta-models/Muse-Glimmer-30B | bf16 | 59.6 | bf16 mlp 6, bf16 attn 1, bf16 vision 1, bf16 lm_head 1, bf16 embed 1 | 33.2% | 33.2% | 34.4% | 34.6% | 34.4% | 1.3 / 1.2 |
| Qwen/Qwen3.5-27B | bf16 | 55.6 | bf16 mlp 5, bf16 attn 1, bf16 lm_head 1, bf16 embed 1, bf16 vision 1, bf16 mtp 1 | 33.1% | 33.1% | 34.4% | 34.5% | 34.4% | 1.4 / 1.3 |
| Qwen/Qwen3.8-27B | bf16 | 55.6 | bf16 mlp 5, bf16 attn 1, bf16 lm_head 1, bf16 embed 1, bf16 vision 1, bf16 mtp 1 | 33.1% | 33.1% | 34.4% | 34.5% | 34.4% | 1.4 / 1.3 |
| google/gemma-4-12B-it | bf16 | 23.9 | bf16 mlp 7, bf16 attn 1, bf16 embed 1, bf16 vision 1 | 33.2% | 33.2% | 34.3% | 34.4% | 34.3% | 1.2 / 1.1 |
| Qwen/Qwen3.5-9B | bf16 | 19.3 | bf16 mlp 4, bf16 attn 2, bf16 lm_head 1, bf16 embed 1, bf16 vision 1, bf16 mtp 1 | 33.0% | 33.0% | 34.3% | 34.4% | 34.4% | 1.4 / 1.4 |
| ibm-granite/granite-4.2-8b | bf16 | 17.6 | bf16 mlp 7, bf16 attn 1, bf16 lm_head 1, bf16 embed 1 | 33.1% | 33.1% | 34.7% | 34.8% | 34.7% | 1.7 / 1.6 |
| MiniMaxAI/MiniMax-M3 | bf16 holding FP8 values | 854.2 | bf16 expert 5, bf16 attn 1, bf16 mlp 1, bf16 lm_head 1, bf16 embed 1, bf16 vision 1 | 33.4% | 33.4% | 59.3% | 59.3% | 59.2% | 25.9 / 25.8 |
| Qwen/Qwen3.6-35B-A3B-FP8 | FP8, 128x128 block scales | 37.5 | fp8 expert 6, fp8 attn 1, fp8 mtp 1, fp8 mlp 1, bf16 lm_head 1 | 16.8% | 16.8% | 18.5% | 19.4% | 19.5% | 2.6 / 2.7 |
| Qwen/Qwen3.8-27B-FP8 | FP8, 128x128 block scales | 30.9 | fp8 mlp 6, fp8 attn 2, fp8 mtp 1, bf16 lm_head 1 | 19.4% | 19.4% | 21.4% | 21.8% | 21.9% | 2.4 / 2.4 |
| mistralai/Mistral-Small-4-119B-2603 | FP8, a scale a tensor | 120.9 | fp8 expert 7, fp8 attn 1, fp8 mlp 1, bf16 lm_head 1 | 16.7% | 16.7% | 19.4% | 19.5% | 19.3% | 2.8 / 2.7 |
| RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic | FP8, a scale a row | 28.6 | fp8 expert 7, fp8 attn 1, fp8 mlp 1, bf16 lm_head 1 | 17.6% | 17.7% | 20.6% | 21.0% | 20.9% | 3.3 / 3.2 |
| MiniMaxAI/MiniMax-M3-MXFP8 | MXFP8 | 443.7 | fp8 expert 7, fp8 attn 1, fp8 mlp 1, bf16 lm_head 1 | 15.9% | 18.3% | 21.1% | 21.4% | 21.4% | 5.5 / 5.5 |
| nvidia/Gemma-4-31B-IT-NVFP4 | NVFP4 | 32.6 | nvfp4 mlp 9, bf16 attn 1 | 21.2% | 23.2% | 24.3% | 24.7% | 24.7% | 3.5 / 3.5 |
| nvidia/Qwen3.6-35B-A3B-NVFP4 | NVFP4 | 23.4 | nvfp4 expert 6, fp8 attn 1, nvfp4 lm_head 1, nvfp4 mlp 1, bf16 mtp 1 | 6.0% | 10.3% | 11.7% | 12.1% | 12.7% | 6.1 / 6.7 |
| nvidia/Qwen3.8-27B-NVFP4 | NVFP4 | 21.9 | nvfp4 mlp 4, fp8 attn 4, nvfp4 lm_head 1, bf16 embed 1 | 11.9% | 14.3% | 16.3% | 16.9% | 16.9% | 5.0 / 5.0 |
| nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 | NVFP4 | 21.6 | nvfp4 expert 6, fp8 attn 1, nvfp4 mlp 1, nvfp4 lm_head 1, bf16 mtp 1 | 6.2% | 10.3% | 12.2% | 13.0% | 13.5% | 6.8 / 7.3 |
| moonshotai/Kimi-K3 | MXFP4 | 1,560.9 | mxfp4 expert 9, bf16 attn 1 | 2.4% | 6.5% | 12.9% | 14.2% | 14.1% | 11.7 / 11.6 |
| deepseek-ai/DeepSeek-V4.1-Flash | MXFP4 | 510.3 | mxfp4 expert 3, fp8 embed 2, mxfp4 mtp 1, fp8 attn 1, fp8 mlp 1, fp8 mtp 1, bf16 other 1 | 6.5% | 10.0% | 13.1% | 14.8% | 15.0% | 8.3 / 8.5 |
| amd/MiniMax-M3-MXFP4 | MXFP4 | 242.7 | mxfp4 expert 8, mxfp4 mlp 1, bf16 attn 1 | 2.9% | 7.2% | 16.2% | 17.1% | 17.1% | 14.2 / 14.2 |
| Qwen/Qwen3.5-35B-A3B-GPTQ-Int4 | INT4 GPTQ | 24.4 | int4 expert 9, bf16 attn 1 | 10.1% | 17.4% | 20.9% | 21.1% | 21.0% | 11.0 / 10.9 |
| QuantTrio/Qwen3.6-35B-A3B-AWQ | INT4 AWQ | 25.4 | int4 expert 9, bf16 attn 1 | 11.6% | 13.2% | 18.1% | 19.0% | 18.3% | 7.3 / 6.7 |
| cyankiwi/Qwen3.6-27B-AWQ-INT4 | INT4 AWQ | 20.4 | int4 mlp 6, int4 attn 2, int4 mtp 1, bf16 lm_head 1 | 4.4% | 7.1% | 15.6% | 16.7% | 15.9% | 12.3 / 11.5 |
| RedHatAI/Qwen3.8-27B-INT4 | INT4 compressed-tensors | 19.5 | int4 mlp 6, int4 attn 3, bf16 lm_head 1 | 11.8% | 19.0% | 21.9% | 22.1% | 22.1% | 10.2 / 10.2 |
| google/gemma-4-12B-it-qat-w4a16-ct | INT4 compressed-tensors | 10.3 | int4 mlp 7, int4 attn 2, bf16 lm_head 1 | 14.1% | 17.2% | 23.8% | 24.2% | 25.5% | 10.2 / 11.5 |
| RedHatAI/Qwen3.5-9B-quantized.w8a8 | INT8 W8A8 | 14.0 | int8 mlp 8, int8 attn 1, bf16 attn 1 | 20.6% | 24.3% | 26.0% | 26.3% | 26.2% | 5.7 / 5.6 |
| unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q4_K_M.gguf | GGUF Q4_K_M | 5.7 | Q4_K mlp 3, Q6_K lm_head 1, Q5_K attn 1, Q6_K mlp 1, Q4_K embed 1, Q4_K attn 1, Q6_K attn 1, Q8_0 attn 1 | 0.0% | 0.5% | 5.0% | 5.9% | 5.9% | 5.9 / 5.9 |
| unsloth/Qwen3.5-9B-GGUF Qwen3.5-9B-Q8_0.gguf | GGUF Q8_0 | 9.5 | Q8_0 mlp 6, Q8_0 attn 2, Q8_0 lm_head 1, Q8_0 embed 1 | 0.0% | 0.8% | 7.0% | 7.2% | 7.2% | 7.2 / 7.2 |

| Format | Samples | Stored | xz -9e | zstd -22 --long | zpaq -m5 | Static floor | CM |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bf16 | 27 | 16.000 | 11.354 | 12.484 | 10.518 | 10.481 | 10.493 |
| bf16 holding FP8 values | 3 | 16.000 | 7.096 | 7.975 | 6.508 | 6.483 | 6.500 |
| FP8, 128x128 block scales | 6 | 8.001 | 6.646 | 6.640 | 6.555 | 6.533 | 6.521 |
| FP8, a scale a tensor | 3 | 8.000 | 6.589 | 6.557 | 6.498 | 6.477 | 6.487 |
| FP8, a scale a row | 3 | 8.006 | 6.648 | 6.651 | 6.566 | 6.531 | 6.541 |
| MXFP8 | 3 | 8.250 | 6.656 | 6.639 | 6.551 | 6.503 | 6.505 |
| NVFP4 | 9 | 4.500 | 4.246 | 4.251 | 4.193 | 4.157 | 4.141 |
| MXFP4 | 8 | 4.250 | 3.832 | 3.801 | 3.770 | 3.726 | 3.723 |
| INT4 GPTQ | 3 | 4.156 | 3.589 | 3.560 | 3.553 | 3.505 | 3.510 |
| INT4 AWQ | 6 | 4.606 | 4.300 | 4.309 | 4.220 | 4.158 | 4.210 |
| INT4 compressed-tensors | 6 | 4.314 | 3.795 | 3.785 | 3.736 | 3.707 | 3.701 |
| INT8 W8A8 | 3 | 8.002 | 7.053 | 7.049 | 6.960 | 6.943 | 6.946 |
| GGUF Q4_K_M | 3 | 5.394 | 5.359 | 5.347 | 5.287 | 5.094 | 5.099 |
| GGUF Q8_0 | 3 | 8.500 | 8.235 | 8.162 | 8.082 | 7.875 | 7.870 |
| FP8 parts of NVFP4 and MXFP4 checkpoints | 4 | 8.074 | 6.659 | 6.641 | 6.565 | 6.520 | 6.510 |

| Format | A table a tensor | A table a kind | One table a checkpoint |
| :--- | ---: | ---: | ---: |
| bf16 | 10.542 | 10.589 | 10.776 |
| bf16 holding FP8 values | 7.686 | 7.686 | 7.845 |
| FP8, 128x128 block scales | 6.562 | 6.564 | 6.577 |
| FP8, a scale a tensor | 6.480 | 6.546 | 6.722 |
| FP8, a scale a row | 6.538 | 6.540 | 6.548 |
| MXFP8 | 6.510 | 6.511 | 6.511 |
| NVFP4 | 3.957 | 3.957 | 3.958 |
| MXFP4 | 3.754 | 3.755 | 3.755 |
| INT4 GPTQ | 3.468 | 3.468 | 3.468 |
| INT4 AWQ | 3.879 | 3.879 | 3.879 |
| INT4 compressed-tensors | 3.602 | 3.605 | 3.605 |
| INT8 W8A8 | 6.997 | 6.998 | 6.998 |
| GGUF Q8_0 | 7.669 | 7.669 | 7.669 |
