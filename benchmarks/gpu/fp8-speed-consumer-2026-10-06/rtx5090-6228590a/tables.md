# NVIDIA GeForce RTX 5090 (sm_120, L2 96 MiB)

vLLM's kernels: Mistral-Small-4-119B-2603: FlashInferFP8ScaledMMLinearKernel; Qwen3.6-35B-A3B-FP8: CutlassFp8BlockScaledMMKernel; Qwen3.8-27B-FP8: CutlassFp8BlockScaledMMKernel; gemma-4-26B-A4B-it-FP8-dynamic: CutlassFP8ScaledMMLinearKernel

Stock vLLM's own selection on each checkpoint (bench_f8.py --served): Qwen3.6-35B-A3B-FP8: CutlassFp8BlockScaledMMKernel; Qwen3.8-27B-FP8: CutlassFp8BlockScaledMMKernel; gemma-4-26B-A4B-it-FP8-dynamic: CutlassFP8ScaledMMLinearKernel; Mistral-Small-4-119B-2603: no line

Shapes x tokens left out of the sums: 0 of 140 (none)

## Per-shape sums: a runner's times added over vLLM's added (each distinct shape once; dense: the expert shapes left out)

| model | tp | shapes | runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | :--- | :--- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 | 1 | all | ours | 1.19 | 1.24 | 1.96 | 2.75 |
| Qwen3.6-35B-A3B-FP8 | 1 | all | decode + vLLM's kernel | 1.70 | 1.60 | 1.57 | 1.62 |
| Qwen3.6-35B-A3B-FP8 | 1 | dense | ours | 1.25 | 1.29 | 2.00 | 2.73 |
| Qwen3.6-35B-A3B-FP8 | 1 | dense | decode + vLLM's kernel | 1.74 | 1.64 | 1.60 | 1.66 |
| Qwen3.6-35B-A3B-FP8 | 2 | all | ours | 1.17 | 1.25 | 2.15 | 3.31 |
| Qwen3.6-35B-A3B-FP8 | 2 | all | decode + vLLM's kernel | 1.73 | 1.65 | 1.64 | 1.64 |
| Qwen3.6-35B-A3B-FP8 | 2 | dense | ours | 1.27 | 1.34 | 2.21 | 3.30 |
| Qwen3.6-35B-A3B-FP8 | 2 | dense | decode + vLLM's kernel | 1.84 | 1.75 | 1.74 | 1.73 |
| Qwen3.8-27B-FP8 | 1 | all | ours | 1.03 | 1.04 | 1.83 | 2.43 |
| Qwen3.8-27B-FP8 | 1 | all | decode + vLLM's kernel | 2.14 | 2.09 | 2.05 | 2.16 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | ours | 1.22 | 1.34 | 2.21 | 2.69 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | all | decode + vLLM's kernel | 1.69 | 1.61 | 1.63 | 1.59 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | ours | 1.25 | 1.35 | 2.15 | 2.61 |
| gemma-4-26B-A4B-it-FP8-dynamic | 1 | dense | decode + vLLM's kernel | 1.70 | 1.62 | 1.64 | 1.62 |
| Mistral-Small-4-119B-2603 | 1 | all | ours | 0.35 | 0.41 | 0.67 | 1.02 |
| Mistral-Small-4-119B-2603 | 1 | all | decode + vLLM's kernel | 1.22 | 1.22 | 1.22 | 1.22 |
| Mistral-Small-4-119B-2603 | 1 | dense | ours | 0.33 | 0.39 | 0.64 | 0.99 |
| Mistral-Small-4-119B-2603 | 1 | dense | decode + vLLM's kernel | 1.20 | 1.20 | 1.20 | 1.20 |
| Mistral-Small-4-119B-2603 | 2 | all | ours | 0.32 | 0.40 | 0.65 | 1.02 |
| Mistral-Small-4-119B-2603 | 2 | all | decode + vLLM's kernel | 1.20 | 1.20 | 1.20 | 1.20 |
| Mistral-Small-4-119B-2603 | 2 | dense | ours | 0.45 | 0.54 | 0.94 | 1.51 |
| Mistral-Small-4-119B-2603 | 2 | dense | decode + vLLM's kernel | 1.25 | 1.26 | 1.26 | 1.27 |

## Per shape: geometric mean of a runner's time over vLLM's, all models (in brackets: shapes at most vLLM's; then the range)

| runner | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| ours | 0.89 (14/35; 0.22-1.49) | 0.98 (11/35; 0.29-1.59) | 1.64 (5/35; 0.41-2.63) | 2.32 (3/35; 0.66-3.96) |
| decode + vLLM's kernel | 1.62 (0/35; 1.09-2.62) | 1.57 (0/35; 1.09-2.58) | 1.56 (0/35; 1.09-2.56) | 1.58 (0/35; 1.09-2.53) |

## Per shape: ours over the decode route (in brackets: shapes where the route is faster; then the range)

| kind | M=1 | M=8 | M=32 | M=64 |
| :--- | ---: | ---: | ---: | ---: |
| block | 0.62 (0/19; 0.38-0.82) | 0.68 (0/19; 0.39-0.88) | 1.16 (17/19; 0.72-1.52) | 1.61 (18/19; 0.94-2.44) |
| channel | 0.72 (0/8; 0.64-0.82) | 0.85 (0/8; 0.70-0.96) | 1.40 (8/8; 1.11-1.64) | 1.77 (8/8; 1.23-2.32) |
| tensor | 0.31 (0/8; 0.20-0.47) | 0.38 (0/8; 0.27-0.56) | 0.62 (0/8; 0.38-0.98) | 0.97 (3/8; 0.61-1.58) |

## Every shape (microseconds)

| model | tensor | tp | N x K | M | vLLM | ours | decode + vLLM's kernel | decode alone | copies (MB) | left out |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 1 | 15.2 | 20.8 | 26.9 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 8 | 16.4 | 22.4 | 26.5 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 32 | 16.3 | 33.7 | 26.0 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | 9216 x 2048 | 64 | 16.3 | 47.3 | 27.9 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 1 | 15.2 | 13.6 | 24.4 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 8 | 16.2 | 17.3 | 25.2 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 32 | 16.4 | 26.4 | 25.3 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | o | 1 | 2048 x 4096 | 64 | 16.6 | 40.7 | 25.5 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 1 | 18.9 | 24.1 | 32.5 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 8 | 20.1 | 25.3 | 32.3 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 32 | 20.0 | 42.2 | 30.3 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | 12288 x 2048 | 64 | 24.4 | 59.1 | 38.8 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 1 | 13.7 | 19.0 | 24.3 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 8 | 14.7 | 20.4 | 24.8 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 32 | 14.8 | 31.3 | 24.7 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkv | 1 | 8192 x 2048 | 64 | 14.8 | 44.4 | 25.9 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 1 | 9.8 | 13.2 | 18.6 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 8 | 10.9 | 15.5 | 19.3 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 32 | 10.9 | 23.1 | 19.4 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_z | 1 | 4096 x 2048 | 64 | 10.9 | 34.7 | 19.3 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 1 | 9.2 | 6.7 | 13.1 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 8 | 10.0 | 8.9 | 13.9 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 32 | 10.0 | 18.4 | 14.0 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 1 | 1024 x 2048 | 64 | 10.2 | 31.3 | 14.2 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 1 | 5.0 | 5.8 | 8.2 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 8 | 5.9 | 6.5 | 8.9 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 32 | 6.0 | 10.0 | 9.1 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_down | 1 | 2048 x 512 | 64 | 6.0 | 14.9 | 9.0 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 1 | 9.7 | 13.4 | 19.4 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 8 | 10.8 | 15.5 | 20.2 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 32 | 10.9 | 25.6 | 20.3 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | qkv | 2 | 4608 x 2048 | 64 | 10.9 | 38.1 | 20.1 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 1 | 9.4 | 8.6 | 15.1 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 8 | 10.3 | 10.9 | 15.8 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 32 | 10.3 | 21.9 | 15.9 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | o | 2 | 2048 x 2048 | 64 | 10.4 | 35.3 | 16.1 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 1 | 11.2 | 16.7 | 21.3 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 8 | 12.3 | 18.3 | 22.1 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 32 | 12.3 | 26.8 | 22.2 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 2 | 6144 x 2048 | 64 | 12.5 | 38.2 | 22.4 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 1 | 9.1 | 6.2 | 12.2 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 8 | 9.9 | 8.4 | 13.0 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 32 | 9.9 | 17.8 | 13.1 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_gate_up | 2 | 512 x 2048 | 64 | 10.0 | 30.4 | 13.2 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 1 | 3.9 | 5.6 | 6.9 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 8 | 4.7 | 6.7 | 7.6 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 32 | 4.7 | 11.7 | 7.7 | - | 302 |  |
| Qwen3.6-35B-A3B-FP8 | expert_down | 2 | 2048 x 256 | 64 | 4.8 | 18.9 | 7.8 | - | 302 |  |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 1 | 49.6 | 52.7 | 87.8 | - | 367 |  |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 8 | 50.9 | 53.9 | 87.0 | - | 367 |  |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 32 | 51.1 | 94.9 | 82.2 | - | 367 |  |
| Qwen3.8-27B-FP8 | qkv | 1 | 14336 x 5120 | 64 | 56.0 | 131.5 | 105.6 | - | 367 |  |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 1 | 24.8 | 26.5 | 44.7 | - | 315 |  |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 8 | 25.9 | 28.3 | 45.5 | - | 315 |  |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 32 | 26.1 | 47.7 | 45.7 | - | 315 |  |
| Qwen3.8-27B-FP8 | o | 1 | 5120 x 6144 | 64 | 26.2 | 67.2 | 45.9 | - | 315 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 1 | 56.1 | 58.9 | 110.8 | - | 336 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 8 | 56.9 | 60.5 | 109.3 | - | 336 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 32 | 57.2 | 107.7 | 103.6 | - | 336 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | 16384 x 5120 | 64 | 58.4 | 148.1 | 126.6 | - | 336 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 1 | 36.9 | 40.0 | 60.8 | - | 315 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 8 | 37.9 | 41.6 | 60.1 | - | 315 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 32 | 38.1 | 70.8 | 58.9 | - | 315 |  |
| Qwen3.8-27B-FP8 | gdn_in_qkv | 1 | 10240 x 5120 | 64 | 38.7 | 97.8 | 67.4 | - | 315 |  |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 1 | 24.7 | 27.0 | 41.8 | - | 315 |  |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 8 | 25.7 | 28.6 | 42.5 | - | 315 |  |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 32 | 25.9 | 47.9 | 42.7 | - | 315 |  |
| Qwen3.8-27B-FP8 | gdn_in_z | 1 | 6144 x 5120 | 64 | 26.0 | 67.0 | 43.0 | - | 315 |  |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 1 | 113.9 | 114.5 | 297.8 | - | 357 |  |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 8 | 115.4 | 116.8 | 298.1 | - | 357 |  |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 32 | 116.7 | 215.0 | 298.7 | - | 357 |  |
| Qwen3.8-27B-FP8 | gate_up | 1 | 34816 x 5120 | 64 | 124.6 | 295.4 | 315.8 | - | 357 |  |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 1 | 63.5 | 60.6 | 146.1 | - | 357 |  |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 8 | 64.5 | 62.6 | 147.1 | - | 357 |  |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 32 | 65.0 | 111.9 | 148.2 | - | 357 |  |
| Qwen3.8-27B-FP8 | down | 1 | 5120 x 17408 | 64 | 65.8 | 153.6 | 149.3 | - | 357 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 1 | 18.6 | 23.5 | 31.0 | - | 323 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 8 | 19.1 | 26.4 | 29.7 | - | 323 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 32 | 19.1 | 39.1 | 29.7 | - | 323 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | 8192 x 2816 | 64 | 20.1 | 55.0 | 32.7 | - | 323 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 1 | 20.3 | 24.0 | 33.9 | - | 311 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 8 | 21.0 | 25.6 | 32.1 | - | 311 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 32 | 21.0 | 42.3 | 32.1 | - | 311 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | 9216 x 2816 | 64 | 22.0 | 60.4 | 34.9 | - | 311 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 1 | 12.4 | 16.1 | 21.7 | - | 311 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 8 | 13.1 | 18.8 | 22.4 | - | 311 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 32 | 13.0 | 33.7 | 22.4 | - | 311 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | 2816 x 4096 | 64 | 17.6 | 52.1 | 28.4 | - | 311 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 1 | 20.6 | 23.3 | 36.0 | - | 323 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 8 | 22.0 | 26.2 | 37.4 | - | 323 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 32 | 21.8 | 41.8 | 37.6 | - | 323 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | o_global | 1 | 2816 x 8192 | 64 | 31.1 | 60.3 | 49.2 | - | 323 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 1 | 12.1 | 16.7 | 20.4 | - | 309 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 8 | 12.7 | 19.0 | 20.5 | - | 309 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 32 | 12.7 | 29.0 | 20.7 | - | 309 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_gate_up | 1 | 4224 x 2816 | 64 | 14.2 | 42.5 | 24.8 | - | 309 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 1 | 8.8 | 12.4 | 15.3 | - | 303 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 8 | 9.4 | 15.0 | 15.7 | - | 303 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 32 | 9.0 | 22.1 | 15.8 | - | 303 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | mlp_down | 1 | 2816 x 2112 | 64 | 11.7 | 34.0 | 18.9 | - | 303 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 1 | 8.9 | 8.8 | 13.8 | - | 305 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 8 | 9.4 | 12.4 | 14.2 | - | 305 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 32 | 9.4 | 23.3 | 14.2 | - | 305 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_gate_up | 1 | 1408 x 2816 | 64 | 13.4 | 38.7 | 18.4 | - | 305 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 1 | 5.7 | 6.4 | 9.4 | - | 303 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 8 | 5.9 | 8.0 | 9.4 | - | 303 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 32 | 5.8 | 15.1 | 9.4 | - | 303 |  |
| gemma-4-26B-A4B-it-FP8-dynamic | expert_down | 1 | 2816 x 704 | 64 | 7.0 | 24.9 | 10.7 | - | 303 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 1 | 58.4 | 12.6 | 63.5 | - | 303 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 8 | 59.2 | 17.3 | 64.4 | - | 303 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 32 | 59.4 | 24.3 | 64.5 | - | 303 |  |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | 1344 x 4096 | 64 | 59.6 | 39.2 | 64.8 | - | 303 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 1 | 17.6 | 8.1 | 23.1 | - | 302 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 8 | 17.6 | 9.4 | 23.2 | - | 302 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 32 | 17.6 | 18.9 | 23.1 | - | 302 |  |
| Mistral-Small-4-119B-2603 | q_b | 1 | 4096 x 1024 | 64 | 17.6 | 29.6 | 23.2 | - | 302 |  |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 1 | 9.4 | 5.9 | 12.8 | - | 302 |  |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 8 | 9.4 | 7.0 | 13.0 | - | 302 |  |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 32 | 9.3 | 11.9 | 12.8 | - | 302 |  |
| Mistral-Small-4-119B-2603 | kv_b | 1 | 6144 x 256 | 64 | 9.2 | 19.0 | 12.6 | - | 302 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 1 | 51.9 | 18.3 | 66.1 | - | 302 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 8 | 52.0 | 20.5 | 65.9 | - | 302 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 32 | 52.0 | 33.7 | 66.1 | - | 302 |  |
| Mistral-Small-4-119B-2603 | o | 1 | 4096 x 4096 | 64 | 52.2 | 49.6 | 66.2 | - | 302 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 1 | 29.1 | 12.7 | 37.8 | - | 302 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 8 | 29.0 | 15.0 | 37.8 | - | 302 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 32 | 29.1 | 22.6 | 37.8 | - | 302 |  |
| Mistral-Small-4-119B-2603 | expert_down | 1 | 4096 x 2048 | 64 | 29.2 | 34.3 | 37.9 | - | 302 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 1 | 17.4 | 6.2 | 21.0 | - | 302 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 8 | 17.3 | 7.4 | 21.1 | - | 302 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 32 | 17.3 | 13.0 | 21.2 | - | 302 |  |
| Mistral-Small-4-119B-2603 | q_b | 2 | 2048 x 1024 | 64 | 17.3 | 20.6 | 21.3 | - | 302 |  |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 1 | 8.9 | 5.6 | 11.8 | - | 302 |  |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 8 | 8.9 | 6.6 | 11.8 | - | 302 |  |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 32 | 8.8 | 11.6 | 11.8 | - | 302 |  |
| Mistral-Small-4-119B-2603 | kv_b | 2 | 3072 x 256 | 64 | 8.8 | 18.8 | 11.9 | - | 302 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 1 | 51.6 | 13.1 | 60.2 | - | 302 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 8 | 51.7 | 16.8 | 60.3 | - | 302 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 32 | 51.8 | 25.8 | 60.5 | - | 302 |  |
| Mistral-Small-4-119B-2603 | expert_gate_up | 2 | 2048 x 4096 | 64 | 52.0 | 40.1 | 60.7 | - | 302 |  |

## Numerics: error against an exact float64 product of the dequantized weights (max: of the sum of |x w|; rms: of the outputs' rms)

| model | tensor | M | vLLM kernel | vLLM max | vLLM rms | ours max | ours rms |
| :--- | :--- | ---: | :--- | ---: | ---: | ---: | ---: |
| Qwen3.6-35B-A3B-FP8 | qkv | 1 | CutlassFp8BlockScaledMMKernel | 4.0e-03 | 2.5e-02 | 4.3e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 8 | CutlassFp8BlockScaledMMKernel | 4.5e-03 | 2.6e-02 | 5.1e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 32 | CutlassFp8BlockScaledMMKernel | 4.8e-03 | 2.6e-02 | 5.1e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | qkv | 64 | CutlassFp8BlockScaledMMKernel | 5.1e-03 | 2.6e-02 | 4.9e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 1 | CutlassFp8BlockScaledMMKernel | 2.9e-03 | 2.5e-02 | 2.1e-04 | 1.6e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 8 | CutlassFp8BlockScaledMMKernel | 2.6e-03 | 2.6e-02 | 3.0e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 32 | CutlassFp8BlockScaledMMKernel | 3.0e-03 | 2.6e-02 | 3.7e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | o | 64 | CutlassFp8BlockScaledMMKernel | 3.2e-03 | 2.6e-02 | 3.3e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 1 | CutlassFp8BlockScaledMMKernel | 4.3e-03 | 2.6e-02 | 3.9e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 8 | CutlassFp8BlockScaledMMKernel | 4.9e-03 | 2.5e-02 | 4.9e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 32 | CutlassFp8BlockScaledMMKernel | 5.2e-03 | 2.6e-02 | 5.0e-04 | 1.7e-03 |
| Qwen3.6-35B-A3B-FP8 | gdn_in_qkvz | 64 | CutlassFp8BlockScaledMMKernel | 5.1e-03 | 2.6e-02 | 6.4e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 1 | CutlassFp8BlockScaledMMKernel | 2.5e-03 | 2.6e-02 | 2.5e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 8 | CutlassFp8BlockScaledMMKernel | 2.7e-03 | 2.6e-02 | 3.5e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 32 | CutlassFp8BlockScaledMMKernel | 2.9e-03 | 2.6e-02 | 4.5e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | qkv | 64 | CutlassFp8BlockScaledMMKernel | 2.8e-03 | 2.6e-02 | 3.6e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 1 | CutlassFp8BlockScaledMMKernel | 2.0e-03 | 2.5e-02 | 2.6e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 8 | CutlassFp8BlockScaledMMKernel | 2.6e-03 | 2.6e-02 | 2.6e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 32 | CutlassFp8BlockScaledMMKernel | 2.6e-03 | 2.6e-02 | 3.4e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | o | 64 | CutlassFp8BlockScaledMMKernel | 2.5e-03 | 2.6e-02 | 3.1e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 1 | CutlassFp8BlockScaledMMKernel | 2.6e-03 | 2.6e-02 | 2.6e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 8 | CutlassFp8BlockScaledMMKernel | 3.0e-03 | 2.6e-02 | 3.5e-04 | 1.6e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 32 | CutlassFp8BlockScaledMMKernel | 3.0e-03 | 2.6e-02 | 3.1e-04 | 1.7e-03 |
| Qwen3.8-27B-FP8 | gdn_in_qkvz | 64 | CutlassFp8BlockScaledMMKernel | 2.9e-03 | 2.6e-02 | 3.2e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 1 | CutlassFP8ScaledMMLinearKernel | 2.7e-03 | 2.6e-02 | 2.8e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 8 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.7e-02 | 5.2e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 32 | CutlassFP8ScaledMMLinearKernel | 3.9e-03 | 2.6e-02 | 3.8e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qkv_sliding | 64 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.6e-02 | 4.2e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 1 | CutlassFP8ScaledMMLinearKernel | 3.0e-03 | 2.7e-02 | 2.7e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 8 | CutlassFP8ScaledMMLinearKernel | 3.8e-03 | 2.6e-02 | 3.9e-04 | 1.6e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 32 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.7e-02 | 4.0e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | qk_global | 64 | CutlassFP8ScaledMMLinearKernel | 3.7e-03 | 2.7e-02 | 4.9e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 1 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.6e-02 | 2.4e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 8 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.7e-02 | 2.9e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 32 | CutlassFP8ScaledMMLinearKernel | 2.8e-03 | 2.7e-02 | 3.2e-04 | 1.7e-03 |
| gemma-4-26B-A4B-it-FP8-dynamic | o_sliding | 64 | CutlassFP8ScaledMMLinearKernel | 3.2e-03 | 2.6e-02 | 3.2e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 1 | FlashInferFP8ScaledMMLinearKernel | - | - | 2.5e-04 | 1.6e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 8 | FlashInferFP8ScaledMMLinearKernel | 3.4e-03 | 2.6e-02 | 3.4e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 32 | FlashInferFP8ScaledMMLinearKernel | 3.5e-03 | 2.6e-02 | 3.8e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | qkv_a | 64 | FlashInferFP8ScaledMMLinearKernel | 3.8e-03 | 2.7e-02 | 3.7e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 1 | FlashInferFP8ScaledMMLinearKernel | 7.5e-03 | 2.6e-02 | 6.4e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 8 | FlashInferFP8ScaledMMLinearKernel | 7.4e-03 | 2.7e-02 | 7.0e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 32 | FlashInferFP8ScaledMMLinearKernel | 7.8e-03 | 2.7e-02 | 8.6e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | q_b | 64 | FlashInferFP8ScaledMMLinearKernel | 8.2e-03 | 2.7e-02 | 9.2e-04 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 1 | FlashInferFP8ScaledMMLinearKernel | 1.2e-02 | 2.8e-02 | 1.3e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 8 | FlashInferFP8ScaledMMLinearKernel | 1.3e-02 | 2.7e-02 | 1.3e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 32 | FlashInferFP8ScaledMMLinearKernel | 1.2e-02 | 2.6e-02 | 1.3e-03 | 1.7e-03 |
| Mistral-Small-4-119B-2603 | kv_b | 64 | FlashInferFP8ScaledMMLinearKernel | 1.4e-02 | 2.6e-02 | 1.6e-03 | 1.7e-03 |
